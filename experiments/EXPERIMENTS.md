# Experiment guide

Each linked `experiment.toml` is the authoritative input: source commit, image, node count, native arguments, environment and procedure.
This guide explains purpose and interpretation; the root [README](../README.md) covers setup, execution and cache management.
Run commands below from the repository root, with `SITE` and the appropriate checkout configured as described there.
Directory names combine a readable workload/variant label with `-m<12-character Megatron commit>-i<12-character image SHA-256>`; image-native workloads omit the Megatron suffix.
Existing run artifacts retain the experiment names recorded when they were created, including the earlier numbered names.

## Dense restart checks

| Experiment | Purpose |
|---|---|
| [dense-baseline-mc56351143a8e-ie84d7ba62309](dense-baseline-mc56351143a8e-ie84d7ba62309/experiment.toml) | Small one-node training and exact checkpoint/restart check |
| [dense-multinode-mc56351143a8e-ie84d7ba62309](dense-multinode-mc56351143a8e-ie84d7ba62309/experiment.toml) | Two-node version of the dense restart check |

Both use BF16, mock data, ordinary dense attention and AdamW, with continuous/prefix/resumed trials.
Both explicitly select [validation/dense_restart.py](validation/dense_restart.py), which owns their instrumentation and numerical checks.
They are correctness smoke tests, not convergence or performance benchmarks.

## Distributed collectives

[distributed-collectives-i0662f3a37ef3](distributed-collectives-i0662f3a37ef3/experiment.toml) runs the image CI's RCCL all-reduce and GPU-aware MPI all-to-all commands on two nodes.
It uses image-provided executables without a Megatron checkout, with a separate 180-second timeout per command.
RCCL output is under `output/0/` and MPI output under `output/1/`.
Inspect RCCL's numerical error count and complete size sweep separately from command exits and MPI timings.

## Mid-4B MoE baseline

[moe-mid4b-mc56351143a8e-ie84d7ba62309](moe-mid4b-mc56351143a8e-ie84d7ba62309/experiment.toml) selects prepared SwissAI Megatron source and the published Primus 26.8 Apertus application image.
It uses full Mid-4B geometry: 32 layers, hidden size 2560, 256 experts with top-8 routing, sequence length 8192, and a repeating three-KDA/one-global-attention pattern.
The first layer is dense; the remaining 31 are MoE.
The initial run uses mock data, the SwissAI magnitude/direction-decoupled optimizer, and 26 updates with a final checkpoint.
It retains the 718-update learning-rate warmup, so this run stays within warmup.

The experiment defaults to 16 nodes / 64 GPUs, using TP1/PP2/CP1/EP8/ETP1 and global batch 1152: DP32 with 36 microbatches per update.
Four nodes is the minimum rank count for this layout, not an established memory-capacity result.
Reuse a checkout containing the pinned commit:

```bash
uv run --locked apertus plan experiments/moe-mid4b-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit
uv run --locked apertus run experiments/moe-mid4b-mc56351143a8e-ie84d7ba62309 \
  --site "$SITE" --submit --checkout "$MEGATRON"
```

This uses ordinary `[procedure] kind = "train"`; it has no numerical or restart validator.
Configuration checks do not establish full-shape GPU correctness, convergence, or throughput.

## Early communicator warmup comparison

[moe-mid4b-comm-warmup-m1072583a9632-ie84d7ba62309](moe-mid4b-comm-warmup-m1072583a9632-ie84d7ba62309/experiment.toml) selects the fork's backport of Imanol's communicator warmup onto the baseline SwissAI revision.
It retains the baseline model, batch, 26-update protocol and image, and enables early collective/pipeline/expert connections with `BEVERIN_COMM_WARMUP=1`, `BEVERIN_COMM_WARMUP_PARTS=ar,p2p,a2a` and `NCCL_RUNTIME_CONNECT=0`.
The module's synthetic memory-pressure test is explicitly disabled with `SS_BALLAST_GB=0`.
This is a communication-initialization diagnostic; successful warmup alone does not establish training or checkpoint correctness.
Use the prepared worktree containing the pinned commit:

```bash
uv run --locked apertus run experiments/moe-mid4b-comm-warmup-m1072583a9632-ie84d7ba62309 \
  --site "$SITE" --submit --reservation cscs --checkout ../megatron/comm-warmup
```

## Rebuilt Primus 26.8 Apertus image

These experiments select the rebuilt `megatron-primus268-rocm` application image `cb5a384016a99328a970d823ecf264a45e43b23b41e8e0c73e2baf14c33e77c0` from the shared `apertus-amd-images` store.
They retain the corresponding workload and source settings above, with the dense checks on baseline Megatron and Mid-4B on the communicator-warmup backport.
The image makes reference optimizations opt-in, so these experiments omit the old image's `B2_AUTOTUNE_SKIP=0` startup-hook workaround.
The collective checks deliberately use the same Apertus application image as training.

| Experiment | Nodes | Procedure |
|---|---|---|
| [dense-baseline-mc56351143a8e-icb5a384016a9](dense-baseline-mc56351143a8e-icb5a384016a9/experiment.toml) | 1 | Dense checkpoint/restart check |
| [dense-multinode-mc56351143a8e-icb5a384016a9](dense-multinode-mc56351143a8e-icb5a384016a9/experiment.toml) | 2 | Distributed dense checkpoint/restart check |
| [distributed-collectives-icb5a384016a9](distributed-collectives-icb5a384016a9/experiment.toml) | 2 | RCCL all-reduce and GPU-aware MPI all-to-all |
| [moe-mid4b-comm-warmup-m1072583a9632-icb5a384016a9](moe-mid4b-comm-warmup-m1072583a9632-icb5a384016a9/experiment.toml) | 16 | Full-shape Mid-4B training with communicator warmup |

```bash
uv run --locked apertus run experiments/moe-mid4b-comm-warmup-m1072583a9632-icb5a384016a9 \
  --site "$SITE" --submit --reservation cscs --checkout ../megatron/comm-warmup
```

Image build/publication checks establish artifact integrity; use these experiments to assess GPU execution, communication and workload-specific correctness.

## Mid-4B allocator and compilation comparison

These three 16-node variants retain the communicator-warmup source, rebuilt image, and 26-update workload.
They make the two historical memory-related defaults explicit to separate allocator behavior from compilation.

| Experiment | Expandable segments | Disable `torch.compile` | Disable P2P |
|---|---|---|---|
| [moe-mid4b-mem-both-m1072583a9632-icb5a384016a9](moe-mid4b-mem-both-m1072583a9632-icb5a384016a9/experiment.toml) | True | 1 | 0 (site default) |
| [moe-mid4b-mem-alloc-m1072583a9632-icb5a384016a9](moe-mid4b-mem-alloc-m1072583a9632-icb5a384016a9/experiment.toml) | True | 0 | 1 |
| [moe-mid4b-mem-eager-m1072583a9632-icb5a384016a9](moe-mid4b-mem-eager-m1072583a9632-icb5a384016a9/experiment.toml) | False | 1 | 1 |

Use `--cache-mode cold` for each comparison arm.
The supplied site configuration enables RCCL P2P with `NCCL_P2P_DISABLE=0` and excludes Open MPI's memory patcher with `OMPI_MCA_memory=^patcher` to avoid its HSA/libfabric lock inversion during IPC setup.
The allocator-only and compile-disable-only arms explicitly disable P2P and should be compared with the combined-settings P2P-disabled control below.
All retain the Open MPI patcher exclusion.
All select [validation/memory_trace.py](validation/memory_trace.py), which samples process memory, available cgroup counters, GPU allocator counters, and node/shared-memory usage every ten seconds, with private-cache sizes every minute.
Newly frozen runs also retain node-level buddyinfo and the base page size every ten seconds, without allocating memory or treating fragmentation as a run-admission check.
It preserves ordinary verified source execution and does not synchronize GPU work or reset memory peaks.
The `memory-start-<rank>.json`, `memory-<rank>.jsonl`, and `memory-end-<rank>.json` artifacts retain effective settings and memory evidence, including partial traces after a killed process.
Its final check verifies telemetry completeness and configured settings; it does not validate numerical equivalence or checkpoint contents.
The prior uninstrumented new-image run is a historical comparison, not a simultaneously measured fourth arm.

[moe-mid4b-mem-nop2p-m1072583a9632-icb5a384016a9](moe-mid4b-mem-nop2p-m1072583a9632-icb5a384016a9/experiment.toml) is the P2P-disabled control for the combined-settings arm.
It overrides the site's P2P setting with `NCCL_P2P_DISABLE=1`, retaining expandable segments, disabled `torch.compile`, the Open MPI patcher exclusion, and identical memory instrumentation.
Run it with `--cache-mode cold` to match the combined-settings comparison.

## Compilation and performance work

[moe-mid4b-compile-m1072583a9632-icb5a384016a9](moe-mid4b-compile-m1072583a9632-icb5a384016a9/experiment.toml) retains the allocator-only workload with expandable segments, compilation enabled, and P2P disabled.
It adds `TORCH_LOGS=recompiles,graph_breaks` and opt-in per-rank compiler telemetry through `APERTUS_COMPILE_TRACE=1`.
The memory validator records Dynamo counters and Inductor's generated-kernel count before training, after each native `train_step`, and on exit in `compilation-<rank>.jsonl`.
It neither resets compiler counters nor synchronizes GPU work.
These version-specific counters establish compilation activity and when it changes; they do not establish numerical equivalence.
Run identical inputs with `--cache-mode cold`, then `--cache-mode warm` after successful publication.
Warm mode restores portable Triton objects, AITER modules, and dataset helpers; it does not restore Inductor graph caches or autotuning state.
Compare early-step compilation and late-step throughput separately.

The [performance plan](PERFORMANCE.md) covers the Mid-4B reference comparison and the separate Chonk baseline.

### Advisory node observations

Use `uv run --locked python -B -m apertus_amd.node_screen survey --reservation cscs` from a shell without an inherited allocation context for a bounded, immediate CPU-only survey.
It records raw buddy orders, GPU-local NUMA topology/memory, uptime and skipped busy/unreachable nodes under `artifacts/investigations/performance-next/node-screening/`.
The survey never waits for ideal nodes or controls benchmark admission; refresh it periodically without rerunning historical benchmarks.
Inside an explicitly owned CPU-only scheduler step, `python -B -m apertus_amd.node_screen sample --samples 3 --interval 10` emits bounded periodic snapshots instead.
Do not use intrusive allocation-based capacity probes on nodes concurrently running benchmarks.

### Independent fusion and recomputation trials

[moe-chonk-kda-gaf-m1072583a9632-icb5a384016a9](moe-chonk-kda-gaf-m1072583a9632-icb5a384016a9/experiment.toml) removes only the baseline's `--no-gradient-accumulation-fusion` argument, selecting native enabled accumulation fusion.
[moe-chonk-kda-moeact-m1072583a9632-icb5a384016a9](moe-chonk-kda-moeact-m1072583a9632-icb5a384016a9/experiment.toml) replaces only `moe` with `moe_act` in selective recomputation, retaining `layernorm linear_attn` and disabled fusion.
Both retain the baseline's P2P-disabled environment and 60-update measurement/checkpoint duration, making them independent of PR #73 restoration.
The args-only changes share the existing cache compatibility namespace; this does not guarantee reuse of new specializations or arbitrary Apex Torch-extension builds.
The optional [grouped-linear probe](validation/grouped_linear_probe.py) checks selected image backend forward/input gradients and multi-microbatch accumulated weight gradients against FP32 references.
[chonk-grouped-linear-probe-icb5a384016a9](chonk-grouped-linear-probe-icb5a384016a9/experiment.toml) runs that frozen script as a bounded native command on one node, with a private temporary compiler cache per rank.
Its per-rank numerical reports are under `output/0/grouped-linear-probe-<rank>.json`; inspect these separately from scheduler exits.
It is not a full-model, routing, SSSGLU, recomputation, optimizer or restart-equivalence validator.
Training telemetry alone does not establish numerical equivalence or justify accepting either optimization.

[chonk-apex-wgrad-probe-icb5a384016a9](chonk-apex-wgrad-probe-icb5a384016a9/experiment.toml) independently tests the installed native Apex FP32-accumulating weight-gradient operation used by the TP1 LM head.
It separates private extension JIT/load, CPU call duration, synchronized GPU timing and additive-reference checks for a small control followed by the exact input4096x3584/output-gradient4096x200064 shape.
Each rank makes three native calls per shape, including nonzero initialization and cancellation, with tiled FP32 references and a 600-second command bound.
Incremental `output/0/apex-wgrad-probe-<rank>.jsonl` and live JSON retain the last phase if an operation or reference stalls.
The probe leaves Apex's audited workspace handling unchanged and does not establish distributed optimizer, stream-lifetime or full-model correctness.

[moe-recompute-probe-m1072583a9632-icb5a384016a9](moe-recompute-probe-m1072583a9632-icb5a384016a9/experiment.toml) is a four-node/16-GPU single-MoELayer component proxy, not full Chonk training.
It retains actual Chonk expert/latent/hidden widths, EP16 and native TEGroupedMLP/SSSGLU/histogram routing, but uses PP1 and two microbatches per native recomputation mode.
Its explicit validator initializes and verifies committed Megatron imports, constructs layer 4 only, and compares `moe` with `moe_act` from identical weights, inputs and RNG.
It checks outputs, input/all-parameter gradients, local FP32 accumulation of BF16 leaf gradients, routing/state/RNG and once-per-microbatch accounting.
The pinned `pretrain_gpt.py` is verified but not executed; no GPT model, optimizer, DDP reduction, beta-finalization, checkpoint or restart equivalence is claimed.
Per-rank numerical evidence is `output/moe-recompute-<rank>.json` and JSONL; all ranks must pass the explicit checker before cache publication.
The diagnostic reports also retain native original/replay/candidate logits/routing fingerprints, pre/post-backward output comparisons and bounded raw rows for routing differences.
Detached top8/top9-prefix controls distinguish boundary-tie branch differences; they do not replace native routing or relax its equivalence gates.
Diagnostic snapshots and hashing can synchronize execution and are not performance instrumentation.

[moe-recompute-probe-me242f3c3c113-icb5a384016a9](moe-recompute-probe-me242f3c3c113-icb5a384016a9/experiment.toml) changes only the source pin to the user-committed histogram checkpoint-routing fix.
It retains the original EP16 proxy workload, image, compilation, disabled P2P, numerical tolerances and exact routing/state/once-only accounting gates.
Diagnostic branch metadata follows the selected committed source; the observer does not replace native routing.
This experiment provides component validation, not full-model correctness or an accepted recomputation optimization.
Tied-boundary regression and inference/non-histogram compatibility remain separate acceptance requirements.

[moe-chonk-kda-me242f3c3c113-icb5a384016a9](moe-chonk-kda-me242f3c3c113-icb5a384016a9/experiment.toml) and [moe-chonk-kda-moeact-me242f3c3c113-icb5a384016a9](moe-chonk-kda-moeact-me242f3c3c113-icb5a384016a9/experiment.toml) prepare the full 60-update recomputation comparison with the same committed routing fix on both sides.
They preserve the original default compiler policy, disabled P2P, disabled accumulation fusion and preselected update windows; the pair differs only by `moe` versus `moe_act`.
Do not interpret corrected-versus-historical training differences as a pure recomputation speedup.
Preparation and pure planning do not authorize these runs or establish full-model numerical correctness; compiler-control and repetition-protocol review remain separate gates.

[moe-chonk-kda-me242f3c3c113-i910bfcf936d3](moe-chonk-kda-me242f3c3c113-i910bfcf936d3/experiment.toml) changes only the published image identity for the full16-node/64-GPU ordinary-alltoall baseline workload.
It keeps full-MoE recomputation, default compilation, P2P disabled and accumulation fusion off, with60updates/checkpoint60 and AUTOcache in the new image namespace.
It tests installed-image integration and quiescent seed/cache publication, not matched-warm image speedup, native UCCL equivalence, checkpoint tensor integrity or production readiness.

### PR #73: restore P2P without the disable workaround

[collectives-pr73-icb5a384016a9](collectives-pr73-icb5a384016a9/experiment.toml) tests the complete relevant PR #73 settings on the existing newer Primus 26.8 image, explicitly setting `NCCL_P2P_DISABLE=0`.
It runs checked RCCL all-reduce, all-to-all, variable all-to-all and repeated send/receive size sweeps in separate bounded steps.
Default two-node execution exercises intra-node IPC and inter-node CXI; use `--nodes 1` for the separate intra-node arm.
Inspect numerical error counts, actual P2P connections, final exits and effective environment before proceeding to 16-node Chonk.
Collective success alone does not resolve the previous full-workload memory-pressure stall or establish a throughput gain.
The objective is reliable full-model training without needing `NCCL_P2P_DISABLE=1`; individual flag ablations follow only if diagnosis or simplification requires them.

[moe-chonk-kda-pr73-m1072583a9632-icb5a384016a9](moe-chonk-kda-pr73-m1072583a9632-icb5a384016a9/experiment.toml) retains the full 60-update KDA baseline and changes only the relevant network environment.
Run it after collective correctness passes, retaining ordinary full-MoE recomputation and disabled accumulation fusion so the P2P restoration is assessed independently.
Record actual IPC connections, memory/progress, checkpoints and finalization separately; do not accept a stalled run just because the synthetic collectives passed.

[moe-chonk-kda-pr73act-m1072583a9632-icb5a384016a9](moe-chonk-kda-pr73act-m1072583a9632-icb5a384016a9/experiment.toml) is the short 26-update follow-up that replaces only whole-MoE recomputation with `moe_act`, preserving the PR #73 environment and full geometry.
It isolates the interaction with backward recomputation seen in the stalled PR #73 trial; it is not a 60-update performance repetition or an accepted combined optimization.
The independent `moeact` experiment above uses the original baseline environment and is not its exact single-variable P2P control.

[moe-chonk-kda-pr73act-nop2p-m1072583a9632-icb5a384016a9](moe-chonk-kda-pr73act-nop2p-m1072583a9632-icb5a384016a9/experiment.toml) is the exact 26-update matched control, changing only `NCCL_P2P_DISABLE=1` from `pr73act`.
All other PR #73 flags, recomputation, geometry, compilation and duration are identical, isolating a P2P-dependent interaction without attributing it to a particular transport operation.
[moe-chonk-kda-pr73-eager-m1072583a9632-icb5a384016a9](moe-chonk-kda-pr73-eager-m1072583a9632-icb5a384016a9/experiment.toml) separately retains PR #73/full-MoE settings with `TORCH_COMPILE_DISABLE=1` and a documented shorter 26-update train/save duration.
It tests a compilation interaction; it is not a performance comparison or proof that every kernel path is eager.

[sssglu-compile-probe-m1072583a9632-icb5a384016a9](sssglu-compile-probe-m1072583a9632-icb5a384016a9/experiment.toml) independently checks an explicit experimental Inductor single-candidate policy against tiled FP64 references for native SSSGLU forward/input/probability gradients.
It uses one node/four independent GPUs, cold empty caches, nine dense/shared/expert contiguous/strided cases and repeated direct/custom-autograd calls without executing GPT, process groups or an optimizer.
The checker requires compiled-launch observations, singleton candidates and no benchmark calls as well as fixed numerical gates; its900-second watchdog is separate from the scheduler bound.
[moe-chonk-kda-pr73-sc-m1072583a9632-icb5a384016a9](moe-chonk-kda-pr73-sc-m1072583a9632-icb5a384016a9/experiment.toml) is the full 26-update counterpart, retaining native compilation and selecting `APERTUS_INDUCTOR_POLICY=single_candidate_backend` relative to the compiled PR #73 full-MoE geometry/settings at the shorter duration.
The selected validator applies and records this policy before native imports, while an absent policy preserves baseline compiler behavior.
Component checks must pass before the full diagnostic is launched; neither establishes recomputation/router equivalence or justifies accepting changed reduction rounding/performance without further checks.

[moe-chonk-kda-pr73-sc-gaf-profile-m1072583a9632-icb5a384016a9](moe-chonk-kda-pr73-sc-gaf-profile-m1072583a9632-icb5a384016a9/experiment.toml) enables native gradient-accumulation fusion on that same full-model registered-backend configuration and adds first-update CPU/GPU profiling on ranks 0/16/32/48.
Its purpose is startup/backward attribution, not throughput measurement or numerical-equivalence acceptance.
The native profiler exports after the first update returns and therefore does not guarantee a durable partial GPU trace during a hung backward; its later NVTX instrumentation also remains native and is not claimed bounded.
Use a predeclared first-update decision deadline, inspect actual trace coverage/correlations and retain compiler/checkpoint/finalization outcomes independently.

[chonk-kda-projection-tune-icb5a384016a9](chonk-kda-projection-tune-icb5a384016a9/experiment.toml) isolates the unchanged source-derived 4096×3584→11036 KDA projection through native TE forward, dgrad and FP32 accumulated wgrad, with a separately labeled unfused control.
It uses disposable TE tuning outputs, fresh-process default/replay phases, independent numerical references and exact native cache/selected-algorithm consumption gates; it neither changes vendor libraries nor validates a captured full-training layout.
The default `installed-sources` mode rejects missing or drifted native sources; the explicit experiment instead selects `pinned-image-build-provenance`, checking immutable image/build-info/ELF identities and actual mapped TE libraries without claiming source-to-binary attestation.
Both modes retain the same numerical and table-consumption gates, and numerical failure stops before tuning or timing acceptance.
Component timing, cache consumption and numerical success remain separate from full-model equivalence or training-speedup acceptance.

[chonk-kda-projection-forward-i910bfcf936d3](chonk-kda-projection-forward-i910bfcf936d3/experiment.toml) separately qualifies only the default native TE forward for that source-derived projection on the combined image.
Each independent local GPU rank uses fresh descriptor-proof and quiet-measurement workers under one shared deadline, checks numerical/provenance/algorithm agreement, and retains every selected timing sample.
Default algorithm identity is observed rather than forced; this does not qualify tuning, gradients, accumulated wgrad or production layouts, and does not waive the full-projection control's failure.

[chonk-kda-projection-failure-capture-icb5a384016a9](chonk-kda-projection-failure-capture-icb5a384016a9/experiment.toml) stops at the projection probe's default-prefix unfused accumulation gate, preserving its original numerical failure rather than tuning or timing.
It retains bounded raw BF16 input vectors, native and reference contribution bits, and independent copy/add diagnostics for exact offline analysis without additional native GEMMs.
Select the explicit five-minute `cscs` launch envelope; capture completion or nonreproduction is not numerical acceptance and never authorizes tuning or a retry.

[chonk-te-source-inventory-icb5a384016a9](chonk-te-source-inventory-icb5a384016a9/experiment.toml) inventories fixed TE source paths, distribution records, build metadata and scoped ELF-file identities on the existing immutable image without importing vendor code or using GPUs.
It requires explicit five-minute/2-GiB site overrides; file absence or mismatched provenance is retained as an observation rather than an inventory-execution failure.
Inventory success does not waive the projection probe's source gate, select runtime-loaded libraries or attest binary/source correspondence.

[chonk-grouped-kernel-source-inventory-i910bfcf936d3](chonk-grouped-kernel-source-inventory-i910bfcf936d3/experiment.toml) retrieves bounded installed TE grouped-linear/Triton Python source and distribution metadata without vendor imports or GPU calls.
Known caller hashes, stable regular-file reads, selected wheel-RECORD correspondence and four-rank agreement gate later source export and API review.
It addresses grouped-backend source readiness, not kernel execution, numerical/performance acceptance or actual production EP16 routing/layout capture.

[chonk-grouped-forward-selector-i910bfcf936d3](chonk-grouped-forward-selector-i910bfcf936d3/experiment.toml) independently qualifies the installed public TE GroupedLinear baseline and existing Triton selector on synthetic16-group FC1/FC2 balanced and ragged counts.
Fresh selector workers use identical canonical BF16 operands, natural config=None and complete per-expert FP32-reference element/RMS checks, including empty experts; selected Python-consumer counts are not native-kernel attestation.
This forward-only component does not waive accumulated-wgrad failures or establish production EP16 layouts, backward or performance.

[chonk-grouped-forward-performance-i910bfcf936d3](chonk-grouped-forward-performance-i910bfcf936d3/experiment.toml) prepares matched public-API timing on the same fixed synthetic cases with fresh ABBA/BAAB workers, complete pre/post accuracy and immutable operand checks.
Timing includes ordinary dispatch, wrapper weight-stack/transpose, allocations and output release; compiler/cache observations and branch wrappers stay outside batches.
Its conservative fixed replica-consistency analysis is conditional on the declared schedule, not statistical significance, a universal speedup threshold or a training-throughput claim.
Execution requires separate source/freeze/placement approval under the reviewed600-second allocation,510-second command and480-second shared-worker caps; no automatic retry or extension is selected.

[combined-uccl-aiter-components-i910bfcf936d3](combined-uccl-aiter-components-i910bfcf936d3/experiment.toml) selects the published combined candidate for two serial, bounded two-node/eight-rank component checks: private-copy AITER core reuse/RMSNorm, then exact UCCL dispatch/combine and post-destroy storage/stream/RCCL lifetime.
The selected wrapper pins embedded scripts/seed metadata, preserves four visible devices and chooses the Slurm local GPU before executing each embedded test in the same process.
It does not source the broad reference-optimization environment, change vendor code or publish compiler caches.
The UCCL test uses a deterministic 32-token/hidden-512 proxy; passing it would not establish autograd, the Megatron dispatcher, Chonk geometry, full-model training or performance acceptance.

[uccl-library-ep-i910bfcf936d3](uccl-library-ep-i910bfcf936d3/experiment.toml) runs the pinned UCCL library's own EP tests with their original numerical assertions.
It uses two independent four-GPU intra-node groups, then eight-GPU inter-node, low-latency and dual-buffer construction tests on the same exclusive two-node allocation.
High-throughput cases use4096tokens/hidden7168/top8/256experts; low-latency uses128tokens/hidden7168/top8/288experts; dual-mode uses8tokens/hidden1024/16experts.
The intra-node entry point retains its original tuning tail; inter-node and low-latency call the original correctness functions once through their supported skip_benchmark argument, without entering unbounded pressure loops.
Frozen test/helper sources are copied into the run before submission, with the buffer ownership patch matching the installed image; no library or test assertion is rewritten.
The explicit [CXI repeat](uccl-library-ep-cxi-i910bfcf936d3/experiment.toml) adds UCCL_EP_TRANSPORT=cxi, UCCL_SOCKET_IFNAME=hsn and UCCL_CXI_THREADING=safe, preserving the original experiment and failed run.
The original experiment omitted UCCL's transport selector; FI_PROVIDER=cxi alone does not select its native backend.

The [EP16 all-to-all control](moe-chonk-kda-ep16-alltoall-i910bfcf936d3/experiment.toml) and [native UCCL candidate](moe-chonk-kda-ep16-uccl-i910bfcf936d3/experiment.toml) compare dispatchers in otherwise matched 64-GPU KDA Chonk mock-data training.
Both use the routing fix, image 910, `moe_act` and the same explicit CXI environment; the all-to-all control does not consume the flex backend selector.
These inputs target 60 updates and retain windows 11–26 and 45–60; they do not by themselves authorize an allocation.
Each command has a360-second limit under an explicit30-minute cscs allocation, with private retained caches and an initial all-rank guard refusing pre-existing EP shared-memory barrier files.
Upstream similarity/error checks are distinct from the Megatron elementwise/RMS comparison; success or failure does not retrospectively change that comparison or establish full-model correctness/performance.

[combined-aiter-attention-i910bfcf936d3](combined-aiter-attention-i910bfcf936d3/experiment.toml) separately checks direct AITER causal BF16 attention forward/backward at `[1,128,2,128]`, without selecting the bundled Turbo/FP8/reference environment.
It preserves the candidate's existing forward tolerance and pinned backward policy against CPU FP64 SDPA plus an independent BF16 math control, retaining original elementwise gradient mismatch counts.
Each rank selects its local GPU and uses private JIT caches; passing this small attention component would not validate the complete reference Turbo stack, native dispatcher or full training geometry.

The [Turbo CK FC1 component](chonk-turbo-ck-forward-i910bfcf936d3/experiment.toml) compares explicit Turbo CK against the default TE grouped path on synthetic Chonk-shaped BF16 forward cases.
The candidate includes copying the separate expert weights on every call; the result does not measure FC2, backward, or training.

[chonk-kda-attention-shape-i910bfcf936d3](chonk-kda-attention-shape-i910bfcf936d3/experiment.toml) qualifies nominal B1/S4096/Q28/KV7/D128 BF16 causal attention through baseline-enabled TE and the public direct AITER API, without timing or selecting a production optimization.
It retains one monolithic full-head BF16 MATH GQA control and original whole-Q/K/V gradient gates; only independent FP32/FP64 CPU references are tiled.
An installed-Torch tiny self-check and read-only effective precision check precede GPU tensor creation; source pins and planning-memory guards do not establish native peak fit, correctness or binary/source correspondence.
Workers use private caches under the unchanged command supervisor, and failed scratch is retained rather than harvested or published.

[chonk-output-gate-qualification-i910bfcf936d3](chonk-output-gate-qualification-i910bfcf936d3/experiment.toml) prepares an isolated forward/dx/dgate check of the repaired output-gate helper against the unchanged default-compiled e242 method and independent FP64 formulas with final BF16 casts.
It embeds the hash-verified applied helper and minimally adapted original method, not a live Megatron checkout or a patched training arm.
Actual global-attention shapes, strided offset views, short/tail cases and saturated/near-zero gates use full-tensor element/RMS checks plus informational numeric and raw-bit BF16 mismatch/ULP diagnostics.
The pointwise thresholds remain proposed pending explicit component-policy approval; CPU/source checks do not establish native accuracy, production flag dispatch, checkpoint/recompute interaction or performance.
Private compiler caches remain job-owned and retained until a separate quiescent cleanup decision; no cache publication or full-model source-provenance shortcut is selected.

[chonk-output-gate-performance-i910bfcf936d3](chonk-output-gate-performance-i910bfcf936d3/experiment.toml) times the same isolated helper against the unchanged default-compiled reference with fixed operands, full pre/post numerics and matched ABBA/BAAB autograd batches.
Its primary synchronized host-completion clock includes Python/autograd/copy/allocation costs; the event span is not summed kernel time.
All raw samples, memory peaks and readonly compiler/cache proof remain authoritative even when the component-specific performance rule does not qualify a gain; neither a valid measurement nor numerical success authorizes image promotion.

[chonk-sssglu-qualification-i910bfcf936d3](chonk-sssglu-qualification-i910bfcf936d3/experiment.toml) checks the isolated default-off same-equation SSSGLU patch using embedded exact base/applied/helper sources and declared default-compile/disabled-NVTX import adapters.
Fresh plain and weighted workers cover D1920 nominal4096/32768rows plus129/1/0tails, a nominal3D shared input and an offset/strided weighted upstream gradient.
Output, both input-gradient halves and FP32 probability gradients use complete tiled FP64 references and unchanged element rtol0.015625/atol0.00390625 plus RMS rtol0.015625/atol1e-5 gates against both arms and their compiled baseline.
Default-off/on/fallback dispatch, compiler/cache evidence, immutable operands and strict closed worker/rank records are required; no timing, production imports, selective checkpoint/recompute, optimizer or full-model equivalence is established.
Private caches/source exports are retained under600-second allocation/510-second command/480-second shared-worker caps, with separate one-shot frozen-input and fresh-placement approval required before execution.

[combined-aiter-attention-parallel-i910bfcf936d3](combined-aiter-attention-parallel-i910bfcf936d3/experiment.toml) retains that attention workload and numerical policy while changing only private-build concurrency from two to eight workers per rank and enabling verbose AITER build logs.
It rejects unexpected prebuild-thread division and Ninja-consumer source drift, retains the original 720-second command/15-minute allocation bounds, and does not promise compilation completion or authorize automatic retries.

[combined-aiter-attention-workers16-i910bfcf936d3](combined-aiter-attention-workers16-i910bfcf936d3/experiment.toml) is a separately reviewed sixteen-worker compiler-concurrency candidate after the eight-worker forward build exhausted most of the unchanged command budget.
It preserves the numerical workload, source/image/seed pins, private-cache ownership and 720-second/15-minute limits; CPU checks do not establish its memory peak or GPU numerical acceptance.

[native-dispatcher-me242f3c3c113-i910bfcf936d3](native-dispatcher-me242f3c3c113-i910bfcf936d3/experiment.toml) compares native ordinary-alltoall and flex/deepep MoELayers through two full-MoE-recomputed forward/backward microbatches on two nodes/eight GPUs.
It retains Chonk hidden3584/latent1792/expert1920 geometry, real routing and experts, all parameter/input/probability gradients, explicit local FP32 gradient sums, metadata coverage and post-buffer-destruction storage/RCCL checks.
The source-staged train/check lifecycle uses cold private caches and ordinary checked publication; a direct-exec GNU foreground timeout bounds the worker to660seconds in the outer supervisor's process group under a separately selected30-minute `cscs` allocation.
EP8 success would not establish production PP4/VPP2/EP16 geometry, Megatron DDP/main_grad equivalence, full-model correctness or throughput acceptance.

[native-dispatcher-combine8-me242f3c3c113-i910bfcf936d3](native-dispatcher-combine8-me242f3c3c113-i910bfcf936d3/experiment.toml) preserves that native component and numerical policy while explicitly selecting the supported EP8 combine configuration `20,4,256,8,128`.
It records and checks effective configuration getters before hints and after work, rejects dispatch overrides, and tests the source-derived send6→8 correction without patching the library or silently applying it to production EP16 defaults.

[native-dispatcher-boundary-me242f3c3c113-i910bfcf936d3](native-dispatcher-boundary-me242f3c3c113-i910bfcf936d3/experiment.toml) preserves every combine8 numerical gate while observing canonical source-oriented latent-projection inputs, outputs and replay gradients.
It retains raw bits for64 deterministic source tokens per boundary in owned snapshots capped at64MiB/rank, serializing only after all native backwards at the existing quiescent boundary.
Capture completion is diagnostic evidence, not numerical acceptance; contributor reconstruction remains blocked by unproved permutation/source-token maps, and no failed result or tolerance is changed.

[chonk-profiler-inventory-icb5a384016a9](chonk-profiler-inventory-icb5a384016a9/experiment.toml) reads bounded selected ROCm profiler file metadata and parser text on the baseline image without invoking a collector, importing vendor libraries or using GPUs.
Its explicit three-minute `cscs` envelope and scoped absence observations establish neither global tool availability nor safe partial trace export, GPU correctness or fusion attribution.

[chonk-profiler-sdk-inventory-icb5a384016a9](chonk-profiler-sdk-inventory-icb5a384016a9/experiment.toml) selects three literal packaged-SDK roots and eight explicit filenames per root instead of assuming installation under `/opt/rocm*`.
It retains bounded parser/header/version metadata without invoking help/version commands, native tools, vendor imports or GPUs; source identity and parser options do not prove native attach/flush safety or trace correlation.

[moe-mid4b-profile-m1072583a9632-icb5a384016a9](moe-mid4b-profile-m1072583a9632-icb5a384016a9/experiment.toml) adds only native PyTorch-profiler arguments to the compilation diagnostic.
It profiles one update after step 10 on ranks 0/1 and 32/33, covering both pipeline stages and local-rank-0 versus local-rank-1 behavior.
Traces are retained under `output/torch_profile/`; use unprofiled runs for throughput comparisons.

## Swiss-AI Chonk alignment and native FNUZ qualification

The [resident preparation](chonk-swissai-resident-preparation-i910bfcf936d3/experiment.toml) pins image 910 and e242 with the corrected GH200 Chonk benchmark workload.
The runner refuses to launch it.
That preparation does not implement resident FP8 arithmetic, and its proposed BF16 path is not a reference-equivalent substitute.
See the [source and capability assessment](chonk-swissai-resident-preparation-i910bfcf936d3/ALIGNMENT.md) and [separate qualification plan](chonk-swissai-resident-preparation-i910bfcf936d3/QUALIFICATION.md).

The [native FNUZ expert probe](chonk-native-fnuz-experts-i910bfcf936d3/experiment.toml) uses image 910 with a signed Megatron source gate and exercises all six expert GEMMs independently of communication.
The [native FNUZ codec](chonk-native-fnuz-codec-i910bfcf936d3/experiment.toml) and [EP16/CXI transport](chonk-native-fnuz-ep16-i910bfcf936d3/experiment.toml) probes qualify the wire and BF16 reverse path without expert GEMMs.
The expert and transport GPU component gates passed independently.
The [combined schedule rehearsal](chonk-native-fnuz-schedule-i910bfcf936d3/experiment.toml) is a launch-blocked, eight-node PP2/VPP2 gate that keeps EP16 and actual Chonk GEMM and transport widths.
The [full-layout fit probe](chonk-native-fnuz-fit-i910bfcf936d3/experiment.toml) is a separately launch-blocked 16-node cold run with two full-size updates and a terminal synchronous checkpoint, without shared-cache publication.
The [OOM diagnostic](chonk-native-fnuz-oom-i910bfcf936d3/experiment.toml) preserves the full workload for one update and streams pre-failure memory and schedule observations without checkpoint I/O.
The [chunked LM-head component](chonk-chunked-lm-head-i910bfcf936d3/experiment.toml) passed loss, gradient, and Torch allocation gates at the actual Chonk head shape on one gfx942 rank.
The [chunked full-layout fit probe](chonk-native-fnuz-chunked-fit-i910bfcf936d3/experiment.toml) cleared the dense-head OOM boundary in `759718`, but did not complete an update after the second pipeline stage ran out of sampled node memory and peers reported dispatch timeouts.
It is guarded against relaunch.
The [resident FNUZ FC1 replay component](chonk-native-fnuz-fc1-recompute-i910bfcf936d3/experiment.toml) tests recomputing one expert's saved preactivation at a time rather than retaining all preactivations through warmup.
Job `759931` verified the standalone native replay fixture, then failed on a test-counter isolation bug before the lifecycle and allocation gates.
The declaration is guarded against relaunch.
The [corrected FC1 replay component](chonk-native-fnuz-fc1-recompute-check-i910bfcf936d3/experiment.toml) pins the signed test-only fix and keeps the same acceptance gates.
It passed as job `759951`, saving 527 MiB of retained Torch allocation across two actual-shape forwards, and is guarded against relaunch.
The [full-layout FC1 replay fit probe](chonk-native-fnuz-replay-fit-i910bfcf936d3/experiment.toml) adds qualified replay to the chunked-head workload and checks the previously failing forward warmup boundary.
Job `759990` cleared the previous warmup memory boundary, then failed to complete an update after all final-stage ranks hit a native dispatch timeout during backward/forward overlap.
It is guarded against relaunch.
The [full-layout notification diagnostic](chonk-native-fnuz-notify-trace-i910bfcf936d3/experiment.toml) adds scoped, nonblocking GPU markers at that dispatch boundary, without changing the workload or transport timeout.
Job `760048` missed the GPU identity rendezvous on 44 of 64 ranks and was cancelled before training; no dispatch markers were collected.
It is guarded against relaunch.
The [600-second startup retry](chonk-native-fnuz-notify-trace-startup600-i910bfcf936d3/experiment.toml) was cancelled before training as `760061`, with 12 of 64 GPU identities still missing at the extended deadline.
It is guarded against relaunch.
The [full-size startup and dispatch trace](chonk-native-fnuz-notify-startup-trace-i910bfcf936d3/experiment.toml) keeps the same deadlines and adds startup phase markers without skipping training.
Job `760303` was cancelled before training; 24 ranks had started Torch import but had not reached GPU enumeration at the identity deadline.
The trace also measured a 493-second spread in the first in-container Bash marker.
It is guarded against relaunch.
The [cached-image comparison](chonk-native-fnuz-notify-image-cache-i910bfcf936d3/experiment.toml) uses `--cache-image` to mount the same verified SquashFS bytes from the shared checkout, with the same startup and dispatch traces.
Job `761653` on the `cscs` reservation cleared startup but reproduced the call-8 timeout on all 16 final-stage ranks.
The caller and prior-UCCL markers before layout were still incomplete on every rank, as was the layout marker.
No full-size update completed, and the declaration is guarded against relaunch.
The [call-8 dependency trace](chonk-native-fnuz-dependency-trace-i910bfcf936d3/experiment.toml) marks the existing stream waits and reverse BF16 exchange before the same FNUZ dispatch.
Job `761667` reproduced the timeout on all 16 final-stage ranks: compute passed call entry but did not reach the marker before forward preprocessing, and Megatron communication did not pass the backward post-combine wait.
No full-size update completed, and the declaration is guarded against relaunch.
The [call-8 compute prefix trace](chonk-native-fnuz-prefix-trace-i910bfcf936d3/experiment.toml) adds markers for batch/plan preparation, initial loss backward, and the existing forward receive callback.
Job `761767` reproduced the timeout on all 16 final-stage ranks: the compute marker before the forward receive callback completed, the one after it did not, and the callback consumed one handle per rank.
No full-size update completed, and the declaration is guarded against relaunch.
The [matched pipeline P2P trace](chonk-native-fnuz-p2p-pair-trace-i910bfcf936d3/experiment.toml) observes both sides of the stage-2-to-stage-3 transfer for virtual forward 7.
Job `762041` matched all 16 stage-3 receives to their local wait handles, but no stage-2 sender reached the matching `isend` hook before the timeout.
The sender GPU markers cleared the backward receive callback; no full-size update completed.
The declaration is guarded against relaunch.
The retained sender logs show all 16 stage-2 ranks building the same native CK attention-backward kernel, with compiler descendants still sampled just before the receiver timeout in `762041`.
Cold attention-backward compilation is the leading explanation for the missing send; exact-shape prewarming has not yet been qualified.
The [one-node CK backward reuse check](chonk-ck-attention-bwd-reuse-i910bfcf936d3/experiment.toml) requires a cold native B2/S4096 build, finite gradients, and reuse of the verified object in a fresh private cache.
Job `762106` passed: native CK backward took 28.530 s cold and 26.651 ms with the verified object in a fresh private cache, with identical finite output and gradient norms.
No shared cache was published; the result does not establish full-layout fit or cache reuse.
The declaration is guarded against relaunch.
The [full-layout CK prewarm diagnostic](chonk-native-fnuz-ck-prewarm-i910bfcf936d3/experiment.toml) verifies the exact qualified object before seeding each rank privately and retesting the matched P2P boundary.
Job `762207` was submitted once on the `cscs` reservation with a one-time approved overlap with the one-node export `761959`.
All 16 builders reported compilation success but failed the pinned artifact check before training; the effect of prewarming on the call-8 stall remains untested.
The declaration is guarded against relaunch.
The [independent CK build comparison](chonk-ck-attention-bwd-compare-i910bfcf936d3/experiment.toml) records two cold object identities on one node with four GPUs visible, then verifies private-copy reuse of each qualified build in a fresh process.
Job `762221` passed: both same-node builds had the same key and size but different SHA-256 digests, and both verified copies were reused without a rebuild.
The declaration is guarded against relaunch, and the full-layout prewarm gate remains unchanged.
The separate [node-verified full-layout retest](chonk-native-fnuz-ck-node-verified-i910bfcf936d3/experiment.toml) qualifies each node's cold build in a fresh warm process and verifies that node's exact bytes in its four private rank caches before testing the same matched P2P boundary.
Job `762323` qualified all 16 node builds and cleared the matched call-8 boundary, but later native forward dispatch timeouts on stage-2 call 10 and stage-3 call 9 prevented any update from completing.
The declaration is guarded against relaunch, and full-layout fit remains unverified.
The [later P2P diagnostic](chonk-native-fnuz-later-p2p-trace-i910bfcf936d3/experiment.toml) adds matched native forward-9, forward-8 and backward-2 observations and GPU markers at both later dispatches, without changing the workload, CK prewarm or deadlines.
Job `762353` recorded the stage-1 forward-9 send after the stage-2 timeout, while stage 3's later forward receive had already completed on GPU when its own dispatch timed out.
No update completed, and the declaration is guarded against relaunch.
The [later phase diagnostic](chonk-native-fnuz-later-phase-trace-i910bfcf936d3/experiment.toml) adds bounded stage-1 host and compiler timing and final-stage loss-backward, stream-wait and reverse-BF16 markers without changing the workload or deadlines.
Job `762390` localized the late send to long MLP and attention backward host spans, and the final-stage GPU wait to the reverse BF16 dispatch boundary after loss backward and receive completion.
No update completed, and the declaration is guarded against relaunch.
The [BF16 event-order component](chonk-bf16-event-order-i910bfcf936d3/experiment.toml) compares native current-stream event capture before versus after gated BF16 materialization, with a contiguous no-copy control.
The check was submitted once as job `762498` with an explicitly approved one-node overlap with profiling job `762400`, and is guarded against relaunch.
Job `762498` passed: capture before the copy let the consumer marker complete early, capture after the copy held it, and every payload was exact after release.
The component establishes the local ordering gap, not the cause of the full-layout stall.
The [EP16 follow-up](chonk-bf16-event-order-ep16-i910bfcf936d3/experiment.toml) verifies gated capture-after-copy in native cached BF16 dispatch with two outstanding FNUZ routing handles, an ungated contiguous control, and exact BF16 and FP32 reverse-combine checks.
It was submitted once as job `762588` under the normal node cap and is guarded against relaunch; it does not run Megatron or read a held payload.
Job `762588` passed all 32 rank-handle cases on native CXI: gated completion stayed pending until release, and the BF16 and FP32 results were exact.
The component does not establish that the training gradient required materialization or that the event ordering explains the full-layout timeout.
The [Megatron EP16 regression](chonk-native-fnuz-reverse-ready-ep16-i910bfcf936d3/experiment.toml) pins verified signed `041c1d753028e4190efb342accb98a6f362ee7dd` and exercises actual `FusedCombine.backward` with both gradient layouts.
Job `762600` passed all 48 exact cases across 16 native CXI ranks via actual Megatron autograd; Slurm completed in 1:11 with exit `0:0`, and no shared cache was published.
The component is guarded against relaunch and does not establish full-layout fit.
The separate [guarded full-layout retest](chonk-native-fnuz-reverse-ready-i910bfcf936d3/experiment.toml) now clears the Megatron EP16 prerequisite and adds observations of the final-stage BF16 gradient layout, stage-1 receive, and first FC2 dgrad.
The [exact 16-node launch](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#signed-full-layout-reverse-readiness-retest-exact-launch-review) was reviewed against frozen `762390` and submitted once as job `762603` in `cscs`.
Job `762603` cleared call 8 but reproduced both later native dispatch timeouts with no completed update.
The final-stage BF16 gradient was already contiguous on all ranks, and the stage-1 host trace narrowed a 207–208 s gap to before the first FC2 dgrad launch.
The [failed run and traces](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#762603-result-the-signed-readiness-fix-did-not-clear-the-full-layout-timeout) are retained and guarded against relaunch; full-layout fit remains unverified.
The [guarded full-layout stall trace](chonk-native-fnuz-stall-trace-i910bfcf936d3/experiment.toml) splits stage-1 MLP autograd and expert input preparation, and brackets final-stage producer readiness, cached internode dispatch, and the existing consumer wait.
The [exact launch](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#full-layout-stalled-phase-diagnostic-exact-launch-review) preserves the signed source, workload, and deadlines and was submitted once as job `764472`; the guard is restored.
Job `764472` located a 194.92–198.21 s stage-1 gap after layernorm recompute and before resident-expert backward, but timed out at final-stage call 8 before the new call-9 reverse trace.
The [retained result](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#764472-result-stage-1-gap-precedes-resident-backward-final-stage-call-8-timed-out) records the late matching send despite verified CK prewarm and no completed update.
The [guarded handoff trace](chonk-native-fnuz-handoff-trace-i910bfcf936d3/experiment.toml) observes actual stage-1 unpermute and resident autograd nodes and the stage-2 call-8 sender's schedule and callbacks.
The [exact full-layout launch](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#full-layout-autograd-handoff-and-call-8-sender-exact-launch-review) was approved and submitted once as job `764539`; the one-shot guard is restored.
Job `764539` cleared call 8 and located the stage-1 delay inside TE's permutation helper (192.67–195.50 s host time), not the subsequent autograd handoff.
It reproduced both later dispatch timeouts and completed no update; the [retained result](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#764539-result-stage-1-host-delay-is-inside-te-permutation-later-timeouts-recur) separates these observations from the still-unverified cause.
The [600-second host-timeout diagnostic](chonk-native-fnuz-timeout600-i910bfcf936d3/experiment.toml) retains the `764539` full workload and outer deadlines, and samples the existing stage-2 and stage-3 GPU markers during the longer UCCL host wait.
The [exact launch](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#full-layout-600-second-uccl-host-timeout-exact-launch-review) was approved and submitted once as job `764642`; the one-shot guard is restored.
In the [interim trace](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#764642-interim-observation-both-dispatches-cleared-after-the-old-cutoff), both previously failing dispatches returned after about 120–122 seconds and their GPU dependencies completed on all ranks; there was not yet a full-update verdict.
The [late backward drain failed](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#764642-outcome-late-backward-drain-failed-after-both-dispatches-recovered) with HSA queue faults and a rank-60 child SIGKILL; no update completed and an OOM kill was not confirmed.
The [24-layer backward-drain control](chonk-reduced24-native-fnuz-drain-i910bfcf936d3/experiment.toml) keeps the `764642` timeout, batch, parallel degrees and mitigations, with depth reduced and the final backward and optimizer boundaries traced.
Its [exact launch](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#24-layer-backward-drain-control-exact-launch-review) was approved and submitted once as job `764651`; the guard is restored.
The [reduced 24-layer dense-head fit probe](chonk-reduced24-native-fnuz-dense-fit-i910bfcf936d3/experiment.toml) keeps the original full batch and parallel degrees but uses the signed source before output chunking and FC1 replay.
Job `760023` failed the startup GPU identity rendezvous before training; reduced-model fit was not tested.
It is guarded against relaunch and cannot establish full Chonk fit.
The [GPU identity startup-only diagnostic](chonk-reduced24-startup-trace-i910bfcf936d3/experiment.toml) times the same startup phases and stops before training, without increasing the 300-second rendezvous deadline.
Job `760046` passed startup on all 64 ranks; the prior multi-minute skew did not recur and reduced-model fit remains untested.
It is guarded against relaunch.
The [cached-image reduced fit probe](chonk-reduced24-native-fnuz-cached-fit-i910bfcf936d3/experiment.toml) preserves `760023`'s signed 24-layer workload and deadlines, stages the verified image locally, and retains startup tracing through training.
The [exact `cscs` launch](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#cached-image-24-layer-dense-head-fit-exact-launch-review) was approved and submitted once as job `762626`; the one-shot guard is restored.
Job `762626` cleared startup and the first unchunked dense head on every final-stage rank, then timed out in final-stage call 8 before any update completed.
No OOM was logged; the [retained evidence](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md#762626-result-first-dense-head-passed-final-stage-call-8-timed-out) does not establish reduced- or full-model fit.
The [execution record](chonk-swissai-resident-fnuz-i910bfcf936d3/EXECUTION.md) tracks accepted component evidence and the rehearsal review separately from the still-unqualified full-layout fit, restart, cache reuse, and throughput.
The earlier inputs and measurements below are historical and unchanged.

## KDA Chonk baseline: primary target

[moe-chonk-kda-m1072583a9632-icb5a384016a9](moe-chonk-kda-m1072583a9632-icb5a384016a9/experiment.toml) uses the full trained model geometry: 42 layers with three leading dense layers, hidden size 3584, dense FFN 17280, and 256 routed experts/top-8 with expert/shared FFN 1920 and latent size 1792.
It uses 31 KDA / 11 global layers, 28/7 global-attention heads, head dimension 128, sequence length 4096, NoPE, and the production PP4/VPP2 layout.
The benchmark selects 16 nodes / 64 GPUs, EP16, microbatch 1, and global batch 512: DP16 with 32 microbatches per update.
The current production size and `md_decoupling` recipe supply learning rates, polar-express Muon, row hyperspheres, fan-in radii, and softplus gains.
The 409,600-sample warmup and 244,137,984-sample decay horizons become 800 and 476,832 updates at this benchmark batch.
It uses mock data from random initialization, native BF16 expert GEMMs/all-to-all, no CPU offload, no gradient-reduction overlap or gradient-accumulation fusion, and selective `layernorm linear_attn moe` recomputation.
Goldfish loss is disabled for this mock-data baseline because it requires special-token metadata from a real HuggingFace tokenizer.
Compilation, expandable segments, communicator warmup, and memory/compiler telemetry follow the measured Mid-4B baseline; P2P and pipeline-communication overlap are disabled explicitly.
Histogram quantile balancing and per-channel KDA `A_log` are explicit current-stack choices.
The latter matches the current production KDA launcher but differs from the older checkpoint inspection, which reports per-head `A_log`.
This is a full-model throughput/memory baseline with explicit benchmark deviations, not a production-checkpoint or numerical-reproduction test.
The unprofiled baseline runs 60 updates with a final checkpoint, retaining updates 11–26 and preselecting updates 45–60 to assess late timing drift.
Use two warm-cache repetitions for this extended comparison; the initial 26-update invocation retains its frozen inputs in its run directory.

[moe-chonk-kda-profile-m1072583a9632-icb5a384016a9](moe-chonk-kda-profile-m1072583a9632-icb5a384016a9/experiment.toml) retains the initial 26-update duration and adds native PyTorch-profiler arguments.
It profiles one update after step 10 on ranks 0/1, 16/17, 32/33, and 48/49, covering all four pipeline stages and neighboring local ranks.
Use its traces to identify actual Chonk kernel paths and shapes; use the unprofiled baseline for throughput.

## Historical SWA Chonk baseline

[moe-chonk-swa-m1072583a9632-icb5a384016a9](moe-chonk-swa-m1072583a9632-icb5a384016a9/experiment.toml) brings the historical SWA workload logged by reference job 677561 onto our pinned communicator-warmup source and Primus 26.8 image.
This is distinct from the primary KDA Chonk target.
It has 42 layers, hidden size 3584, 256 experts/top-8, sequence length 8192, and sliding/global fused attention.
The explicit layout selects PP4/VPP2 with EP16 on 16 nodes / 64 GPUs, using global batch 256 and microbatch 1.
Its optimizer uses quintic Muon with five Newton–Schulz steps, direct gains, flat hyperspheres, and constant learning rates.
The old `--hypersphere-radius-from-init` spelling maps to `--hypersphere-radius-mode init`, whose radius multiplier remains `sqrt(min(d_out,d_in)/hidden_size)`.
Quantile balancing explicitly selects `legacy_average` to retain the older raw-logit estimator rather than today's histogram default.
The historical expert-bias flag and update rate are omitted: the old quantile-routing path did not consume that separate bias, and the new source rejects combining them.
It retains reference gradient-reduction overlap, default gradient-accumulation fusion, disabled pipeline communication overlap, and compilation disabled.
The baseline uses our explicit P2P-disabled transport setting, expandable segments, communicator warmup, private caches, and memory telemetry.
It adds a final checkpoint and zero data-loader workers to the reference's 26-update mock-data workload.
This establishes a baseline on our stack; historical source/image identity and numerical equivalence to the reference remain unresolved.

## Outcomes and evidence

Each invocation freezes its inputs and records its outcome under `<data-root>/runs/<experiment>/<run>/`.

| Artifact | Meaning |
|---|---|
| `resolved.json` | Exact executed inputs, source/image identity and resource settings |
| `submission.json` | Batch submission receipt and job ID |
| `launch.json` | Actual Slurm launch and controller outcome |
| `result.json` | Execution/check outcomes and cache publication status |
| `output/checks.json` | Numerical/checkpoint validation when an explicit validator provides it |
| `output/rank-*.log`, `output/launcher.log`, `output/slurm.log` | Training, launcher and batch evidence |

Use these records for results rather than maintaining a second pass/fail history in documentation.
Scheduler completion, completed updates, validated checkpoints and numerical equivalence are distinct claims.
