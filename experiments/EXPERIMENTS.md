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

[moe-mid4b-profile-m1072583a9632-icb5a384016a9](moe-mid4b-profile-m1072583a9632-icb5a384016a9/experiment.toml) adds only native PyTorch-profiler arguments to the compilation diagnostic.
It profiles one update after step 10 on ranks 0/1 and 32/33, covering both pipeline stages and local-rank-0 versus local-rank-1 behavior.
Traces are retained under `output/torch_profile/`; use unprofiled runs for throughput comparisons.

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
