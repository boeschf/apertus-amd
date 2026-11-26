# Native FNUZ Chonk execution record

The [brief](../../AMD-CHONK-BRIEF.md) specifies the workload.
The earlier BF16 [settings preparation](../chonk-swissai-resident-preparation-i910bfcf936d3/experiment.toml) is unchanged and is not a baseline.
The historical [backend audit](../chonk-swissai-resident-preparation-i910bfcf936d3/RESIDENT-FP8.md) describes the selected TE list-GEMM route, but its FN transport and saved-activation proposals are superseded.

## Ownership and source admission

All three worktrees are under `/ritom/scratch/cscs/boeschf/megatron/`, based on `e242f3c3c1136aeae498529bfde5e06bda9a7cca`.

| Worktree | Branch, under `experiment/` | Ownership |
| --- | --- | --- |
| `amd-chonk-fnuz-transport` | `amd-chonk-fnuz-transport` | Codec, asymmetric dispatch, transport fixtures |
| `amd-chonk-fnuz-experts` | `amd-chonk-fnuz-experts` | Six routed GEMMs, SSSGLU, saved operands, expert-delayed drain, expert fixtures |
| `amd-chonk-fnuz-integration` | `amd-chonk-fnuz-integration` | Selectors, FP32 delayed-ready DDP, optimizer, schedule |

No GPU run may use these working trees as executable source.
For each slice, record the expected change list and reviewed blob digests before requesting a user-signed commit.
After handoff verify the full commit ID, signature against the user's established signing identity, ancestry, complete reviewed diff, and committed blob digests; check that no intended edits were left out.
Record both component SHAs as ancestors of the signed integration pin.
The runner snapshots the pinned commit, checks its tree and archive, freezes the manifest and tooling hashes, materializes source in job-owned scratch, and verifies the executed entry point and imported Megatron modules.
Image `910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490` and UCCL `52d4184c89deb5631be6dac6a9b1c40896c9e26e` are unchanged.

## Integrated schedule contract

Select `--moe-use-fp8-dispatch --moe-fp8-dispatch-fnuz --moe-resident-fp8-fnuz`.
Keep `--overlap-grad-reduce --overlap-moe-expert-parallel-comm --delay-wgrad-compute`, the default aligned gradient reduction and native P2P overlap.
Keep global fusion and graphs off (`--no-gradient-accumulation-fusion --cuda-graph-impl none`).
The adapter's delayed callback reports completion after its **FP32** destination has been written on the ordered compute stream, including logical completion for empty experts.
DDP validates all expected contributions and orders the reduction/finalization stream after those writes without consuming a BF16 `param.grad` sentinel.
The BF16 optimizer copy explicitly advances weight versions because its `.data` copy is not visible to PyTorch version counters.
The separately selected transport codec decodes to BF16 before the expert adapter, so a transport pass cannot count as a GEMM pass.
The [full-layout declaration](experiment.toml) keeps the fixed architecture, batch, and topology, and is launch-blocked pending the signed combined source and the staged gates.

## Gates and resource boundaries

| Gate | Minimum intended placement | Required evidence before moving on |
| --- | --- | --- |
| Codec | 1 node, 1 GPU, `cscs` reservation | Exact typed FNUZ payload, FP32 scale and BF16 decode, generated gfx942 path |
| Expert component | 1 disjoint node, 1 GPU, `cscs` | Both FCs' `TN`, `NN`, `NT` in native kernels, matching-format arithmetic and probability gradients, zero padding, ragged/empty behavior, refresh/lifetime and FP32 accumulation; FN deltas reported separately |
| Transport | 4 nodes × 4 ranks, `cscs` | EP16/CXI, width 1792 / 56 scale bytes, 8192 source rows and top-8, exact association, async/reverse BF16 and FP32 probability gradients |
| Combined schedule rehearsal | After component passes; 8 nodes for PP2/VPP2 with full EP16, or a narrower isolated schedule probe if sufficient | Actual per-layer expert/latent/sequence/MBS shapes, distinct compute and transport streams, delayed writes/ready counts, optimizer consumption and two successive weight generations; a reduced-depth rehearsal is not Chonk fit or throughput evidence |
| Full-layout fit | 16 nodes × 4 ranks on general `mi300`, only after combined gate | Untouched 42-layer workload at MBS2/GBS4096 and PP4/VPP2; peak per-rank HBM, initialization, compilation, collectives, optimizer, timing and checkpoint costs |
| Checkpoint / cache | Same qualified workload for restore and comparable repetitions | `torch_dist` structure and readable tensor/momentum/gain payloads; actual resume compared with a continuous control; verified cache hydration and quiescent, checked, immutable publication |
| Throughput | 16 × 4 on general `mi300`, two disjoint-in-time warm 60-update repetitions | Both contiguous windows 11–26 and 45–60, no omitted slow samples; placement plus full run evidence |

Exact component argv and seeds belong to the signed component entry points, with a child deadline smaller than each freshly specified allocation deadline.
Probes use `procedure.kind = "probe"` to execute the manifest-verified entry point without checkpoint arguments or shared cache mounting/publication.
Before submission record occupancy and eligible nodes, excluding unrelated allocations and counting all active or submitted investigation allocations toward the 16-node combined cap.
Distributed component failures stop peers via the owned runner; do not reuse failed layouts or adjust acceptance thresholds after seeing a GPU result.
Choose longer full-layout deadlines only from fresh fit and timing data.

## Existing evidence

- Source and ELF evidence identified the six TE/HipKittens operations; GPU job `756732` independently verified the actual dispatch and numerical gate.
- CPU probes compared two rank-3 expert matrices against per-expert rank-2 matrices on identical prescribed FP32 gradients for FC1 and FC2 at the actual Chonk widths.
  Over three updates they check gate/up blocks, gain axes and moments, momentum, bare row normalization, and FP32/BF16 weights without changing optimizer math.
  This used image 910's Emerging Optimizers and CPU Torch 2.11, not the GPU training stack.
- CPU tests cover the aligned DDP callback contract and explicit BF16 model-parameter version increment.
- No schedule, fit, restore, cache-reuse, or performance verdict has been accepted for this recipe.

## Expert component review boundary

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-experts`.
Branch: `experiment/amd-chonk-fnuz-experts`.
Parent: `e242f3c3c1136aeae498529bfde5e06bda9a7cca`; no component dependencies.
Suggested signed commit subject: `Add resident gfx942 FNUZ routed expert GEMMs`.

| Reviewed path | Expected Git blob |
| --- | --- |
| `megatron/core/transformer/moe/experts.py` | `63be6b5fcd11d36e9bf6f06ccacd06fe0a2e43e5` |
| `megatron/core/transformer/moe/resident_fp8.py` | `14f6592af2fd3181b4031898ef6beb70d3d1df4c` |
| `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py` | `2ec1d640099c9ad3b6bc6fc8542f066e6c192700` |

The independent probe's CPU mode passed with software GEMMs and mocked streams.
The GPU probe pins actual-width ragged, skewed and all-empty fixtures, matching-format 1.8% relative L2 and 4% scaled-max limits, direct FP32 beta checks, and six HIP traces with native kernel names.
It reports FN deltas separately.
It exercises in-memory model state loading; `torch_dist` restoration remains a distinct gate.

The first three expert attempts pinned verified user-signed commit `86c89189fe41e4145a02f4f5d705f509f2c5d339`.
The signature matches the e242 signer (`FB605768E5C3B6BE52CA4B09B5C8CB8659FA42E6`), its sole parent is e242, all three blobs match the table, and the worktree was clean at verification.
The initial plan requested one reserved node, one task and one GPU, 48 CPUs, 20 minutes of allocation time, and a 600-second probe child deadline.
The approved corrective plan uses core binding because Slurm rejected `map_ldom:0` for the one-task, 48-CPU step:

```bash
uv run --locked apertus plan experiments/chonk-native-fnuz-experts-i910bfcf936d3 \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:20:00 --cache-mode cold
```

The runtime retains per-rank logs and the six trace files beneath the unique `artifacts/experiment-data/runs/chonk-native-fnuz-experts-i910bfcf936d3/<run-id>/output/`.
The source snapshot and frozen tooling are verified before execution; this probe cannot write or read the shared compiler-cache mount.

| Job | Run suffix | Outcome |
| --- | --- | --- |
| `756602`, `nid002774` | `20261120T135519Z-fc2c1f31` | The `srun` client exited with SIGSEGV in Pyxis before the step. The preserved core shows the EDF path code dereferenced missing `HOME` in the allowlisted environment. |
| `756603`, `nid002774` | `20261120T135741Z-5db52c7d` | A private project-owned `HOME` resolved the first failure. Slurm then rejected `map_ldom:0` with `Entire node must be allocated for map_ldom` for the partial-CPU step. Pyxis also logged that both `EDF_IMAGESTORE` and `SCRATCH` were absent and it would continue without its image cache. |
| `756604`, `nid002774` | `20261120T135953Z-54ee3ebe` | Corrective binding worked and the expert quantizer's GPU fixture passed on gfx942. The job failed at the first GEMM call: `te_general_grouped_gemm` rejected `m_splits=None` at the Python/C++ boundary, before any GEMM launch. |

Neither of the first two attempts entered the container or started a GPU probe rank.
Keep both failed runs; do not reuse their directories or change the component source, image, or numerical limits.
The one-rank recovery retains the private launcher `HOME`, passes the documented site `SCRATCH` path for the image lookup, and selects `--cpu-bind cores` with 48 CPUs.
Core binding is only a single-rank diagnostic change; it does not change the four-rank EP16 or full-model binding policy.
The corrective submission executes outside the agent sandbox with normal host filesystem access, using the launcher allowlist below plus Slurm-managed and frozen image variables.
From `/ritom/scratch/cscs/boeschf/apertus-amd`, the approved expert command was:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-experts-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-experts \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:20:00 --cache-mode cold
```

### Expert binding repair: signed follow-up

The image's Python wrapper forwards its `m_splits` argument unchanged, but the C++ entry point takes a non-optional `std::vector<int64_t>` even with separate destinations.
Supply the integer list of padded token counts from the leading axis of logical B for `TN`, `NN` and `NT`.
This changes no native kernel, scale layout, precision, tensor, or numerical criterion.
The CPU lifecycle and new six-operation list-call contract check passed, as did `git diff --check`.

Use the same expert worktree and branch, with parent `86c89189fe41e4145a02f4f5d705f509f2c5d339`.
Suggested signed subject: `Pass explicit token splits to TE list GEMM`.

| Follow-up path | Expected Git blob |
| --- | --- |
| `megatron/core/transformer/moe/resident_fp8.py` | `555eaf6cae57b7fbaf6a07b1c7e3d2354c6106c4` |
| `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py` | `7d1a4fc22055042e605acce7609314314aa1cfd0` |

The [expert declaration](../chonk-native-fnuz-experts-i910bfcf936d3/experiment.toml) now pins user-signed `8acb7858ce4aeece86694ea4cdf01c35fc18a58f`, verified against the same signing identity, sole parent, and both blobs above.
Expert job `756614` on `nid002774` tested this pin in run `20261120T140944Z-cdff4bf6`, with the prior corrected resources and unchanged numerical limits.
All six exact component fixtures passed, and the six HIP traces each contain two gfx942 `micro_tk` GEMMs with native `__hip_fp8_e4m3_fnuz` operands.
Forward and dgrad traces show BF16 output, and the two wgrad traces show FP32 output with the fixture verifying nonzero prior FP32 destination accumulation.
The composed skewed fixture failed at expert 15's FC2 output: relative L2 `0.0180838443338871` exceeded `0.018`, while scaled maximum `0.02393617108464241` was below `0.04`.
This stops the composed expert gate.
No relaxation, source change to the training adapter, or inference about the cause follows from that one observation.

### Expert oracle diagnostic: signed follow-up

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-experts`.
Branch: `experiment/amd-chonk-fnuz-experts`.
Parent: `8acb7858ce4aeece86694ea4cdf01c35fc18a58f`.
Suggested signed subject: `Diagnose skewed native FNUZ FC2 oracle mismatch`.
The only changed file is `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py`, reviewed Git blob `40902001b206fcb74bb2221c4f43a180c564b257`.
The normal component run retains the original inputs and tolerances; the CPU lifecycle, list-call contract and syntax checks passed.
The diagnostic uses the exact failing 1025-token fixture and compares activation and weight cast bytes and FP32 decoded operands, native and independent FC1/FC2 results with both shared and independent operands, and SSSGLU on identical BF16 FC1 output.
It writes one `FNUZ_EXPERT_ORACLE_DIAGNOSTIC` record and cannot qualify the expert component on its own.

The [diagnostic declaration](../chonk-native-fnuz-expert-oracle-i910bfcf936d3/experiment.toml) pins `50b26ce93c9d6d7356bd5b0d32240631c247083f`, verified against the e242 signer, the sole parent, and the one reviewed blob.
The approved run uses the same one-node `cscs` placement, one GPU, 48 CPUs, `cores` binding, private launcher environment, 600-second child deadline and 20-minute outer deadline as the signed expert probe.
The diagnostic is a different command from qualification: it needs gfx942 and will execute outside the agent sandbox with normal host filesystem access in the pinned image, using only the selected launcher environment below plus Slurm-managed and frozen image variables.
It runs from the Apertus workspace, freezes the committed source, and writes only to a fresh run directory under `artifacts/experiment-data/runs/chonk-native-fnuz-expert-oracle-i910bfcf936d3/` and job-owned scratch.
It cannot hydrate or publish shared compiler caches.

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-expert-oracle-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-experts \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:20:00 --cache-mode cold
```

Do not retry the failing acceptance case until the source or oracle cause is identified.
Diagnostic job `756723` completed in 1:04 on `nid002778`, with clean teardown and no cache publication.
The [diagnostic log](../../artifacts/experiment-data/runs/chonk-native-fnuz-expert-oracle-i910bfcf936d3/20261120T145058Z-bcd6752f/output/rank-0.log) reproduces the `0.0180838443338871` FC2 relative-L2 failure exactly, and the adapter and explicit native chain are bit-identical.
SSSGLU is bit-identical given the same BF16 FC1 output.
Given the same native quantized operands, FC1 and FC2 relative-L2 errors against FP32 dequantization and matmul are `5.0565e-5` and `4.6704e-5`, respectively.

### Matching-format oracle repair: signed follow-up

The original Torch expression `240.0 / amax` implements an FP32 reciprocal and then an FP32 multiply; the installed TE quantizer uses the directly rounded FP32 division `__fdiv_rn(240, amax)`.
For example, on the preserved fixture with a block maximum of `2.578125`, the two multipliers are `93.09090423583984` and `93.09091186523438`.
With input `-0.494140625`, they produce scaled values of `-45.999996185302734` and `-46.0`, on different sides of the FNUZ round-to-even decision, giving byte codes `235` and `236`.
With the exact seeded 1025-row input on CPU, the old-versus-corrected oracle comparison reproduces **1,381 differing payload bytes** and **447,192 differing FP32 decoded values**, matching the retained GPU TE-versus-old-oracle comparison.
Both relative-L2 input errors are approximately `0.00241653`; do not infer a whole-component GPU pass from this match.

Only the independent matching-format oracle changes: use directly rounded FP32 division, with the TE finite-scale cap for the source dtype.
The native adapter, FN comparison, fixtures and `0.018` relative-L2 / `0.04` scaled-max limits are unchanged.
The CPU lifecycle, list-GEMM and new exact midpoint regression passed; the focused Apertus suite passed 104 tests.

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-experts`.
Branch: `experiment/amd-chonk-fnuz-experts`.
Parent: `50b26ce93c9d6d7356bd5b0d32240631c247083f`.
Suggested signed subject: `Match TE FNUZ oracle scale division rounding`.
The only changed file is `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py`, reviewed Git blob `bffcb5c30e6c99d58723f9a29a04c30556a97b54`.

The [expert qualification declaration](../chonk-native-fnuz-experts-i910bfcf936d3/experiment.toml) now pins signed commit `cfbf192c765b4add2de5f24f47e3462e3303b430`, verified against the sole parent, the reviewed blob, and the e242 signer.
Job `756730`, run `20261120T150729Z-60396e23`, reruns the complete expert gate under explicit approval, not just the skewed case.
The [expert command above](#expert-component-review-boundary) uses the same private launcher environment, one `cscs` reservation node, one gfx942 GPU, 48 CPUs, `cores` binding, 600-second child deadline and 20-minute outer deadline as job `756614`.
It executes outside the agent sandbox with normal host filesystem access in the same pinned image, without shared cache hydration or publication, and writes to a fresh directory under `artifacts/experiment-data/runs/chonk-native-fnuz-experts-i910bfcf936d3/`.
If the repaired oracle misses a fixed limit anywhere, retain the failure, diagnose it and keep the combined training gate blocked.

Job `756730` on `nid002774` ended after 1:10 with clean teardown and all six native traces preserved.
It passed the exact native fixtures, the revised oracle checks for the active ragged and skewed experts, and the first two delayed-wgrad comparisons, then stopped in the all-empty snapshot loop.
The test used `tuple.index(param)`, which attempted to convert elementwise comparison between different Torch parameters to a Boolean.
This is a harness failure, not a numerical verdict on the all-empty drain or the subsequent refresh/restore checks.
See the [rank log](../../artifacts/experiment-data/runs/chonk-native-fnuz-experts-i910bfcf936d3/20261120T150729Z-60396e23/output/rank-0.log).

### All-empty snapshot check: signed follow-up

Use the same expert worktree and branch, with parent `cfbf192c765b4add2de5f24f47e3462e3303b430`.
Suggested signed subject: `Compare empty FNUZ drain snapshots in callback order`.
The only changed file is `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py`, reviewed Git blob `498fbf404c0b4843969f84040c2101a040a958e3`.
The test now compares consecutive drains in verified callback order without comparing parameter tensors for lookup.
The CPU lifecycle probe exercised the same helper with multiple parameters, and `git diff --check` passed.
No training adapter, fixture, numerical limit, or resource setting changed.

The [expert qualification declaration](../chonk-native-fnuz-experts-i910bfcf936d3/experiment.toml) pins signed commit `37a8e6fdca240ca4e582d1d7191e8fdc30577384`, verified against its sole parent, the reviewed blob, and the e242 signer.
Job `756732` on `nid002774`, run `20261120T151442Z-8de91f26`, executes the complete gate after explicit approval with the [same outside-sandbox command and artifact layout](#expert-component-review-boundary), private launcher environment, one gfx942 GPU and 48 CPUs on one exclusive `cscs` reservation node, 600-second child deadline, and 20-minute allocation.
It completed in 1:10 with clean teardown and no cache publication.
The [rank log](../../artifacts/experiment-data/runs/chonk-native-fnuz-experts-i910bfcf936d3/20261120T151442Z-8de91f26/output/rank-0.log) and all six HIP traces passed verification against the signed source snapshot (`29379668b3afa7c6804c64eed6383a771eaea5f17bd527a0691e4db27836d01d`).
The gate covers the native codec, all six native GEMM fixtures, ragged and skewed numerics, two-stream delayed FP32 accumulation, the all-empty drain, BF16 mutation/refresh, and in-memory state restore.
In-memory state restore is not a distributed checkpoint or training restart.

| Output | Worst relative L2 vs matching FNUZ oracle | Worst scaled maximum vs matching oracle | Worst relative L2 between FNUZ and FN oracles, descriptive only |
| --- | ---: | ---: | ---: |
| Forward | 0.06395% | 0.28736% | 9.38% |
| Input gradient | 0.05450% | 0.42017% | 12.65% |
| Probability gradient | 0.02103% | 0.11323% | 10.98% |
| FP32 accumulated weight gradient | 0.06374% | 1.09770% | 12.26% |

The matching-format limits remain 1.8% and 4%, respectively.
These FN-versus-FNUZ fixture differences do not establish training equivalence or training degradation.
Combined training remains blocked pending the signed integration and schedule qualification.

## Integration code: signed handoff

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration`.
Branch: `experiment/amd-chonk-fnuz-integration`.
Parent: `e242f3c3c1136aeae498529bfde5e06bda9a7cca`.
Suggested signed subject: `Integrate resident FNUZ selectors and delayed FP32 gradient readiness`.

| Path | Expected Git blob |
| --- | --- |
| `megatron/core/distributed/distributed_data_parallel.py` | `765972a443e0dcb9d7e26894c5a4bb2ebcaf8de1` |
| `megatron/core/optimizer/optimizer.py` | `14ce4aaabbc5af04df617a1c33eb7d397b465005` |
| `megatron/core/transformer/transformer_config.py` | `d20e2516d9fc01a97bf02746a943b9dde6d1a0ce` |
| `megatron/training/arguments.py` | `29fd04446c3509f5f51fb8d5218d01b4721992bf` |
| `tests/unit_tests/distributed/test_resident_fp8_grad_ready.py` | `67ccfd4a22d5e2f4eee871e0c4a717fb0280244c` |
| `tests/unit_tests/optimizer/test_resident_fp8_optimizer_mapping.py` | `ced220e70b5b4d91d86d1fdbc1577b70ff850eb6` |

This slice adds fail-closed selectors, the delayed FP32 gradient-ready callback and stream waits, and explicit weight version invalidation after the optimizer writes authoritative BF16 parameters.
The four DDP/version contract tests and four optimizer-layout tests passed on CPU; `git diff --check` passed.
The standalone integration commit `1a3e50dd4cf7d51a3ce80b08523889eb16198784` was verified against its sole e242 parent, all six reviewed blobs, and the established signer.
Neither the standalone integration code commit nor an expert/transport component commit is a combined launch pin.
The signed integration code must be combined through a user-signed merge that retains expert `37a8e6fdca240ca4e582d1d7191e8fdc30577384` and transport `696ff8b7c711754f50b748b6ecaccd72fd00d3ba` as ancestors.
No training job is authorized by this handoff.

## Combined schedule rehearsal: launch review

The clean integration worktree now pins signed merge `4b21bb5781ee6db66952a9b6ab128b688c7dc1d1`, tree `9f0c1d1a76c9aec1c299c45dc24c14bff8b46060`.
The established signer, ordered integration/expert/transport parents, and all 15 changed parent blobs were verified.
The [rehearsal declaration](../chonk-native-fnuz-schedule-i910bfcf936d3/experiment.toml) and the still-blocked full-layout declaration pin this source and image 910.

The rehearsal runs two seed-42 mock-data updates with four MoE layers (three KDA and one attention), PP2/VPP2 `Et|t|t|tL`, EP16, TP/ETP/CP/EDP1, DP16, S4096/MBS2/GBS256 and eight microbatches per update.
The full declaration differs in layers, attention/MoE frequency, PP/layout, GBS, run length, and checkpointing; the rehearsal uses the same environment and remaining optimizer and kernel arguments.
The signed parser, tokenizer and core config were exercised on CPU with device capability, installed TE version and TE fused API availability stubbed, not with a model forward or GPU execution.

The [validator](../validation/chonk_fnuz_schedule.py) rejects any rank that does not finish exactly two updates with the expected interleaved schedule, two drained chunks per rank and 256 delayed-ready events for 32 parameters in each chunk per update.
It requires no outstanding gradient-reduce handles, verifies that all 64 expert master weights receive the exact finalized FP32 gradient views in `md_decoupling`, and checks advanced versions on unchanged BF16 storage, finite resident weights, finite positive gradient norms, and finite LM loss on ranks 16–31 only.
The native EP16 buffer sizes and configuration must match the component gate, and each rank must report CXI initialization, verified imports and clean distributed teardown.
Any nonfinite result, skip, rerun, stale or missing generation, lifecycle error, timeout or missing rank is a stop condition; retain evidence and diagnose before proposing a retry.
This gate does not use checkpoint I/O or shared cache hydration/publication, and will not serve as full-layout fit, restart or throughput evidence.

One new exclusive reservation allocation is requested: Beverin `csstaff` / `mi300` / `cscs`, eight MI300A nodes, four ranks/GPUs and 192 CPUs per node (48 per rank), all allocatable memory and a 40-minute outer limit.
The probe child is limited to 1,500 seconds after source/dataset-helper preparation, with the remainder reserved for setup, checking and teardown.
Slurm chooses the exclusive nodes; the reservation had 32 available nodes at review, the unrelated interactive job `755661` occupied `nid002772`, and held CI `738614` used none.
No investigation job is currently allocated; even counting the unrelated interactive node conservatively the request totals nine of 16.
Recheck occupancy and reservation capacity before submitting, and do not reuse or modify either unrelated job.

The batch and compute steps execute **outside the agent sandbox with normal host filesystem access**.
The launcher inherits only the five variables below, plus Slurm-managed fields and the runner's explicit settings; the container adds image defaults, the declaration and `config.toml` environment, EDF defaults, and runtime-owned rank, rendezvous and private cache paths.
No agent credentials, uenv, home or library path are forwarded.
From `/ritom/scratch/cscs/boeschf/apertus-amd`, the proposed exact command after approval is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-schedule-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration \
  --submit --reservation cscs --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:40:00 --cache-mode cold
```

The runner generates `job.sh` (`--exclusive`, `--uenv-passthrough=ignore`, `block:block`) and launches one `srun` step with `pmix`, `disable_rdzv_get`, one GPU per task and `--gpu-bind=none` from the approved run.
Run files go under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-schedule-i910bfcf936d3/<run-id>/`; committed source snapshots go under the adjacent `experiment-data/sources/`.
Node-local work and private rank caches use the runner-owned `/dev/shm/apertus-amd/<uid>-<job-id>-<run-id>-<node>` directories.
The user approved this exact launch; only the rehearsal guard was removed.
The runner submitted job `756745` once on `nid[002920,002922,002924,002926,002928,002930,002932,002934]`, disjoint from job `755661`.
The [immutable run](../../artifacts/experiment-data/runs/chonk-native-fnuz-schedule-i910bfcf936d3/20261120T153816Z-e28c5123/) froze source archive SHA-256 `f2970059a14ab437667335974024b1d8546e0214aa9a724f1da2461435523596` and validator SHA-256 `fce93af7c89a301413d93876bf1fc9b26dfe0b48a6338722e8937d3300a6abdb`.
The rehearsal guard is back in place after submission; this approval does not authorize a retry or the full-layout run.
Startup records show 32 gfx942 ranks, 48 bound CPUs per rank, cold private caches, and eight dataset-helper preparations.
Independent CPU parsing of the full declaration with the same hardware/TE capability stubs resolved decoder chunks `(3,5)`, `(6,6)`, `(6,6)`, `(5,5)` by physical PP rank, 31 KDA and 39 MoE layers, padded vocabulary 200064, and 128 microbatches at MBS2/GBS4096.
This confirms the full input contract only, not its GPU memory or timing.

Job `756745` exited `1:0` after 9:08 and did not pass the two-update gate.
All 32 rank logs show CXI initialization, and an offline check with the frozen validator accepted every rank's update-1 record: 256 delayed-ready events per chunk, all 64 FP32 optimizer gradient views per rank, advanced BF16 versions, and finite resident weights.
The gradient norm was 2.635089 on every rank, and the LM loss was 12.715135 on all 16 loss-stage ranks.
After the optimizer step, Megatron's `report_memory()` called `torch.cuda.device_memory_used()` because of `--log-device-memory-used`; the pinned image's Torch attempted to import `amdsmi` and raised `ModuleNotFoundError` on all 32 ranks.
There is no update-2, post-training transport snapshot, or successful teardown verdict.
The 417.9-second first update includes cold startup work and is not a steady-state timing sample.
The narrow retry candidate is to omit `--log-device-memory-used` while keeping Torch allocator statistics and all training settings unchanged; do not modify the pinned image or retry without approval.
The full-layout declaration also contains this flag; total device memory telemetry for fit needs a separately verified source.

The user approved one retry with that sole argument removal and the same outside-sandbox command and limits.
The reservation review found a 16-node image-910 Chonk EP16/UCCL profiling job `756820` under another project user; treat it as potentially part of the investigation until it finishes or its scope is clarified.
The retry was held until `756820` was terminal and a fresh scheduler check confirmed headroom under the 16-node cap.
Job `756991` was then submitted once on the same eight nodes as `756745`, disjoint from the other active allocations.
The [retry artifacts](../../artifacts/experiment-data/runs/chonk-native-fnuz-schedule-i910bfcf936d3/20261120T160622Z-18e279da/) retain the same source archive, validator, runner tooling, image, resources, and environment; only `--log-device-memory-used` was removed.
The rehearsal guard is back in place.

**The reduced schedule gate passed in job `756991`.**
Slurm completed in 9:20 with exit code `0:0`, and the runner and [all-rank checker](../../artifacts/experiment-data/runs/chonk-native-fnuz-schedule-i910bfcf936d3/20261120T160622Z-18e279da/output/checks.json) both passed.
All 32 ranks reported two accepted updates, expected EP16/CXI configuration and buffer sizes, valid imports, and clean distributed teardown; all eight node leads recorded runner cleanup.
Each rank delivered 256 delayed-ready notifications per chunk per update and passed 64 completed FP32 gradient views to the optimizer before advancing the resident BF16 weight versions.
Resident weights stayed finite, and there were no skipped updates or reruns.

| Update | LM loss on loss-stage ranks | Gradient norm on all ranks |
| --- | ---: | ---: |
| 1 | 12.715216 | 2.635098 |
| 2 | 12.707118 | 2.608146 |

There was no shared-cache publication.
Startup and cold compilation dominate the first update, and the reduced rehearsal's displayed timing is not a throughput measurement or a basis for the full-layout walltime.
The next gate is a separately reviewed short 16-node, 42-layer PP4/VPP2 fit at S4096/MBS2/GBS4096 with allocator and total-device memory telemetry kept distinct.
Full-layout fit, distributed checkpoint/restart, cache reuse, and throughput are still unqualified.

## Full-layout fit: launch review

The [fit declaration](../chonk-native-fnuz-fit-i910bfcf936d3/experiment.toml) retains 42 layers, 39 MoE and 31 KDA layers, 64 ranks, PP4/VPP2/EP16, DP16, S4096/MBS2/GBS4096 and 128 microbatches per update on the same signed source and image.
Its only differences from the full measurement argv are the two-update length, the final save at update 2, the runner-supplied destination, and omission of `--log-device-memory-used` because image 910's Torch cannot import `amdsmi`.
There is no load, asynchronous save, shared cache access, or profiling.
The probe is for memory and timing, not integrity, restoration, training parity, or steady-state throughput.

The [fit observer](../validation/chonk_fnuz_fit.py) requires all 64 ranks to report the fixed layout, native EP16/CXI, two finite unskipped updates without reruns, one final save and clean teardown.
It records startup through first training-step entry, scheduler and update boundary durations, compiler counters, EP16 buffer sizes, allocator peaks through save and teardown, and HIP `mem_get_info()` snapshots separately.
An unavailable HIP snapshot is recorded explicitly and does not turn allocator peaks into a device-usage estimate.
It measures the save and inventories the checkpoint without reading tensor payloads.
Expert weights, FP32 masters, and optimizer state put the checkpoint on the order of 1 TB; that is a storage estimate, not a measurement or quota confirmation.
Stop and retain the artifacts on nonfinite values, missing ranks, an incomplete save, OOM, or deadline; no blind retry or extension.

The selected source's parser, tokenizer, and config passed on CPU with device and TE availability stubbed.
The focused runner, probe and validator suite passed 173 tests.
The cold fit budget is 2,700 seconds from training child start through checkpoint and teardown, within a 60-minute outer allocation.
Neither bound was taken from historical smaller-batch throughput; the full fit will provide the timing for a later bounded run.

Placement is one new `csstaff` allocation on general `mi300`, with no reservation, 16 exclusive x86_64 MI300A nodes, four GPUs/ranks and 192 CPUs per node, all allocatable memory, `block:block`, `map_ldom:0,1,2,3`, `pmix`, `disable_rdzv_get` and `--uenv-passthrough=ignore`.
The live review initially held this launch while checking whether another user's profiling job `757060` counted toward the investigation cap.
The user clarified that the cap counts only jobs submitted for this investigation across both Slurm pools.
Do not touch the unrelated interactive and CI jobs, reuse other allocations, or submit while doing so would exceed the combined investigation cap.

The batch and compute steps execute **outside the agent sandbox with normal host filesystem access**.
The host allowlist is exactly `HOME`, `USER`, `LOGNAME`, `PATH` and `SCRATCH` as below, in addition to Slurm-managed fields; the container uses the identical image and site/declaration environment as the rehearsal plus the runner-owned launch values.
From `/ritom/scratch/cscs/boeschf/apertus-amd`, the proposed command is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-fit-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold
```

Frozen inputs, logs and the checkpoint live under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-fit-i910bfcf936d3/<run-id>/`; checkpoint shards are in `output/checkpoints/iter_0000002/`.
Source snapshots live under the adjacent `experiment-data/sources/`; private preparation and caches are under the runner-owned `/dev/shm/apertus-amd/<uid>-<job-id>-<run-id>-<node>` on each node.
The user approved this exact one-shot operation.
At submission, no other job submitted for this investigation was active or pending, and 58 general-partition four-GPU nodes were idle.
Job `757063` was submitted once on `nid[002408,002412,002414,002416,002418,002420,002422,002424,002426,002440,002442,002444,002446,002448,002450,002452]`.
The [fit run](../../artifacts/experiment-data/runs/chonk-native-fnuz-fit-i910bfcf936d3/20261120T164117Z-91ee450f/) froze the reviewed source archive and fit validator; the fit guard has been restored, and the 60-update measurement declaration stays blocked independently.
All 64 ranks reached the first training step and logged native CXI initialization.
During update 1, the 16 ranks in the third physical pipeline stage (ranks 32–47) reported `DeepEP error: timeout (dispatch CPU)` inside `fused_fp8_dispatch_fwd`.
No completed update, checkpoint, or full-layout fit was verified.
The checkpoint directory contains only the initial `progress.txt`, not a tracker or shards.

### `757063` postmortem: OOM kill and dispatch timeout

Slurm had already stopped the job when cancellation was requested: `FAILED` after 10:33, exit `1:0`, with no active job left to cancel.
All 16 node leads recorded runner cleanup.
The final `launcher.log` reports an `oom_kill` event and marks `nid002450` (tasks 56–59, physical PP stage 3) `Out Of Memory`.
The runner identifies rank 59's training child as exiting with signal 9.
This is direct evidence of an OOM kill, not a Torch allocator `hipErrorOutOfMemory`; the accounting record does not identify which allocation or memory consumer pushed the node over its limit.
The 16 DeepEP tracebacks come from a different physical stage (PP stage 2), waiting in the CPU-side `internode_prepare` path.
Loss of a downstream rank could explain the upstream dispatch timeout, but the runner's error propagation and log modification times do not prove the ordering of the OOM event and transport stall.

Only ranks 32–47 retained fit-observer records, all with zero completed updates.
At first-step entry on those ranks, the Torch allocator had 24.45–24.65 GiB reserved and HIP reported 80.91–82.75 GiB free out of 128 GiB per device.
No equivalent memory sample survived from the OOM node; neither these snapshots nor Slurm's per-task MaxRSS measure its late node-wide peak.
The failing PP stage also has the largest parameter count per rank: about 3.21 billion including the output head, versus 3.01 billion on each middle stage and 2.42 billion on the first stage.

The reduced rehearsal is not a memory proxy.
At balanced routing, 8,192 input tokens with top-8 give about 65,536 routed rows per rank per MoE layer and microbatch.
The two saved rowwise FNUZ operands plus FP32 scales are roughly 363 MiB per layer-microbatch, and the four queued wgrad operands plus their scales are roughly 602 MiB per delayed contribution, before padding and physical allocator layout.
Using the pinned 1F1B warmup table, the nominal saved-operand volume alone across physical stages is 7.09, 19.14, 14.89 and 8.86 GiB per rank, respectively, before ordinary activations, workspaces, communication, queued wgrad, or host-side and compiler memory.
These are source-derived size estimates, not an observed pre-OOM breakdown or a diagnosis of which tensor caused the kill.

The next diagnostic should obtain the timestamped Slurm/kernel OOM record for `nid002450` if available and persist lightweight node/cgroup usage, limit, `memory.events`, per-rank allocator peaks, and schedule progress before failures rather than only on Python exit.
Correlate the OOM event with rank progress before deciding whether to fix resident allocation/lifetimes or transport.
Do not lengthen the dispatch timeout, change the fixed batch/layout, or submit a blind rerun.

### Placement retry excluding `nid002450`

The user requested one rerun of the same fit with `nid002450` excluded.
The original fit command was used with the sole launch option `--exclude nid002450`, which the runner records in the resolved resources and passes to Slurm as `#SBATCH --exclude=nid002450`.
The focused suite passed 47 CPU tests, and the resolved plan matched `757063` apart from placement exclusion and the runner code needed to forward it.
Job `759575` was submitted once on general `mi300` with no other active investigation allocation, and Slurm confirmed that none of its 16 allocated nodes is `nid002450`.
The [retry artifacts](../../artifacts/experiment-data/runs/chonk-native-fnuz-fit-i910bfcf936d3/20261123T083349Z-52941268/) retain the same source archive, validator, image, workload, environment, and deadlines as `757063`.
This tests whether failure recurs elsewhere; it does not isolate a node effect or fix the OOM.
The fit launch guard is back in place.

Job `759575` failed after 9:47 (`1:0`) with `nid002450` excluded.
The initial wait for ranks 32–35 on `nid002456` cleared; all 64 ranks entered the first step and reported native CXI initialization.
Slurm recorded an `oom_kill` and marked `nid002484` (tasks 56–59, fourth physical PP stage) `Out Of Memory`.
The runner recorded child `SIGKILL` for ranks 55 and 59, but the launcher names only `nid002484` as OOM, so the rank-55 signal alone does not establish a second OOM.
Slurm sampled a maximum task RSS of 50,748,420 KiB on rank 59; this is not a node or GPU memory peak.
No rank reported a DeepEP dispatch timeout, no update completed, and `output/checkpoints/` contained only `progress.txt`.
All 16 node leads recorded runner cleanup.
The same PP stage failed with OOM on two different nodes; excluding `nid002450` is not a remedy, and the OOM can recur without a logged dispatch timeout.
The allocation source is still unknown; a further diagnostic should record the hard-to-retain pre-failure memory evidence described above.

### Full-layout OOM telemetry: launch review

The [OOM probe](../chonk-native-fnuz-oom-i910bfcf936d3/experiment.toml) keeps the signed source, image, environment, full 42-layer PP4/VPP2/EP16 model, S4096/MBS2/GBS4096, and optimizer unchanged.
It runs one update, with no checkpoint I/O, cache hydration/publication, new precision policy, compiler cap, or transport timeout changes.
Every supervisor appends five-second process-tree samples and a final sample when it observes exit or peer failure.
The 16 node leads add node memory, `/dev/shm`, and visible cgroup usage, limit, peak, event, and category counters.
The [validator](../validation/chonk_fnuz_oom.py) flushes per-rank allocator and virtual-microbatch entry/exit records before teardown, and samples HIP free device memory at step entry and about every five seconds when schedule calls return.
The timelines use host wall-clock timestamps and retain the last available observation; they cannot guarantee the instantaneous peak, a final sample on `SIGKILL`, or cgroup visibility.
The probe passes only if all ranks complete one finite update, report native CXI, and tear down cleanly.
If it fails, use the retained JSONL, Slurm and rank logs to distinguish host/cgroup pressure from rising device allocations, without treating a missing final sample as a zero.

183 focused CPU and scheduler-stub tests passed, including a supervisor test that retains samples after the training child is killed.
No investigation job was active or pending at review, and 35 eligible general-partition nodes were idle.
Recheck the 16-node cap before launch and leave unrelated jobs untouched.
The request retains the prior fit limits: 16 exclusive general `mi300` nodes on `csstaff`, 64 GPUs, 48 CPUs per rank, all allocatable memory, `pmix`, `disable_rdzv_get`, uenv passthrough `ignore`, a 60-minute allocation, and a 2,700-second child deadline.
Keep `nid002450` excluded as in `759575`; do not select nodes based on the observed failure.
The Slurm task runs **outside the agent sandbox with normal host filesystem access**.
Use the same host allowlist as the fit plus Slurm-managed and frozen runtime environment.
From `/ritom/scratch/cscs/boeschf/apertus-amd`, the proposed exact one-shot command is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-oom-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

The immutable run goes under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-oom-i910bfcf936d3/<run-id>/`.
Live samples are `output/host-memory-rank-*.jsonl` and `output/gpu-progress-rank-*.jsonl`.
The user approved the exact command, and job `759592` was submitted once after confirming no other active investigation job and 33 eligible idle general-partition nodes.
Slurm confirmed `nid002450` is excluded; it placed the diagnostic on the same 16 nodes as `759575`.
The [run](../../artifacts/experiment-data/runs/chonk-native-fnuz-oom-i910bfcf936d3/20261123T085423Z-4fc52399/) froze the verified source archive and reviewed telemetry wrapper.
The one-shot guard is back in place; neither a retry nor a full fit is authorized by this review.

#### `759592` live evidence

All 64 ranks entered update 1 and initialized native CXI.
The supervisor observed rank 59 exit with signal 9 at 10:04:08.898 site time, on `nid002484`, during the last physical stage's fifth warmup call.
No update completion or DeepEP dispatch timeout appeared in the rank logs.
The final-stage node leads sampled `MemAvailable` minima of 0.027, 0.089, 0.948, and 6.720 GiB across those four nodes.
On the rank-59 node, `MemAvailable` fell from 141.8 GiB to 0.95 GiB in about 20 seconds while node `Shmem` rose from 9.6 to 69.9 GiB; `/dev/shm` mount usage only moved from 3.52 to 3.53 GiB.
The node's task/job cgroup files were not visible in that container.
Rank 59's last HIP snapshot, about 37 seconds before exit, still showed 63.35 GiB free of 128 GiB; the last logged Torch reserved peak was 46.62 GiB.
Those device samples did not measure the kill instant or rule out device-backed allocations contributing to shared physical memory pressure.
Inductor reported 32 compile threads despite `MAX_JOBS=8`; rank 59's sampled process tree included 33 descendants with a summed `RssAnon` of roughly 12.3 GiB, which is not unique physical memory.
The abrupt rise in `Shmem`, nearly flat `AnonPages`, and unchanged `/dev/shm` usage distinguish the observed spike from compiler RSS and ordinary `/dev/shm` files alone; these counters do not establish the mapping owner.
Correlate a timestamped node OOM record and the large shared mappings with warmup before attributing the surge to a particular library.

#### Compiler-cap diagnostic: launch review

The user requested cancellation and a compiler-capped resubmission, and Slurm acknowledged the cancellation of `759592` at 15:14.
At the review, the old job was still `COMPLETING` on two nodes; wait for all of its nodes to be released before requesting 16 more.
The replacement adds `TORCHINDUCTOR_COMPILE_THREADS=4` to the diagnostic declaration, leaves `MAX_JOBS=8` unchanged, and checks the effective Inductor value on every rank before Megatron starts and again at update entry.
The one-update/no-save workload, telemetry, source, image, launcher allowlist, output root, deadlines, placement constraints, and runner tooling otherwise match `759592`.
The focused CPU suite passed 56 tests.
This is a controlled attempt to recover headroom, not an explanation of the `Shmem` surge.
The exact command and resources are the same as in [the OOM telemetry review](#full-layout-oom-telemetry-launch-review), with the updated diagnostic declaration and validator; the run still executes outside the sandbox with normal host filesystem access.
The user approved the command and, after `759592` fully released its nodes, the runner submitted job `759605` once.
The [new run](../../artifacts/experiment-data/runs/chonk-native-fnuz-oom-i910bfcf936d3/20261123T091852Z-4616adc9/) is independent of the cancelled run; the launch guard is back in place.

All 64 ranks in `759605` verified the four-thread limit at startup and at update entry, and all reported native CXI initialization.
The supervisor observed rank 55 on `nid002484` exit with `SIGKILL` at 10:28:19.832 site time, at the same final-stage warmup boundary as `759592`: call 5, forward virtual microbatch 4, no backward call.
The last live sample retaining the victim's RSS shows five descendants, versus 33 on rank 59 in the uncapped diagnostic.
On `nid002484`, the lowest sampled `MemAvailable` was 1.44 GiB with 69.69 GiB `Shmem` and 3.53 GiB used on the `/dev/shm` mount; another final-stage node fell to 0.52 GiB available.
The final-stage shared-memory surge therefore persisted despite the effective compiler cap.
No completed update or DeepEP dispatch timeout appeared in the logs.
Slurm ultimately recorded another OOM kill on `nid002484` and terminated `759605` after 16:39 with exit `1:0`.
The next discriminator is the allocation or mapping owner in this forward-only boundary, not another compiler or placement retry.

#### Mapping and output-head attribution: launch review

The pinned PP4/VPP2 overlap schedule gives physical stage 3 five warmup calls: microbatches 0–3 on chunk 0, then microbatch 0 on chunk 1.
Both telemetry runs failed during that fifth call, the first warmup call whose forward includes the last virtual chunk and LM head.
From the signed source, dense logits at `[4096, 2, 200064]` occupy 3.05 GiB in BF16 and the native cross-entropy FP32 conversion occupies 6.10 GiB, per rank, when materialized.
The schedule call also runs MoE before the head; neither those tensor sizes nor a call-5 entry distinguishes the culprit.

The revised observer timestamps and flushes FNUZ dispatch, output postprocess, native loss and FP32-logits boundaries for the first eight schedule calls on all 16 final-stage ranks.
It records tensor shape, dtype and logical bytes where available, plus allocator counters, without retaining tensors or adding GPU synchronization.
The existing host telemetry continues on every rank.
Every ten seconds, the supervisors of the first and last local rank on each final-stage node record up to 32 of the largest resident mappings of 128 MiB or larger from the training leader's `/proc/<pid>/smaps`.
The samples retain backing path, device, inode, offset and RSS/PSS counters, not mapping contents; unavailable reads are recorded explicitly.
Correlate the growth and owner with the last successful boundary.
Do not sum shared mappings across ranks as distinct physical bytes or treat unlabelled, unobserved, or driver-only mappings as zero.
The eight sampled ranks are 48, 51, 52, 55, 56, 59, 60, and 63.

This attribution run keeps the four-thread cap, full architecture and batch, image, source, deadlines, environment, and placement request from `759605`.
It still attempts one update without checkpoint I/O, shared-cache publication or a new numerical path.
The scope is diagnostic; timing is not a benchmark.
If attribution yields no actionable memory change, review a separately labelled shallower model before changing architecture.
Prefer preserving hidden, expert and vocabulary widths, sequence and batch, optimizer, and parallel layout; measure it as a reduced-model result, not full Chonk throughput.
If evidence instead points to the output head, review a bounded-memory loss implementation separately before assuming fewer layers are the best remedy.

194 focused CPU and scheduler-stub tests passed, including mapping parsing and persistence of trace entry on failure.
At review `759605` was terminal, no other investigation job was active or pending, and 34 eligible general-partition nodes were idle.
Use the same exact command, explicit launcher environment and output root in [the OOM telemetry launch review](#full-layout-oom-telemetry-launch-review), with the revised declaration, validator and runner instrumentation.
This is one outside-sandbox run with normal host filesystem access on 16 exclusive `csstaff` / general `mi300` nodes, `nid002450` excluded, a 60-minute allocation and a 45-minute child deadline.
Keep the launch guard until approval and recheck the investigation cap immediately before submitting.
The user approved the command, and the runner submitted job `759635` once after verifying the frozen inputs and confirming no other active investigation job.
The [attribution run](../../artifacts/experiment-data/runs/chonk-native-fnuz-oom-i910bfcf936d3/20261123T094707Z-a5749313/) has its own frozen tooling and retained telemetry; the one-shot launch guard is back in place.

#### `759635` attribution evidence

All 64 ranks entered update 1 and initialized native CXI.
On warmup call 5, all 16 final-stage ranks returned from the traced FNUZ dispatch calls, entered the LM head, and returned from the native loss's FP32 logits conversion.
The trace confirms `[4096, 2, 200064]` BF16 logits (3.05 GiB) and FP32 logits (6.10 GiB) with approximately 6.1 GiB more Torch-reserved memory across the conversion.
Rank 55 on `nid002484` never recorded a loss return and its supervisor observed `SIGKILL` at 10:54:32.131 site time.
Four final-stage ranks returned from the loss and entered call 6; no update completed or DeepEP dispatch timeout was logged.

The eight watched training processes each showed up to two new 8 GiB shared `/dev/zero (deleted)` mappings after the FP32 conversion, with growing resident pages and separate backing inodes on the sampled processes.
On `nid002484`, `MemAvailable` fell from 135.28 GiB just before the head to a sampled 1.31 GiB, and `Shmem` rose from 9.61 GiB to 69.69 GiB while the `/dev/shm` mount remained near 3.53 GiB.
The call trace narrows the failure to the first native loss window after FP32 conversion, rather than a logged dispatch failure.
The mapping metadata identifies anonymous shared mappings, not the library or allocation call that created them.
The first targeted mitigation to review is a bounded-memory output/loss path on the unchanged model, with loss and gradient qualification before training.
Keep a shallower model as a separate fallback if the loss-path change cannot be qualified or does not recover enough headroom.

### Bounded LM output: signed handoff

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-chunked-lm-head`.
Branch: `experiment/amd-chonk-chunked-lm-head`.
Parent: signed integrated Chonk `4b21bb5781ee6db66952a9b6ab128b688c7dc1d1`.
Suggested signed subject: `Bound full-vocabulary Chonk loss with a recomputed token-chunked head`.

| Reviewed file | Expected Git blob |
| --- | --- |
| `megatron/core/models/gpt/gpt_model.py` | `2313863356ff06fa2a9ad956bc341903dd23a044` |
| `megatron/core/models/gpt/chunked_lm_head.py` | `a1dc177bca6a8f209d9c531c87580192cd090c7a` |
| `megatron/core/transformer/transformer_config.py` | `7ff48cab82c85a1d9efef7266ba231d93dbfe2db` |
| `tests/test_utils/python_scripts/qualify_chunked_lm_head.py` | `fd342a338272af90060ee2421134dbb6d377b3e2` |

`--lm-head-token-chunk-size 512` splits the output projection and existing native fused cross-entropy across token positions, not vocabulary.
It is off by default and restricted to the compatible TP1 BF16 training path.
The existing output parameter, state-dict keys and optimizer grouping are unchanged.
Forward saves hidden states, labels and the weight reference rather than any vocabulary-sized logits or softmax; backward recomputes each chunk and emits one output-weight gradient through the original autograd path.
The BF16 and FP32 logits for a 512-position tile total approximately 0.57 GiB, excluding workspaces and backward buffers.
The backward path accumulates BF16 partial weight-gradient products in one FP32 buffer; the reduction order and partial rounding differ from the original unsplit BF16 gradient.

Four CPU/Gloo fixtures passed with Torch `2.9.1+cpu`, compilation disabled, and the signed parent's native fused CE as the reference.
The worst absolute loss difference was `9.54e-7`; input gradients matched in these fixtures, and worst wgrad relative L2 and scaled maximum were 0.282% and 0.500%.
The CPU checks also exercised the real GPT postprocess branch, dense behavior when disabled and for eval/no-label output, flag registration, supported config checks, finite gradients, one weight gradient hook, ragged chunks, and absence of saved dense logits.
This does not establish numerics or memory behavior for image 910 on gfx942, actual GPU DDP behavior, or full-layout fit.

The [GPU component declaration](../chonk-chunked-lm-head-i910bfcf936d3/experiment.toml) now pins signed `aee1e9abab07e644653f71da98e35835059de10d`, tree `736a978d9e0e4b0ab0f846b1001e6fa49f460aae`.
The signature matches the established signer of the Beverin base and integrated Chonk merge; the sole parent and four reviewed blobs match, and the worktree is clean.
The next GPU gate is one gfx942 rank on image 910 with actual shape `[4096, 2, 3584]`, vocabulary 200064, the same native fused CE, and cold private caches.
Predeclared limits are maximum absolute loss error `5e-4`; dgrad relative L2/scaled maximum `1%/3%`; wgrad `1.5%/3%`; all finite; no saved vocabulary-sized logits; at least 3 GiB reduction in the incremental Torch allocation peak.
The allocation peak does not include node-wide shared mappings or non-Torch allocations.
Passing the component permits a separate review of full-layout memory and finite updates with the existing telemetry, not a claim that the shared mapping surge is resolved.

#### Chunked output GPU component: launch review

Use one exclusive node in Beverin `csstaff` / `mi300` / `cscs`, one gfx942 GPU, 48 CPUs bound to cores, all allocatable node memory, a 25-minute allocation and a 900-second child deadline.
There is no dataset build, training run, checkpoint I/O, shared cache hydration or publication.
The key outcomes are full-shape native loss and gradients, persistence of the original weight gradient hook, bounded logits/saves, and incremental Torch allocated peak; any error or missed limit is a failed gate, not authorization to retry.
The Slurm execution is **outside the agent sandbox with normal host filesystem access**.
The command below uses the same explicit host allowlist as prior component checks, with Slurm metadata and the pinned container's image and declaration environment.
From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposal is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-chunked-lm-head-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-chunked-lm-head \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:25:00 --cache-mode cold
```

Outputs will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-chunked-lm-head-i910bfcf936d3/<run-id>/`, committed snapshots under the adjacent `experiment-data/sources/`, and private caches and temporary files under runner-owned `/dev/shm/apertus-amd/` directories.
Recheck the investigation cap and placement before launching and leave unrelated jobs alone.
The user approved this exact component launch.
Job `759658` was submitted once on `nid002774`, with no other active investigation allocation and no overlap with the unrelated interactive job.
The [frozen run](../../artifacts/experiment-data/runs/chonk-chunked-lm-head-i910bfcf936d3/20261123T101757Z-f6e84b63/) records source archive SHA-256 `270309b01f6b34877f7f5ee5b9b510f1e19fcdc165b5b0879fecdc1b942705a6`; the launch guard is back in place.
Job `759658` passed on gfx942 in 2:07 with clean runner teardown and no shared cache publication.
The [rank report](../../artifacts/experiment-data/runs/chonk-chunked-lm-head-i910bfcf936d3/20261123T101757Z-f6e84b63/output/rank-0.log) recorded maximum absolute loss error `2.86e-6`, input-gradient relative L2/scaled maximum `0.00270%/0.180%`, and weight-gradient relative L2/scaled maximum `0.122%/0.379%`.
The incremental Torch allocated peak fell from 9.24 GiB to 4.06 GiB, a reduction of 5.18 GiB (56%), exceeding the predeclared 3 GiB gate.
The component also passed the bounds on chunk size and forward-saved tensors and observed one ordinary output-weight gradient notification.
This is single-GPU component evidence, not evidence of lower node-wide shared memory or a completed full-layout update.

#### Chunked full-layout fit: launch review

The [new declaration](../chonk-native-fnuz-chunked-fit-i910bfcf936d3/experiment.toml) pins signed `aee1e9abab07e644653f71da98e35835059de10d` and opts in with `--lm-head-token-chunk-size 512`.
Its resolved inputs match the frozen `759635` attribution run except for the separate experiment ID, source pin, new flag, validator digest and review guard.
It uses unchanged image `910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490` and the same 42-layer workload, S4096/MBS2/GBS4096, PP4/VPP2/EP16, native FNUZ experts and dispatch, CXI, optimizer and four effective Inductor compile threads per rank.
The source signature, sole parent, tree and four reviewed blobs were reverified against the clean worktree.
The planned validator SHA-256 is `169b05624941ce5e84e7101a2b420ced57644dd0134316e6fb23867f62281307`; 218 focused CPU and scheduler-stub tests passed.

The pass gate is one completed finite, unskipped update with no rerun or discarded batch, CXI initialization and clean completion on all 64 ranks, all required telemetry, and no checkpoint or shared cache hydration/publication.
The effective native CLI and transformer config must both use 512 tokens and native fused CE.
At warmup call 5 (forward virtual microbatch 4, no backward), all 16 final-stage ranks must show one head entry and exit enclosing 16 ordered CE calls with BF16 and FP32 `[256, 2, 200064]` logits, respectively 204,865,536 and 409,731,072 bytes per chunk.
Compare host memory and shared-mapping samples, Torch allocated/reserved peaks and HIP free memory with the comparable window in `759635`, recording missing samples explicitly.
There is no minimum memory-reduction claim for this run beyond fitting the update.
An OOM, failed rank, timeout, nonfinite/skip/rerun, or missing gate evidence means stop, retain artifacts and localize the last successful boundary without a retry or walltime extension.
Checkpoint integrity, actual restart, cache reuse and throughput remain separate gates even if this run passes.

One exclusive allocation in Beverin `csstaff` / general `mi300` (no reservation) requests 16 nodes, 4 MI300A GPUs and ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, all allocatable memory, `pmix`, and `disable_rdzv_get`, excluding `nid002450`.
The allocation walltime is 60 minutes and the training child deadline is 2,700 seconds.
At review no investigation job was active or pending, and 25 eligible general nodes were idle after the exclusion.
Recheck both general and reserved occupancy against the combined 16-node investigation cap immediately before submission and leave unrelated jobs untouched.
This topology is not available in the agent's current allocation; Slurm will execute **outside the agent sandbox with normal host filesystem access**.
The launcher environment uses the explicit allowlist below; the pinned container uses the declaration's environment plus scheduler/runner metadata and `--uenv-passthrough=ignore`.
From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposal is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-chunked-fit-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-chunked-lm-head \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-chunked-fit-i910bfcf936d3/<run-id>/`, source snapshots to the adjacent `experiment-data/sources/`, and private scratch and rank caches to job-owned `/dev/shm/apertus-amd/` directories.
The user approved this exact one-shot launch.
Job `759718` was submitted once after rechecking that no investigation job was active or pending and 26 eligible general nodes were idle.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-chunked-fit-i910bfcf936d3/20261123T104601Z-ecb47e77/) is separate from the dense attribution run; the launch guard is back in place.

#### `759718` first-loss evidence

All 64 ranks entered update 1 with the effective 512-token cap and four Inductor threads.
All 16 final-stage ranks completed warmup call 5, and the frozen validator's trace check confirmed exactly 16 ordered BF16/FP32 logits chunks per rank.
In the first 30 seconds after call-5 head entry, the four final-stage nodes in `759718` had at least 67.97 GiB sampled `MemAvailable` and at most 15.14 GiB sampled `Shmem`; the dense comparator `759635` sampled near zero available and up to 73.68 GiB `Shmem` in the same relative window.
The largest sampled shared `/dev/zero (deleted)` mapping on each of the eight watched ranks was 0.5 GiB versus 8 GiB in the dense comparator, while `/dev/shm` mount use in that window peaked at 4.32 GiB versus 3.54 GiB.
These are sampled, phase-aligned observations across different placements, not whole-update peaks, proof of mapping ownership, or a full-layout fit verdict.

#### `759718` full-layout fit verdict: failed

All 64 ranks reported CXI initialization, but none completed update 1.
The second physical pipeline stage (PP rank 1) completed eight warmup calls on all ranks and stalled in call 9 (forward virtual microbatch 8, no backward).
At call entry, the four node leads showed 70.40–72.07 GiB allocated and 72.33–74.82 GiB reserved by Torch per rank.
On `nid002448` and `nid002452` (ranks 20–27), sampled node `MemAvailable` reached zero about 18 seconds into that call and all eight rank supervisors stopped writing live samples at about 18–23 seconds, with no recorded child exit before cancellation.
On those nodes `Shmem` remained around 9.6 GiB and `/dev/shm` around 3.52 GiB, unlike the prior final-stage loss spike.
The other eight ranks in the same EP group (16–19 and 28–31 on `nid002446` and `nid002454`) reported `DeepEP error: timeout (dispatch CPU)` in `internode_prepare` roughly 113–119 seconds into call 9.
The sample loss and memory exhaustion preceded those tracebacks by around 90 seconds; this points to node memory pressure upstream of the observed transport timeout, but no kernel OOM event or process exit was recorded to establish the cause.
The final stage progressed beyond its first loss and reached the first backward overlap calls.
The owned probe was cancelled under its stop rule after 11:22 when the timeouts had been recorded but the runner had not exited.
There was no checkpoint directory or authoritative runner pass result, and no rerun or walltime extension was submitted.

Keep the validated chunked head as the mitigation for the first-loss spike, not as proof of full-layout fit.
Before considering a new GPU run, review the PP-rank-1 warmup memory budget and whether a resident allocation-lifetime change can recover headroom without changing the declared model or numerical path.
One candidate is to retain the original FNUZ FC1 input and regenerate the saved FC1 preactivation with the same native GEMM, versioned weights and row quantizer when backward needs it.
At balanced routing and ignoring padding, omitting that saved preactivation for six MoE layers across nine warmup forwards would remove about 13.05 GiB of retained tensors per rank before recompute transients.
That estimate is source arithmetic, not a prediction of observed node-memory savings; an extra GEMM in backward and unchanged numerical and weight-lifetime behavior would need separate qualification.
Review a separately labelled shallower fallback if that cannot be justified; do not raise the DeepEP timeout or repeat this probe unchanged.

#### Resident expert FC1 replay: signed handoff

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute`.
Branch: `experiment/amd-chonk-fnuz-recompute`.
Sole parent: signed chunked-head commit `aee1e9abab07e644653f71da98e35835059de10d`.
Suggested signed subject: `Replay native resident FNUZ FC1 during backward`.

| Reviewed file | Expected Git blob |
| --- | --- |
| `megatron/core/transformer/moe/experts.py` | `84dbe2386958d4f7ce955c8daba98f12bf23c21a` |
| `megatron/core/transformer/moe/resident_fp8.py` | `23cfdee9825092984d01bd06cc733e6072e4bc5f` |
| `megatron/core/transformer/transformer_config.py` | `be4d74b7aecf67a6cafe30540a4ca950c4782d98` |
| `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py` | `a631254f6a4b993358041860c42cc021a08b0f1c` |

`--moe-resident-fp8-fnuz-recompute-fc1` is off by default and requires resident FNUZ.
Forward retains the same FNUZ FC1 input and FP32 router probabilities, but does not retain the row-quantized FC1 preactivation when replay is enabled.
Backward checks the weight version before replaying FC1 one active expert at a time, using the original native `TN` projection flags, operand bytes and cached weight generation.
It then applies the same padded-tail zeroing and BF16 → rowwise FNUZ → BF16 boundary as the stored path before the existing SSSGLU derivative and delayed FP32 wgrad.
No new forward GEMM, transport, model/optimizer layout, or offload path is introduced.

CPU lifecycle and arithmetic fixtures passed with replay enabled and disabled using software GEMMs and simulated streams.
Those tests check the saved operand slot, operation counts, empty/ragged cases, `main_grad` accumulation, callback ordering and rejection before replay on a stale generation.
The CLI default and fail-closed config were checked, four CPU DDP/version tests passed, and 83 focused Apertus tests passed.
The optional CPU optimizer mapping rerun hit the same `newton_schulz_tp(..., tp_mode=...)` API mismatch on the signed parent and candidate with the local Megatron lockfile's `emerging-optimizers` dependency; there is no new optimizer-equivalence result.

The [GPU component declaration](../chonk-native-fnuz-fc1-recompute-i910bfcf936d3/experiment.toml) now pins `4c34a4238fd22f2ea033b95b044a4087b4bf56a1`, tree `a52966621ada3cf86cbbba9765ed54e52bb2273d`.
The established signature, sole parent, exact four changed blobs and clean worktree were verified.
The declaration remains blocked until the exact outside-sandbox launch is approved.
On one MI300A with image 910, it will require all six original native traces plus the replay helper's native FNUZ `TN` trace, exact regenerated decoded operand on a padded 129-row fixture, and the existing matching FNUZ arithmetic limits (relative L2 1.8%, scaled maximum 4%) with ragged, skewed and empty routes.
It also checks delayed FP32 accumulation, notification and weight-generation refresh, in-memory restore, and no retained FC1 preactivation in the replay mode.
For a separate allocator gate, it compares two in-flight forwards at 65,536 routed rows per rank (balanced across 16 local experts at widths 1792/1920), after priming identical resident weight caches.
The expected unpadded retained reduction is 495 MiB; the predeclared gate is at least 450 MiB lower retained Torch allocation after the two forwards, with finite backward/drain on both paths.
Forward and total Torch peaks are descriptive and separate from retained allocation.
No single-GPU component result can establish a full-node memory reduction or a completed full-layout update.

#### Resident expert FC1 replay: GPU launch review

Use one exclusive Beverin `csstaff` / `mi300` / `cscs` reservation node, one MI300A GPU, 48 CPUs bound to cores and all allocatable node memory, with a 25-minute allocation and 900-second child deadline.
The agent's interactive allocation is not part of the investigation and will not be used.
At review there were no active or pending investigation jobs and 16 idle reserved four-GPU nodes.
Recheck occupancy across general and reserved jobs against the combined 16-node cap immediately before submission.
Slurm runs **outside the agent sandbox, with normal host filesystem access** and `--uenv-passthrough=ignore`.
The approved image digest is unchanged; the isolated launcher environment uses the allowlist below and the container uses the declaration environment plus scheduler/runner metadata.
There is no training update, checkpoint I/O, shared cache hydration or publication.
Any missed gate, timeout or failure stops this one-shot qualification; retain artifacts without retrying or extending the deadline.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposal is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-fc1-recompute-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:25:00 --cache-mode cold
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-fc1-recompute-i910bfcf936d3/<run-id>/` with native GPU traces under `output/expert-traces`.
Committed source snapshots go to the adjacent `experiment-data/sources/`, and private rank scratch and caches to job-owned `/dev/shm/apertus-amd/`.
The user approved this exact launch.
Preflight stopped before submission because `uv` could not refresh the launcher environment; the partial `.venv` was preserved under `artifacts/experiment-data/venv-recovery.JleEOt/` and rebuilt at the original path with `uv sync --locked`, with no change to the reviewed command or GPU inputs.
The runner submitted job `759931` once after rechecking that no investigation job was active or pending and 32 reserved four-GPU nodes were idle.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-fc1-recompute-i910bfcf936d3/20261123T122220Z-e828ee17/) retains its exact inputs; the launch guard is back in place.

#### `759931` failed gate and test-only signed handoff

Job `759931` failed in 4:17 on `nid002774`, with completed runner cleanup.
The six original native trace fixtures passed, and the replay trace shows one native HipKittens FNUZ GEMM with BF16 output.
The exact decoded comparison on the padded 129-row replay fixture returned successfully before the failure.
`prove_native_replay` intentionally uses the real adapter helper, but the signed script had already installed `ObservedBackend`, adding a replay GEMM and row cast to the counters reserved for the subsequent lifecycle cases.
The first lifecycle assertion therefore failed on an unexpected pre-existing replay count, before lifecycle backward or the retained-allocation gate ran.
No lifecycle, memory, or full-layout pass is claimed.

The test-only correction runs the standalone trace with the unwrapped native backend, then attaches the observer before the lifecycle workload.
It also prints an explicit success line for the exact padded-operand comparison.
No production blob, fixture input, numerical limit, or allocation bound changed.
A CPU mock reproduced the extra replay count with the signed parent and verified empty lifecycle counters with the correction.
Both CPU lifecycle modes also passed after the change.

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check`.
Branch: `experiment/amd-chonk-fnuz-recompute-check`.
Sole parent: `4c34a4238fd22f2ea033b95b044a4087b4bf56a1`.
Only changed file: `tests/test_utils/python_scripts/qualify_resident_fnuz_experts.py`, expected Git blob `928565c5cc7be43ec3f48cd5e9b4153c5e1a24d6`.
Suggested signed subject: `Isolate FC1 replay trace from lifecycle counters`.
The failed declaration and run are retained.
Signed commit `ceb8035c182c40711db4d256d99fc44e4b8db27e` (tree `f3b8a0d4e7aa07fe157c81b95e12cc8d478aa86d`) matches the established signer, sole parent, and single reviewed blob in the clean worktree.

#### Corrected FC1 replay component: launch review

The [new declaration](../chonk-native-fnuz-fc1-recompute-check-i910bfcf936d3/experiment.toml) preserves `759931`'s image, component arguments, environment, and resource bounds.
The signed source changes only the test harness; the numerical and retained-allocation gates are unchanged.
Use one exclusive Beverin `csstaff` / `mi300` / `cscs` reservation node, one MI300A GPU, 48 CPUs bound to cores, all allocatable memory, a 25-minute allocation and a 900-second child deadline.
At review there were no active or pending investigation jobs and 32 idle reserved four-GPU nodes.
Recheck general and reserved occupancy against the combined 16-node investigation cap before submission; do not touch unrelated jobs.
Slurm will run **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
The launcher uses the explicit environment below, and the pinned container uses the declaration environment plus scheduler/runner metadata.
No full training, checkpoint, or shared-cache access is requested; any failure ends this one-shot run without retry or walltime extension.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-fc1-recompute-check-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:25:00 --cache-mode cold
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-fc1-recompute-check-i910bfcf936d3/<run-id>/`, with native kernel traces under `output/expert-traces`.
Source snapshots go to the adjacent `experiment-data/sources/`, and private rank scratch and caches go to job-owned `/dev/shm/apertus-amd/`.
The user approved this exact command.
Job `759951` was submitted once after rechecking that there were no active or pending investigation jobs and 32 reserved nodes were idle.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-fc1-recompute-check-i910bfcf936d3/20261123T123447Z-3f645803/) is separate from `759931`, and the launch guard is back in place.

#### `759951` FC1 replay component verdict: passed

Job `759951` completed in 3:04 on `nid002774`, with clean runner teardown and no shared cache publication.
The frozen source archive digest is `d8224113d37f3750f7291bb6764ea38b236f02242ddd52101cabd1381fbdaac4`.
All six original native FNUZ traces passed, and the additional `TN` FC1 replay trace passed with an exact regenerated decoded operand for the padded fixture.
The ragged, skewed and empty lifecycle passed, including delayed FP32 accumulation, router probability gradients, ready notifications, generation refresh, and in-memory weight restoration.
Across matching FNUZ oracle checks, the worst relative L2 was 0.064% and worst scaled maximum was 1.098%, within the respective 1.8% and 4% limits.
FN-format differences remain separate descriptive evidence.
Across the two actual-shape in-flight forwards, retained Torch allocation fell from 1,206 to 679 MiB: **527 MiB saved**, above the 450 MiB gate.
The incremental forward and total Torch peaks fell by 519.0 and 424.25 MiB, respectively; both backward/drain cases were finite.
This qualifies the component, not full-node headroom, a full-layout update, or throughput.

#### Full-layout fit with FC1 replay: launch review

The [new declaration](../chonk-native-fnuz-replay-fit-i910bfcf936d3/experiment.toml) retains the chunked head and adds only `--moe-resident-fp8-fnuz-recompute-fc1` to the training argv.
The resolved inputs match frozen `759718` except for the source pin, replay flag, validator digest, experiment ID and review guard.
The source pin is signed `ceb8035c182c40711db4d256d99fc44e4b8db27e`, which qualified replay in `759951`.
The 42-layer model, S4096/MBS2/GBS4096, PP4/VPP2/EP16 layout, optimizer, native precision and transports, image, and four effective Inductor threads per rank are unchanged.
The validator SHA-256 is `298ec71dfd0460f7b9fec26d2d4632b4987d8cf6f723bf8796eff3b69dd39e1f`; 225 focused CPU and scheduler-stub tests passed.

Require one finite, unskipped update with no rerun, discard, or checkpoint, all 64 ranks using CXI and completing cleanly, and all existing telemetry and first-loss chunk checks.
Require the effective replay flag plus one completed replay in a backward-bearing call on every rank; sampling the first replay avoids per-expert logging throughout the update.
Require all 16 PP-rank-1 ranks to exit forward-only warmup call 9 and record dispatch boundaries there for diagnosis.
Keep the existing final-stage mapping probes and all-rank host/GPU telemetry; compare the recorded memory at the previous pressure boundary against `759718`.
No claimed minimum node-wide memory reduction; the fit gate is the completed update with the qualified recipe.
Stop at an OOM, rank error, timeout, nonfinite/skip/rerun, or missing gate evidence; retain the artifacts without an automatic retry or extension.
Checkpoint integrity, actual restart, cache reuse, and throughput remain separate gates.

Request one exclusive Beverin `csstaff` / general `mi300` allocation with no reservation, 16 nodes, four MI300A GPUs and ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, all allocatable memory, and `nid002450` excluded.
The allocation walltime is 60 minutes and the training child deadline is 2,700 seconds.
At review no investigation job was active or pending, but only 12 eligible general nodes were idle.
Recheck occupancy immediately before any submission and launch only when at least 16 eligible nodes are idle and the combined investigation cap permits it; do not change placement or touch unrelated jobs.
Slurm executes **outside the agent sandbox with normal host filesystem access**; the runner uses `--uenv-passthrough=ignore`, the same pinned image and declared environment, and the explicit host allowlist below.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-replay-fit-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-replay-fit-i910bfcf936d3/<run-id>/`, source snapshots to the adjacent `experiment-data/sources/`, and private scratch and rank caches to job-owned `/dev/shm/apertus-amd/` directories.
The user approved this exact command.
Job `759990` was submitted once after rechecking that no investigation job was active or pending and 24 eligible general nodes were idle.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-replay-fit-i910bfcf936d3/20261123T124709Z-7c772b9b/) is separate from `759718`, and the launch guard is back in place.

#### `759990` warmup evidence

All 64 ranks initialized native CXI and entered update 1 after a prolonged startup.
All 16 final-stage ranks completed the first loss call, and the frozen validator confirmed the expected 16 BF16/FP32 chunks on every one of those ranks.
All 16 ranks of the second physical pipeline stage cleared forward-only call 9, where `759718` stalled.
At call-9 entry, Torch allocated 59.06–59.78 GiB per rank on its four node leads, versus 70.40–72.07 GiB in `759718`.
In the first 30 seconds after call-9 entry, the lowest sampled node `MemAvailable` across that stage was 38.26 GiB, versus 0 GiB in the prior placement.
These are phase-aligned samples from different placements, not whole-update peaks or a full-layout fit verdict.
The final-stage ranks also recorded the first FC1 replay return in backward.

#### `759990` full-layout fit verdict: failed

No rank completed update 1.
All ranks on physical stages 2 and 3 recorded an FC1 replay return, but physical stages 0 and 1 did not reach that observed backward boundary.
All 16 final-stage ranks entered call 8 (forward virtual microbatch 7, backward virtual microbatch 2), reached a native FNUZ forward dispatch, and did not record a dispatch return.
About 101 seconds later, all 16 reported `DeepEP error: timeout (dispatch CPU)` in `internode_prepare`.
The neighbouring stage's first FC1 replay returned near the time the final stage entered call 8, but its call 8 had not exited by cancellation.
These are host-side timestamps and do not establish completion of queued GPU work or identify the first blocked stream or transport operation.

The lowest sampled `MemAvailable` during the training step was 38.26 GiB across the second-stage nodes and 81.59 GiB across the final-stage nodes.
Host telemetry remained live until cancellation, and no pre-cancellation OOM or child `SIGKILL` was recorded.
This differs from the two previous memory-pressure failures, but the samples cannot rule out a device-side stall or an unsampled transient.
The owned probe was cancelled at 35:38 under its stop rule after the dispatch errors were recorded.
No checkpoint or authoritative runner pass result was written, and no retry or deadline extension was submitted.

Keep the qualified output-head and FC1 memory changes, but do not treat the cleared warmup boundaries as full-layout fit.
Before another launch, review operation ordering at the first backward/forward overlap across physical stages 2 and 3.
Any proposed diagnostic should distinguish a queued producer or pipeline dependency from stalled CXI progress at the final-stage dispatch, without extending the transport timeout or replaying the same run unchanged.

##### `759990` dispatch-timeout attribution

The retained traces show one forward dispatch entry and no return on each final-stage rank in call 8, with all 16 host entries within 93 ms and all errors 101.06–101.08 seconds after the corresponding entry.
Each rank had returned from the native CE calls for all 16 backward recompute chunks before entering that dispatch.
These are Python boundaries, not GPU completion timestamps.
They rule out a missing or very late host caller at this dispatch boundary, but not a late GPU producer or peer notification.

Reading the installed wrapper and retained C++ source directly from image 910 shows that `internode_prepare` queues `notify_dispatch` behind the dispatch-layout event, then polls mapped receive-count words on the CPU.
The payload receive buffers are allocated only after `internode_prepare` returns.
The failure is therefore in the count-notification phase, before payload exchange, and does not implicate the width-1792 FNUZ payload or scale-copy kernel at this boundary.
The notification kernel drains prior requests with `quiet`, then performs cross-node and node-local synchronization before exchanging counts.
The timeout alone does not distinguish an earlier stream dependency, a notification-kernel stall, or transport progress.

In the pinned PP4/VPP2 schedule, final-stage call 8 pairs virtual chunk 1 forward microbatch 3 with chunk 1 backward microbatch 2.
The combined layer schedule issues backward combine and expert dgrad before forward dispatch, and only enters delayed expert wgrad and reverse dispatch after that forward dispatch returns.
Pipeline sends and MoE communication also share a Megatron communication stream, while UCCL has its own stream waiting on the layout event.
The later operations in the same layer pair cannot be assumed to have run, and the earlier host returns do not show when their device work completed.
If the reduced run does not settle the next decision, a diagnostic should record only the implicated calls and tag the paired layers, prior combine, producer/layout event, UCCL notification-stream predecessor, and pipeline send/receive completion.
Query GPU event completion without a device-wide synchronization in the measured path; do not silently change the scheduler, native format, or timeout.
The installed UCCL `EventHandle` binding exposes a stream wait but no query, so a Python observer would need separate Torch event markers at the producer and layout boundaries.
The installed DeepEP Python wrapper records its no-argument `EventHandle()` on the current PyTorch stream explicitly, not on the legacy default stream used by the C++ no-argument constructor.
There is no source evidence that the caller accidentally selected the legacy stream through that wrapper.
An owned marker queued on the UCCL stream immediately after layout can distinguish stalled predecessor work from an unreturned notification after layout completion.
Neither marker identifies the notification's internal `quiet`, barrier, or count-exchange phase; that would require separate native evidence.

##### Full-layout notification diagnostic: launch review

The [new declaration](../chonk-native-fnuz-notify-trace-i910bfcf936d3/experiment.toml) matches the frozen `759990` source, training arguments, image, resources, cache policy and deadlines.
It adds opt-in runner instrumentation only (`APERTUS_CHONK_NOTIFY_TRACE=1`, validator SHA-256 `6c1c0de5b848c46629f69467ef803711d58fbf9487c6c831f1614ca7256fa4f6`), with no Megatron or image changes.
The clean signed `ceb8035c182c40711db4d256d99fc44e4b8db27e` worktree was reverified and 236 focused CPU and scheduler-stub tests passed.

On ranks 48–63 only, trace the first FNUZ dispatch in call 8 (forward virtual microbatch 7, backward 2).
Record three timing-disabled Torch events: on the Megatron caller stream before layout, on the UCCL stream before layout, and on the UCCL stream after layout.
Query those events at dispatch return or exception, from latest to earliest, without a device-wide synchronization or an added cross-stream wait.
If the same timeout recurs and layout is complete on all 16 ranks, the block is after layout on the notification path; this does not identify CXI as the cause.
If layout is incomplete, separate evidence of unfinished caller work, unfinished prior UCCL work, and an incomplete layout or stream dependency.
Missing markers, unavailable queries, or a failure before the boundary are inconclusive for that rank; preserve them as such.
If the timeout does not recur, retain the trace but do not infer the original cause.
A fit pass still requires usable markers on every final-stage rank and all of the previous one-update acceptance checks.
No checkpoints, shared cache, retry, or deadline extension.

The agent allocation cannot provide the full topology; the launch runs **outside the agent sandbox with normal host filesystem access**.
It uses an exclusive allocation on 16 general Beverin `mi300` nodes under `csstaff`, four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, a 60-minute allocation, and a 2,700-second training deadline.
`nid002450` remains excluded, and the runner uses `--uenv-passthrough=ignore`.
At review there were 27 eligible idle general nodes and no investigation allocation; recheck immediately before submission and keep the combined cap at 16 nodes.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-notify-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-notify-trace-i910bfcf936d3/<run-id>/`, source snapshots to the adjacent `experiment-data/sources/`, and private scratch and rank caches to job-owned `/dev/shm/apertus-amd/`.
After reviewing the startup pass and confirming job `760046` had left the queue, the approved notification diagnostic was submitted once as job `760048`.
There were 28 eligible idle general nodes and no active or pending investigation jobs at submission.
The [frozen full-layout run](../../artifacts/experiment-data/runs/chonk-native-fnuz-notify-trace-i910bfcf936d3/20261123T141723Z-63d86c6c/) is separate from `759990` and its launch guard is back in place.

##### `760048` notification diagnostic: startup failed before dispatch

At 14:23:22 UTC, four ranks on `nid002458` reached the unchanged 300-second GPU identity deadline and recorded 44 missing identities; the 16 other ranks already waiting recorded peer failure.
Only 20 ranks on five nodes published identities, although 60 ranks on 15 nodes published environment records.
`nid002458` published environment records at 14:17:55 UTC and identity records about 27 seconds later.
The other 14 nodes with environment records published them about 171 seconds later; four of those nodes then published identities about 44 seconds after their environment records.
At cancellation, ten nodes had environment records but no identities, and `nid002512` had neither.
These file timestamps distinguish delays before and after environment publication, but do not identify the blocking call or exclude delayed filesystem visibility.
The same 16 physical nodes had just passed the startup-only probe `760046`, so placement alone does not explain the failure.

The owned job was cancelled under the stop rule; Slurm recorded `CANCELLED` at 6:49, separately from the retained startup failure records.
There was no cache selection, training child, validator verdict, or dispatch observation.
This run neither reproduces nor localizes the `759990` timeout, and it adds no fit evidence for either model depth.
Before another training submission, correlate the retained UTC timestamps with site-side Slurm/EDF and filesystem telemetry, or review a bounded startup trace that also covers the normal launch path and keeps the 300-second wait.
No retry or deadline extension was submitted.

##### `760048` one-shot retry: launch review

The [retry declaration](../chonk-native-fnuz-notify-trace-startup600-i910bfcf936d3/experiment.toml) extends only the per-rank GPU identity rendezvous from 300 to 600 seconds.
`760023` missed the original wait by 7–9 seconds, `759990` passed it by roughly 1.5 seconds, and `760048` was cancelled before the 44 missing ranks could establish their actual arrival times.
The new bound tests whether late arrivals can complete startup; it does not resolve the cause of the skew.
The signed Megatron source, image, training arguments, dispatch markers, 60-minute allocation, 2,700-second child deadline, and DeepEP timeout remain unchanged.
The signed source and reviewed test blob were reverified, and 249 focused CPU and scheduler-stub tests passed.
The original fit and diagnostic gates remain in force; stop on any terminal rank failure, with no further retry or extension.

The full topology requires execution **outside the agent sandbox with normal host filesystem access**, on 16 exclusive general Beverin `mi300` nodes under `csstaff`, four MI300A ranks per node and 48 CPUs per rank, with `nid002450` excluded.
Recheck that the prior allocation has cleared and the combined 16-node investigation cap permits the launch.
Use cold private caches only, with no checkpoint, hydration, or publication.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-notify-trace-startup600-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-notify-trace-startup600-i910bfcf936d3/<run-id>/`.
The user approved the exact command.
Job `760061` was submitted once after job `760048` cleared, with no other investigation allocations and 34 eligible idle nodes.
The [frozen retry](../../artifacts/experiment-data/runs/chonk-native-fnuz-notify-trace-startup600-i910bfcf936d3/20261123T143237Z-1738a131/) is retained separately and the launch guard is back in place.

##### `760061` retry outcome: startup failed at 600 seconds

All 64 ranks published environment records, but only 52 published GPU identities before the first deadline at 14:43:02.784 UTC.
Ranks 60–63 on `nid002512` published identities about 507.5 seconds after the first identity, beyond the original 300-second wait.
Ranks 20–23, 28–31, and 36–39 on `nid002476`, `nid002484`, and `nid002488` published environment records about 510.5 seconds after the first identity, but published no identities.
The retained records include two rendezvous deadlines and 50 peer failures.
The same 16 nodes passed `760046`; the file timestamps do not identify the calls responsible for the intermittent startup delays.

The owned job was cancelled under the stop rule; Slurm recorded `CANCELLED` at 10:30.
There was no cache selection, training child, dispatch marker, or authoritative runner or validator verdict.
The extension admitted four late identities but did not clear startup, so it provides no evidence about the `759990` dispatch stall or fit at either depth.
Do not extend the deadline again on this evidence.

##### Full-size retry after reported Capstor improvement: launch review

The user reported that `/capstor` appears healthier and requested another attempt.
The prior jobs mounted the pinned image from `/capstor/store`, but wrote environment and GPU identity records to `/ritom/scratch`.
Image reads could have delayed both container launch and in-container Torch or ROCm library loading; the retained timestamps do not establish that cause.
The [new declaration](../chonk-native-fnuz-notify-startup-trace-i910bfcf936d3/experiment.toml) adds the startup-only probe's Bash and Python phase markers to the normal training path, rather than stopping after GPU identity validation.
If startup fails, use the per-rank launcher timestamps and failure records to localize the delay; a missing initial Bash marker does not establish which earlier component blocked.
If startup passes, continue to the unchanged call-8 dispatch diagnostic and one-update fit checks.
A pass after reported recovery would be consistent with a storage contribution, not proof of it.
The signed Megatron source, reviewed blob, image path and digest, training arguments, native dispatch instrumentation, 600-second GPU identity wait, 2,700-second training child deadline, and 60-minute allocation are unchanged from `760061`.
The new runner path passed 253 focused CPU and scheduler-stub tests.
No checkpoint, shared cache, automatic retry, or deadline extension; stop on a terminal rank error.

The full topology is unavailable inside the agent allocation.
Use 16 exclusive general Beverin `mi300` nodes under `csstaff`, four MI300A ranks per node, 48 CPUs per rank and all allocatable memory, excluding `nid002450`.
This launch runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
At review 28 eligible nodes were idle and no investigation jobs were active or pending; recheck the 16-node cap and placement before submission.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-notify-startup-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-notify-startup-trace-i910bfcf936d3/<run-id>/`, with startup markers in `output/launcher.log`.
After exact user approval, job `760303` was submitted once with no other investigation allocation active or pending and 36 eligible idle nodes.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-notify-startup-trace-i910bfcf936d3/20261123T160607Z-96808324/) is retained separately, and the launch guard is back in place.

##### `760303` startup trace: delay before Bash and during Torch import

All 64 in-container Bash markers arrived, with a 493.27-second spread from 16:06:21.326 to 16:14:34.596 UTC.
All 64 ranks then entered Torch import, but 24 ranks on six nodes had not reached the device-enumeration marker at the first identity deadline, 16:16:33.141 UTC.
Those ranks had emitted their import-start markers 118–377 seconds earlier.
The other 40 ranks reached GPU enumeration 9.57–40.72 seconds after import start, enumerated in 0.21–0.28 seconds, and published their identities.
Tooling and source-manifest verification took at most 0.068 seconds on all ranks, environment writes at most 0.067 seconds, and the 40 measured identity writes at most 0.028 seconds.
The trace localizes the observed delays to before the first in-container Bash marker and to the Torch import interval, not to the subsequent GPU enumeration or measured rendezvous-file writes.
The results are consistent with intermittent reads of the image on `/capstor/store`, but the trace does not identify a syscall, file, or storage server as the cause.

Slurm recorded the owned job as `CANCELLED` at 10:41 after four rendezvous deadlines and 36 peer failures were retained.
There was no training child, dispatch observation, or authoritative runner or validator verdict.
No new training launch is authorized.
If the site supports it, a bounded run using the exact same image bytes from an independently verified copy on healthy shared storage would remove the direct `/capstor/store` image dependency without changing the software.
Keep the startup markers and deadlines fixed for that comparison; one successful run would support, but not prove, a storage explanation.

##### Cached-image comparison: launch review

The [new declaration](../chonk-native-fnuz-notify-image-cache-i910bfcf936d3/experiment.toml) keeps the `760303` workload, traces, and all deadlines.
The only workload input that moves is the EDF image path, from the registry to shared `/ritom/scratch` in this checkout.
The digest is unchanged.
`--cache-image` stages the 9.44 GB SquashFS before submission if absent, verifies its complete SHA-256, and publishes it without replacing any existing file.
An existing copy is independently verified without accessing the registry, and no check failure falls back to Capstor.
The image copy is separate from the compiler cache, which remains cold and private.
261 focused CPU and scheduler-stub tests passed.

The full topology is unavailable inside the agent allocation, and the Slurm launch runs **outside the agent sandbox with normal host filesystem access**.
Use one exclusive 60-minute allocation under `csstaff` on 16 general Beverin `mi300` nodes, four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, and `nid002450` excluded.
The user requested immediate queueing rather than waiting for idle nodes, with the general queue as fallback if the `cscs` reservation is full.
At reservation review, 32 of its 33 nodes were unallocated, and no investigation job was active or pending.
Use only one queue, and recheck the combined 16-node cap before submitting.
No checkpoint, shared compiler cache, automatic retry, or deadline extension.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-notify-image-cache-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

The already reviewed general-queue fallback is the same command without `--reservation cscs`.
Registry input: `/capstor/store/cscs/cscs/public/team-ml/apertus-amd-images/objects/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
The copy was staged before submission and verified at 9,442,426,880 bytes against that SHA-256.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-notify-image-cache-i910bfcf936d3/<run-id>/`.
After exact user approval, job `761653` was submitted once to the `cscs` reservation after rechecking that no other investigation jobs were active or pending.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-notify-image-cache-i910bfcf936d3/20261124T091253Z-5afc63fd/) uses the verified checkout image, and the launch guard is back in place.
All 64 ranks passed GPU identity validation and started training children.
Torch import took 5.65–6.15 seconds per rank, and the first-to-last identity publication spread was 0.81 seconds.
This clears startup for this run and is consistent with a storage contribution to earlier delays, but the reservation also changed the physical nodes, so this is not an isolated test of causality.

##### `761653` dispatch trace: GPU work outstanding before layout

All 64 ranks entered the first update and logged native CXI initialization.
Final-stage ranks 48–63 entered the target FNUZ dispatch in call 8 (forward 7, backward 2) within 225 ms and queued layout on a stream distinct from the caller.
All 16 calls failed in `internode_prepare` with `DeepEP error: timeout (dispatch CPU)` about 101 seconds after entry; none returned.
On every rank all three event queries succeeded and returned `false`: the caller marker before layout, the prior-UCCL marker before layout, and the UCCL marker after layout.
The timeout therefore did not occur with GPU layout already complete.
Because both pre-layout markers were incomplete, the trace cannot tell whether caller work, earlier communication, or a dependency between streams stopped progress first.
All ranks had returned from 16 backward CE chunks at the host boundary; this is not evidence that those kernels completed on the GPU.

The minimum sampled final-stage node availability during the stall was 81.09 GiB, and no training log recorded an OOM or SIGKILL before cancellation.
The owned allocation was cancelled at 10:27 under the stop rule; Slurm recorded the training step as `FAILED` (`1:0`) and the rank-zero runner result as `failed`, without a validator pass.
There was no completed update or checkpoint, and the allocation cleared.
The next source review should map the stream dependencies before layout, particularly the preceding backward work and communication waits, before proposing another GPU run.
No further launch was submitted.

##### Call-8 stream dependency trace: launch review

The pinned schedule places the first FNUZ dispatch on Megatron's communication stream, separate from both the computation stream and the native UCCL stream.
`ScheduleNode.stream_acquire_context` queues a wait on the current microbatch's event before running each node and records that event again on exit.
Before the target dispatch, `TransformerModelChunkSchedulePlan.run` schedules forward preprocessing and LM-head backward; the first overlapped layer then runs backward post-combine and the reverse BF16 dispatch, forward attention and quantization, and backward expert MLP and post-dispatch.
`FusedCombine.backward` queues the BF16 exchange via `Buffer.dispatch` and, when asynchronous, queues an existing caller-stream wait on its completion event.
The old caller marker was already downstream of the forward quantization wait, so its non-completion alone cannot identify a computation stall.

The [new declaration](../chonk-native-fnuz-dependency-trace-i910bfcf936d3/experiment.toml) matches `761653` except for instrumentation (`APERTUS_CHONK_DEPENDENCY_TRACE=1`, validator SHA-256 `3cbea32c83d78db5139fb49a0329e43dd27437a954b7d3e826d8c0d13b3e0a88`).
Only ranks 48–63 and only call 8 (forward 7, backward 2) record timing-disabled events at schedule entry, around the existing node waits and bodies, and on both sides of the reverse BF16 dispatch on the UCCL stream.
Each marker records its actual stream identity; all are queried latest-to-earliest when the first target dispatch returns or throws, or if the schedule exits before it.
The observer adds no synchronization, new wait, stream change, image change, or change to native operation order.
If the timeout recurs, compare the last completed events **on each stream**, not a global host order.
A completed pre-wait marker with a pending post-wait marker narrows the blockage to that wait boundary; a completed UCCL pre-exchange marker with a pending post-exchange marker narrows it to the exchange or its native dependencies.
Missing or failed marker queries are inconclusive, and a timeout does not by itself prove a dependency cycle or fabric fault.
A fit pass still requires the original one-update gates and usable dependency observations on all final-stage ranks.
265 focused CPU and scheduler-stub tests passed; the pinned source signature, tree, parent, and reviewed blob were reverified.

Use the verified checkout copy of image `910…` via `--cache-image`, cold private compiler caches, and no checkpoint or shared compiler cache.
The run needs 16 exclusive Beverin `mi300` nodes under `csstaff` in the `cscs` reservation, four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, and `nid002450` excluded.
It runs **outside the agent sandbox with normal host filesystem access**, without the agent uenv.
At review, 32 reservation nodes were unallocated and no investigation jobs were active or pending; recheck the combined cap before submitting once.
Keep the 600-second GPU identity rendezvous, 2,700-second child deadline, 60-minute allocation, and native DeepEP timeout.
Stop on terminal rank failure; no automatic retry or extension.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-dependency-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-dependency-trace-i910bfcf936d3/<run-id>/`.
After exact user approval, job `761667` was submitted once on the `cscs` reservation with no other investigation allocation active or pending.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-dependency-trace-i910bfcf936d3/20261124T095615Z-d49a4910/) uses the verified checkout image, and the launch guard is back in place.

##### `761667` dependency trace: compute stops before forward preprocessing

All 16 final-stage ranks reproduced the call-8 `internode_prepare` timeout about 101 seconds after dispatch entry, with all entries within 90 ms.
All 16 produced identical dependency signatures, with no event recording or query errors.

| Stream | Last completed marker | First pending marker |
| --- | --- | --- |
| Megatron compute | Call entry | Before forward preprocessing's wait |
| Megatron communication | Before backward post-combine's wait | After that wait |
| Native UCCL | None observed in this trace | Before the reverse BF16 exchange |

The compute marker immediately before the forward preprocessing wait was already pending, so that wait cannot explain the first observed compute-stream gap.
The pending post-combine wait is downstream of the backward microbatch event, which the pinned schedule records on the compute stream after LM-head backward.
The compute gap begins earlier than that backward pass: between combined-call entry and the first forward preprocessing wait, the pinned code prepares the forward batch and plan, starts loss backward, and runs the forward pipeline receive callback.
The callback can enqueue a wait on an existing P2P receive handle even when it returns promptly to Python.
The trace does not distinguish those candidates, identify a dependency cycle, or establish that the BF16 exchange or CXI fabric initiated the stall.
The next useful distinction is whether the compute stream reached the forward receive callback and, if it did, whether it got past its existing wait; bracket the earlier batch/plan and loss-backward boundaries to interpret either result.

The minimum sampled final-stage node availability during the stall was 80.97 GiB, with no logged OOM or SIGKILL.
The owned allocation was cancelled at 10:25 under the stop rule; the step was `FAILED` (`1:0`), and rank zero finalized `failed` without running the fit validator.
No update or checkpoint completed, the allocation cleared, and no further job was submitted.

##### Matching send for the `761767` receive: source and retained-log review

The pinned scheduler gives physical PP stages 0–3 respectively 11, 9, 7, and 5 warmup forwards.
Virtual forward 7 is zero-based microbatch 3 in virtual chunk 1.
Physical PP stage 3 (global ranks 48–63) posts its receive for forward 7 in the post-forward callback of call 7 (`f=6, b=1`), then consumes the queued handle in the pre-forward callback of call 8 (`f=7, b=2`).
The matching send is from physical PP stage 2 (global ranks 32–47, receiver rank = sender rank + 16) in the post-forward callback of its call 8 (`f=7, b=0`).
All 16 rank pairs share the expected distinct pipeline communicator names in the retained RCCL logs.

On PP stage 2, call 8 runs the overlapped forward layers and backward-0 work before that send callback.
The callback runs on Megatron's schedule communication stream after its wait on the forward chunk event, and posts the send through the native `torch.distributed.isend` path.
Before reaching the send callback, the same combined call consumes the backward-0 gradient receive posted at the end of warmup.
Its expected matching gradient send is from PP stage 3 call 6 (`f=5, b=0`), not the stalled call 8.
This schedule-level reconstruction does not identify a direct cycle between the forward-7 receive and a backward-0 gradient produced by that same final-stage call.

The retained host records show all 16 senders entered call 8 about 144.25–144.33 seconds before their respective receiver reached the wait callback.
Their first FC1 replay returned on the host 0.30–0.36 seconds before the receiver reached that callback.
No sender logged a call-8 exit, but the send occurs before that exit, so its absence cannot tell whether `isend` was invoked.
The retained RCCL INFO logs identify the pipeline communicators but do not record the individual send or receive for this microbatch.

A targeted GPU follow-up should trace both sides of this P2P pair: the sender's backward-0 wait, forward-chunk readiness and actual `isend` entry/return, and the receiver's matching `irecv` and existing `Work.wait()` separately from output pseudo-deallocation.
That would distinguish a missing host-side send from a posted send whose producer or transfer does not complete.
No additional allocation was used for this review.

##### Call-8 compute prefix trace: launch review

The [prefix diagnostic](../chonk-native-fnuz-prefix-trace-i910bfcf936d3/experiment.toml) keeps the frozen `761667` workload and adds only `APERTUS_CHONK_PREFIX_TRACE=1` and the updated validator (SHA-256 `a5d0250c4c67488ef15453303135d6a39494f61371c544e29f83b953620073f5`).
On ranks 48–63 in call 8 only, add timing-disabled GPU events before and after forward batch/plan preparation, initial loss backward, and the existing forward receive callback.
Log the callback's pending receive-handle count before and after it runs without calling or modifying the handle.
The existing dependency observer queries these markers with the others; no GPU synchronization, new waits, or schedule changes are added.
The loss marker covers the initial autograd loss call, not the subsequent chunked LM-head backward.
If the callback consumes a handle, the pre-callback event completes, and the post-callback event stays pending, the compute gap is consistent with the receive wait; the callback also deallocates the prior output, so this alone does not prove the wait is the cause.
If the pre-callback event is already pending, use the earlier pairs to narrow the gap instead.
Missing or failed queries are inconclusive; a training pass requires the existing fit gates and usable prefix observations.
273 focused CPU and scheduler-stub tests passed; the clean signed Megatron pin, parent, tree, and reviewed blob were reverified.

Use one exclusive 16-node Beverin `mi300` allocation in the `cscs` reservation under `csstaff`, with four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, and `nid002450` excluded.
The full topology is unavailable inside the agent allocation, so this runs **outside the agent sandbox with normal host filesystem access**, without the agent uenv.
The verified checkout image, 600-second identity wait, 2,700-second training child deadline, 60-minute allocation, and native DeepEP timeout are unchanged.
No checkpoint, shared compiler cache, automatic retry, or extension; stop on terminal rank failure.
Recheck the combined 16-node investigation cap before submitting once and let Slurm manage the queue.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-prefix-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-prefix-trace-i910bfcf936d3/<run-id>/`.
After exact user approval, job `761767` was submitted once on the `cscs` reservation with no other investigation allocation active or pending.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-prefix-trace-i910bfcf936d3/20261124T102323Z-55286645/) uses the verified checkout image, and the launch guard is back in place.

##### `761767` prefix trace: compute stops at the forward receive callback

All 16 final-stage ranks entered the target call-8 FNUZ dispatch within 36.4 ms and reproduced the `internode_prepare` timeout about 101 seconds later.
All 16 had the same event signature, with no marker recording or query errors.
The compute stream completed the markers after batch/plan preparation and initial loss backward, and immediately before `pp_pre_forward`.
The first pending compute marker was immediately after `pp_pre_forward`.
On the host, each callback consumed exactly one queued forward receive handle and returned in 0.081–0.109 ms.
The communication stream still stopped at the backward post-combine wait, and neither UCCL marker around the reverse BF16 exchange completed.

The pinned callback calls `recv_prev_wait_handle.wait()` and then pseudo-deallocates the previous output.
The receive wait is the leading suspect, but the trace brackets the whole callback, not the native wait alone.
It does not show whether the matching send was posted or could make progress, whether the receive was paired correctly, or whether a dependency cycle exists.
The next useful source review is to identify the matching upstream send for virtual microbatch 7 and its dependencies, then decide whether more GPU instrumentation is needed.

The minimum sampled final-stage node availability during the stall was 82.88 GiB, with no OOM or SIGKILL reported in rank logs.
The owned allocation was cancelled at 10:28 under the stop rule; Slurm recorded the step as `FAILED` (`1:0`), and rank zero finalized `failed` without a fit-validator verdict.
No update or checkpoint completed, the allocation cleared, and no further job was submitted.

##### Matched pipeline P2P trace: launch review

The [P2P-pair diagnostic](../chonk-native-fnuz-p2p-pair-trace-i910bfcf936d3/experiment.toml) keeps the `761767` workload and adds `APERTUS_CHONK_P2P_PAIR_TRACE=1` and a new validator digest `099f66aa88d43265c8bd473a6a3e1bf261a0f48ee661e7955329916c81b5d3be`.
On stage 2, trace call 8 (`f=7, b=0`): mark the existing backward-receive and forward-chunk waits and the sender stream immediately before `isend`.
Log entry and return of the expected native `isend` on ranks 32–47 and `irecv` on ranks 48–63, including peer, shape, and dtype.
On the receiver, verify that call 8 consumes the same local work handle returned by call 7's `irecv`, and mark immediately after the original `Work.wait()` returns, at entry to output pseudo-deallocation.
Neither the native call nor its work handle is replaced.
One daemon per sender queries timing-disabled GPU events once per five seconds, for at most eight minutes, so a sender blocked on the host can still leave evidence.
There is also an immediate sender sample after `isend` returns.
No GPU synchronization, work-handle completion polling, or additional cross-stream wait is added.

A missing host-side `isend` is distinct from a posted send whose producer-stream marker stays pending.
If the post returns and the producer marker completes but the receiver remains blocked after `Work.wait()`, investigate the native P2P transfer or its stream dependencies without inferring a fabric fault.
Require matching peers and receive-handle identity before interpreting a completed trace; absent or unusable samples remain inconclusive.
A fit pass still requires the existing one-update gates and usable paired evidence.
276 focused CPU and scheduler-stub tests passed; the signed Megatron commit, parent, tree, and reviewed blob were reverified.

The full workload needs 16 exclusive general Beverin `mi300` nodes under `csstaff`, four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, and `nid002450` excluded.
Slurm executes it **outside the agent sandbox with normal host filesystem access**, without the agent uenv.
Keep the verified checkout image, cold private compiler caches, 600-second identity rendezvous, 2,700-second training child, 60-minute allocation, and native DeepEP timeout.
No checkpoint, shared compiler cache, retries, or extensions; stop on terminal rank failure.
At review the `cscs` reservation had 14 unallocated nodes, so the reviewed fallback is the general queue, where Slurm may hold the job pending.
Separately, reservation job `761781` is a 16-node Chonk run from another checkout and account; its relationship to this investigation is not established.
Before submitting to either queue, confirm whether it counts toward the combined cap or that it has cleared; leave that job untouched.
Use at most one queue per approved launch.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the general-queue command is:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-p2p-pair-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image
```

If 16 `cscs` reservation nodes are available at submission, the reservation variant is the exact same command with `--reservation cscs` appended.
EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-p2p-pair-trace-i910bfcf936d3/<run-id>/`.
After exact launch approval and confirmation that job `761781` had cleared, the user authorized a one-time exception to overlap the 16-node diagnostic with the one-node trace export `761959`, for a combined maximum of 17 nodes.
Job `762041` was submitted once to the `cscs` reservation, which had 31 unallocated nodes at review.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-p2p-pair-trace-i910bfcf936d3/20261124T114548Z-e0c5659b/) is retained separately, the export was left untouched, and the launch guard is back in place.
The 16-node cap remains in effect for future submissions.

##### `762041` matched P2P trace: stage 2 did not reach the forward send

All 16 stage-3 receivers posted the expected `[4096, 2, 3584]` BF16 `irecv` from their stage-2 peer, and call 8 consumed the same local work handle returned by call 7.
The host reached the hook immediately following the original `Work.wait()` in 0.050–0.071 ms, but the GPU marker there remained pending on every receiver, while the marker before the callback completed.
Because that pending marker was inserted before pseudo-deallocation, the observed gap is at the receive-wait boundary, not in output cleanup.

None of the 16 senders emitted a matching `isend` entry or reached the forward-chunk readiness wait that precedes the send.
Each sender retained 50 samples with no query errors; the last samples, about four seconds after the corresponding receiver timeout, showed GPU completion through the sender's backward-0 receive callback.
The last non-sampler sender record on every rank was a host return from its first FC1 replay in call 8; it does not establish GPU completion of that replay or identify the next host-side block.
This evidence does not support a posted forward-7 send that failed in transit.
The next source/log check is to narrow the sender-side interval from that replay return to the forward-chunk readiness wait, before proposing another allocation.

All 16 final-stage ranks reproduced the call-8 DeepEP timeout at about 101 seconds, with at least 82.15 GiB sampled node availability and no logged OOM or SIGKILL.
The allocation was cancelled at 10:25 under the stop rule; the step was `FAILED` (`1:0`) and rank zero finalized `failed` without a fit-validator verdict.
No update or checkpoint completed, and the 16-node allocation cleared.
The one-node trace export was left untouched.

##### Sender-side review: cold attention backward compilation

The last FC1 replay marker is not the end of sender host progress.
All 16 stage-2 logs in each of `762041`, `761767`, `761667`, and `761653` show SSSGLU and KDA QKV preparation recompiles after that marker, followed by a build of the same native CK `fmha_bwd_d128_bf16_batch…gfx9` attention-backward blob.
The same blob was requested by all 16 stage-2 ranks in `759990`.
The build output is under each rank's private `xdg/te_ck_jit`, not the shared image cache.

Node telemetry caught `hipcc` or `clang` descendants on all 16 stage-2 ranks during the receiver timeout window in the four reservation runs.
In `762041`, the first observed compiler sample on each sender was 0.1–5.1 seconds before its corresponding receiver logged the error, and the last was 0.1–0.6 seconds before it.
These samples place compilation at the end of the window, but do not show when the build started or account for the full wait.
The pinned schedule runs backward attention within the overlapped layer loop, before it can queue the forward-chunk readiness wait and post the pipeline send.
The compiler observations and missing send records make cold ordinary-attention backward JIT the leading explanation for why the send was late.
The timeout and subsequent cancellation do not establish a compiler failure, the compile duration if left to finish, or whether prewarming would clear the update.

The smallest useful GPU check is to time the exact native TE attention backward on one node with the image and training shape unchanged (`B=2`, `S=4096`, BF16, 28 query heads, 7 KV heads, head dimension 128, causal, no dropout).
Require the retained CK blob key, finite gradients, a complete cold build, and reuse on a fresh process with a verified private copy.
The existing B1 attention shape probe is not evidence for this exact B2 build.
`xdg/te_ck_jit` is not supported by the current portable-cache policy; do not publish the whole XDG tree or reuse an interrupted build.
If the component check passes, review a diagnostic-only verified prewarm before another full-layout attempt, preserving native operations and the existing deadlines.
No GPU job, cache publication, or software change was made for this review.

##### One-node CK attention backward reuse: launch review

The [component diagnostic](../chonk-ck-attention-bwd-reuse-i910bfcf936d3/experiment.toml) runs image `910…`'s public TE attention on one MI300A at `B=2`, `S=4096`, BF16, 28 query heads, 7 KV heads, head dimension 128, causal, and no dropout.
It requires a cold build and mapping of the exact retained `fmha_bwd_d128_bf16_batch…gfx9.so.40ee1ee0` object and finite, nonzero output and gradient norms.
After the cold worker and compiler descendants exit, copy only the verified object into a fresh private cache.
Require a fresh warm process to map that object without recompilation or mutation and match the cold backend and norms within 0.5% relative tolerance.
Record synchronized forward and backward walltimes, rank logs, and the object's SHA-256.
There is no Megatron source execution, collective, checkpoint, or shared cache publication.
102 focused CPU and scheduler-stub tests passed; validator SHA-256 is `d854a6af83b358be9a57aed89908996ab9ef1e50ab200e88e051c129267d4ec5`.

Use one exclusive `mi300` node in the `cscs` reservation under `csstaff`, with one visible MI300A, 48 bound CPUs, `nid002450` excluded, and a 25-minute allocation.
The runner bounds the single component command at 1,200 seconds, with separate 900-second cold and 180-second warm worker limits.
The component needs the pinned container, 48 bound CPUs, and an exclusive allocation; do not borrow the unrelated interactive allocation.
Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
Recheck investigation occupancy before submitting once; the one-node export, if active, keeps this attempt within the normal 16-node cap.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-ck-attention-bwd-reuse-i910bfcf936d3 \
  --submit --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:25:00 --exclude nid002450 --cache-image --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-ck-attention-bwd-reuse-i910bfcf936d3/<run-id>/`, with component reports under `output/0/ck-attention-reuse/`.
Each copy is private to the command under `/dev/shm/apertus-amd/`; no object is installed or published.
After exact user approval, job `762106` was submitted once to the `cscs` reservation.
The one-node trace export was still running at submission, so the combined request was two investigation nodes under the normal cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-ck-attention-bwd-reuse-i910bfcf936d3/20261124T121720Z-17fe4429/) uses the verified checkout image, and the launch guard is back in place.

##### `762106` result: exact CK backward build and reuse passed

Slurm completed the one-node job on `nid002776` in 1:11 with exit `0:0`; the component summary and runner both passed.
Both processes selected native TE fused attention (`NVTE_CK`) and mapped the expected backward object on gfx942.
Synchronized backward took 28.530 s with the cold build and 26.651 ms with the verified private copy in a fresh process.
The output and all three BF16 gradient norms were finite, nonzero, and identical between the two runs.
The warm log recorded no CK rebuild, and the object remained byte-identical (`114880` bytes, SHA-256 `c87378fb7245bac1ca43b6065fc4f7454d2bf7b840ec1ce45e68a72c6eb517b7`).
The private scratch was removed after both workers exited, and no object or shared cache was published.
This proves component reuse, not full-layout cache reuse or that the 16-node update will clear the pipeline stall.
The next discriminating check is to prewarm and verify this exact object before full-layout training, leaving the workload, schedule, and deadlines unchanged.

##### Full-layout CK prewarm: launch review

The [prewarm diagnostic](../chonk-native-fnuz-ck-prewarm-i910bfcf936d3/experiment.toml) preserves the frozen `762041` workload, startup trace, matched P2P trace, and deadlines.
It adds only the opt-in `APERTUS_CHONK_CK_PREWARM=1` procedure and its validator (SHA-256 `5fff0a3db44d99766c19b1fd97ca0244b25937716c977b27cb3550554b5da36a`).
Before distributed training, one worker per node runs the same native TE attention backward used in `762106`.
Each builder has a 240-second subprocess limit inside the existing 2,700-second rank-child budget.
Require the exact qualified 114880-byte object and SHA-256, finite output and gradient norms matching the component fixture, and no surviving compiler on node handoff.
Copy only this object into each of the 64 private rank caches without replacement, rehash each copy, and require all ranks to rendezvous before entering training.
The handoffs use the existing 300-second coordination bound.
Training must map the unchanged object, with no rebuild of that key in the training logs.
The original finite, unskipped update and matched-P2P acceptance gates remain in force.
If the digest or reuse check fails, do not fall back or claim that the prewarm tested the pipeline stall.
If the diagnostic clears the stall, retain the result as full-layout fit with this specific prewarm, not generic cache reuse or proof that compilation explains the entire cold stall.
290 focused CPU and scheduler-stub tests passed; the signed source, tree, parent and reviewed blob were reverified.

Use 16 exclusive Beverin `mi300` nodes in the `cscs` reservation under `csstaff`, with four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, and `nid002450` excluded.
The full topology is not available in the agent allocation, so Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
Use the verified checkout image and retain the 60-minute allocation, 600-second GPU identity wait, 2,700-second child deadline, and native DeepEP timeout.
There is no checkpoint, shared cache publication, automatic retry, or deadline extension.
The one-node trace export `761959` was still running at review; the earlier one-time cap exception applied only to `762041`.
Confirm that the export has cleared and recheck the combined 16-node cap before submission.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-ck-prewarm-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

Image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained reports and logs: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-ck-prewarm-i910bfcf936d3/<run-id>/`.
Builders and rank caches stay under job-owned `/dev/shm/apertus-amd/`.
The user approved the exact launch once export `761959` clears.
While export `761959` was still running, the user explicitly authorized a one-time overlap of its one node with this 16-node run, for a combined maximum of 17.
Job `762207` was submitted once to the `cscs` reservation after rechecking occupancy and the approved inputs.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-ck-prewarm-i910bfcf936d3/20261124T124901Z-fa2ed994/) is retained separately; the export was left untouched and the launch guard is back in place.
The ordinary 16-node cap still applies to future submissions.

##### `762207` result: CK prewarm rejected before training

All 64 ranks completed startup and all 16 builders logged success compiling the requested CK attention-backward key.
All 16 then failed the pinned artifact check against the object from `762106`.
The failure record did not store the newly built byte counts or digests, so it does not distinguish a size mismatch from a digest mismatch or establish why the outputs differed.
No builder published a ready record, no rank seeded the object, and training did not start.
The owned job was cancelled at 1:28; the step failed with exit `1:0` and rank zero finalized `failed`, with no training validator verdict.
The export was left untouched.
This run did not test whether CK prewarming resolves the call-8 stall.

Before another 16-node attempt, use a bounded component check to compare independently built objects from the pinned stack and test fresh-process reuse of a verified copy.
Record the exact key, size, digest, and finite-gradient results for every build, including rejected objects.
If independent builds differ legitimately, keep byte-for-byte verification from each qualified build to its private consumers rather than assuming a common digest or admitting unverified objects.
No shared cache was published or retry submitted.

##### Independent CK builds: launch review

The [component comparison](../chonk-ck-attention-bwd-compare-i910bfcf936d3/experiment.toml) uses the same cached image `910…` and native `NVTE_CK` BF16 causal attention backward at `B=2`, `S=4096`, with 28 query heads, 7 KV heads, head dimension 128, zero dropout, and RNG seed 123.
Local rank zero runs two independent cold builds on device zero with all four MI300As visible, matching the GPU visibility, rank-zero placement, and CPU binding of the `762207` builders.
The other three ranks do not import Torch, build a kernel, or run a collective.
Each cold build gets a separate empty private cache and must log success for the exact key, map the created executable inode, and produce finite BF16 output/Q/K/V gradients with norms within 0.5% of the `762106` fixture.
The worker reports record node, GPU visibility, target, Torch/HIP, TE import, and CK builder provenance.
The controller records the exact key, byte count and SHA-256 of each regular artifact after worker/descendant quiescence, even when it rejects a completed build.
If a worker leaves live descendants, stop without reading or copying its cache.
For each qualified build, copy only its object without replacement and hash both source and copy.
A fresh warm process must map that exact copy, issue no CK JIT compile, match the cold native backend and norms within 0.5%, and leave the object identity and bytes unchanged.
Record synchronized forward/backward timings and compare both cold object identities to each other and to historical job `762106`; the historical digest is not an admission gate for this diagnostic.
The result cannot isolate GPU visibility from other differences between jobs or retrospectively identify the `762207` binaries.
There is no change to the full-layout prewarm gate, no training or checkpoint, and no object or compiler cache publication.

Use one exclusive x86-64 Beverin `mi300` node in the `cscs` reservation under account `csstaff`, with four GPUs, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, all allocatable memory, and `nid002450` excluded.
The step uses `pmix`, `disable_rdzv_get`, and `--gpu-bind=none` so the rank-zero worker sees all four GPUs.
The bounds are 25 minutes for the allocation, 1,200 seconds per rank command, and 240 seconds per cold worker and 180 seconds per warm worker.
The four-GPU layout cannot be exercised from the current agent sandbox; Slurm runs **outside the agent sandbox with normal host filesystem access** and discards the agent uenv.
Recheck combined investigation occupancy across general and reservation queues before submitting once and keep the ordinary 16-node cap.
The one-node trace export `761959`, if still running, is left untouched.
The guarded declaration SHA-256 is `c96c038b87983e30225d5fb1ad85a16a77fa8363471647cbc438ddb68f543d02`; validator SHA-256 is `1bca93fcec857689bd7c39147e2ce74233a678f59e51bf934834749dd6f4208d`.
304 focused CPU and scheduler-stub tests passed; the broader suite encountered an unrelated launch fixture missing `manifest.json`.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact reviewed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-ck-attention-bwd-compare-i910bfcf936d3 \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:25:00 --exclude nid002450 --cache-image --reservation cscs
```

EDF input: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Run directory: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-ck-attention-bwd-compare-i910bfcf936d3/<run-id>/`, with reports under `output/0/ck-attention-compare/`.
The run uses no Megatron checkout.
Temporary objects stay in private `/dev/shm/apertus-amd/ck-compare-<job-id>-*/` paths and are not durable artifacts.
After explicit approval and an occupancy recheck, job `762221` was submitted once in `cscs`.
The one-node export `761959` was left untouched, for a combined request of two investigation nodes under the ordinary cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-ck-attention-bwd-compare-i910bfcf936d3/20261124T132702Z-7e0c585e/) is retained and the launch guard is restored.

##### `762221` result: different CK objects both qualified for private reuse

Job `762221` completed on `nid002776` in 2:02 with exit `0:0`; the component and runner both passed.
All four workers saw four MI300As, selected native `NVTE_CK` on gfx942, and mapped the exact private executable inode.
The cold logs each record one build of the expected key; neither warm log records a CK compilation.

| Build | Bytes | Cold SHA-256, also verified on its warm copy | Cold backward | Warm backward |
|---|---:|---|---:|---:|
| A | 114880 | `79f8f9b15b974ae0f2c602b0ab76e025330edf4dca700fe01456ec4b1c70a26e` | 28.681 s | 23.137 ms |
| B | 114880 | `05cb007bdc0770d297454be8314002d2614374b772f1eeba513420b6e6a60c6e` | 27.293 s | 22.610 ms |

Both same-node cold builds had the same key and size but differed in SHA-256 from each other and from `762106`.
All four output/Q/K/V gradient checks passed, with a maximum relative norm difference of `1.84e-7` from `762106`; the norms were not all bit-identical.
Each warm process reused its own verified copy with unchanged bytes and file identity.
No object or shared cache was published and the trace export was left untouched.
The historical digest is not a valid universal identity for fresh builds, but this result does not identify which bytes changed, explain the rejected `762207` outputs, or establish that prewarming clears the call-8 stall.
The next full-layout review should require each node's quiescent native build to pass the component qualification before handoff, then verify that build's exact bytes in each private consumer and keep the matched P2P and finite-update gates.
No full-layout retry was submitted.

##### Full-layout node-verified CK prewarm: launch review

The [new declaration](../chonk-native-fnuz-ck-node-verified-i910bfcf936d3/experiment.toml) is separate from the consumed `762207` declaration.
Compared with the frozen `762207` plan, it changes only the CK prewarm mode (`node-verified`), its validator, and run/guard metadata.
The pinned image, signed Megatron source, 42-layer workload, memory mitigations, native CXI path, startup diagnostics, matched P2P trace, and no-checkpoint one-update schedule remain unchanged.
The signed source is `ceb8035c182c40711db4d256d99fc44e4b8db27e`, whose signer, parent, tree, and reviewed blob were reverified in the dedicated clean worktree.

Before training, local rank zero on each of the 16 nodes runs one cold and one fresh-process warm qualification of the `762221` fixture.
Require the pinned Torch/HIP, TE consumer and CK builder, exact native `NVTE_CK` route, and finite BF16 output and Q/K/V gradients with norms within 0.5% of the reference and of each other.
Require the exact `fmha_bwd…gfx9.so.40ee1ee0` key, a regular 114880-byte ELF, and exact successful cold-build evidence.
The controller waits for the worker and compiler descendants to exit before it inspects or copies the object.
Verify that the fresh process maps the executable inode of its exact private copy without rebuilding or changing it.
Record the observed key, size and SHA-256 for quiescent rejected artifacts as well as for qualified builds.
Allow different SHA-256 digests across nodes, but bind each qualified digest to its originating host and verify the exact bytes, without replacement, in each of that host's four private rank caches.
Do not enter training unless all 16 node qualifications and all 64 rank-copy handoffs pass.

The cold and warm worker pair shares the existing 240-second node prewarm budget, and rendezvous retain the 300-second coordination bound.
The GPU identity wait remains 600 seconds and the rank-child deadline remains 2,700 seconds inside a 60-minute allocation.
Require all 64 training ranks to map their unchanged private executable inode and never rebuild the target key.
Interpret stage-2 call-8 send readiness, the matching stage-3 receive completion, and the finite unskipped update as separate observations.
Accept a full-size update only if the existing matched P2P, workload, memory, native CXI, and one-update validator gates also pass.
If anything fails, retain reports and logs; do not fall back, retry, extend a deadline, or publish any object or shared cache.
This test does not qualify checkpoint/restart, portable cache reuse, or throughput.

The guarded declaration SHA-256 is `e7d6982bf8912925caa02312416b30f27353f050c8e36cb87d32462794d498b4`; the validator SHA-256 is `58fd2187550d6fd9c3fde7e4c00e4076b80f7ede3d9edecabdb428c6eff4a48f`.
316 focused CPU and scheduler-stub tests passed.

Use 16 exclusive x86-64 Beverin `mi300` nodes under `csstaff` in `cscs`, four MI300As per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
The full layout cannot run within the current agent allocation, so this Slurm launch runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
Trace export `761959` still occupies one investigation node at review; leave it untouched and submit only after it clears and combined occupancy is rechecked.
Neither earlier one-time 17-node exception applies.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-ck-node-verified-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF input: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-ck-node-verified-i910bfcf936d3/<run-id>/`, with node qualification, rank, trace, and validator evidence under `output/`.
Objects stay in job-owned `/dev/shm/apertus-amd/` scratch.
Trace export `761959` completed with exit `0:0`, and the queue recheck found no other investigation job.
After explicit approval, job `762323` was submitted once in `cscs` under the normal 16-node cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-ck-node-verified-i910bfcf936d3/20261124T140121Z-97694085/) is retained and the launch guard is restored.

##### `762323` result: call 8 cleared; later dispatch timeouts blocked the update

All 16 nodes qualified distinct 114880-byte CK objects in 55.86–57.85 s per node, and all 64 ranks verified their private copies and entered the first update.
Training reported the verified mapping on 48 ranks (all of stages 1–3), with no target CK rebuild in the rank logs; stage 0 did not report a mapping before the run stopped.
All 16 stage-2 ranks posted the matched call-8 sends and all 16 final-stage ranks demonstrated the expected receive handle, completed GPU wait marker, and successful dispatch return.
Stage-2 call-8 entry to send return took 211.46–211.72 s, with the sends returning 1.04–1.21 s after the CK mappings were recorded.
The final-stage call-8 notify enqueue to return took 96.29–96.48 s.
This proves that the previous boundary cleared with verified prewarm, not that CK compilation accounted for the entire prior delay.

Stage 2 completed call 9, then all 16 ranks timed out in forward FNUZ `internode_prepare` on call 10 `(f=9,b=2)`.
Stage 3 completed call 8, then all 16 ranks hit the same timeout on call 9 `(f=8,b=3)`.
Both groups failed about 101.1 s after entering their respective calls, with first errors about 0.14 s apart by logged wall clock.
The minimum sampled node availability during the failed calls was 46.59 GiB on stage 2 and 82.10 GiB on stage 3, with no logged OOM or pre-cancellation SIGKILL.
SSSGLU and KDA recompile logs were present on both stages, but the last pre-error process samples showed only Python processes and the scoped call-8 markers do not identify which prerequisite was pending at the later timeouts.
The native dispatch timeout is not evidence by itself of a CXI failure, an OOM, or another CK build.

No rank completed an update.
The owned allocation was cancelled after the fatal errors at 13:06; Slurm recorded step `FAILED` (`1:0`), and rank zero finalized `failed` without a fit-validator verdict.
The allocation cleared, no shared cache was published, and no retry was submitted.

The signed schedule identifies stage-1 call 10 as the forward-9 activation producer for stage-2 call 10, and stage-3 call 8 as its backward-2 gradient producer.
Stage-3 call 9 expects the forward-8 activation from stage-2 call 9; backward-3 is local to the last virtual stage and does not require a pipeline receive.
All 16 stage-1 forward-9 producer calls returned on the host 12.14–12.19 s after the stage-2 errors by logged cross-node wall clock.
The stage-2 forward-8 and stage-3 backward-2 producer calls had returned on the host about 101 s before the corresponding errors.
Those host returns do not date individual `isend` posts or establish GPU completion, because the combined schedule posts P2P before returning.
A later-call trace should first distinguish whether stage 1 posted forward-9 in time and whether the stage-2 and stage-3 receive, producer, and prior-UCCL GPU markers cleared.

##### Full-layout later P2P trace: launch review

The [later P2P diagnostic](../chonk-native-fnuz-later-p2p-trace-i910bfcf936d3/experiment.toml) preserves the frozen `762323` workload, node-verified CK prewarm, call-8 trace, and deadlines.
The only plan changes are `APERTUS_CHONK_LATER_P2P_TRACE=1`, the validator digest, and run/guard metadata.
The new observation covers the following edges on all 16 pipeline lanes, with the send and receive post on their native callbacks and the wait in the consuming call.
The contexts `(call, forward, backward)` match the pinned schedule and were checked against the retained `762323` logs for every lane.

| Edge | Send context | Receive post context | Receive wait context |
|---|---|---|---|
| Forward 9 | Stage 1 `(10,9,0)` | Stage 2 `(9,8,1)` | Stage 2 `(10,9,2)` |
| Forward 8 | Stage 2 `(9,8,1)` | Stage 3 `(8,7,2)` | Stage 3 `(9,8,3)` |
| Backward 2 | Stage 3 `(8,7,2)` | Stage 2 `(9,8,1)` | Stage 2 `(10,9,2)` |

Record BF16 `[4096,2,3584]` shape, peer, native post entry/return, receiver-local `Work` identity, and the queue counts around each existing receive wait.
Record GPU events before and after the existing callbacks on their actual streams, and query these along with the first target dispatch's producer, prior-UCCL, and layout markers on return or exception.
If the combined call exits before dispatch, record the available receive markers without assuming dispatch was reached.
The observer does not insert waits, synchronize devices, change queues, replace native handles, retain tensors, or trace beyond the named edges and target dispatches.
Use the observations to distinguish an absent or late native send from a posted send with incomplete receive/producer work or a pending prior-UCCL/layout boundary.
Use cross-node timestamps only for coarse ordering; a host-side return is not proof of GPU completion.
A passing full-layout verdict still requires one finite unskipped update and all prior CK, P2P, and memory checks, plus valid later-trace identities and query results.
No checkpoint, cache publication, deadline extension, or retry is authorized.

329 focused CPU and scheduler-stub tests passed; the dedicated clean Megatron checkout's signed commit `ceb8035c182c40711db4d256d99fc44e4b8db27e`, tree, parent, and reviewed blob were reverified.
The guarded declaration SHA-256 is `8d334fc6aae4e4ae7348345722a7032ee379368b06562ed7f7bdab67497a50e9` and the validator SHA-256 is `cda7cae1c158481f909352a60124ab228ee9938f59543af3a790aafa0d3bedb1`.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300As per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second rank-child bound, 600-second GPU identity wait, 240-second node prewarm bound, 300-second coordination bound, and native DeepEP timeout.
The full layout cannot run in the current agent allocation, so Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
No investigation job was active at review; recheck combined occupancy before submitting once and leave unrelated allocations untouched.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-later-p2p-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF input: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-later-p2p-trace-i910bfcf936d3/<run-id>/`, with rank and validator evidence under `output/`.
Compiler objects remain in job-owned `/dev/shm/apertus-amd/` caches.
After explicit approval and the occupancy recheck, job `762353` was submitted once in `cscs` within the 16-node cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-later-p2p-trace-i910bfcf936d3/20261124T143358Z-62c5d3bc/) is retained, and the launch guard is restored.

##### `762353` result: late forward-9 send; separate final-stage dependency still pending

All 16 node CK qualifications and all 64 private-copy handoffs passed.
Training mapped its seeded CK object on all 48 ranks in stages 1–3 before the stop, with no target rebuild logged on any rank.
The original call-8 match cleared again, including successful dispatch and complete producer, prior-UCCL, and layout markers on all 16 final-stage ranks.
No rank completed the first update.

| Edge | Observation across all 16 lanes |
|---|---|
| Forward 9 into stage-2 call 10 | The expected receive handle matched; the event before its wait completed, but the event after stayed pending at timeout. All stage-1 native send entries were about 10.6–10.8 s after the corresponding receiver errors by cross-node wall clock. |
| Backward 2 into stage-2 call 10 | The expected gradient receive handle matched and its sender returned around 101 s before the receiver timeout. Both wait-bracketing GPU markers stayed pending, so the trace does not show a separate backward receive stall. |
| Forward 8 into stage-3 call 9 | The expected receive handle matched and both wait-bracketing GPU markers completed, ruling out this receive as the unfinished dependency at this boundary. |

All 16 ranks on each of stages 2 and 3 then failed in `internode_prepare`, with their producer, prior-UCCL, and layout markers pending and no marker query errors.
The minimum sampled node availability during the failed calls was 47.03 GiB on stage 2 and 80.69 GiB on stage 3, with no logged OOM or pre-cancellation SIGKILL.
Stage 1 spent 320.92–321.77 s from call-10 entry to native send entry.
The first FC1 replay returned on the host 110.13–110.73 s before the training CK mapping was logged, and send entry followed that mapping by 0.96–1.59 s.
The stage-1 logs show SSSGLU and KDA recompiles, but the timings do not establish how much of the delay came from compilation or where stage 3's subsequent producer stopped.

The owned allocation was cancelled after the fatal errors at 12:54 and cleared.
The step was `FAILED` (`1:0`), and rank zero finalized `failed` without a fit-validator verdict.
No shared cache was published and no retry was submitted.
Before another full-layout launch, separate the stage-1 delay within call 10 from the final-stage call-9 local loss-backward, compute-stream, and reverse-BF16 dependencies.

##### Full-layout later phase trace: launch review

The [later phase diagnostic](../chonk-native-fnuz-later-phase-trace-i910bfcf936d3/experiment.toml) keeps the frozen `762353` workload, node-verified CK prewarm, matched P2P traces, and deadlines.
Only `APERTUS_CHONK_LATER_PHASE_TRACE=1`, the validator digest, and run/guard metadata change.
The new observations cover all 16 lanes at each of the two boundaries below.

- **Stage 1, call 10 `(f=9,b=0)`:** Record host elapsed time around the existing forward and backward P2P callbacks, slow schedule-node spans, and start/end observations from the pinned TorchDynamo compiler callback API.
  Use native send entry, FC1 replay and CK mapping records from the existing traces to place those spans relative to the late forward-9 send.
  The observer caps logging at 128 callback events and 32 schedule-node spans of at least 250 ms per rank, emits a zero-compile count if none are observed, and deregisters its callbacks on exit.
  The timestamps measure host intervals, not GPU runtime or the complete cost of all compiler subprocesses.
- **Stage 3, call 9 `(f=8,b=3)`:** Record GPU markers around batch/plan, local loss backward, the existing forward receive callback, and the schedule-node waits on the compute and communication streams.
  Bracket reverse BF16 UCCL work and query the markers with the existing dispatch producer, prior-UCCL and layout observations on return or exception.
  This distinguishes incomplete local loss/backward or reverse-communication work from a forward receive that already completed.

The observer adds no waits or device synchronizations and does not change compiler settings, queues, native work handles, kernels, or numerical gates.
A fit verdict still requires one finite unskipped update, all prior gates, and usable phase records.
No checkpoint, cache publication, deadline extension, or retry is authorized.
335 focused CPU and scheduler-stub tests passed; the dedicated clean Megatron checkout's signed commit `ceb8035c182c40711db4d256d99fc44e4b8db27e`, tree, parent, and reviewed blob were reverified.
The guarded declaration SHA-256 is `4d45a38e25042773fcc22704aa43cf3654b6c2208a9fea8d57136624ea6f074c` and the validator SHA-256 is `b739d31730e90c3cb89f4078e760ac73f6818114524e8ee465a2a286486ca839`.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300As per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second rank-child bound, 600-second GPU identity wait, 240-second node prewarm bound, 300-second coordination bound, and native DeepEP timeout.
The full layout cannot run in the current agent allocation, so Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
No investigation job was active at review; recheck combined occupancy before submitting once and leave unrelated jobs untouched.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-later-phase-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-recompute-check \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF input: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Retained run: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-later-phase-trace-i910bfcf936d3/<run-id>/`, with rank and validator evidence under `output/`.
Compiler objects remain in job-owned `/dev/shm/apertus-amd/` caches.
After explicit approval and the occupancy recheck, job `762390` was submitted once in `cscs` within the 16-node cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-later-phase-trace-i910bfcf936d3/20261124T145835Z-8d1dd443/) is retained, and the launch guard is restored.

##### `762390` result: long sender backward spans and a pending reverse BF16 boundary

All 16 node CK qualifications and all 64 private-copy handoffs passed.
All ranks entered the first update, and stages 1–3 recorded 48 verified CK mappings, with no rebuild of the target key in any rank log.
Call 8 cleared again.
All 16 ranks on each of stage 2 (call 10) and stage 3 (call 9) hit the native DeepEP CPU dispatch timeout before any rank completed an update.
At those boundaries the minimum sampled node availability was 46.51 GiB on stage 2 and 82.51 GiB on stage 3, with no OOM or pre-cancellation SIGKILL logged.

On all 16 stage-1 senders, call-10 entry to forward-9 native send entry took 323.04–324.18 s, and the sends began 12.30–12.61 s after the stage-2 errors by cross-node wall clock.
The largest scoped **host** schedule-node spans were `mlp backward` at 210.50–211.35 s and `attn backward` at 88.35–96.81 s.
Recorded Dynamo callback spans totaled 3.46–5.84 s per rank; they do not account for the long intervals and are not a census of all compiler, loader, or runtime work.
The trace narrows the late send to backward execution before the send, but it does not identify a stalled GPU kernel or the particular host operation responsible for either long span.

On all 16 final-stage ranks, local loss backward, the matched forward-8 receive, and the Megatron `moe_combine backward` entry/wait markers completed on GPU.
The event before reverse BF16 dispatch on the UCCL stream completed on 9 of 16 ranks, while the event after it remained pending on all 16.
The dependent `mlp backward` marker before its stream wait completed on all 16; the after-wait marker remained pending on all 16.
The forward dispatch producer, prior-UCCL, and layout markers also remained pending on all 16, with no trace record/query errors.
This localizes the final-stage dependency to work at or ahead of the reverse BF16 dispatch, not the prior forward receive or loss backward.
It does not establish a CXI fault or show whether the reverse operation itself or an upstream stream dependency caused the delay.

The owned job was cancelled after the fatal errors at 12:59, and the allocation cleared.
Slurm recorded the step as `CANCELLED` (`0:9`).
Rank failure records and native traces were retained; `result.json` and `checks.json` were absent after the allocation cleared.
No fit-validator verdict or shared-cache publication exists for this run.
No retry was submitted.
Use the retained traces and pinned source to resolve the backward host spans and reverse-dispatch event chain before proposing another GPU run; do not extend dispatch deadlines on this evidence.

##### Offline audit: reverse BF16 producer event and the late sender

This audit used the retained `762390` records, signed source `ceb8035c182c40711db4d256d99fc44e4b8db27e`, and source files read from the verified `910…` SquashFS.
No GPU process ran and no Megatron source, timeout, or launch declaration changed.

**Final-stage reverse BF16.** The pinned `MoEFlexTokenDispatcher.token_combine` defaults to `async_finish=True` and `allocate_on_comm_stream=True`, and `FusedCombine.backward` carries those flags forward from its autograd context.
The backward obtains the original routing handle from that context and calls cached BF16 `Buffer.dispatch`; it does not call the uncached `internode_prepare` receive-count path.
Before that call it creates `EventOverlap(EventHandle())`, *then* evaluates `grad_output.contiguous()`.
The raw C++ no-argument constructor in `ep_event_hip.hpp` records on HIP's null stream, but Megatron does not call that constructor.
Its `deep_ep.utils.EventHandle()` import is a Python wrapper that explicitly passes `torch.cuda.current_stream().cuda_stream` to the native constructor, as does `Buffer.capture()`.
The image's native `internode_dispatch` uses the supplied previous event in preference to the explicit caller stream, then queues `cached_notify` and the payload dispatch on UCCL's stream and records the completion event there.
Megatron queues a wait for the returned event on the schedule stream; the next MLP backward waits on the schedule event from that node.

The failed trace places Megatron's `moe_combine backward` on a nonzero schedule stream, separate from both compute stream 0 and UCCL's stream; this is not evidence of a wrong-stream event.
The narrower source-level gap is that the argument copy, if needed, is queued after the producer event has been recorded on the correct stream.
A CPU stub executing the exact pinned Python functions confirmed `record current stream → contiguous → dispatch → wait on reverse completion`; it does not establish GPU ordering.
The retained trace does not show whether this gradient needed a copy or whether the later native queues waited for it via any other dependency.
The nine completed pre-dispatch UCCL markers only certify work already queued on that stream *before* the dispatch's input-event wait, not that the input event released, that cached notify ran, or that payload dispatch completed.
The remaining seven pre-markers also leave older queued work in play.
The next uncached forward dispatch is the operation whose CPU polling timed out; that error is not proof that the cached reverse dispatch itself entered a CPU timeout or that CXI failed.
Other call sites use the same Python event wrapper; no blanket conversion is warranted, and this audit does not establish that the copy-order gap caused the full-layout failure.

**Stage-1 call 10.** Using same-rank timestamps and the measured `mlp backward` host span, the first FC1 replay began 206.95–208.94 s into that node; only 1.61–4.40 s remained until the node returned.
The pinned resident path gets weights and pads and quantizes `dy`, then issues FC2 dgrad *before* entering its per-expert FC1 replay loop.
Thus the ~211 s cannot be assigned to the FC1 replay loop, but the current boundary does not distinguish an upstream/device wait, allocator work, quantization, or the FC2 dgrad launch path.
All 16 senders had `ninja`, `c++`, and `cc1plus` descendants in three early process samples each before the first replay; the process records contain names but not command lines or compiler inputs.
The corresponding stage-2 call-8 host returns preceded stage-1 replay entry by 1.18–1.32 s on logged cross-node clocks, consistent with upstream backward-0 work gating this interval but not proving when the gradient was posted or when it became usable on stage 1.
The stage-1 host receive callbacks each took less than 0.06 ms and do not prove GPU completion.
The long attention backward node returned 10.66–17.71 s *before* the first exact CK mapping was logged; only Python processes were sampled during that long span.
Do not attribute that span to a target CK build or equate the observed compiler callbacks with all attention work.

The smallest discriminating GPU test is a one-MI300A event-ordering component: use the same current-stream event wrapper in both arms, capture before versus after a real noncontiguous-to-contiguous BF16 copy, and use a bounded producer gate to distinguish whether the consumer event can complete before the copy when no other dependency is present.
Include a contiguous no-copy control, validate the final BF16 values after the gate is released, and do not read the payload before it is ready or insert a stream-wide synchronization into the measured path.
If it demonstrates the distinction, extend the existing EP16 exact transport fixture with a separate, reviewed nondefault producer/consumer case for `FusedCombine.backward`, including the existing cached reverse route, interleaved handles, and native FNUZ/BF16/FP32 gradient invariants.
The candidate source change, if taken forward, must capture after materializing the BF16 send tensor on its producing stream, keep the native completion event and collective order, and land in a dedicated worktree with a user-signed commit.
The prior `756615` component acceptance remains valid for its own tested schedule; it was not constructed to distinguish these two event-capture semantics under a held producer.
Separately, a future stage-1 trace should bracket the incoming backward-0 GPU wait and the FC2 dgrad enqueue before replay, rather than retiming the same full layout or changing the deadlines.

##### Single-MI300A BF16 event-order check: launch review

The [component declaration](../chonk-bf16-event-order-i910bfcf936d3/experiment.toml) tests the pinned image's `deep_ep.utils.EventHandle()` and `EventOverlap.current_stream_wait()` without Megatron or a collective.
It uses three distinct nondefault streams and an eight-byte HIP signal, with a runtime capability check and no fallback.
The installed HIP header and exported ABI symbols were checked offline; the device's wait-value capability has not yet been exercised.
The BF16 input is a deterministic `[8192,3584]` fixture, not a replay of the retained training gradient layout.

| Arm | Producer ordering | Expected while the signal is held |
|---|---|---|
| Before copy | Capture readiness, hold, materialize the strided input | Consumer completion passes; producer completion remains pending |
| After copy | Hold, materialize the same strided input, capture readiness | Consumer and producer completion both remain pending; independent consumer prefix completes |
| Contiguous control | Reuse contiguous input, capture readiness, hold unrelated successor work | No copy; consumer completion passes despite the held successor |

The consumer queues only markers while the gate is closed; it does not read the payload.
The gate is released in `finally`, and all three streams must pass bounded event queries before the payload is transferred for exact BF16 comparison.
The independent prefix and initial marker checks are each bounded at five seconds, the held observation is one second per arm, and the release/drain is bounded at ten seconds.
If the gate cannot be used or drained, fail closed under the existing runner without an alternative gate, timeout extension, or retry.
A pass establishes only the local event-order distinction; it does not prove that the training gradient required a copy or that cached reverse dispatch or CXI was the cause of the stall.
There is no training, checkpoint, image change, or shared cache mounted or published.

349 focused CPU and scheduler-stub tests passed.
The guarded declaration SHA-256 is `1d6f9e4e6c50782b8668654e39fa705d948760acd4c5faf43432007f09ba480e` and the validator SHA-256 is `60934e87b7dcecdabcda771e90e6a8077b25806ef9c28f68122d7a14f040fbc0`.
Use one exclusive x86-64 Beverin `mi300` node in `cscs` under `csstaff`, with one visible MI300A, 48 CPUs bound to cores, all allocatable memory, and `nid002450` excluded.
The allocation is ten minutes and the command deadline is 300 seconds.
The pinned GPU container is not available for this check within the current agent sandbox, so Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
Recheck combined investigation occupancy before submitting once and leave unrelated allocations untouched.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-bf16-event-order-i910bfcf936d3 \
  --submit --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:10:00 --exclude nid002450 --cache-image --reservation cscs
```

EDF input: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Run artifacts: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-bf16-event-order-i910bfcf936d3/<run-id>/`, with the component report at `output/0/bf16-event-order.json`.
The user approved the exact launch.
The preflight found `762400`, a separate 16-node Chonk EP16 profiling run in `cscs`, so the combined cap leaves no room for this additional node while it runs.
The user then explicitly authorized one additional node while `762400` was running and reconfirmed the outside-sandbox launch.
Job `762498` was submitted once in `cscs`, with 16 profiling nodes plus one probe node at the occupancy check.
The [frozen run](../../artifacts/experiment-data/runs/chonk-bf16-event-order-i910bfcf936d3/20261124T155339Z-204b1a8d/) is retained and the launch guard is restored.
The profiling job and unrelated allocations were left untouched; the normal 16-node cap applies to future launches.

##### `762498` result: BF16 readiness ordering distinguished

The component and runner passed on `nid002774` (gfx942); Slurm completed the job in 22 seconds with exit `0:0`.
With the gate held, the independent consumer prefix completed in all three arms.
Capturing before the strided-to-contiguous BF16 copy let the consumer's dependent marker complete while the producer remained pending.
Capturing after the copy kept the dependent marker pending until release.
In the contiguous control, no copy occurred and the consumer completed despite unrelated successor work being held.
All streams drained after release, and all three payloads matched the BF16 reference exactly.
The component did not exercise a collective or read a held payload.
It confirms the local ordering gap, not that the training gradient was noncontiguous or that this gap caused the full-layout timeout.
The next discriminating check is the separately reviewed EP16 cached reverse-dispatch case with a noncontiguous BF16 producer.
No shared cache was published and no distributed check was submitted.

##### Native EP16 gated BF16 reverse dispatch: launch review

The [EP16 component](../chonk-bf16-event-order-ep16-i910bfcf936d3/experiment.toml) uses image `910…`'s installed DeepEP/UCCL wrappers and native extension, with no Megatron import, patch, or checkout.
It uses 16 ranks across four nodes, width 1792, 128 input rows per rank, 256 experts, top‑8 routing, and the 56-byte unaligned FP32 scale row.
It verifies the pinned Torch, HIP, DeepEP wrapper, utility and UCCL extension identities, native CXI initialization, GPU-resident scratch, rank placement, and expected buffer sizing.

Create two distinct routing handles before consuming either of them.
For each handle, verify the native FNUZ wire bytes, FP32 scales and routing metadata; then run both a contiguous BF16 control and a gated strided-to-contiguous BF16 copy followed by cached reverse dispatch.
Use the exact original routing handle, native async completion event, and a current-stream readiness capture **after** materialization.
While the gate is held, require the marker before cached dispatch on the UCCL stream to complete and require copy, dispatch and dependent completion markers to remain pending.
Never dispatch behind an event recorded before the copy, and never read the payload before release and bounded drain.
After release, require exact cached BF16 rows, exact BF16 reverse-combine sums, and exact FP32 per-route probability gradients on every rank for both handles.
The component uses a synthetic balanced routing fixture and the native Buffer API, not the full training schedule or a changed `FusedCombine.backward`.
It cannot establish that a copy occurred in the failed training run or that correcting event order resolves the full-layout timeout.

The HIP signal is released in `finally` if the local host call reaches that boundary; a blocked native call remains subject to the outer command deadline.
The run uses a private per-step FileStore with a 180-second rendezvous and process-group bound, the existing bounded local event observations, and a 600-second command deadline.
No training, checkpoint, fallback transport, shared cache, or retry is authorized.
360 focused CPU and scheduler-stub tests passed.
The guarded declaration SHA-256 is `6b7a38bb79a96bb874159f12a7b2f372778b1e3b1abc7995a8cdd7deb7f9cb98` and the validator SHA-256 is `ab22c275c37f5f6e720db8d4ad297d011d786be76a4c19ffb3bc3f9203698782`.

Use four exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four visible MI300As per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
The allocation is 20 minutes.
This multi-node component cannot run in the current agent sandbox, so Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
Chonk profiling job `762400` still occupied 16 investigation nodes at review.
It subsequently left the queue (`FAILED`, exit `1:0`), and the occupancy recheck found no active investigation jobs.
Recheck combined occupancy before submitting once; the one-node exception for `762498` does not apply and the profiling job must be left untouched.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-bf16-event-order-ep16-i910bfcf936d3 \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:20:00 --exclude nid002450 --cache-image --reservation cscs
```

EDF input: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Run artifacts: `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-bf16-event-order-ep16-i910bfcf936d3/<run-id>/`, with per-rank reports at `output/0/bf16-ep16-rank-<rank>.json`.
After explicit approval and the occupancy recheck, job `762588` was submitted once in `cscs` under the normal cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-bf16-event-order-ep16-i910bfcf936d3/20261124T162633Z-c90e0984/) is retained and the launch guard is restored.

##### `762588` result: native cached BF16 dispatch waits for materialization

Slurm completed the four-node component in 31 seconds with exit `0:0`, and all 16 ranks passed on gfx942 with native CXI and GPU-resident communication buffers.
Both outstanding routing handles passed on every rank: 32 rank-handle cases in total, each with 1,024 received rows and 56-byte FP32 scale rows.
In all 672 held samples, prior UCCL work was complete while the copy and both dependent completion markers remained pending.
The cached native call returned on the host in 0.100–0.362 ms while the gate was held; these are enqueue timings, not GPU communication timings.
After release, all streams drained and the FNUZ wire payload and scales, routing metadata, contiguous control and gated cached BF16 rows, BF16 combine sums, and FP32 probability gradients matched exactly.
The run qualifies capture after materialization in the native EP16 component; it did not run Megatron or test whether the failed training gradient needed a copy.
This does not establish the cause of the full-layout stall, nor does it qualify a Megatron source change.
The allocation cleared, and no shared cache was published or follow-up submitted.

##### Megatron reverse-readiness fix: signed source handoff

The proposed change is in the dedicated worktree `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready`, branch `experiment/amd-chonk-bf16-reverse-ready`, based on `ceb8035c182c40711db4d256d99fc44e4b8db27e`.
`FusedCombine.backward` now materializes the contiguous send tensor before recording the current-stream readiness event.
It retains the original routing handle, native dispatch arguments, output event wait, and collective order; no other communication site is changed.
The EP16 fixture adds a separate `ep16-ready-order` mode using actual autograd and checks that strided and contiguous BF16 gradients reach `FusedCombine.backward` on the nondefault schedule stream with their original handles.
It retains the exact FNUZ wire, BF16 dispatch/combine, and FP32 probability-gradient oracles.

| Reviewed Megatron file | Candidate Git blob |
|---|---|
| `megatron/core/transformer/moe/fused_a2a.py` | `d805ecfec99c33d894fef7c0c6d03f6f528bf5ba` |
| `tests/functional_tests/qualify_fnuz_dispatch.py` | `b68111d89b5991e7e3114a7837c2e5f55ec5c1eb` |

CPU stubs over the exact parent and candidate functions checked all eight combinations of source version, input layout, and async mode, including materialization/event order, handle identity, dispatch options, and completion wait.
365 focused Apertus CPU and scheduler-stub tests passed.
No assistant commit, push, GPU job, or publication occurred during source preparation.
Suggested signed commit subject: `Fence contiguous BF16 gradients before cached MoE dispatch`.

User-signed `041c1d753028e4190efb342accb98a6f362ee7dd` has tree `fb163db400a24be7d559837783667c330694cad8` and sole parent `ceb8035c182c40711db4d256d99fc44e4b8db27e`.
The signature from `FB605768E5C3B6BE52CA4B09B5C8CB8659FA42E6`, the two reviewed blobs, the exhaustive changed-file list, and the clean dedicated worktree were verified before pinning both declarations.
First review the exact [four-node Megatron EP16 regression](../chonk-native-fnuz-reverse-ready-ep16-i910bfcf936d3/experiment.toml) launch and obtain separate approval.
Require native CXI, exact numerical reports, and the intended BF16 layout on all ranks for every case before reviewing any full-layout launch.
The guarded [full-layout retest](../chonk-native-fnuz-reverse-ready-i910bfcf936d3/experiment.toml) retains the `762390` workload, CK prewarm, P2P trace and deadlines.
It adds final-stage gradient layout and host entry/return at `FusedCombine.backward`, plus stage-1 backward-0 GPU wait markers and host and GPU markers around the first FC2 dgrad launch.
Those markers are queried without adding stream waits, synchronizations, or tensor retention.
The full-layout acceptance criterion remains one finite, unskipped update with all existing gates intact.
The user approved a 16-node outside-sandbox retest before the prerequisite four-node regression ran.
The completed component result and exact full-layout command review are recorded below.

##### Signed Megatron EP16 readiness regression: launch review

Use four exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four GPUs and four ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
The allocation is 25 minutes and the rank-child deadline is 900 seconds.
The signed `ep16-ready-order` fixture uses a three-minute process-group timeout.
All 16 ranks must complete with native CXI and exact results for the three cases, including a strided gradient and a contiguous gradient on the nondefault schedule stream, with the original handles.
Check each rank's `FNUZ_TRANSPORT_RESULT` as well as the scheduler and runner exits; this is a component regression, not a full-size update.
No training, checkpoint, shared compiler cache, or retry is authorized.
This multi-node regression cannot run in the current agent sandbox, so Slurm runs **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
The guarded declaration SHA-256 is `9923756477787322b41207ad9fb6e56e40bdf8dbff021a60f24c92f5a0045cd8`.
Recheck combined investigation occupancy before submitting once and keep the normal 16-node cap.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-reverse-ready-ep16-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:25:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Frozen inputs and rank reports will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-reverse-ready-ep16-i910bfcf936d3/<run-id>/`, with private rank caches in job-owned `/dev/shm/apertus-amd/`.
After explicit approval and the occupancy recheck, job `762600` was submitted once in `cscs` within the normal cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-reverse-ready-ep16-i910bfcf936d3/20261124T171639Z-c3ec6a16/) is retained and the launch guard is restored.

##### `762600` result: signed Megatron EP16 regression passed

Slurm completed the four-node `cscs` job in 1:11 with exit `0:0`, and the runner passed.
All 16 gfx942 ranks used native CXI and reported three exact cases each: width 1792 strided, width 1792 contiguous, and width 1536 contiguous.
The signed fixture exercised the actual `FusedCombine.backward` via autograd and verified the intended BF16 input layout, nondefault stream, original routing handles, exact FNUZ wire and routing oracles, BF16 dispatch/combine, and FP32 probability gradients.
All 48 rank cases passed; the allocation cleared and no shared cache was published.
This satisfies the component gate, but neither establishes that the training gradient is strided nor proves that the full-size stall is fixed.

##### Signed full-layout reverse-readiness retest: exact launch review

The guarded [declaration](../chonk-native-fnuz-reverse-ready-i910bfcf936d3/experiment.toml) was compared with the frozen `762390` inputs.
Only the signed source pin, `APERTUS_CHONK_REVERSE_READY_TRACE=1`, validator bytes, and run and guard metadata differ.
The resolved workload, generated Slurm step, pinned image, mount policy, and all 11 runner Python files matched the frozen baseline apart from the expected path and trace changes.
The guarded declaration SHA-256 is `892f6fdea58b5382591b03a60f79ae1bcc605e2b1fdfdbbee653a1fb25ce28f3` and the validator SHA-256 is `2e6b8ea473a333decadfae1b210c1e58c74620b31ff810fe6047ea5cc36f2806`.
The signer, clean checkout, exact parent, and both reviewed blobs were reverified.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300As and four ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second rank-child deadline, 600-second GPU identity wait, 240-second node CK prewarm bound, and 300-second coordination bound.
The current agent allocation cannot run the 64-rank topology, so Slurm must execute **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
At review, no investigation job was active or queued and `cscs` had 32 idle reserved nodes.
Recheck combined investigation occupancy and reservation room immediately before any submission, leave unrelated jobs untouched, and restore the launch guard after the one-shot attempt.

Require all existing fit gates, including node-local CK proof and one finite, unskipped full-size update, plus usable reverse-readiness records.
Inspect final-stage call 9 `(f=8,b=3)` for the actual BF16 shape, stride, contiguity, and reverse dispatch GPU completion, and stage-1 call 10 `(f=9,b=0)` for backward-0 wait and first FC2 dgrad host/GPU progress.
Retain call-8 and later matched P2P observations; a host return is not GPU completion.
No checkpoint, shared cache, publish, retry, or deadline extension is part of this run.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-reverse-ready-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Frozen inputs and rank reports will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-reverse-ready-i910bfcf936d3/<run-id>/`, with private rank caches in job-owned `/dev/shm/apertus-amd/`.
The user confirmed that the earlier outside-sandbox approval covers this exact launch.
The pre-submit recheck found no investigation jobs and 32 idle eligible `cscs` nodes, so the job was submitted once as `762603` under the 16-node cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-reverse-ready-i910bfcf936d3/20261124T172348Z-889cb94d/) is retained, and the launch guard is restored.

##### `762603` result: the signed readiness fix did not clear the full-layout timeout

All 16 nodes qualified their independent CK objects and all 64 ranks verified their private copies.
All ranks initialized native CXI and entered update 1; stages 1–3 recorded 48 verified training-process mappings of their prewarmed CK objects.
The frozen validator's partial CK checks found no rebuild of the exact seeded key in any training-rank log.
Call 8 cleared on all 16 final-stage ranks, including the forward receive and reverse BF16 GPU markers.
The later native forward dispatch timed out on all 16 stage-2 ranks at call 10 `(f=9,b=2)` and all 16 stage-3 ranks at call 9 `(f=8,b=3)`.
Minimum sampled node availability in those windows was 47.02 GiB and 83.60 GiB, respectively; no rank logged an OOM.

**Stage 3:** The actual reverse BF16 gradient was `[8192,1792]` with stride `[1792,1]`, already contiguous on all 16 ranks.
The signed `FusedCombine.backward` returned on the host on all 16, but the UCCL marker after cached reverse dispatch completed on 0/16 at the later timeout; the before-marker completed on 8/16.
Loss backward and the forward-8 receive completed on GPU on all 16; the dependent MLP-backward after-wait marker remained pending on all 16.
The readiness-order fix addresses the strided component case, but a late contiguous copy cannot explain this retest's failure.
These markers do not distinguish work ahead of the reverse dispatch from the cached communication itself.

**Stage 1:** The backward-0 host wait returned and the first FC2 dgrad entered **206.93–208.35 s** later on all 16 lanes.
Both GPU wait markers were complete when first queried at FC2 dgrad entry; that does not timestamp their completion within the interval.
The FC2 dgrad host call returned in **0.530–0.685 ms**, its after-marker was complete by call exit on every lane, and FC1 replay entry followed FC2 dgrad entry by about 1–2 ms.
The scoped MLP- and attention-backward host spans were 208.60–209.19 s and 88.64–98.67 s, respectively.
The observed Dynamo callbacks totaled 3.46–4.33 s per rank over call 10, only about 1–3 ms entirely between the wait return and FC2 dgrad entry.
Sampled `ninja`/C++ compiler descendants within the gap do not account for its full duration.
The stage-1 forward-9 sends reached native entry 10.24–10.57 s after their paired stage-2 errors by cross-node wall clock; stage 2's forward-9 GPU after-marker was pending on all 16 at the timeout.
This narrows the dominant first host interval to before the first FC2 dgrad launch, not its host enqueue or FC1 replay, without proving which preparation or runtime operation consumed the interval.

No rank completed an update.
After all 32 fatal rank records were retained, only job `762603` was cancelled at 13:00; the step ultimately recorded `FAILED` (`1:0`) and the runner wrote a failed result.
The allocation cleared, no `checks.json` fit verdict was written, and no cache was published or follow-up submitted.
The pinned stage-1 path calls schedule-node autograd before the resident adapter can check the saved generation, wait on the weight event, pad `dy`, allocate destinations, and quantize the FC2 input rows.
The current trace does not show when the adapter was entered; these are separate boundaries to discriminate before attributing the interval to expert quantization or the allocator.
Before any further full-layout launch, narrow that interval and the stage-3 cached-reverse dependency separately; do not extend deadlines or rerun this declaration.

##### Offline follow-up: separate MLP autograd entry from FC2 input preparation

Reconstructing the beginning of the longest scoped `mlp backward` context from its recorded end and elapsed time puts it 0.99–2.35 s after the stage-1 backward-0 host wait returned.
The first FC2 dgrad launch then entered 204.59–207.29 s into that context, across all 16 lanes.
Each lane had three telemetry samples with `ninja` and C++ compiler descendants during the pre-GEMM gap.
The last such sample came 13.07–17.02 s after the host wait returned, leaving 189.92–195.24 s before GEMM entry with no compiler descendant in subsequent samples.
Process sampling does not rule out in-process compilation, a blocked runtime call, or shorter activity between samples.
The pinned source has a lazy C++ `share_storage_ext` build in `CheckpointWithoutOutput`, and the routed MLP forward attaches a pre-MLP layernorm recomputation hook to the expert output when enabled.
This is a candidate for early compiler activity, not an attribution: the telemetry retained names, not the compiler command line or the extension call boundary.
The useful next sender split is the schedule wait enqueue, autograd engine entry, layernorm recompute hook and storage-extension entry/return, resident backward entry, weight retrieval, padding/allocation, and TE input row quantization.
Observe those boundaries without replacing autograd, adding a device wait, or changing a cache or stream policy.
The stage-3 cached-reverse before-marker split was not node-wide: each of its four nodes had both completed and pending lanes.
The existing marker before the wrapper does not distinguish its earlier queued UCCL work from readiness, native cached notification, and payload dispatch.
Keep that dependency separate from the sender's late forward-9 send; the stage-3 forward-8 receive was already complete.
No full-layout rerun, source change, or timeout change follows from this offline analysis.

##### Full-layout stalled-phase diagnostic: exact launch review

The guarded [stall-trace declaration](../chonk-native-fnuz-stall-trace-i910bfcf936d3/experiment.toml) keeps the signed source and frozen `762603` workload, image, runner, and deadlines.
Beyond run and guard metadata, the only differences are `APERTUS_CHONK_STALL_TRACE=1` and the validator bytes.
The clean Megatron worktree, signer `FB605768E5C3B6BE52CA4B09B5C8CB8659FA42E6`, commit `041c1d753028e4190efb342accb98a6f362ee7dd`, sole parent and both reviewed blobs were reverified.
No Megatron or vendor source changed, and no new signature is required.

- **Stage 1, call 10 `(f=9,b=0)`:** Trace the first MLP backward's stream-wait enqueue and autograd entry, any layernorm recompute and storage-extension work reached there, and the first resident backward.
  Within the resident adapter, bracket saved-weight retrieval, padding, destination allocation and the TE input-row quantization until FC2 dgrad entry.
  Sample the MLP wait markers at body entry, resident entry and FC2 entry; preserve the prior backward-0 and FC2 markers.
- **Stage 3, call 9 `(f=8,b=3)`:** Mark the producer immediately before readiness capture and the UCCL queue at entry to the cached internode wrapper, at the last Python argument before native dispatch, and after the wrapper returns.
  Bracket the existing output-event wait on its consumer stream and sample the markers when the following forward dispatch returns or fails.
  This distinguishes earlier queued work, producer readiness, and the native cached-call boundary; it does not separately observe `cached_notify` and payload execution inside the native call.

The observer records and queries events without adding waits, synchronizations, tensor retention, or changing any native arguments.
GPU markers show whether a stream reached a point when queried, not elapsed kernel times.
Preserve all existing fit gates and require usable scoped trace evidence plus one finite, unskipped update for a fit pass.
On failure, retain completed and open host spans and sampled GPU state; do not retry or change deadlines.
370 focused CPU and scheduler-stub tests passed.
The guarded declaration SHA-256 is `70fc3f7509bc73415ea7b1fe846eb0249769d1ce67d2c07fe37d9cdf35a7f3c8`, and the validator SHA-256 is `694afdc450976b74c5d049826c7e7a6de60bfa1a2f08d79799c7dc0e2aa425ff`.
The dry plan verified the frozen source manifest, unchanged runner file hashes, Slurm step shape, and EDF delta against `762603`.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300A ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second rank-child deadline, 600-second GPU identity wait, 240-second node CK prewarm bound, and 300-second coordination bound.
The current agent allocation cannot run the 64-rank topology, so Slurm must execute **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
At review no investigation job was queued and 32 eligible reserved nodes were idle; recheck both the combined 16-node cap and reservation capacity immediately before submission.
No checkpoint, shared compiler cache, publication, or automatic follow-up is included.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-stall-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Frozen inputs and rank reports will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-stall-trace-i910bfcf936d3/<run-id>/`, with private rank caches in job-owned `/dev/shm/apertus-amd/`.
The user approved this exact outside-sandbox launch.
The recheck found no investigation jobs and 32 idle eligible `cscs` nodes; the run was submitted once as `764472` under the 16-node cap.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-stall-trace-i910bfcf936d3/20261126T074745Z-20677aef/) is retained, and the one-shot guard is restored.

##### `764472` result: stage-1 gap precedes resident backward; final-stage call 8 timed out

All 16 nodes passed CK prewarm and all 64 ranks verified their private copies, started the first update, and initialized native CXI.
Stages 2–3 recorded 32 training-process mappings of the seeded object, and no rank logged a rebuild of the exact target key.
All 16 final-stage ranks timed out in `internode_prepare` at call 8 `(f=7,b=2)`, 101.17–101.39 s after call entry.
No rank logged an OOM, and the minimum sampled node availability during that window was 79.49 GiB.
Loss backward and the host receive wait returned, but the receive completion marker and both reverse BF16 UCCL markers were still pending on all 16 ranks at timeout.
The stage-2 matching native forward send was posted **0.67–1.01 s after** the paired final-stage errors by cross-node wall clock.
Thus call 8 can recur despite CK reuse when the forward send misses the native timeout; the trace does not identify why the sender was late.
Call 9 was not reached, so the new stage-3 cached reverse observer produced no evidence about that separate boundary.

All 16 stage-1 ranks completed the scoped sender trace.
Inside MLP autograd, layernorm recompute took 15.00–16.65 s, nearly all inside `_get_share_storage`.
The interval from the recompute hook returning to resident backward entry was **194.92–198.21 s**.
The MLP GPU wait markers were both pending at body entry and both complete by resident entry; the trace does not timestamp completion within that interval.
Within resident backward, the summed host spans for weight lookup, input padding, output allocation, and TE row quantization were individually at most 0.36 ms, 0.94 ms, 1.51 ms, and 10.25 ms per rank, respectively.
This locates the dominant sender delay before resident input preparation and FC2 dgrad, not inside those operations.
The pinned forward graph places DeepEP's fused output unpermute between resident expert output and the tensor carrying the recompute hook, making that backward and the autograd stream handoff the next boundaries to separate, not proven causes.

After the 16 fatal final-stage records were durable, only job `764472` was cancelled at 11:36.
The step recorded `FAILED` (`1:0`), rank zero wrote a failed result, and the allocation cleared.
No update completed, no `checks.json` fit verdict was written, and no cache or checkpoint was published.
No retry or deadline extension was made.

##### Full-layout autograd handoff and call-8 sender: exact launch review

The guarded [handoff-trace declaration](../chonk-native-fnuz-handoff-trace-i910bfcf936d3/experiment.toml) retains all of `764472`'s inputs except the validator, `APERTUS_CHONK_HANDOFF_TRACE=1`, and run and guard metadata.
The clean signed Megatron checkout, signer, parent, and reviewed blobs were reverified.
The frozen source manifest, runner hashes, Slurm step, and EDF were compared against `764472`.
The guarded declaration SHA-256 is `9244182bb2352f72002b5ac51fda5b2b2662930c5fe2a1ce6e4b027e89f3d65c` and the validator SHA-256 is `431c477ff4cdcd3e04b82f1c3e439b3dc0e5da5eb9a3d8d8b72ca1ef462e87cf`.
374 focused CPU and scheduler-stub tests passed; this does not qualify the GPU trace.

- **Stage 1, call 10 `(f=9,b=0)`:** Bind temporary hooks to the actual generated autograd nodes for TE mask-map unpermute and the resident expert.
  Record host and GPU boundaries around native unpermute backward, its TE Triton permutation helper, and the resident node's prehook.
  Compare these against the retained recompute-return and resident-backward markers.
  If the node binding cannot be verified, record that explicitly and leave the original backward alone.
- **Stage 2, call 8 `(f=7,b=0)`:** Record the existing forward/backward callbacks, forward chunk wait, bounded schedule-node entry/exit records, and Dynamo callback spans up to the matching native forward send.
  Keep the existing matched P2P trace and distinguish host post time from GPU readiness.
- **Stage 3:** Retain the existing receive and cached reverse measurements if their boundaries are reached.

No backward, collective, wait, or stream is replaced; the observer adds no device wait.
The stage-1 GPU events only show completion when queried, not elapsed kernel times.
A fit verdict still requires one finite, unskipped full-layout update and all existing and new validation gates.
Preserve failure evidence, and do not automatically retry or extend deadlines.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300A ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second rank-child deadline, 600-second GPU identity wait, 240-second node CK prewarm bound, and 300-second coordination bound.
The 64-rank topology is unavailable in the current agent allocation, so Slurm must execute **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
At review, running job `764536` occupied 16 nodes counted against the combined investigation cap.
Do not submit or queue this job while the requested nodes would exceed that cap, and leave `764536` untouched.
Recheck both combined occupancy and eligible reservation capacity immediately before any approved submission.
No checkpoint, shared compiler cache, publication, or follow-up submission is included.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-handoff-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Frozen inputs and rank reports will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-handoff-trace-i910bfcf936d3/<run-id>/`, with private rank caches in job-owned `/dev/shm/apertus-amd/`.
The user approved this exact outside-sandbox launch once capacity is available.
The post-approval recheck still found `764536` using the full 16-node investigation budget.
The user then clarified that the cap applies only to investigation jobs submitted by this session's team, not jobs submitted by others.
The final recheck found no other team investigation requests and 16 eligible idle `cscs` nodes.
The approved run was submitted once as `764539`; job `764536` was left untouched.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-handoff-trace-i910bfcf936d3/20261126T081520Z-77287b5a/) is retained and the one-shot guard is restored.

##### `764539` result: stage-1 host delay is inside TE permutation; later timeouts recur

All 16 node CK handoffs and 64 private copies passed, and all 64 ranks entered the update with native CXI.
Stages 2–3 recorded 32 unchanged CK mappings; no rank logged a rebuild of the exact target key.
The frozen verifier accepted the scoped stage-1 handoff and stage-2 call-8 sender evidence on all 16 lanes of each stage.

**Stage 1:** TE's `permute_with_mask_map` took **192.67–195.50 s on the host** inside fused unpermute backward.
From the unpermute posthook to the resident-expert prehook took only **0.125–0.214 ms**, and all five handoff GPU markers were complete at that prehook.
This rules out a long delay in the post-unpermute handoff in this run, but does not identify when GPU work completed or show that a permutation kernel ran for 193 seconds.
The pinned TE helper allocates outputs and invokes a Triton autotuner keyed by `hidden_size`.
All 608 process-tree samples within the helper showed only `python3`; that does not rule out in-process compiler or autotuner work.
The next discriminating check is a one-GPU cold versus same-process warm exercise of the native no-probability BF16 unpermute backward, capturing allocation, Triton compilation, autotuner benchmarking, and ordinary launch separately.
Use the training specialization and routing geometry rather than prewarming a generic shape.
Do not change tuning policy, publish a cache, or infer fresh-process reuse from a same-process warm call.

**Stage 2:** Call 8 entered the matching native forward send **213.34–213.40 s** after call entry, and the stage-3 receive completed on GPU on all 16 lanes.
The longest MLP- and attention-backward host spans were **116.91–117.14 s** and **88.03–89.41 s**, respectively.
The forward chunk wait and post-forward callback returned within 0.14 ms and 1.25 ms, respectively.
Despite clearing call 8, stage 2 timed out at call 10 `(f=9,b=2)` on all 16 lanes with no corresponding stage-1 forward-9 send recorded.

**Stage 3:** All 16 ranks timed out at call 9 `(f=8,b=3)` after the forward-8 receive completed on GPU.
For the contiguous `[8192,1792]` BF16 reverse gradient, 15/16 producer markers and 10/16 UCCL prefix and native-entry markers had completed.
The cached-dispatch after-marker and the consumer's after-wait marker were pending on all 16.
The ten ranks whose producer and UCCL prefix markers both completed narrow the wait to the cached native path on those ranks; the markers do not separate its readiness wait, cached notification, and payload work.
The minimum sampled node availability in the failing stage-2 and stage-3 windows was 46.36 GiB and 81.41 GiB, respectively; no rank logged an OOM.

No update completed.
After confirming all 32 fatal rank records, only `764539` was cancelled at 12:44; its step recorded `FAILED` (`1:0`).
The allocation cleared, rank zero wrote a failed result, and no fit-validator verdict or cache publication exists.
No repeat or deadline extension was submitted.

##### Full-layout 600-second UCCL host timeout: exact launch review

At the user's request, the guarded [600-second timeout diagnostic](../chonk-native-fnuz-timeout600-i910bfcf936d3/experiment.toml) sets `UCCL_EP_CPU_TIMEOUT_SECS=600` for all ranks.
The pinned HIP implementation uses this variable in the host polling loop that raised `DeepEP error: timeout (dispatch CPU)`, and the string is present in the installed `uccl/ep.abi3.so`.
The override applies to UCCL's host receive timeout, not to GPU-side timeout constants, RCCL, or the training process-group timeout.
The same signed Megatron commit `041c1d753028e4190efb342accb98a6f362ee7dd`, image, full workload, runner, and CK prewarm are retained from `764539`.
The clean checkout, signature, parent, and both reviewed blobs were reverified.
Only the timeout override, validator, and run and guard metadata differ.

Retain the `764539` traces and sample the existing stage-2 forward-9/backward-2 receive markers and stage-3 forward-8 receive and cached reverse-dispatch markers at 101, 150, 300, and 580 seconds after entry to the respective traced Python dispatch, only while it is still in progress.
The 101-second sample approximates the previous cutoff; it is not a timestamp inside the native host polling loop.
The sampler queries existing GPU events without adding a device wait or replacing dispatch.
Distinguish a matching send being posted, GPU receipt completing, and dispatch returning.
If stage 2 clears while stage 3 remains pending, keep the two failure paths separate.
If an earlier watchdog or failure ends the run, record that boundary; the override does not guarantee a full 600-second observation window.
A fit pass still requires one finite, unskipped full-layout update and all validation gates.
No checkpoint, shared compiler cache, publication, automatic retry, or further deadline change is authorized.

377 focused CPU and scheduler-stub tests passed.
The guarded declaration SHA-256 is `d3453e6d51269df8cce4493b3c28d0ccd11ca49bfe3c6159e37ae80e0c0754a6` and the validator SHA-256 is `c587c5b27c6ceb1b12c38a5f45edb2b9ca944fcf49e854305a9d093ac0c20438`.
The frozen source manifest, runner hashes, Slurm step, unchanged outer deadlines, and EDF delta were checked against `764539`.
Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300A ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second child deadline, 600-second GPU identity wait, 240-second node CK prewarm bound, and 300-second coordination bound.
The current agent allocation cannot run this topology, so Slurm must execute **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
Recheck the 16-node cap for investigation jobs we submitted and actual idle reservation capacity; leave other jobs untouched.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-timeout600-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Frozen inputs and rank reports will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-native-fnuz-timeout600-i910bfcf936d3/<run-id>/`, with private rank caches in job-owned `/dev/shm/apertus-amd/`.
The user approved this exact outside-sandbox launch.
The pre-submit recheck found no other investigation requests we submitted and 16 eligible idle `cscs` nodes.
The probe was submitted once as `764642`, and the one-shot guard is restored.
The [frozen run](../../artifacts/experiment-data/runs/chonk-native-fnuz-timeout600-i910bfcf936d3/20261126T084820Z-193edd18/) is retained.

##### `764642` interim observation: both dispatches cleared after the old cutoff

All 64 ranks verified their private CK seeds, entered the first update, and eventually recorded the unchanged training-process CK mapping.
At 101 seconds into the scoped stage-2 call-10 dispatch, the forward-9 after-marker and both backward-2 markers were pending on all 16 ranks.
Stage 1 posted all 16 matching forward-9 sends at 120.13–120.44 s relative to the stage-2 dispatch entries by cross-node wall clock.
The stage-2 dispatches returned after 120.45–120.54 s, with the forward-9 and backward-2 GPU receive markers complete on all 16 ranks.
At 101 seconds into the scoped stage-3 call-9 dispatch, the forward-8 receive and reverse producer markers were complete on all 16 ranks, but the cached reverse completion and consumer after-wait markers were pending on all 16.
The stage-3 forward dispatches returned after 121.87–122.02 s, with all cached reverse and consumer GPU markers complete.
This establishes recovery at both previously failing boundaries under the larger host deadline in this run; it does not establish a shared root cause.
At the 15:15 Slurm snapshot all ranks had advanced past both boundaries, but no update had completed and no fit verdict was available.

##### `764642` outcome: late backward drain failed after both dispatches recovered

Stage 0 reached its last observed schedule exit at call 267; stages 1–3 stopped making progress during backward-only calls 259–262 without any rank recording `update_exit`.
Rank 60 last entered call 261 `(forward=None, backward=255)`, logged `HSA_STATUS_ERROR_EXCEPTION`, and the runner recorded its child exit as `-9`.
All 16 stage-1 rank logs contained `HSA_STATUS_ERROR_MEMORY_FAULT`, and nine final-stage rank logs contained the HSA exception.
These are asynchronous queue reports and do not identify the first failing kernel or the cause of the SIGKILL.
Two stage-1 nodes and one final-stage node each sampled zero `MemAvailable`; available cgroup telemetry showed no positive OOM counters, but coverage was incomplete.
Memory pressure was observed; an OOM kill was not established.
The first persisted runner failure was rank 60's child exit, which triggered peer-failure interruption on other ranks.
After preserving the fatal rank record and confirming no update had completed, only `764642` was cancelled at 25:40.
The allocation cleared and the Slurm step ended `CANCELLED` (`0:9`).
The durable rank records were retained, but neither `result.json` nor `checks.json` was written.
No retry or publication was made.
The next attribution target is the late backward drain: associate the HSA queue or fault reports with last enqueued operations and allocation or lifetime evidence before changing the training recipe.

##### 24-layer backward-drain control: exact launch review

The guarded [24-layer drain control](../chonk-reduced24-native-fnuz-drain-i910bfcf936d3/experiment.toml) changes only the four depth-dependent arguments, a diagnostic flag, the validator, and run and guard metadata relative to `764642`.
Use 24 layers, three per virtual chunk (`Ettt|(ttt|)*6,tttL`), with 18 KDA and 21 MoE layers.
Keep the signed `041c1d753028e4190efb342accb98a6f362ee7dd` source, image `910…`, model widths, S4096/MBS2/GBS4096, optimizer, parallel degrees, head chunking at 512 tokens, FC1 replay, CK prewarm, and 600-second UCCL host receive timeout.
The original reduced probe `762626` is not this control: it used a dense head, stored expert preactivations and the old UCCL host deadline.
Verify all 64 CK private copies.
Require the seeded object to be mapped in training on stages 1–3; stage 0 owns only KDA layers in this particular layout, so it has no CK attention training consumer.

Keep the early dispatch and handoff observers from `764642`.
Add bounded host entry/exit records around backward-only calls 259–267, including schedule nodes and delayed wgrad, gradient finalization, norm preparation and the actual optimizer step.
These records add no device wait and do not identify a GPU kernel merely from its last host enqueue.
Compare the HSA faults, sampled memory headroom, and late progress against `764642`.
A pass requires one finite, unskipped update and a successful validator on all 64 ranks; it establishes reduced-model fit only.
A pass would not prove that `764642` OOMed because depth reduction also changes timing.
A recurring fault with substantial headroom would prioritize lifetime and ordering checks.
Do not retry automatically or extend any existing deadline.

380 focused CPU and scheduler-stub tests passed.
The guarded declaration SHA-256 is `8911ba6f18fb99924a372ff3560482e8fe0e2f0b8bf860b9efe7e9bf0e195c6d` and the validator SHA-256 is `69042c3f2ca9cc7111f5868c56a0070dcb77b7d77479a079a12398f5af978c28`.
The clean checkout, signer, parent and reviewed blobs were reverified, and the frozen source manifest, runner hashes, launcher and EDF delta were checked against `764642`.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, four MI300A ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Retain the 60-minute allocation, 2,700-second rank-child deadline, 600-second GPU identity wait, 240-second node CK prewarm bound and 300-second coordination bound.
The agent allocation cannot run this topology, so Slurm must execute **outside the agent sandbox with normal host filesystem access**, without inheriting the agent uenv.
At review, 31 eligible nodes were idle in `cscs` and no investigation jobs we submitted were in the queue.
Recheck capacity and the scope of the 16-node cap before submitting once, and leave other jobs untouched.
No checkpoint, shared compiler cache or publication is included.
Report progress approximately every three minutes and promptly on a terminal result.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-reduced24-native-fnuz-drain-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-bf16-reverse-ready \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
The artifact directory is `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-reduced24-native-fnuz-drain-i910bfcf936d3/<run-id>/`; private rank caches are in job-owned `/dev/shm/apertus-amd/`.
The user approved the exact outside-sandbox launch.
The pre-submit recheck found no other investigation requests we submitted and 16 eligible idle `cscs` nodes.
The control was submitted once as job `764651`, and the launch guard is restored.
The [frozen run](../../artifacts/experiment-data/runs/chonk-reduced24-native-fnuz-drain-i910bfcf936d3/20261126T092801Z-b35edd56/) is retained.

#### Reduced 24-layer dense-head probe: launch review

The user requested a reduced-depth run before investigating the full-layout stall.
The [separate declaration](../chonk-reduced24-native-fnuz-dense-fit-i910bfcf936d3/experiment.toml) uses the previously verified signed integration merge `4b21bb5781ee6db66952a9b6ab128b688c7dc1d1`, tree `9f0c1d1a76c9aec1c299c45dc24c14bff8b46060`, from the clean integration worktree.
That pin predates both output-head chunking and expert FC1 replay; it uses the dense native fused loss and stored expert preactivations while retaining the earlier native FNUZ and correctness work.
Relative to frozen `759635`, only the four depth-dependent training arguments change: 24 layers, `Ettt|(ttt|)*6,tttL`, 18 KDA layers in a repeating three-of-four pattern, and 21 MoE layers after the first three dense layers.
The parallel degrees, widths, vocabulary, S4096/MBS2/GBS4096, optimizer, compiler cap, image and placement request are unchanged.
The pinned source's CPU parser confirmed three decoder layers per virtual chunk and the expected embedding and loss placement.
231 focused CPU and scheduler-stub tests passed; the validator SHA-256 is `a13e981c532cb771149593cb32707980debd3a3aabad9eb162e91f445c6fc964`.

Require one finite, unskipped update with no rerun or discard, native CXI and clean completion on all 64 ranks, and no checkpoint or shared cache.
The validator checks effective reduced layout and masks and requires one dense `[4096, 2, 200064]` BF16/FP32 loss tile on the first head call of each final-stage rank.
The same host, device and mapping telemetry is retained so the original loss window and overlapping schedule can be compared.
An OOM, error, timeout or failed check ends the one-shot probe; there is no automatic retry or walltime extension.
This is evidence about a reduced model, not full Chonk fit or throughput.

Request one exclusive Beverin `csstaff` / general `mi300` allocation: 16 nodes, four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, `map_ldom:0,1,2,3`, and `nid002450` excluded.
Use a 60-minute allocation and 2,700-second training deadline.
At review, no investigation job was active or pending and 31 eligible general nodes were idle.
Recheck the combined 16-node investigation cap and idle placement before submission; unrelated jobs and the agent interactive allocation are not used.
The full topology is unavailable inside the agent allocation; Slurm executes **outside the agent sandbox with normal host filesystem access**, with `--uenv-passthrough=ignore` and the pinned container environment.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-reduced24-native-fnuz-dense-fit-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-reduced24-native-fnuz-dense-fit-i910bfcf936d3/<run-id>/`, source snapshots to the adjacent `experiment-data/sources/`, and private scratch and rank caches to job-owned `/dev/shm/apertus-amd/`.
The user approved this exact command.
Job `760023` was submitted once after rechecking that no investigation job was active or pending and 31 eligible general nodes were idle.
The [frozen run](../../artifacts/experiment-data/runs/chonk-reduced24-native-fnuz-dense-fit-i910bfcf936d3/20261123T133358Z-6515d26e/) is separate from the full-depth runs, and the launch guard is back in place.

#### `760023` reduced-depth probe: startup failure, fit untested

Slurm recorded `FAILED` after 10:31 with step exit `1:0`.
All 64 ranks recorded their environment, but ranks 4–7, 12–15, 52–55, and 60–63 missed the 300-second GPU identity rendezvous on `nid002424`, `nid002444`, `nid002484`, and `nid002488`.
Those ranks' identity files appeared roughly 7–9 seconds after the first recorded deadline, during failure propagation.
The environment and identity file timestamps show node-wide delays both before and after environment recording, but do not distinguish image/library loading, GPU initialization, or shared-filesystem latency.
In the frozen runner, the interval between those records spans Torch import, device selection and property lookup, and affinity/identity writes.
No training child, dense-loss trace, or optimizer update ran, so this is neither a reduced-model OOM nor a reproduction of the full-size dispatch stall.
The runner wrote a failed result, the allocation is gone, and no retry or deadline extension was submitted.
For another reduced-depth attempt, first distinguish those startup phases with bounded instrumentation; do not substitute a longer training or DeepEP timeout for the missing evidence.

##### GPU identity startup-only diagnostic: launch review

Independent review of `760023` found that every rank eventually wrote the same GPU identity key, all 64 retained their 48-CPU affinity masks, and the four late nodes had completed startup in `759718`.
Eight nodes' environment files appeared about 305 seconds later than the others; several nodes then took another roughly 300 seconds to publish identity.
In `759990`, three nodes showed a similar environment-to-identity delay, but the first-to-last identity spread was about 298 seconds, within the rendezvous budget.
Frozen `759635` used the same image, source snapshot, and startup environment as `760023` and completed that rendezvous with about 0.26 seconds of identity spread.
This is evidence of startup skew, not a depth-dependent training failure.

The [startup-only declaration](../chonk-reduced24-startup-trace-i910bfcf936d3/experiment.toml) preserves the `760023` image, signed source/manifest, startup environment, 16-node/64-rank placement geometry, and 300-second per-rank identity wait.
It adds flushed rank markers to `output/launcher.log`: first in-container Bash, Python/settings, integrity checks, environment publication, Torch import, device enumeration/selection/properties, affinity and identity writes, and identity wait outcome.
Python records UTC and monotonic timestamps, and diagnostic marker writes go through the launcher stream rather than relying solely on files written by the rank.
Stop after collective validation of identities; do not create a cache selection, materialize training source, stage scratch, compile dataset helpers, or launch a training process.
An all-rank success establishes only that this startup attempt completed within its bounds.
If the skew does not recur, do not infer a root cause or a 24-layer fit result.
The updated runner and both diagnostics passed 243 focused CPU and scheduler-stub tests.

Use exclusive general Beverin `mi300` nodes under `csstaff`, four MI300A ranks per node, 48 CPUs per rank, all allocatable memory, and `nid002450` excluded.
The allocation is capped at 25 minutes; the identity wait remains 300 seconds and there is no training child.
The full topology is not available within the agent allocation; Slurm executes **outside the agent sandbox with normal host filesystem access** and does not inherit the agent uenv.
At review there were 28 eligible idle general nodes and no active or pending investigation jobs.
Run this probe and the full-layout notification diagnostic sequentially, never exceeding 16 investigation nodes.
Recheck capacity and review startup evidence before using the full-layout slot; do not run an unchanged reduced training retry.

From `/ritom/scratch/cscs/boeschf/apertus-amd`:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-reduced24-startup-trace-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:25:00 --cache-mode cold --exclude nid002450
```

Artifacts go to `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-reduced24-startup-trace-i910bfcf936d3/<run-id>/`.
The user approved both exact diagnostic commands, with the startup probe first.
Job `760046` was submitted once after rechecking that no investigation job was active or pending and 37 eligible nodes were idle.
The [frozen startup run](../../artifacts/experiment-data/runs/chonk-reduced24-startup-trace-i910bfcf936d3/20261123T141240Z-ff4e7bb0/) is separate from `760023`, and its launch guard is back in place.
Job `760046` completed in 3:32, and rank zero recorded a startup-only pass on all 64 gfx942 ranks, with no training command, child or cache publication.
The first-to-last Bash marker spread was 5.84 seconds; the identity publication spread was 29.26 seconds.
Torch import took 8.73–34.31 seconds per rank and was the largest measured in-container phase.
GPU enumeration took at most 0.29 seconds, device selection 0.13 seconds, and each measured affinity or identity write at most 0.14 seconds.
Concurrent log lines occasionally joined, but decoding each prefixed JSON object recovered all startup phase markers for all 64 ranks.
The multi-minute skew seen in `760023` did not recur, so its root cause remains unverified; this was not a reduced-model fit run.

##### Cached-image 24-layer dense-head fit: exact launch review

The [separate guarded declaration](../chonk-reduced24-native-fnuz-cached-fit-i910bfcf936d3/experiment.toml) keeps the original `760023` 24-layer workload, signed integration source, training environment and deadlines.
Unlike `760023`, it uses the local SHA-verified copy of image `910…` and enables startup phase logging through training rather than stopping at identity validation.
Cached-image startup worked on the recent 64-rank full-layout launches, but this does not prove the cause of the earlier skew.
The signed clean worktree still has commit `4b21bb5781ee6db66952a9b6ab128b688c7dc1d1`, tree `9f0c1d1a76c9aec1c299c45dc24c14bff8b46060`, and the same three parents and signer as the frozen reduced probe.
The output head remains dense and expert preactivations remain stored; full batch, width, optimizer and parallel degrees are unchanged.
The current validator's reduced path retains the layout, 18 KDA/21 MoE masks, exact dense full-vocabulary BF16/FP32 loss tile, native CXI on all 64 ranks, and finite unskipped update requirements.
It does not enable CK prewarm or the full-layout-specific later traces.
The original declaration and frozen failure remain unchanged.

Use 16 exclusive x86-64 Beverin `mi300` nodes in `cscs` under `csstaff`, with four MI300A ranks per node, 48 CPUs per rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, `disable_rdzv_get`, all allocatable memory, and `nid002450` excluded.
Keep the 60-minute allocation, 2,700-second child deadline, and original 300-second identity wait.
The current agent allocation cannot run this topology, so Slurm must execute **outside the agent sandbox with normal host filesystem access**, with `--uenv-passthrough=ignore`.
The dry plan matched `760023` except for cached-image staging, startup tracing, reservation, the current frozen runner/validator, and run metadata.
The generated EDF retained the original native environment, injection guards, and mount policy, with no shared compiler cache.
The guarded declaration SHA-256 is `ef4865b83e5a2ba0357727c9cff0779818923d0b934e7a304034306020fa005c` and the validator SHA-256 is `2e6b8ea473a333decadfae1b210c1e58c74620b31ff810fe6047ea5cc36f2806`.
208 focused CPU and scheduler-stub tests passed.
At review there were 32 idle eligible `cscs` nodes and no investigation jobs in the queue.
Recheck combined occupancy before a one-shot submission and leave unrelated jobs untouched.
No checkpoint, cache publication, retry, or automatic deadline extension is authorized.
A pass would establish reduced-model fit only, not full Chonk fit or throughput.

From `/ritom/scratch/cscs/boeschf/apertus-amd`, the exact proposed launch:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-reduced24-native-fnuz-cached-fit-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-integration \
  --submit --gpus-per-node 4 --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 01:00:00 --cache-mode cold --exclude nid002450 --cache-image \
  --reservation cscs
```

EDF image: `/ritom/scratch/cscs/boeschf/apertus-amd/cache/images/sha256/910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490/image.sqsh`.
Frozen inputs and rank reports will be under `/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/runs/chonk-reduced24-native-fnuz-cached-fit-i910bfcf936d3/<run-id>/`, with private rank caches in job-owned `/dev/shm/apertus-amd/`.
The user approved the exact outside-sandbox launch.
The pre-submit recheck found no investigation jobs and 32 idle eligible `cscs` nodes.
The probe was submitted once as `762626` and its launch guard is restored.
The [frozen run](../../artifacts/experiment-data/runs/chonk-reduced24-native-fnuz-cached-fit-i910bfcf936d3/20261124T183631Z-f2ed5e9c/) is separate from `760023` and the full-layout runs.

##### `762626` result: first dense head passed; final-stage call 8 timed out

All 64 GPU identities were published within 0.718 s of each other, all ranks initialized native CXI, and all entered the first update.
All 16 final-stage ranks passed the first dense-head trace check at forward-only call 5, including the single `[4096,2,200064]` BF16 input and FP32 native cross-entropy tile.
This reduced model cleared the first dense-head memory boundary without chunking or FC1 replay.
It did not complete the update: all 16 final-stage ranks reported `DeepEP error: timeout (dispatch CPU)` in `internode_prepare` at call 8 `(f=7,b=2)`, 101.10–102.90 s after call entry.
The minimum sampled node availability during that boundary was 47.61 GiB, and no rank logged an OOM.
This does not establish memory fit over the unexecuted portion of the update.

Cold builds of the exact CK backward target appeared on all 32 ranks in stages 2 and 3, and no node CK prewarm was configured.
Stage 2 exited its own call 8 16.39–19.18 s after the paired final-stage errors by cross-node wall clock.
This resembles the older cold full-layout call-8 behavior, but the reduced run did not record matched send or GPU receive markers and does not establish that CK compilation caused the timeout.
It also did not reach the later full-layout boundary isolated in `762603`.

All fatal rank traces were retained before cancelling only `762626` at 11:48.
The step recorded `FAILED` (`1:0`), and rank zero wrote the failed runner result.
No rank completed an update, no `checks.json` fit verdict was written, and no cache was published or follow-up submitted.
Any separate reduced-depth follow-up should address the cold call-8 boundary explicitly rather than repeat this declaration or extend its deadlines.

## Transport review boundary

Worktree: `/ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-transport`.
Branch: `experiment/amd-chonk-fnuz-transport`.
Parent: `e242f3c3c1136aeae498529bfde5e06bda9a7cca`; no expert or integration dependency.
Suggested signed commit subject: `Add native gfx942 FNUZ MoE transport and qualification probes`.

| Reviewed path | Expected Git blob |
| --- | --- |
| `megatron/core/models/gpt/fine_grained_callables.py` | `9e393fd0080cd4c6e8e444112527bc7d1919b1f7` |
| `megatron/core/transformer/moe/fp8_jit.py` | `dd63fca9d3c86216e409bd98c8290f9ed3216e97` |
| `megatron/core/transformer/moe/fused_a2a.py` | `20ff8ab5f7f940973c75ee091c17bf758eaab3e4` |
| `megatron/core/transformer/moe/token_dispatcher.py` | `52b11896f18711b1532e6e0cef789aed38f049c1` |
| `tests/functional_tests/qualify_fnuz_dispatch.py` | `952fe723d1384c50d3015a49ab665987415e7276` |
| `tests/functional_tests/qualify_fnuz_dispatch.md` | `83bf46e154d463386f98a0ff29dc66f98e109734` |

The source selects native FNUZ, `amax/240` with exact FP32 division, scale `1` for all-zero blocks and a `1e-30` nonzero floor, unpacked FP32 scaling, and typed receive dequantization to BF16.
Both native conversion directions appear in the offline gfx942 compilation using the image's Triton.
CPU checks reproduced the two-width oracle and the EP16 zero-send/zero-receive fixtures; actual codec bytes and CXI traffic have not been measured.
Both probes pin verified user-signed commit `696ff8b7c711754f50b748b6ecaccd72fd00d3ba`.
Its signature matches the e242 signer, its sole parent is e242, and all six blobs match the table.
The probes use exact payload, scale, dequantization, routing and reverse-gradient checks, and require an all-rank pass plus confirmed CXI initialization for EP16.
Codec job `756606` on `nid002774` passed in 2:18, with the exact device result and clean teardown in run `20261120T140215Z-ec2c2f1d`.
The pass covers widths 1792 and 2048, strided inputs, empty batches, and the prescribed finite/NaN cases; no EP16 traffic was exercised.
EP16 job `756615` in run `20261120T141018Z-adc0eb0b` passed in 3:48 on `nid[002920,002922,002924,002926]`, disjoint from the expert run.
All 16 ranks emitted one passing result, each with three exact cases, and all reported native CXI initialization.
The width-1792 buffer allocations agreed on every rank: 64,974,592 NVL bytes and 63,350,400 RDMA bytes.
The evidence includes exact payload and FP32 scales, ragged and zero traffic, handle reuse, BF16 reverse paths, FP32 probability gradients, and clean all-rank teardown.
This is transport component acceptance only; the combined training schedule remains unqualified.

Read-only plans:

```bash
uv run --locked apertus plan experiments/chonk-native-fnuz-codec-i910bfcf936d3 \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:20:00 --cache-mode cold
uv run --locked apertus plan experiments/chonk-native-fnuz-ep16-i910bfcf936d3 \
  --submit --reservation cscs --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:25:00 --cache-mode cold
```

The child deadlines are 600 and 900 seconds respectively; EP16 must not start until the codec passes.
Evidence lives under the respective unique `artifacts/experiment-data/runs/<experiment>/<run-id>/output/`, including per-rank `FNUZ_TRANSPORT_RESULT` records.

The proposed exact submissions use the same host/sandbox boundary and launcher allowlist as the approved expert correction.
From the Apertus workspace:

```bash
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-codec-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-transport \
  --submit --reservation cscs --gpus-per-node 1 --cpus-per-task 48 \
  --cpu-bind cores --time 00:20:00 --cache-mode cold
uv run --locked /usr/bin/env -i \
  HOME=/ritom/scratch/cscs/boeschf/apertus-amd/artifacts/experiment-data/pyxis-home-fnuz.Cjg8lEOV \
  USER=boeschf LOGNAME=boeschf PATH=/usr/bin:/bin SCRATCH=/ritom/scratch/cscs/boeschf \
  /ritom/scratch/cscs/boeschf/apertus-amd/.venv/bin/apertus run \
  experiments/chonk-native-fnuz-ep16-i910bfcf936d3 \
  --checkout /ritom/scratch/cscs/boeschf/megatron/amd-chonk-fnuz-transport \
  --submit --reservation cscs --cpus-per-task 48 \
  --cpu-bind map_ldom:0,1,2,3 --time 00:25:00 --cache-mode cold
```
