# Apertus performance measurements

## Measurement rules

Keep model geometry, optimizer, sequence length, global batch, parallel layout, source, image, and communication settings fixed within each comparison.
Record every intentional change in an explicit experiment input.
Change one runtime optimization at a time, including its required source backport, and compare it against the preceding accepted configuration.
The agreed UCCL upgrade and AITER image-cache seed share one candidate image build; their selected execution paths and component checks remain explicit.
Build and make candidate images available only through this repository's `scripts/build.py` and `scripts/publish.py` tools (or their thin `build-publish.py` wrapper), retaining provenance and immutable published identities.
Measure CPU affinity and allocated nodes because memory placement and host state can affect these MI300A runs.
Retain restored startup-only affinity excursions as advisory observations; escalate persistent or unrestored bindings, changed thread defaults, or excursions during measured updates rather than automatically repeating completed runs.
Use at least two unprofiled repetitions per configuration before accepting small speedups.
Report seconds per update and tokens per second per GPU using global batch times sequence length divided by world size and update duration.
Use the same measurement window in both runs: initially updates 11–26, retaining all earlier timings separately.
If compiler activity or timing drift persists into that window, retain all samples and review a symmetric duration/window extension before subsequent paired runs rather than dropping slow samples or automatically extending one arm.
Report median, mean, and spread; keep startup, checkpoint saving, and cache publication times separate.
Keep cold and warm-cache results separate, and record which cache classes were actually restored.
Do not use profiled update times as throughput measurements.
Collect short CPU/GPU traces on representative ranks from every pipeline stage once compilation has settled.
Use traces to distinguish optimizer cost, expert GEMMs, routing/permutation, attention, communication waits, and CPU launch gaps.
Kernel-duration sums alone do not measure elapsed time when streams overlap.
Record finite updates, numerical comparison, checkpoint integrity, and cache reuse as separate outcomes.

## Mid-4B

Start from `1072583a9632` and image `cb5a384016a9`, with expandable segments enabled, compilation enabled, and P2P disabled.
Finish the cold/warm compilation diagnostic before changing compute paths.
The historical allocator/compiler runs are useful evidence, but their cache/checkpoint finalization outcomes and different nodes prevent treating them as complete repeatability measurements.

Imanol's reference root is `/iopsstor/scratch/cscs/schlag/beverin`.
The Mid-4B launcher is `bench2/b2/run22.sh`, with defaults and source snapshotting in `bench2/b2/sub.sh`.
Use the actual run's `logs/`, `env/`, and snapshot `HEAD` and `worktree.diff` rather than assuming today's launcher or branch matches an older run.
Identify the exact image/compute stack and tuning artifacts before making an apples-to-apples claim.
Reference results and extracted evidence belong in `artifacts/investigations/compilation-performance/`, not in this plan.

Initial differences to measure include gradient-reduction overlap and gradient-accumulation fusion, before stacking custom kernels.
Subsequent candidates include Turbo grouped GEMM and fused weight gradients, SSSGLU kernels, tuned GEMM tiles, routing/CE/norm fusions, attention/KDA kernels, expert communication, and side-stream scheduling.
Dependencies between patches must be recorded; an environment variable with no consumer in the selected source is not an enabled optimization.
Keep compiler policy fixed while testing each optimization, then revisit compilation if a custom kernel replaces a compiled path.
FP8 dispatch, changed recomputation, and alternate attention kernels need numerical checks appropriate to the change.
Historical tuning tables from Torch 2.10/Triton 3.6 must not be assumed valid on our Torch 2.14/Triton 3.8 image.

## KDA Chonk: primary target

Treat trained KDA Chonk as the primary workload, distinct from both Mid-4B and historical SWA Chonk.
KDA Chonk has 42 layers with three leading dense layers, hidden size 3584, 28 attention heads, 7 query groups, head dimension 128, 256 experts/top-8, expert/shared FFN 1920, and latent size 1792.
Its dense FFN is 17280, and its 31 KDA / 11 global attention layers use NoPE with a final global layer.
The production size file selects sequence length 4096 and PP4/VPP2/EP16; batch, offload, recomputation, and optimizer settings must be explicit in each benchmark.
The cloned reference repository is `/iopsstor/scratch/cscs/boeschf/amd-vs-gh200-moe-training`.
Its `bench2/APERTUS2_LADDER.md` and `chonk/ckpt_inspect_59000.txt` distinguish KDA from SWA; its `experiments/` directory is superseded orchestration.
Use primary size/recipe files and checkpoint metadata to resolve model settings, recording discrepancies between executable assignments and stale comments.
The [historical SWA baseline](moe-chonk-swa-m1072583a9632-icb5a384016a9/experiment.toml) uses job 677561's logged workload and is secondary evidence.
The clone's `lab2/BEST.md` reports a later 13.44-second SWA median; neither that result nor 677561 measures KDA Chonk.
Use the reduced model only for targeted kernel or correctness checks, and label it explicitly as a proxy.
Establish full-model memory feasibility and a repeatable unoptimized baseline before carrying over Mid-4B wins.
Prioritize optimizations that benefit both workloads, but measure Chonk independently before accepting them for Chonk.
Keep checkpoint loading and real-data runs separate from mock-data throughput comparisons.
For the extended KDA baseline, run two 60-update warm-cache repetitions and report both updates 11–26 and the preselected 45–60 window.
The extension follows timing drift in the initial 26-update run; apply the same duration and windows to subsequent optimization comparisons.

### Architecture and kernel constraints

For source `1072583a9632`, unchanged KDA Chonk requires TP1 on both AMD and NVIDIA.
The 28 query/KDA heads and 128-wide KDA gate bottleneck jointly permit TP1/2/4, but the 7 global-attention KV groups reject TP2/4 under the current divisibility checks.
This is an architecture/implementation intersection, not a general prohibition on tensor parallelism for KDA.
Keep PP/VPP/EP as the baseline scaling axes; evaluate CP only against its own implementation, memory, and numerical requirements.

Hidden 3584, latent 1792, expert FFN 1920, and dense FFN 17280 are all multiples of 128.
Their non-power-of-two sizes do not by themselves establish an AMD disadvantage; profile actual GEMM tiles, variable expert token counts, and forward/dgrad/fp32-wgrad solution selection.
The low-rank KDA input projection has width `3*3584 + 128 + 128 + 28 = 11036`, giving a BF16 row stride of 22072 bytes, which is not 16-byte aligned.
Measure this projection and its strided Q/K/V convolution inputs explicitly; internal beta-slot padding to 32 would give width 11040 and is a candidate only with correct slicing, backward, and checkpoint mapping.
KDA Chonk's head dimension 128 avoids the historical SWA Chonk head-dimension-112 attention-forward limitation.

Separate kernel availability from kernel performance: current Mid-4B traces contain HIP causal-convolution kernels, FLA KDA recurrence kernels, and AITER attention kernels.
Imanol's older measurements found slower AMD convolution and recurrence kernels but a faster tuned whole KDA layer at Mid-4B shapes; neither finding establishes current Chonk performance.
Profile Chonk's short-convolution backward, state recurrence, global attention, grouped experts, and large vocabulary projection before prioritizing a backend replacement.
Validate alternate attention/KDA kernels against forward and gradient references at the actual shape, layout, and gate parameterization.
Benchmark exact-shape GEMM tuning and semantics-preserving layout changes individually before considering changes to hidden size or head counts.
Any architecture-changing comparison needs its own model definition and must not be presented as a speedup of the unchanged trained model.
Retained source constraints and historical kernel evidence are in `artifacts/investigations/compilation-performance/KDA-ARCHITECTURE.md`.

## Active execution plan (2026-11-12)

The full KDA Chonk baseline and both profiles are complete; retained outcomes are in `artifacts/investigations/compilation-performance/FINAL-SUMMARY.md`.
Current execution outcomes, frozen evidence and scoped acceptance/blockers are indexed in [performance investigation status](../artifacts/investigations/performance-next/STATUS.md); completed subtracks do not waive the remaining numerical, checkpoint or production gates below.
Keep the newer Primus 26.8 stack used by image `cb5a384016a9`; do not add an older-Primus comparison or change vendor Torch for this phase.
Existing measurements remain valid evidence and do not need repeating because node screening was absent.
Keep source, image, resolved settings, node list, compiler/cache evidence, and numerical/checkpoint outcomes explicit for every new run.
Do not commit or push changes on the user's behalf.

### Parallel workstreams

Current priorities are GEMM/backend tuning, native UCCL EP correctness/performance, and AITER kernel validation/performance as three parallel tracks, by user direction on 2026-11-14.
For GEMM tuning, independently qualify each explicitly selected operation's numerical checks, shape/layout and backend consumer before tuning; a separately named forward-only component does not pass or waive the failed accumulated-wgrad/full-projection control.
For UCCL EP, resolve the native output/gradient differences before accepting throughput results.
For AITER, establish which kernels the baseline already consumes and a concrete candidate delta, then validate forward/backward at actual Chonk shapes before performance claims; the existing small attention check is not actual-shape acceptance.
Keep grouped-GEMM tuning under the GEMM track to avoid duplicate or bundled experiments, and retain the existing published image/vendor stack unless a separately reviewed change requires rebuilding.
Defer accumulation-fusion and collector work behind these three tracks; do not schedule another full-model fusion attempt without a separate decision and usable attribution.

| Track | Image rebuild | Work | Dependencies and acceptance |
| --- | --- | --- | --- |
| Finalization tooling | No | Correct bounded check/publication coordination and peer-failure handling; test slow publication and cleanup failure paths | CPU lifecycle tests, then real-run finalization; preserve immutable caches and evidence |
| Combined UCCL + AITER image | Yes, one shared candidate | Pin current upstream UCCL, discard superseded patches, retain justified fixes, and bake compatible AITER JIT seed modules using the vLLM pattern | Keep newer Primus base; component/import, dispatch/combine/gradient/lifetime and seed checks before full-model claims |
| Image-seed hydration | No, consumes candidate image | Copy compatible baked AITER objects into private rank caches before merging compatible published objects | Record seed provenance and reuse; do not copy binaries blindly from a different vLLM image |
| Accumulation fusion | No if selected backend supports it | Enable only gradient-accumulation fusion; inspect AccumulateGrad stream mismatch and actual expert/dense/LM-head paths | Forward and accumulated-gradient checks, memory and repeated unprofiled timing; standalone-add removal alone is insufficient |
| Narrower recomputation | No | Replace only `moe` with `moe_act`, keeping `layernorm linear_attn` | Validate outputs/gradients, increased retained activation memory, and repeated timing |
| PR #73 P2P restoration | Initially no | Test the complete relevant runtime settings with RCCL P2P explicitly enabled | Goal: correct stable full-model training without needing `NCCL_P2P_DISABLE=1`, including memory and teardown |
| GEMM/backend tuning | Usually no | Collect exact forward/dgrad/wgrad shapes and strides; tune hipBLASLt, grouped experts and applicable AITER kernels | Separate JIT reuse from faster kernel selection; rebuild only for demonstrated backend/tool requirements |
| AITER attention and non-GEMM kernels | Not initially | Establish installed TE/flash-attention/AITER consumers and a distinct candidate beyond baseline kernels; validate actual Chonk attention and any supported norm/pointwise/KDA replacement | Preserve forward/backward numerical gates; resolve full-shape GQA reference and BF16-control equivalence before implementation; static source inventory does not establish runtime dispatch |
| Reference/log audit | No | Retain NVIDIA reference configurations; classify stream/communicator, fallback, precision and logging warnings | Historical numbers are targets, not a matched current-Chonk comparison |
| Non-blocking node survey | No | Read buddyinfo, page size, NUMA memory, uptime and scheduler availability periodically | Timestamp/rank by worst GPU-local NUMA domain; informational only, not a submission gate |

Image preparation/builds, tooling/CPU tests, current-image GPU experiments, and the reference/node audits can proceed concurrently.
For UCCL, review the intervening commits from the old pin to the candidate pin, not just patch applicability or current-source similarity.
Classify each retained correction as essential, redundant, potentially harmful or unproven, with upstream commit/PR evidence and required compiler/runtime checks.
Distinguish requirements of our selected full-node execution path from unused compatibility fixes, and do not select the candidate for training acceptance until this review and relevant checks pass.
Keep running build contexts frozen; corrections require a separately retained build attempt through the same build/publish tools.
During execution, provide approximately five-minute progress updates grouped by these workstreams and report failures or blockers sooner.
Updates distinguish submitted/running work, measured GPU outcomes, CPU-only checks, image build/publication status and remaining dependencies.
Run independent GPU tests only in disjoint authorized allocations under reservation `cscs`; do not borrow or disturb unrelated interactive jobs.
Reuse a node set for paired small-gain measurements when practical, without delaying work for ideal nodes.
Do not run intrusive allocation-based memory probes or unrelated GPU tuning on benchmark nodes concurrently with training.
Refresh lightweight node observations before runs and periodically when scheduler access permits, but do not require a clean-node threshold or retroactive reruns.
Buddyinfo is a fragmentation indicator, not proof of GPU huge-page allocation capacity or the cause of the observed approximately 2.7% variation.

### PR #73 acceptance and test order

The objective is removal of the P2P-disable workaround, not merely checking that environment variables can be exported.
Audit the settings against pinned RCCL/aws-ofi/libfabric implementations, then test the complete relevant set with `NCCL_P2P_DISABLE=0` to override any inherited disable.
The candidate set from <https://github.com/eth-cscs/alps-extended-images/pull/73> is:

```text
OFI_NCCL_PROTOCOL=SENDRECV
NCCL_CROSS_NIC=0
NCCL_PXN_DISABLE=1
NCCL_NET_GDR_LEVEL=PHB
NCCL_NET_FORCE_FLUSH=1
HSA_ENABLE_IPC_MODE_LEGACY=1
NCCL_P2P_LEVEL=SYS
OMPI_MCA_memory=^patcher
```

Preserve existing explicit CXI settings and disable both host network-injection hooks in every EDF.
First check intra-/inter-node collective correctness and repeated communication/teardown, then full KDA Chonk on 16 nodes with progress and memory telemetry.
Successful initialization alone does not resolve the earlier P2P-enabled memory-pressure stall.
Compare against retained P2P-disabled measurements, adding a matched control only as required for a specific comparison rather than rerunning the whole baseline campaign.
Use individual flag ablations to diagnose a failure or simplify a working configuration, not as a prerequisite to trying the complete set.
Keep P2P disabled in the independent fusion/recomputation comparisons until the restoration experiment passes.
Promote validated settings into image defaults later; unsupported settings require a targeted stack change, not a silent no-op success claim.
UCCL's direct CXI path must be validated independently rather than assuming it consumes aws-ofi/RCCL settings.

### Follow-on integration

After independent trials, combine accepted fusion, recomputation, tuning and network changes with explicit inputs.
Test the new UCCL dispatcher against ordinary all-to-all separately from simply installing the upgraded library.
Investigate pipeline stage balance, supported communication overlap and the source-derived projection-padding hypothesis after attribution and numerical checks.
Retain full-model checkpoint inventory, tensor-integrity/restart validation, production-checkpoint compatibility and forward/gradient equivalence as distinct goals.
Finite mock-data training and successful cache reuse do not establish convergence, restart equivalence or production readiness.
