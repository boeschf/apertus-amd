# Apertus performance measurements

## Measurement rules

Keep model geometry, optimizer, sequence length, global batch, parallel layout, source, image, and communication settings fixed within each comparison.
Record every intentional change in an explicit experiment input.
Change one optimization at a time, including its required source backport, and compare it against the preceding accepted configuration.
Measure CPU affinity and allocated nodes because memory placement and host state can affect these MI300A runs.
Use at least two unprofiled repetitions per configuration before accepting small speedups.
Report seconds per update and tokens per second per GPU using global batch times sequence length divided by world size and update duration.
Use the same measurement window in both runs: initially updates 11–26, retaining all earlier timings separately.
Extend the run if compiler activity or timing drift persists into that window rather than dropping slow samples after seeing results.
Report median, mean, and spread; keep startup, checkpoint saving, and cache publication times separate.
Keep cold and warm-cache results separate, and record which cache classes were actually restored.
Do not use profiled update times as throughput measurements.
Collect short CPU/GPU traces on representative ranks from both pipeline stages once compilation has settled.
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
