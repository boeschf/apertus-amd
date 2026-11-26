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
The user-supplied Chonk patch directory is `/iopsstor/scratch/cscs/boeschf/amd-vs-gh200-moe-training/chonk/patches/`.
Audit patches individually against the selected source and installed vendor APIs; a clean apply-check does not verify helper destinations, numerical behavior or dependencies.
Prefer an existing supported backend selector over importing a custom autograd/backend bundle when it isolates the same hypothesis.
Keep compatibility fixes separate from performance arms, retain default-off source backports and pin the exact Megatron commit and source archive before execution.
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

## Active execution plan

The full KDA Chonk baseline and both profiles are complete; retained outcomes are in `artifacts/investigations/compilation-performance/FINAL-SUMMARY.md`.
Current execution outcomes, frozen evidence and scoped acceptance/blockers are indexed in [performance investigation status](../artifacts/investigations/performance-next/STATUS.md); completed subtracks do not waive the remaining numerical, checkpoint or production gates below.
The [next execution wave](#next-execution-wave-2026-11-19) updates the original roadmap using the completed component tests and Imanol patch audit.
Keep the newer Primus 26.8 stack used by image `cb5a384016a9`; do not add an older-Primus comparison or change vendor Torch for this phase.
Existing measurements remain valid evidence and do not need repeating because node screening was absent.
Keep source, image, resolved settings, node list, compiler/cache evidence, and numerical/checkpoint outcomes explicit for every new run.
Do not commit or push changes on the user's behalf.

### Parallel workstreams

Current priorities are GEMM/backend tuning, native UCCL EP correctness/performance, and AITER kernel validation/performance as three parallel tracks, by user direction on 2026-11-14.
For GEMM tuning, independently qualify each explicitly selected operation's numerical checks, shape/layout and backend consumer before tuning; a separately named forward-only component does not pass or waive the failed accumulated-wgrad/full-projection control.
For UCCL EP, the user cleared the existing numerical-difference blocker on2026-11-19 after the unchanged upstream EP suite passed with explicit CXI in754226.
Treat those observed differences as acceptable for continuing the performance investigation; retain the original failed comparisons without requiring contributor capture or tolerance calibration as prerequisites.
This is an explicit acceptance decision, not a proved BF16 rounding bound or a claim that the original comparisons passed; matched performance, finite training, checkpoint and lifecycle checks remain separate.
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

### Promotion of confirmed optimizations into images

By user direction, relevant patches, tuning artifacts and supported settings with demonstrated correctness and positive performance effects should eventually become reproducible image contents rather than ad-hoc run modifications.
Qualifying an API, kernel or baseline is not itself a positive performance result; failed, neutral and unmeasured optimization candidates are not promotion-ready.
Required compatibility fixes may accompany a confirmed winner as explicitly tested dependencies, without being credited as independent speedups.
Establish the effect with fixed-source/image/model/compiler/settings comparisons changing one optimization at a time, retaining numerical, memory, checkpoint and lifecycle gates appropriate to the change.
Keep each candidate's correctness, measured effect and image-promotion status separate in run artifacts.

Implement accepted image changes in the separate image-recipe repository, pinning upstream base/source revisions, patch bytes/order, tuning data, build inputs and supporting run evidence.
For tuning tables or JIT seeds, record the actual consuming API and effective selection, hardware/shape/layout scope, library/compiler versions and artifact hashes; capture only quiescent validated objects and preserve safe behavior outside the qualified scope.
Keep workload-specific settings and intended training arguments explicit in experiments unless broader image-default behavior has also been qualified.
Megatron backports must retain an explicit matching source/base-plus-patch identity even when their assets are shipped in an image; do not silently execute a different checkout or replace vendor Torch.
Keep Primus26.8 and the selected vendor Torch/TE/Triton versions unless a separate supported stack change is explicitly reviewed.

Build and publish a new immutable candidate through the normal image tools, independently of experiment execution.
Requalify imports, effective patch/tuning consumption, numerics, memory/lifecycle behavior and retained performance on that rebuilt image before promoting it as a default.
Packaging several individually confirmed winners later does not make their combined interaction or performance established; retain an unchanged image/control and qualify the combined candidate separately.

### Execution updates and placement

During execution, provide approximately five-minute progress updates grouped by these workstreams and report failures or blockers sooner.
Updates distinguish submitted/running work, measured GPU outcomes, CPU-only checks, image build/publication status and remaining dependencies.
Use disjoint exclusive allocations for independent GPU tests; do not borrow or disturb unrelated jobs.
The `cscs` reservation and general `mi300` may run concurrently on disjoint allocations, with at most 16 investigation nodes total across both pools.
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

## Next execution wave (2026-11-19)

### Objective and current decisions

Measure native UCCL in full KDA Chonk while independent teams prepare the next small GEMM and kernel changes from Imanol's patches.
Do not bulk-apply the 60-patch directory or restart the baseline campaign.
Use Megatron e242 and image 910 with Primus 26.8 and the unchanged vendor Torch/TE/Triton stack.
Keep the accepted routing fix in both arms of each new comparison.
The user has accepted the observed UCCL numerical differences after 754226 passed upstream intra-node, inter-node, low-latency and dual-buffer tests.
Contributor capture and tolerance calibration are no longer prerequisites for that performance track.
Retain original failed comparisons and their thresholds; do not relabel them as passing tests.
The output-gate candidate and existing TE grouped-Triton selector have valid negative or nonqualifying timing results and remain parked.
SSSGLU has useful numerical evidence but is not compiler-qualified.

### Parallel ownership and dependencies

Parallel work is enabled for the following independent responsibilities.
Each owner keeps its own proposal, tests and artifacts; the coordinator owns shared plan changes, source-runner changes and final launch approval.
This plan does not itself submit GPU jobs or grant unlimited allocations or retries.

| Owner / track | Work that can start now | Next concrete deliverable | Dependency before GPU execution |
| --- | --- | --- | --- |
| A: UCCL training | Prepare a matched all-to-all/UCCL pair from committed e242; verify EP16 integration and effective transport/config consumers. | Two explicit 16-node/64-GPU inputs, source/image/settings diff, acceptance policy and bounded timing schedule. | Frozen inputs, production EP16 buffer/config review and fresh exclusive placement; no dependency on optional contributor capture. |
| B: GEMM | Review custom Turbo API/storage and specify a narrow forward-only component; keep 0010 forward/dgrad as a secondary alternative. | One exact backend/shape/layout/storage contract, with no custom wgrad or implicit tuning activation. | CPU/source checks, then a separately approved small component allocation; no dependency on UCCL timing. |
| C1: SSSGLU | Diagnose weighted-reference compiler specialization without changing the activation patch initially. | Specialization ledger, source-supported cause and corrected qualification proposal. | Default compiler-budget qualification before timing; retain all declared numerical and fallback checks. |
| C2: Q/K norm | Prepare b3-0004 independently of C1 and the negative output gate. | Installed FLA training contract, parameter/state map and default-off numerical proposal. | Full shape/layout/output/input-gradient/parameter-gradient checks and parameter/checkpoint compatibility. |
| C3: KDA layout | Inventory remaining copies and design a K2 convolution-only candidate separately from G2 padding. | Baseline/candidate copy ledger, exact shape/cast contract and backward cases. | Source/CPU review, then separate native qualification; no dependence on an SSSGLU retry. |
| D: patched source selection | Use an isolated Megatron branch for separately reviewed patches and pin the full resulting commit. | Use the existing commit-only archive and runtime; retain patch provenance and keep the custom source runner prototype parked. | Review the exact delta and native component evidence before any patched full-model run; not a prerequisite for committed-source UCCL. |
| E: pipeline communication | Review the existing overlap consumer and isolate the even-PP communicator-split hypothesis. | Consumer/endpoint/lifetime map and a default-off PP4/VPP2 test proposal. | Separate schedule correctness and completion checks; no bundled EP/backend or wait-removal changes. |

Use parallel CPU preparation freely within these ownership boundaries.
Use `cscs` for ready component jobs and general `mi300` for full-model training, subject to the combined 16-node cap.
A 16-node training allocation leaves no capacity for a concurrent investigation GPU job; CPU work may continue in parallel.
Keep the two matched training arms sequential and never share benchmark GPUs, private caches or mutable source trees between tracks.
If nodes or review capacity are limited, prioritize UCCL training, then the first qualified GEMM or activation component.
The coordinator reports progress about every five minutes during execution and reports blockers sooner.

### A: native UCCL training and timing

1. Prepare both arms from the same e242 source and image 910, with TP1/PP4/VPP2/EP16, sequence 4096, micro-batch 1 and global batch 512.
   Select the accepted `moe_act` recompute policy for both arms, retaining `layernorm` and `linear_attn` recompute; validate the input against current source support and memory requirements before launch.
   This pair compares dispatchers, not the old moe versus moe_act policy or different images.
2. Set the control to ordinary all-to-all and the candidate to the installed native UCCL-backed flex path, with all required selectors explicit.
   Keep P2P policy, compiler settings, optimizer, offload, warmup, model geometry and logging fixed in both arms.
   Require `UCCL_EP_TRANSPORT=cxi`, `UCCL_SOCKET_IFNAME=hsn` and `UCCL_CXI_THREADING=safe` where the native path consumes them.
   Do not add `FI_CXI_ENABLE_WRITEDATA`; 754226 passed without it.
3. Review the production EP16 buffer capacities and dispatch/combine configuration independently of the earlier EP8 proxy.
   Do not copy the EP8 combine tuple into EP16 without the installed API and capacity checks.
   Verify the effective dispatcher, native library, local GPU selection and actual CXI initialization in run evidence.
4. Use the first candidate run as both an integration check and the first timing sample.
   Fail on nonfinite updates, wrong dispatch/import identity, lifecycle failure or a stated memory/time cap; the accepted historical numerical differences alone are not a failure trigger.
   Do not attach contributor observers or intrusive profiling to timed runs.
5. Target 60 updates per timing run and retain both preselected windows 11–26 and 45–60.
   A1/B1 completed; the user deferred B2/A2 confirmation, so do not submit those jobs for now.
   Record actual cache restoration and separate warm from cold evidence; dispatcher-specific JIT objects are not interchangeable by assumption.
   Set explicit wall limits and review rules before submission; never extend one slow arm or drop slow samples after seeing its timings.
6. Report update-time median/mean/spread, tokens/s/GPU, allocated memory, compilation, checkpoint time and finalization separately.
   Use a same-image matched control where needed; the historical best window is context, not a substitute for this control.
   No speedup is accepted from one favorable run or from a library microbenchmark.

### B: Imanol GEMM candidates

The existing TE grouped-Triton selector was the first low-risk test of Imanol's grouped-GEMM idea; 745752 showed it slower in every measured case.
This does not prove that every custom Turbo kernel is slower, but it removes that selector from immediate promotion.
The first design choice is a distinct custom Turbo implementation, not another run of the same TE flag.
Review `0002-turbo-grouped-gemm-experts` and extract only the backend-selection intent from `0004-turbo-gg-backend-select`.
Historical helpers target older private APIs; establish the installed Primus-Turbo 0.5.1.dev7 contract rather than assuming 0.2.0 compatibility.
Do not port the broad origami-exception monkeypatch or hide a failed requested backend behind a silent fallback.
Prove common storage, offsets and bounds for any stacked-weight view; pointer adjacency alone is not ownership proof.
If zero-copy storage cannot be proved, use an explicit copied-weight control and include that cost, or stop that design.
Keep grouped FC1/FC2, dense/latent projections and the vocabulary projection separate in the candidate ranking.
Use actual Chonk dimensions, ragged and empty expert groups, current strides and BF16 inputs with the correct output/gradient dtypes.
Start with a separately named forward-only component, not the complete 0002 custom training-autograd integration.
Keep a narrow forward/dgrad subset of `0010-t4-te-lmhead-wgrad` as a separate secondary candidate; identify the actual selected module because its class-wide route is not automatically LM-head-only.
Keep original wgrad, FP32 accumulation, DDP hooks and optimizer ordering unchanged in the first candidate.
Treat the TE workspace patch as compatibility only and apply it equally to both arms if it is required.
Before GPU tuning, identify the actual consumer for any solution table or kernel setting on this pinned image; do not reuse Torch 2.10/Triton 3.6 tables or assume Torch TunableOp controls TE's native path.
After component correctness, time both device events and synchronized completion, including required packing/copies and realistic expert imbalance.
Advance only a repeatable positive result that can affect elapsed training time; faster host return alone is insufficient.
The accumulated-wgrad/full-projection failures remain separate blockers for any future fused-wgrad candidate.
Prepare a bounded metadata observer at the existing host split boundary in parallel, without another device-to-host conversion, synchronization or tensor retention.
If separately approved for a suitable non-timing or already instrumented production run, record 16 local experts, zero-inclusive splits, FC1/FC2 strides, post-SSSGLU layout, stage/VPP/microbatch and recompute context.
Do not launch training just to collect those shapes or silently add the observer to timed UCCL arms; keep uncaptured cases labelled synthetic.

### C: Imanol activation, norm and KDA candidates

For 0007 SSSGLU, first determine why the weighted compiled reference exhausted the default `recompile_limit(8)` in 746139.
Use retained guard logs to separate intended shape/dtype variants from an avoidable specialization or harness-state problem.
The warning names the compiled baseline `weighted_sssglu` with weights expected BF16/actual FP32; it does not by itself prove a Triton-kernel defect.
Include the final `prob_bf16` fallback and the baseline/default-off/candidate module contexts in the ledger.
Preserve BF16 activations, FP32 probability-gradient semantics, both input-gradient halves, all declared cases and the default compiler policy.
A corrected proposal must prove each production-like compiled signature stays within budget; it must not merely hide warnings, reset caches, raise limits or omit failing coverage.
Auxiliary-worker isolation may be reviewed as a test protocol, but it cannot substitute for a representative mixed-call sequence within the default production compiler budget.
Only then repeat bounded native qualification, followed by copy-inclusive paired timing of plain and weighted D1920 cases.
The 376 passing numerical metrics are supporting evidence, not a reason to bypass this compiler check.
Require complete supervisor/rank closure in the replacement proposal; retained rank 0/1 closure was incomplete after 746139 fail-fast.

Prepare b3-0004-qknorm-fla independently while the SSSGLU investigation runs.
Verify the exact helper path and installed API; map Q28/KV7/head128 layouts, normalization epsilon, learned parameters, optimizer grouping, state-dict keys and all parameter/input gradients.
The historical helper replaces `q_layernorm`/`k_layernorm` with `fla.modules.RMSNorm` and copies parameter values; preserve original parameter identity and framework metadata where possible, rather than treating equal initial values as optimizer/checkpoint compatibility.
Use actual Q/K row populations 114688×128 and 28672×128, with output, dinput and 128-element dgamma checks under a separate norm-specific numerical policy.
Keep unsupported-input fallback and default-off selection; apply only the narrow candidate to an owned e242 export.
The b3-0005 output gate remains parked because 745621 did not establish a positive effect, especially for strided inputs.
Do not gate Q/K norm preparation on rerunning that negative candidate.

Keep b3-0002 direct AITER attention as a consumer-difference question first: the baseline already executes AITER attention kernels.
Do not count a different Python entry point or prebuilt cache seed as a new faster kernel.
Prefer a K2 convolution-only design over importing the G2 padding bundle.
Retain `norm_qk=False`, projection width 11036, parameters and checkpoint shapes; cover projected `[4096,1,11036]`, convolution weights `[10752,1,4]`, Q/K/V `[1,4096,3584]` and tail 284.
Check projected-input and weight gradients, missing-output/tail-only gradients, causal/document boundaries and original cast points, not just forward outputs.
Keep G2 padding as a separate later candidate, ranked against measured critical-path cost and exact stride requirements.
Padding 11036 to 11040 must preserve beta28 slicing, outputs, backward and checkpoint parameter mapping; it is not a model-width change or an automatic fix.
Do not combine KDA padding, recompute removal, tile changes and convolution fusion into one trial.
The b3-0003 internal KDA recompute option is a separate memory/performance alternative; it does not remove outer `linear_attn` checkpointing and needs installed-API and saved-state memory review.
Do not port k6 chunked-convolution private APIs in this wave.

### D: use pinned commits from an experiment branch

Prefer a local experiment branch based on the exact tested e242 commit over a new base-plus-patch runner mode.
Keep the original checkout and historical tooling intact; use a separate worktree for any explicitly approved branch changes, with one isolated commit per reviewed patch and compatibility fixes recorded separately.
Pin the full resulting commit in the experiment, not the movable branch name, and let the existing runner freeze its source archive and mount it read-only.
Retain the original patch bytes, branch-base commit, exact resulting delta and runtime import evidence.
For a source-kernel comparison, run both arms from the same pinned commit and change only its default-off selector.
Keep image 910 and its vendor packages fixed; no image rebuild, external merge, or push is needed for the experiment.
Do not include negative or unqualified candidates in a tested branch just to assemble the patch stack.

### E: pipeline communication and later optimizer work

Prepare a6-0002 even-PP communicator split separately from native UCCL.
Confirm that the selected schedule uses the unbatched `_p2p_ops` consumer; prove matching communicator/peer choices and rank-wide setting agreement before testing PP4/VPP2 fill, steady state and flush.
Retain send/receive completion, buffer ownership, gradients and failure teardown.
The existing warmup/flush-overlap implementation can be tested later as its own scheduling change; the historical a4-0004 CLI alias is not missing functionality.
Do not bundle communicator split with that overlap switch, lazy waits, sync removal or an EP backend replacement.

Keep Muon workspace reuse and k7 batched expert updates in a bounded CPU/source queue, not the first GPU wave.
Their review must cover reentrancy, asynchronous scratch ownership, exact update arithmetic, per-parameter state and checkpoint restore before allocation timing can count as a win.
Keep T6/T7/T8 dispatchers, FP8 transport, dispatch stashes, delayed/fused wgrad, side streams and global update-consensus removal out of this wave.
Existing communication warmup consumers must not be duplicated.

### Release and stop rules

Each GPU proposal requires focused CPU checks, explicit experiment inputs, immutable source/image/tooling identity, fresh placement review and bounded resources.
The user grants standing approval for in-scope runs within the combined 16-node cap; check existing and pending investigation requests before submission to prevent excess overlap.
Give each component proposal a stated case count, numerical/compiler requirements, timing schedule, success rule and maximum attempts before using GPUs.
New failures stop that candidate for review; independent ready tracks may continue without automatic retries or budget growth.
After a component win, test the real import/dispatch, recomputation and parameter/checkpoint behavior, then run a separate full-model comparison.
Promote only confirmed winners and their required compatibility fixes into the separate image-recipe repository through the normal build/publish tools.
Do not rebuild or publish a combined patch image while its only candidates are negative, compiler-unqualified or unmeasured.
Requalify a rebuilt image, then test interactions between accepted changes separately; independent gains cannot simply be added.

### Review inputs

The retained [parent patch audit](../artifacts/investigations/performance-next/restart-20261116/imanol-patch-audit/PARENT-REVIEW.md) and its GEMM, AITER/KDA and communication reports define patch dependencies and unsafe bundles.
The [isolated patch inventory](patches/README.md) records staged default-off candidates and compatibility fixes.
The current status supersedes the early audit's pending-test recommendations; the audit files remain historical evidence.
This wave does not reopen the accepted UCCL numerical decision, promote negative timings or erase compiler qualification failures.
