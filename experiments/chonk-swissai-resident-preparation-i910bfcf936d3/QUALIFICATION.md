# Swiss-AI Chonk: GPU qualification proposal

**Not authorized to execute.**
This document is a review proposal, not a job, a frozen launch envelope or a walltime approval.
The only checked-in experiment is the [blocked preparation](experiment.toml).
Before any later execution, freeze the exact comparison, source/image/validator/cache inputs, placement, environment, commands, and deadlines for review.
Slurm would execute outside the agent sandbox with normal host filesystem access.
Use separate, explicitly named freezes for each authorized diagnostic; do not edit prior inputs or promote a BF16 diagnostic into the FP8 reference.

## Order and stop conditions

| Gate | Proposed scope | Evidence required to proceed |
|---|---|---|
| 0. Inputs and reference | CPU-only: verify the published GH200 digest when available, parser/config mapping, precise launch and env; no runtime dependency install. | Accept exact argv and named differences. Confirm `TP1/PP4/VPP2/EP16/ETP1/CP1/DP16/EDP1`, 128 microbatches and 16,777,216 tokens/update. New hardware-dependent defaults must not be carried from CUDA by name. |
| 1. Compute and transport components | Start on one `cscs` reservation node (in `mi300`) for shape-faithful compute, KDA/CE/numerics and accumulation. Use a separate four-node EP16 component for CXI with the actual 8192 tokens/source rank, top-8, latent 1792 and four ranks/node; include BF16 forward/reverse control and FP8 payload-plus-scale cases only after its arithmetic path is defined. | Verify actual fwd/dgrad/wgrad kernel paths, precision and zero/ragged semantics; confirm UCCL route, P2P enabled behavior, returned probabilities/gradients, and all-rank cleanup. No generic old-stack, no broad bundle and no blind retry. |
| 2. Staged schedule and memory | 16 general `mi300` nodes/64 GPUs, not concurrent with the components. Use MBS2/GBS4096 and full geometry from the start for fit, with only a short reviewed horizon. First isolate BF16 resident operation and applicable overlap changes, then check the requested combined schedule; add EP overlap before delayed wgrad and do not bundle changes into a causal timing claim. | Two or more completed updates after initialization to exercise warmup, steady schedule, drain, optimizer state and a subsequent weight use. Prove actual backend and grad-ready order, numerical gates, no skips/reruns/nonfinite, no OOM or allocator pressure. Never shrink MBS to make this count as memory fit. |
| 3. Precision, fusion and graph | Test resident FP8 expert compute and FP8 transport as distinct changes; recheck the full combination after individual acceptance. Isolate accumulation fusion before testing the exact partial graph scope. No graph job until separately approved. | Prove refresh, delayed-wgrad and optimizer correctness with stable `main_grad` ownership; measure graph-only memory and replay effects. Test new router distributions across capture/replay, comparing state after each microbatch and optimizer update. Stop on any silent fallback, stale histogram/FP8 scales, or missing callback. |
| 4. Distributed checkpoint | Same model and placement geometry; separate continuous versus short-prefix/resume qualification with a new checker that reads only owned trusted checkpoint data. | First inventory (and retain manifest/digests), then DCP metadata/storage and model/optimizer tensor integrity, then real restore of optimizer, quantile/router state, scheduler, RNG, iteration and sample counters. Compare fixed updates, weights/gains and optimizer state versus uninterrupted execution. Validate FP8 derived-buffer refresh on restore; no `--no-load-optim` or fresh-state fallback. |
| 5. Throughput | Only after runtime, precision, fit and restart gates close or the user explicitly accepts a named deviation. Two unprofiled **warm cache** 60-update repetitions per configuration on matched nodes, same geometry, schedule and cache policy; change only the approved variable. | Report median/mean/spread and tokens/s/GPU for **11–26 and 45–60** (262144 tokens/GPU/update), and keep all samples if drift or compile activity persists. Document checkpoint, startup, cache hydration/publication and failure time separately. Do not count profiled updates or silently choose faster subsets. |

The 16 investigation-node cap applies across reservations and general `mi300` at all times.
No allocation reuse, scheduler interruption, extra run, or wall limit extension is implied by this proposal.
Begin with a local prelaunch review of the BF16 path and exact schedule, not with a parity benchmark.

## Numerical cases and observables

- Pin initial values and upstream gradients independent of GPU RNG differences; use both deterministic reference-derived fixtures and representative real split vectors collected later.
  Keep identical explicit BF16 model values, a reference quantization oracle, and separate native AMD format witnesses.
- Test `FC1:1792→3840`, weighted SSSGLU `3840→1920`, `FC2:1920→1792`, with 16 local experts.
  Cover all-empty, zero groups among live groups, one live or hot expert, uneven and skewed and alignment-edge token counts, and captured upper-tail routing.
  Include zero/near-zero and saturated values, clipping, scale floors and round boundaries, noncontiguous inputs where selected, and actual input/output destinations.
- Check all-element forward and dgrad outputs, expert wgrads, gradients of top-k routing probabilities, BF16 reverse payload, FP32 main-gradient accumulation across distinct microbatches, reduction-ready ordering, and matching effects of one or more complete layer-wise `md_decoupling` steps.
  Distinguish tolerances for approximation in the FP8 format from backend correctness against that format.
  Agree on reference oracles and acceptance tolerances before running; no post hoc relaxation or dropping failing cases.
- Verify actual mode/arguments, UCCL CXI/P2P, EP/P2P and grad-reduce dependencies, KDA per-head gate, conv1d and norm path, recompute call counts, and weight versions.
  Record `MAX_JOBS`, actual worker concurrency and affinity rather than treating environment labels as dispatch proof.

## Memory and walltime admission

Request all allocatable node memory (`--mem=0` in scheduler terms), not host staging, oversubscription, or a GPU memory cap.
Treat MI300A HBM as shared with host/node activity; collect per-rank device allocated/reserved peaks and free memory, node/NUMA and cgroup pressure and peaks, `/dev/shm`/cache pressure, compiler workers, collective buffers, graph pools, and transient optimizer/checkpoint peaks.
The staged validator's allocator counters and process RSS are only part of this admission.

For a scale check, each local MoE layer has `16 × 1792 × (3840 + 1920)` routed expert weight elements: **315 MiB BF16 weights + 630 MiB FP32 main grads** before optimizer, other parameters and runtime state.
The heaviest physical stages contain 12 MoE layers, about **11.1 GiB/rank** for those two routed slices alone; two byte-sized FP8 orientations, if required for all those weights, add about **3.7 GiB/rank** before scale/padding.
MBS2 doubles input tokens and makes the unsharded BF16 logits `[8192,200064]` about **3.05 GiB** on the final stage if materialized, before fusion and autocast buffers.
The 128 microbatches are a work count, not an activation-memory multiplier; for memory, determine the number of simultaneously live microbatches and stage-specific buffered work.
Inspect storage quotas and per-node footprint for final checkpoint, integrity inventory and restart copies; routed BF16 weights alone are about **192 GiB across the checkpoint**, excluding optimizer and all other tensors.

Use a fresh fit/timing screen before freezing a 60-update wall limit.
For review, reserve separate allowances for startup/compile/hydration, the measured upper per-update duration times the remaining work, explicit synchronous checkpoint timing, validation/cache publication and teardown.
The old B1 ≈22 s/update, multiplied by eight tokens/update as a rough illustration, would already be ≈3 hours for 60 updates **before** fixed overhead; this is not a prediction, lower bound or time request.
Check the site's partition/QoS wall limit and available storage with read-only commands when preparing a real request.
Select headroom before submission from limiting-rank measurements and retain any terminal or incomplete run without retry.

## Separate verdicts

Retain distinct records for configuration acceptance, runtime dispatch, finite accepted updates including skips/reruns, numerical agreement, memory feasibility, checkpoint file inventory, model-tensor integrity, optimizer-state integrity, successful restart, cache reuse and steady-state throughput.
The new CPU-tested checker inventories `torch_dist` files and reads all step metrics; it does **not** implement native checkpoint tensor integrity or a Chonk restore procedure.
For the latter, prepare a separate DCP-aware validation freeze and artifact manifest before authorization; do not reuse the dense restart checker or load a sealed run in place.
