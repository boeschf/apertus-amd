# Swiss-AI Chonk on Beverin: alignment and remaining gaps

**Review only; the runner rejects both batch submission and execution in an existing allocation.**
The single [experiment](experiment.toml) implements the fixed model and training inputs from the brief and represents the source-expressible overlap settings.
It still selects resident **BF16** experts and BF16 dispatch, with accumulation fusion and explicit graph capture off.
It is not reference-equivalent, an accepted BF16 substitute, or a launch candidate yet.
The AITER tuning task is parked.
The [resident FP8 follow-up](RESIDENT-FP8.md) resolves the installed TE/HipKittens candidate for all six expert GEMMs and the separate E4M3FN wire-codec design; neither is implemented or qualified.

## Identities and exact inputs

- GH200: Megatron `c56351143a8e15899ef3d7d7f97867772f8e0e5c`, checkout `megatron/baseline`, NGC26.06-based Apertus-2 `megatron-cuda` target; published digest pending.
- Beverin: Megatron `e242f3c3c1136aeae498529bfde5e06bda9a7cca`, available in `megatron/comm-warmup`; image `910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490` in the site's `apertus-amd-images` store.
- The exact two-commit Git interval contains `1072583a9` (optional communicator warmup in `initialize.py` and `beverin_comm_warmup.py`) and `e242f3c3c` (histogram route selection at ties is consistent in forward/recompute; statistics accumulation remains separately gated).
- The local `pretrain/apertus-2/pretraining/chonk/chonk.sh` was read, not invoked (`sha256:c307314d5463dc981479f0d8ab19d1b3e9eab4ba0d42cfaf1df038fa81eb4f1e`).
  The brief, not that production launch script, governs the intentional benchmark deviations: mock data, no Goldfish, fixed geometry, and 60 iterations.
  Do not copy its runtime dependency install, CUDA cache archive, CUDA launch defaults, or production checkpoint logic.

The [TOML](experiment.toml) contains every workload argv and explicit environment entry; [site configuration](../../config.toml) owns Slurm, storage paths, PMIx and topology.
The following command prints the exact local plan and validator digest without running either Slurm or Megatron (`0` is only a planning placeholder):

```sh
uv run --locked apertus plan experiments/chonk-swissai-resident-preparation-i910bfcf936d3 --job-id 0 --cache-mode warm
```

`resources.time` in that plan is an inherited site placeholder, **not** the approved wall limit.
The selected resources are 16 exclusive `mi300` nodes on `csstaff`, 4 ranks/GPUs and 192 CPUs per node, effective target 48 logical CPUs/rank, `map_ldom:0,1,2,3`, `block:block`, `pmix`, network `disable_rdzv_get`, uenv passthrough `ignore`, and `--mem=0G`.
There is no reservation or dependency in this plan.
The runner retains Slurm's allocated GPU visibility and local-rank mapping; it builds dataset helpers with `make -C megatron/core/datasets`, uses the committed source archive, and verifies imports.
EDF keeps `entrypoint=true`, `HWLOC_COMPONENTS=-gl`, both network injection hooks off, and `com.hooks.netstack.source=host` to avoid mounting an unused foreign netstack.

The full explicit environment is in the TOML and site config, with `MAX_JOBS=8` as a preparation choice, not an assumption about effective compile concurrency.
Check the real Ninja `-j` and Torch/Inductor worker parallelism separately; `MAX_JOBS` is not a universal cap.
The image's `BASH_ENV` sources `alps-runtime.env`; for example, it defaults to `FI_PROVIDER=cxi`, `FI_CXI_RDZV_PROTO=alt_read`, `FI_CXI_RX_MATCH_MODE=hybrid`, and the `AWS Libfabric` NCCL network.
These remain inherited and the runner retains the effective `FI_*`, RCCL/NCCL, visibility and affinity records per rank; resolve conflicting host environment at launch review rather than assuming those unset-only defaults won.
Do not load any CUDA allocator or launcher defaults, Inductor policy, tuning table, Q/K or SSSGLU candidate override.

The CUDA UCCL dispatch/combine overrides and timeout are omitted.
Image 910 `deep_ep/buffer.py` binds native UCCL through its dispatch/combine config selectors, whose [EP16 non-EFA defaults](../../artifacts/investigations/performance-next/next-wave-20261119/uccl/README.md) are `20,36,288,20,128` and `20,4,288,12,128` (effective receive counts 140/132).
The AMD path is not the CUDA 24-SM resource plan; neither the larger CUDA chunks nor the old EP8 combine override is admitted.
Prove resolved EP16 hints, allocation size and native transport again at MBS2.

## Delta from the previous B1 input

| Area | Previous Beverin | New preparation |
|---|---|---|
| Model | Dense FFN 17280; KDA `A_log` per channel; gate projection without legacy bias; initial std `0.016703827619526525` | Dense FFN **8960**; `A_log` per head (`[28]` instead of `[28,128]` at TP1/CP1); legacy KDA gate-out bias on, other linear biases still off; std `0.0167038`. These affect state layout; no previous checkpoint may load. |
| Batch and pipeline | MBS1, GBS512, 32 microbatches/update, 2,097,152 tokens/update | MBS2, GBS4096, **128** microbatches and **16,777,216** tokens/update; same exact PP4/VPP2 layout, TP1/CP1/ETP1/EP16, DP16/EDP1 and sequence 4096. |
| Schedule and optimizer | `layernorm linear_attn moe_act`; P2P overlap off; no delayed wgrad or EP overlap | `layernorm linear_attn qkv shared_experts`; native P2P overlap default, EP overlap, delayed wgrad and grad-reduce overlap requested. Layer-wise `md_decoupling`, Muon, gains and other fixed optimizer settings unchanged; new telemetry requested. |
| Routing | Quantile histogram, aux coeff 0, implicit 1000 bins | Coeff `0.001`, histogram EMA `0` and 1000 bins pinned, per-layer logging on. `router.py:587–607,1027–1030,1081–1090`: quantile-only does not select a named `aux_loss`, `seq_aux_loss` or `global_aux_loss`, so this coefficient does **not** add a penalty on the selected source. |
| Loss, reruns and data | Mock data; built-in check at default; no fused CE or explicit norm-check request | Single mock-data iterator, fused native vocab-parallel CE, reference NaN-check disable and grad-norm threshold 5.0. Independent finite/skip/rerun gate remains active in the proposed validator. |
| Runtime | P2P disabled with previous transport and compiler telemetry | P2P enabled with the requested AMD network settings; explicit KDA switches; 18 OMP/NumExpr threads per rank; the same pinned stack, communicator warmup, allocator and passive compiler policy. |
| LR and checkpoint | 800/476832 iteration horizons; `torch` | 100/59604 iterations (409600/244137984 samples); `torch_dist`, synchronous, final-only iteration 60, fresh initialization and runner-owned save path. |

The model keeps 42 layers (31 KDA, 11 global), 39 MoE layers, 256 experts/top-8, and unchanged projection/latent widths outside those listed above.
The pinned `NullTokenizer` adds one EOD token, so `tokenizer.vocab_size` is `200063 + 1 = 200064`.
It is already divisible by `128 × TP` at TP1, so the padded vocabulary is **200064**.
The proposed validator also checks the actual runtime `padded_vocab_size`; the arithmetic and CPU plan are not runtime observation.
`cross_entropy_fusion_impl` defaults to `native` (`model_parallel_config.py:238`); the GPU check must cover fused loss/gradients at vocab 200064 and MBS2, not merely note the flag.

## Capability matrix

Source locations refer to e242 unless marked image 910.
“Staged” is configuration and source evidence, not selected-runtime evidence.

| Capability | Selected consumer / evidence | Unclosed requirement |
|---|---|---|
| Resident FP8 expert forward | **Unavailable as a configured equivalent.** `transformer_engine_spec_provider.py:75` picks `TEGroupedMLP` when offloading is off. The reference's `OffloadingExpertsMLP` FP8 path calls DeepGEMM through `fp8_utils.py:177–253`, including when its debug resident branch invokes `experts_fp8_util.py`. | Need a resident grouped FC1/FC2 training consumer with 1×128 activation and 128×128 weight scaling, same SSSGLU and probability weighting, original BF16 parameters, and refresh at the same optimizer boundary. Do not promote debug or enable generic FP8 to reach this. |
| Resident FP8 dgrad | **No equivalent selected.** The reference uses M-grouped calls with separately quantized/transposed operands; image 910 TE ships `Float8BlockQuantizer`, advertises block-scaled FP8 on gfx94x (`quantization.py:202–214`) and exposes grouped GEMM. AITER `moe_gemm_a8w8_blockscale` accepts scale tensors and routing metadata but is a forward operator, not the training backward contract. | Map actual TE grouped dgrad scale/layout and runtime solutions for both expert projections, and prove dX and dprob. The gfx942 AITER dtype table uses **E4M3FNUZ**, not E4M3FN; native formatting, `/448` reference range, zeros/subnormals, rounding and clipping require explicit comparison. Do not reinterpret bits. |
| Resident FP8 wgrad | **No equivalent selected.** Reference `experts_offloading_fp8_util.py:1148–1244` builds K-grouped reductions over per-expert token counts and accumulates into FP32 main gradients. Image TE `grouped_linear.py:763–903,1217–1218` supplies training wgrad/output buffers and deferred callbacks, not proof of the exact grouped FP8 K-reduction, layout or scaling. | Need one proven per-expert reduction into the same FP32 destination across 128 microbatches, including zeros and ragged K, first-write and accumulation semantics, DDP-ready hooks, and one matching layer-wise optimizer step. |
| FP8 forward dispatch / BF16 reverse | **Off in this plan; transport API present, arithmetic unqualified.** `fine_grained_callables.py:800–893` quantizes with `per_token_cast_to_fp8(use_ue8m0=False)`, receives then **dequantizes to BF16 before expert processing**, and uses BF16 reverse dispatch with probability gradients; it does not hand the original transport bytes straight to expert GEMM. `fused_a2a.py:282–365` calls `Buffer.dispatch((x_fp8, x_sf),...)`/`combine(...)`; image `deep_ep/buffer.py:1194–1240,1283–1470` exposes tuple/FP32 scale transport. The Megatron caster uses `tl.float8e4nv` and `/448` (`fp8_jit.py:156–177`), not an attested gfx942 implementation. | Prove or supply native E4M3FN conversion/dequantization and delivered bytes/scales, probability gradients and BF16 reverse semantics at EP16/MBS2, under the combined schedule. Can test transport independently with known payloads; enabling it with BF16 experts would still not qualify FP8 expert arithmetic. |
| EP overlap and delayed wgrad | **Staged, unqualified.** `transformer_config.py:2935–3003` requires EP>1, compatible dispatcher, VPP for PP>1, and no whole-MoE recompute. `schedules.py:1351` selects combined interleaved 1F1B; ordinary TE grouped wgrad is queued and drained by `GroupedLinear.backward_dw()` in image 910. Layer-wise `md_decoupling` permits grad-reduce overlap (`arguments.py:1612–1629,1747–1785`). | Prove EP stream order, buffer lifetimes, wgrad queue drains and all dense/shared/expert grad-ready/reduce events before optimizer use. TE version gates are not runtime support or proof of actual overlap. |
| Accumulation fusion | **Explicitly off.** The TE/Apex paths have different consumers: TE uses `main_grad` in grouped/dense linear; `tensor_parallel/layers.py:48,564–568,912` uses the Apex-compatible `fused_weight_gradient_mlp_cuda` entry where selected. Earlier accumulated-BF16/full-projection failures and the AccumulateGrad stream mismatch investigation are not cleared. | Prove consumers, accumulation destinations and lifetime for KDA, expert FCs, shared/dense layers and LM head with delayed wgrad and overlapped reduction. No absent-flag parity claim; re-enable only after comparison with the unfused path. |
| Pipeline overlap | **Staged via native default.** No `--no-overlap-p2p-communication`; `arguments.py:1921` derives `batch_p2p_comm=False` when overlap is on. PP4/VPP2 selects the interleaved P2P schedule; prior B1 with P2P disabled and batched is not this test. | Verify RCCL IPC/P2P peer selection, disjoint local binding, waits/tag/order across warmup/steady/drain, last-stage loss and rank teardown with EP overlap. |
| Requested recomputation | **Staged.** `kimi_delta_attention.py:1186–1260` handles KDA QKV/core recompute; `attention.py:1417,1485` handles dense attention QKV; `transformer_layer.py` handles layernorm; `moe_layer.py:238–240,467,706–708` handles shared experts. Neither `moe` nor `moe_act` is included. | Prove selected paths, expected call counts, no duplicate routing-state updates or delayed-wgrad closures, matching gradients, and memory at MBS2/VPP2. For `KDA_CONV1D_CUDA=1` image 910 ships `causal_conv1d` autograd forward/backward, but the selector/import and actual HIP kernels, SiLU and cast points must be witnessed. |
| Router/preprocess graph capture | **Explicitly off; API present.** Image 910 `transformer_engine/pytorch/graph.py:402–404,1131` ships `make_graphed_callables` using the Torch graph API (still named `torch.cuda` on ROCm). Megatron `cuda_graphs.py:1651–1690,2125–2136,2292–2309` resolves partial scopes and possible delayed-wgrad graphs; `arguments.py:1754–1775` requires PP>1, sufficient TE version and accumulation fusion in the relevant combined graph case. | Qualify exactly `moe_router moe_preprocess` only, with dispatch outside the graph, changing histogram inputs and probabilities, no stale capture state, correct gradient-ready events, resource retention, replay benefit and coexistence with fixed recompute/EP schedule. Torch compile is not this graph mode. |
| Checkpoint save | **Staged.** `arguments.py:1618` permits layer-wise `md_decoupling` with `torch_dist`; `checkpointing.py:1014–1053` packs model, optimizer sharded state and LR scheduler; runner owns `--save`. | Check final synchronous file inventory and timing separately, then review model/storage/optimizer integrity. No CUDA checkpoint conversion, early saves, async saving or automatic resume. |
| Checkpoint load | **Not enabled in the benchmark.** Megatron's distributed load builds the sharded model/optimizer state (`checkpointing.py:1215–1254,1858–1915`) rather than using dense `torch` optimizer sidecars. | Separate continuous-vs-prefix/resume qualification at the same layout and same schedule; verify actual loaded optimizer and restored training state before claiming restart. |

### Proposed resident FP8 design, not a port

First review image-910 TE block-scaled quantizer and grouped training entry points (`general_grouped_gemm`, grouped and discrete output variants) against each of the six exact expert operations.
Validate whether they select a gfx942 implementation with the required input formats, backward scaling, no hidden BF16 fallback and FP32 wgrad output before choosing them.
Use AITER block-scaled forward only as an alternative primitive if a candidate can supply the separate dgrad/wgrad contracts and is compatible with SSSGLU; do not enable SwiGLU.
If gfx942 needs FNUZ for native GEMM, quantify the conversion/arithmetic difference and request acceptance before calling it equivalent; a BF16 implementation of a reference dequantization formula is a numerical oracle, not evidence of FP8 hardware arithmetic.

If viable, propose one isolated, default-off Megatron resident expert adapter branch on e242.
Keep BF16 parameters, FP32 main grads, activations and derived weight payloads resident and preserve expert identities, parameter/optimizer layout, not just output shape.
Separate transport quantization/dequantization from expert quantization, and enumerate FC1/FC2 fwd, dgrad, wgrad with their group metadata, cast points, BF16/FP32 destinations and callback ownership.
Match the reference's first-microbatch refresh of quantized weights (`experts_offloading_fp8_util.py:404–425,471–535,653–655`): invalidate derived weights on optimizer mutation or load, refresh before the corresponding first use, and retain the correct version for backward and replay.
Quantify any extra row/column FP8 replicas and graph pools.
No offload flag, debug code, generalized CUDA-port project or fallback inside an “FP8” arm is proposed.
If there is no compatible implementation of a required direction, return that concrete gap for review before writing an adapter.

### Cache and evidence

No new cache backend or vendor artifact is introduced.
Use the existing image/GPU-architecture/source/environment-qualified key, job-owned `/dev/shm` scratch, verified hydration, quiescent capture, required checks before publication, immutable generations and existing-wins whole-object merges.
Do not import CUDA artifacts; any future FP8 JIT adapter must add its backend, quantization, signature, ABI and config identity to compatibility review before capture.
Do not publish mutable tuning state or delete prior generations.
A future freeze must retain resolved runtime env, source/import evidence, cache receipts, launch/dispatch proof and validator outputs, not just a passed process status.
