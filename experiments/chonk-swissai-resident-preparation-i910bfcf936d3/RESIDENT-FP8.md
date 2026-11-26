# Image 910 resident FP8: backend decision

**Propose one resident expert adapter using TE’s list-based block-scaled GEMM API and the installed gfx942 HipKittens kernels.**
All six operations have a source- and binary-supported native **E4M3FNUZ** candidate.
There is no installed native **E4M3FN** version of that path on gfx942, and stock `GroupedLinear` does not reproduce the Swiss-AI saved-activation and wgrad contract.
This is a design for review, not implemented, not GPU-qualified, and not a claim of equivalent precision.
If matching E4M3FN expert numerics is mandatory without any format deviation, **none of the six native operations is an exact match**; retain E4M3FN quantize/dequantize plus high-precision matmul as an oracle, not as native FP8 performance evidence.
The [preparation](experiment.toml) remains blocked; this review does not select an alternative training baseline.

## Exact implementation and routing

The image’s TE build identifies [`ROCm/TransformerEngine@130099ce48dc806398611de9960f8e0c486ad5f1`](https://github.com/ROCm/TransformerEngine/tree/130099ce48dc806398611de9960f8e0c486ad5f1) and `gfx942;gfx950`.
The installed `grouped_linear.py` and `fp8_padding.py` match that commit byte for byte.
The source chain below is corroborated by installed ELF symbols and error strings; it is not an observed HIP dispatch.

1. Call `transformer_engine.pytorch.cpp_extensions.gemm.general_grouped_gemm(A, B, out, ..., layout=..., use_split_accumulator=True)` with **lists** of TE `Float8BlockwiseQTensor` operands.
   Construct them with `transformer_engine.pytorch.tensor.float8_blockwise_tensor.Float8BlockQuantizer`, TE `DType.kFloat8E4M3`, explicit row/column usage, block dimension `1` or `2`, and `force_pow_2_scales=False`.
   Pass one output buffer per nonempty expert (`single_output=False`), `[None] * len(A)` for `quantization_params`, `out_dtype=out[0].dtype`, and `use_bias=False`.
   The native E4M3 type on gfx942 is FNUZ.
2. `gemm.py:594,685–703` takes the dtype from `out[0]` unless `D_dtype` is explicitly set and calls `transformer_engine_torch.te_general_grouped_gemm`.
   The C++ binding filters empty operand pairs and calls `nvte_multi_tensor_gemm` (`pytorch/csrc/extensions/gemm.cpp:501–669`).
3. Leave `NVTE_USE_CUTLASS_GROUPED_GEMM`, `NVTE_USE_HIPKITTENS_GROUPED_GEMM` and `NVTE_USE_CK_GROUPED_GEMM` unset.
   With those selectors off, `nvte_multi_tensor_gemm` uses `multi_stream_cublas_gemm`, which calls a ROCm `cublas_gemm` **per nonempty expert** and fences its compute streams back to the calling stream (`common/gemm/cublaslt_gemm.cu:998–1069,1112–1146`).
   Despite the function name, two block-scaled inputs go to `kittens_blockwise_fp8_gemm`, **before** the hipBLASLt path (`common/gemm/rocm_gemm.cu:2014–2076,2114–2124`).
4. The installed `libkittens_gemm.so` exports `BlockwiseGemmCdna3::run` and contains CDNA3 `micro_tk` instantiations for `__hip_fp8_e4m3_fnuz` pairs with both BF16 and float outputs.
   `cdna3/blockwise_fp8_gemm.cpp:404–455` handles 1D×2D and 1D×1D.
   The helper uses FP32 MFMA accumulators, multiplies block scales into the partials in FP32, and stores BF16 or FP32 as requested.
   Its optional beta epilogue reads the output dtype’s previous value; the proposed unfused lane below uses `beta=0`.

`GroupedTensor` and its discrete-in/out variants are **not** the ROCm route (`grouped_linear.py:105–125`, `pytorch/csrc/extensions/gemm.cpp:674ff`).
The optional TE Triton grouped selector explicitly excludes FP8 (`grouped_linear.py:449`).
The CK grouped FP8 route is not the selected block-scaled contract: its descriptors use a single scale per operand (`ck_grouped_gemm_fp8.cpp:275–298`) and its front end declines FP8 accumulation (`ck_grouped_gemm.cpp:55–60`).
Grouped MXFP8 is a different recipe and is not a substitute.
This design is a batched host interface to multiple kernels, not one native ragged grouped launch and not a speedup claim.

## Six operations

`m` is the number of valid rows for **one** of 16 local experts; `p=ceil(m/128)*128`, with `p=0` for empty experts.
Below, dimensions use row-major math and `p` includes only that expert’s explicit zero padding.
`X[p,1792]` is the permuted BF16 input after latent projection, `A[p,3840]` is FC1’s BF16 output, `S[p,1920]` is weighted SSSGLU, and `Y[p,1792]` is FC2’s BF16 output.
`dS = dY @ W2` and `dA` is produced by SSSGLU backward together with the routing-probability gradient.
Both FCs use no linear bias.
All rows use the exact callable and native implementation above, **FNUZ inputs**, FP32 internal partials, and `use_split_accumulator=True`.
`grad=True` is used for dgrad/wgrad, with no bias or DGELU epilogue.

| Operation | TE list arguments, layout, and row-major result | Block scales passed to HipKittens (FP32 dequant multipliers) | Output / schedule |
|---|---|---|---|
| FC1 fwd | `A=W1[3840,1792]`, `B=X[p,1792]`, `TN`: `X @ W1.T` | Rowwise `X [14,p]` (1×128); rowwise `W1 [30,14]` (128×128) | BF16 `[p,3840]`, overwrite, execute in forward. |
| FC2 fwd | `A=W2[1792,1920]`, `B=S[p,1920]`, `TN`: `S @ W2.T` | Rowwise `S [15,p]` (1×128); rowwise `W2 [14,15]` (128×128) | BF16 `[p,1792]`, overwrite, execute in forward. |
| FC2 dgrad | `A=W2`, `B=dY[p,1792]`, `NN`: `dY @ W2` | Rowwise `dY [14,p]` (1×128); columnwise `W2 [15,14]` (128×128 in transposed view) | BF16 `[p,1920]`, overwrite before SSSGLU backward. |
| FC1 dgrad | `A=W1`, `B=dA[p,3840]`, `NN`: `dA @ W1` | Rowwise `dA [30,p]` (1×128); columnwise `W1 [14,30]` (128×128 in transposed view) | BF16 `[p,1792]`, overwrite; return only valid rows. |
| FC2 wgrad | `A=S_b[p,1920]`, `B=dY[p,1792]`, `NT`: `dY.T @ S_b` | Columnwise `S_b [p/128,1920]` and `dY [p/128,1792]`, each one scale per feature per 128 expert tokens | FP32 `[1792,1920]` per expert; queue and drain through `backward_dw`; add into FP32 `main_grad` without a BF16 intermediate. |
| FC1 wgrad | `A=X_b[p,1792]`, `B=dA[p,3840]`, `NT`: `dA.T @ X_b` | Columnwise `X_b [p/128,1792]` and `dA [p/128,3840]`, each one scale per feature per 128 expert tokens | FP32 `[3840,1792]` per expert; same delayed/accumulation path. |

The physical storage shapes follow installed `Float8BlockQuantizer.get_scale_shape` and `get_columnwise_shape` (`float8_blockwise_tensor.py:125–204`).
TE selects row/column data and its associated scales separately for each operand (`rocm_gemm.cu:465–471,517–523`); never transpose FP8 byte storage and reuse the wrong scales.
Forward/dgrad have 1D inputs and 2D weight scales; wgrad has **two 1D quantizers along the token reduction**, produced from the appropriate reconstructed BF16 operands, not a transpose of rowwise FP8 payloads.
The native API requires its column-major `m` and `k` to be multiples of 16, but TE’s block quantizer `is_quantizable()` requires both matrix axes divisible by **128**.
Stock TE `Fp8Padding` defaults to 16 for this recipe and its low-level padding is not the zero-fill contract this design requires.
Explicit **per-expert** 128 padding satisfies both constraints and preserves the reference block boundaries for wgrad.

### Ragged and empty handling

Use the actual 16 splits and retain expert order; group padding is independent of every other split, including zero and tail groups.
Zero padded rows before quantization at every participating boundary (`X`, `S`, `dY`, `dA`, reconstructed operands and padded probabilities), not just the final output.
Apply SSSGLU and its probability gradient only to valid rows or mask their padding exactly; unpad outputs and gradients before handing them back.
The installed C++ GEMM binding skips a pair with an empty operand, zeros a nonempty destination when `accumulate=false`, preserves it when `accumulate=true`, and returns without native GEMM if all groups are skipped (`extensions/gemm.cpp:568–599`).
The adapter should skip empty **quantization** explicitly and omit empty experts from the GEMM lists; an all-empty batch launches no cast or GEMM, contributes zero wgrad, and must still close all expected scheduling and grad-ready events.
Prove those cases independently of the mixed nonempty case.
`tex.split_quantize` can bulk-allocate blockwise storage but the native blockwise cast still runs per split (`extensions/cast.cpp:540–617,1647–1693`); do not rely on its `valid_split_sections` fast padding variant to zero blockwise operands.

## Minimal adapter boundary

Use one default-off expert-only adapter on a reviewed e242 branch **if requested after this review**.
Reuse TE’s BF16 expert parameter objects, ordering, init, model/optimizer/checkpoint keys and FP32 `main_grad` ownership.
Keep the existing latent projections, routing, shared experts, combine, schedule, and optimizer out of the adapter.
Do not set global `--fp8`, call CUDA DeepGEMM, set a hidden environment recipe or enable host offload.
Scope the differential to expert quantization and the six GEMMs, with the reference’s weighted SSSGLU forward/backward and saved-activation semantics.

- **Saved operands are part of the numerical contract.**
  The Swiss-AI forward uses BF16 `A` for `S`, but saves row-quantized `A` and decodes it to BF16 for SSSGLU backward when `moe_act` is not recomputed (`experts_offloading_fp8_util.py:1464–1482,1553–1580`).
  Preserve this cast point on device with the same E4M3FN software row codec described below for the saved `A` only; it need not use a GEMM-native datatype.
  Compute `S_b` using weighted SSSGLU on that decoded `A`, not by saving or reusing the original forward `S` (`:1148–1165`).
  The reference also reconstructs `X_b` from **FC1’s retained forward row payload**, then quantizes along the token axis for wgrad (`:1595–1601,1699–1707`).
  Decode the candidate’s retained FNUZ row payload and quantize `X_b` columnwise; a one-pass cast of original BF16 `X` would change the recipe.
  Retain the source’s SSSGLU equation and dtype boundaries including FP32 probability gradient accumulation, with no SwiGLU substitution.
- **Weight lifetime.**
  Build rowwise and columnwise 128×128 views from authoritative BF16 weights with the same version, on the first use for the next update.
  Invalidate on accepted parameter mutation or restore, not on every microbatch; retain each version until the last scheduled user exits.
  No FP8 overwrite of the BF16 representation, and no change to optimizer parameter layout.
- **Optimizer comparison.**
  Keeping the resident TE parameter layout does not prove Swiss-AI update parity: the reference uses fused rank-3 expert weights with the `merged_offload_expert` path; the resident TE path uses individual rank-2 parameters.
  `md_decoupling.py:1473–1499,2177–2200,2268–2285` routes both to matrix updates, with per-expert gain axes for the fused representation.
  Compare mapped per-expert gate/up splits, gains, momentum and weight updates on identical accumulated FP32 gradients before adding the FP8 effect; if they differ, stop and scope that difference separately.
- **Delayed wgrad without enabling global fusion.**
  Queue explicit `NT` descriptors with retained columnwise buffers and stream dependencies.
  At `backward_dw`, call the installed list API with FP32 destinations and `accumulate=False`, then explicitly add the FP32 results into the existing FP32 main gradients before notifying DDP.
  Reuse bound scratch at drain time, not a retained FP32 result matrix for every queued microbatch.
  Use Megatron’s registered delayed-wgrad hooks; after the add, use its existing `grad_added_to_main_grad` convention with a dtype-correct **zero** sentinel if using the stock ready hook, which asserts non-`None` `param.grad` during overlap (`distributed_data_parallel.py:324–354,444–462`).
  Prove one contribution and one expected ready notification per parameter per scheduled drain, including zero experts and the all-empty last microbatch; reject any duplicate add or any early ready event.
  This is explicit local FP32 accumulation, not the global fused-accumulation mode.
  Later integration may test native `beta=1` directly into `main_grad`, separately.

Stock `GroupedLinear`’s saved activations and quantizer usage are not a drop-in for the reference reconstruction above.
Its delayed wgrad is real (`WeightGradStore.put` → `GroupedLinear.backward_dw()`), but with fusion **off** it allocates gradient output in activation dtype and later sets `param.grad` in parameter dtype (`grouped_linear.py:1128–1144,1199–1218,1919–1956`).
With fusion **on**, it can send `main_grad` through the same native list GEMM with beta/accumulate; this is a source-supported separate route, not a result for the pinned Megatron schedule.
The minimal work is the scoped adapter, explicit padding and saved-operand reconstruction, and delayed FP32 queue/hook integration—not six new GEMM kernels.

## E4M3FN and E4M3FNUZ are not interchangeable

| Item | Swiss-AI reference | Selected gfx942 TE path |
|---|---|---|
| Format | E4M3FN, exponent bias 7, finite max 448, signed zero, smallest positive subnormal `2^-9` | E4M3FNUZ, exponent bias 8, finite max 240, unsigned zero, smallest positive subnormal `2^-10`; the negative-zero bit pattern from FN is NaN in FNUZ. |
| Scale | GPU reference casts use `amax/448` (dequant multiplier), not power-of-two scales; token and weight casts use 1 for all-zero tiles, and token/weight/wgrad cast paths floor tiny scales at `1e-30` (wgrad zeros get the floor). | TE computes the *quantization* multiplier `240/amax` using `__fdiv_rn` and stores inverse FP32 scales for GEMM; it handles zero and non-finite maxima with scale 1 and caps an overflowing multiplier. Force power-of-two scaling **off** locally; the block recipe enables it by default. This is not the same zero/tiny-block policy or a constant byte conversion. |
| Rounding and clipping | Reference GPU code scales in FP32 and casts via Triton `tl.float8e4nv`; its explicit clamp is commented out. Compare to reference cast semantics, not an arbitrary `.to(torch.float8_e4m3fn)` on out-of-range values. | TE’s HIP FP8 constructor selects finite saturation and the non-stochastic RNE path. The gfx942 fast conversion clips finite inputs to ±240 and does not apply that clamp to NaN/Inf. Do not infer equivalence for subnormals, signed zero or exceptional values; compare native GEMM partial sums and BF16 output casts too. |

Never view FN bytes as FNUZ or vice versa.
To compare expert arithmetic, use separate FN and FNUZ quantization oracles from the same high-precision operands, plus a GPU check against the **matching format** for each backend.
Record clipping counts, midpoint behavior, zeros/subnormals, maximum and tiny finite values and the downstream forward/dgrad/wgrad/optimizer deltas.
An accepted FNUZ precision difference is a user decision, not implied by a correct FNUZ implementation.
The relevant TE source includes `common/amd_detail/hip_float8.h`, `transpose/quantize_transpose_{vector,square}_blockwise.cu`, and `recipe/recipe_common.cuh:16–76`; image 910’s HIP `amd_hip_fp8.h` supplies the finite-saturating FNUZ conversion.

## Independent transport: keep E4M3FN on the wire

**Proposed cast/decast: the existing e242 `fp8_jit.per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)` and `per_token_dequant_from_fp8(q, sf, gran_k=128, out_dtype=torch.bfloat16)`, compiled for gfx942 with image 910’s Triton.**
No conversion to FNUZ, no native FP8 matrix multiply, and no wire reinterpretation are required for this lane.
The matching [Triton commit `669b31acc1dd1b3fd93286afd5db67f65d9f7557`](https://github.com/triton-lang/triton/blob/669b31acc1dd1b3fd93286afd5db67f65d9f7557/third_party/amd/lib/TritonAMDGPUToLLVM/ConvertFpCastOpToLLVM.cpp) lowers FP32 → E4M3FN on CDNA3 through `Fp32ToFp8E4M3fnRtneSW` with finite saturation (`:655–719,745–777,2107–2130,2181–2188`).
The inverse selects software E4M3FN → FP16 → FP32 on CDNA3 (`:909–930,1395–1428`); all E4M3FN finite values fit exactly in FP16.
The installed AMD compiler advertises `fp8e4nv`, but actual image codegen and byte results still need a future component check.

The selected Megatron combined-schedule nodes cast the **post-latent** BF16 input and dequantize received data to BF16 **before** expert processing (`fine_grained_callables.py:800–893`, `moe_layer.py:437–442`).
The cast computes FP32 scales `amax/448`, with `1` for zero and `max(scale,1e-30)` before reciprocal and multiplication, then writes real E4M3FN bytes.
The receive path converts E4M3FN to FP32, multiplies by the delivered FP32 scale, and rounds to BF16 on store.
Pin the exact cast kernel and verify the BF16 rounding and scale floor against the reference; do not assume all backends compile division/reciprocal identically.
Require contiguous feature stride for the payload and source activation; record both scale strides and compare what the consumer sees.

At MBS2/S4096 each source has up to 8192 token rows before routing, width 1792.
For each transmitted row the contract is **1792 FP8 bytes + 14 FP32 scales (56 bytes)** before indices, routing weights and UCCL metadata.
The Python wrapper passes `x.element_size()`, the scale pointer/count and both scale strides to native UCCL; the HIP internode path copies payload vectors and scale words separately, then reconstructs the returned tuple (`deep_ep/buffer.py:1194–1283,1696–1732,1966–2000`; `opt/uccl/ep/src/internode.hip:910–935,1573–1615`).
The **actual 56-byte scale row** takes the non-16-byte-aligned receive-scale branch; testing only a 512-wide proxy, with a 16-byte scale row, misses it.
The reverse is `fused_fp8_dispatch_bwd` → `Buffer.combine` on **BF16** gradients, with FP32 top-k probability gradients and the same dispatch handle.
Transport-only success must not be reported as expert FP8 success.

If separately approved later, qualify in this order:

1. **Codec only:** compare E4M3FN bytes, FP32 scale words and decoded BF16 values for fixed finite BF16 fixtures; include halfway ties, max-range values, all zero, tiny finite BF16, signs/subnormals, zero rows and exact 1792 width.
   Use a reference codec with the same FP32 evaluation order and explicit SATFINITE semantics; reject non-finite model inputs separately rather than relying on saturation to hide them.
   Retain actual gfx942 dispatch and generated dtype/lowering evidence; FNUZ-typed data is a hard failure.
2. **UCCL wire only:** send known typed FN payloads and independent, nonuniform FP32 scales with an identity per `(source rank, token, destination rank)` and per-feature-block scale pattern.
   Check payload bytes and FP32 scale bits **before** decode, with correct row association, top-k indices/FP32 probabilities, counts, handles, and no duplicate/lost rows when several top-k experts share a destination.
   Test intra-node plus EP16 CXI internode, zero send and zero receive, skewed and ragged routes, normal fresh dispatch and repeated independent handles under the actual stream/event pattern.
   Verify buffer hints and capacity for 8192 source rows/top-8 and the previous EP16 defaults without adding the old EP8 workaround.
3. **Composed transport:** compare receive decode to a local codec oracle, then test BF16 reverse combine, probability-gradient association, reuse/teardown and generation lifetime.
   Sender/receiver payload and scale equality is exact; hierarchically reduced BF16 reverse results require their own predeclared error check and are not a bytewise wire assertion.
   No expert GEMM or full-layout memory allocation is required to qualify these pieces.

## Evidence boundaries

Read-only audit of image `910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490`, e242, and the exact reported TE/Triton revisions.
Key installed SHA-256 values:

| Image member (site-packages-relative except where noted) | SHA-256 |
|---|---|
| `transformer_engine/pytorch/module/grouped_linear.py` | `bc52bcbeb1e5c6e696c3869814d9e9bebe37d507a06beb5e16ccef43de988dea` |
| `transformer_engine/pytorch/cpp_extensions/gemm.py` | `957974b26d32ab74f1521780d96e2d9e56a30e3a8420d7bb066c69cb4c6a5e9d` |
| `transformer_engine/libtransformer_engine.so` | `140dc626463ae6072121ca66a988c71954268f7b2c27b4d447894a97f2cba69a` |
| `transformer_engine/lib/libkittens_gemm.so` | `23da03c0140a871d9882890a2ea84a471f69f92f3a2096b8c176468adea84d9c` |
| `deep_ep/buffer.py` | `d30ad5231f476c726c246240eec9b6a2f9124e16752900ca562653623e77a768` |
| `opt/uccl/ep/src/internode.hip` (image-root relative) | `ec1e4d9ed1bb344b556cb4c4e32b0850cfc65440866d31307a68639d03e16e51` |

Before enabling anything, prove the six exact GPU call paths, finite and numerical behavior, ragged and empty groups, accumulated FP32 gradients, and an optimizer-step comparison using real Chonk widths and split cases.
Run no full-layout BF16 memory screen pending separate review of its purpose and resources.
Graphs and global accumulation fusion are subsequent integration qualifications.
No experiment switches, code, image, or Slurm state were changed by this review.
