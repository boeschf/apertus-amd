# Isolated Chonk source candidates

These patches target Megatron commit `e242f3c3c1136aeae498529bfde5e06bda9a7cca` independently.
They are staged candidates, not an automatically applied patch stack.
Normal experiments still select the original committed source; the runner does not consume this directory.

| Patch | Purpose | Selection and qualification |
| --- | --- | --- |
| `chonk-fused-output-gate-e242.patch` | Repair and isolate Imanol Schlag's sigmoid-gate kernel at the global-attention output gate. | Defaults off; `BEVERIN_FUSED_OUT_GATE=1` admits only ordinary same-device BF16 CUDA/HIP tensors with matching positive element counts; all other inputs retain the original path. The isolated helper passed nominal-shape forward/both-gradient checks against the default-compiled reference, but its matched component timing did not qualify a positive effect. Do not promote it on this evidence; production dispatch and recomputation/checkpoint interaction remain unqualified. |
| `chonk-te-gemm-workspace-e242.patch` | Preserve the original workspace argument only when the installed `general_gemm` signature accepts it. | Compatibility only, not a backend switch or measured speedup; signature inspection remains on eligible calls. |
| `chonk-fused-sssglu-e242.patch` | Isolate Imanol's same-equation plain/weighted SSSGLU kernels at expert/shared D1920. | Defaults off; narrow contiguous BF16/FP32 admission preserves original-path fallback. CPU apply/math/guard tests and source review passed; native output, both input-gradient halves, FP32 probability gradient, recomputation and copy-inclusive performance remain unqualified. Not a SwiGLU or public AITER substitution. |

All three derive from `/iopsstor/scratch/cscs/boeschf/amd-vs-gh200-moe-training/chonk/patches/`.
Attribution, original bytes, exact base hashes, apply-checks and CPU evidence are retained under `artifacts/investigations/performance-next/restart-20261116/imanol-patch-integration/`.
Focused tests are `tests/test_chonk_output_gate_patch.py` and `tests/test_chonk_te_gemm_workspace_patch.py`.
The independent SSSGLU candidate has `tests/test_chonk_fused_sssglu_patch.py` and retained provenance/review under `artifacts/investigations/performance-next/restart-20261116/sssglu-isolated-candidate-review/`.
Their CPU stub/formula/source checks do not establish GPU correctness or throughput.
The separate [output-gate numerical input](../chonk-output-gate-qualification-i910bfcf936d3/experiment.toml) embeds only the pinned helper and original reference method for component checks; it does not make an uncommitted patched training source executable.
Its pointwise numerical policy and resource execution require separate approval, and it does not test production flag dispatch or checkpoint/recompute integration.
The [performance investigation status](../../artifacts/investigations/performance-next/STATUS.md) links retained correctness, timing and promotion evidence; numerical acceptance alone is not performance acceptance.

Apply only to a separately owned export of the exact base, never an external or live training checkout.
Before execution, retain and verify the immutable base archive, patch bytes/order and resulting source/import identities through an explicit reviewed execution path.
Do not create a commit merely to bypass that provenance requirement, and do not combine compatibility and performance changes in a matched comparison.
