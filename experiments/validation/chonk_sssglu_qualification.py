"""Self-contained SSSGLU numerical component; GPU execution needs separate approval."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import inspect
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

BEFORE = '# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.\n\n\n# pylint: disable=missing-function-docstring, missing-class-docstring\n\nimport torch\n\nfrom megatron.core.jit import jit_fuser\nfrom megatron.core.utils import nvtx_decorator\n\n###### BIAS SSSGLU FUSION/ NO AUTOGRAD ################\n# SSSGLU is a gated linear unit whose gate is a shifted, scaled softsign:\n#   gate(x)  = softsign(x - 1) + 0.5 = (x - 1) / (1 + |x - 1|) + 0.5   (== sssglu_act in activations.py)\n#   SSSGLU(y_1, y_2) = gate(y_1) * y_2\n# Like RLGLU (and unlike SwiGLU/SSGLU) the gate is NOT of the form ``x * squash(x)`` -- it is the\n# gate itself, applied directly, so there is no separate SiLU-style helper. The gate derivative is\n#   gate\'(x) = 1 / (1 + |x - 1|)**2\n# which is cheap and needs nothing beyond ``|x - 1|`` (already computed for the gate). Built exactly\n# like the SwiGLU/SSGLU/RLGLU fusions (fused_bias_swiglu.py / fused_bias_ssglu.py /\n# fused_bias_rlglu.py): @jit_fuser forward/backward pairs wrapped in torch.autograd.Function, since\n# the gate has no cross-feature reduction. The gate math is inlined here (rather than importing\n# sssglu_act) to keep the jit_fuser scripting self-contained; it must stay in sync with sssglu_act\n# in activations.py.\n\n\n@jit_fuser\ndef sssglu(y):\n    """Performs SSSGLU (Shifted-Scaled-Softsign-Gated Linear Unit) activation function.\n\n    Args:\n        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.\n\n    Returns:\n        torch.Tensor: Result of SSSGLU activation: gate(y1) * y2, where y1, y2 are the split\n            halves and gate(x) = softsign(x - 1) + 0.5.\n    """\n    y_1, y_2 = torch.chunk(y, 2, -1)\n    u = y_1 - 1\n    return (0.5 + u / (1 + torch.abs(u))) * y_2\n\n\n@jit_fuser\ndef bias_sssglu(y, bias):\n    """Performs SSSGLU activation with bias addition.\n\n    Args:\n        y (torch.Tensor): Input tensor.\n        bias (torch.Tensor): Bias tensor to be added to input.\n\n    Returns:\n        torch.Tensor: Result of bias addition followed by SSSGLU activation.\n    """\n    y = y + bias\n    return sssglu(y)\n\n\n@jit_fuser\ndef weighted_sssglu(y, weights):\n    dtype = y.dtype\n    res = sssglu(y) * weights\n    return res.to(dtype)\n\n\n@jit_fuser\ndef sssglu_back(g, y):\n    """Computes the gradient for the SSSGLU activation function.\n\n    With gate(x) = softsign(x - 1) + 0.5, the gate derivative is\n        gate\'(x) = 1 / (1 + |x - 1|)**2.\n    So d/dy1 [gate(y1) * y2] = gate\'(y1) * y2 and d/dy2 [gate(y1) * y2] = gate(y1).\n\n    Args:\n        g (torch.Tensor): Gradient tensor from the subsequent layer.\n        y (torch.Tensor): Input tensor that was used in the forward pass.\n\n    Returns:\n        torch.Tensor: Gradient with respect to the input tensor.\n    """\n    y_1, y_2 = torch.chunk(y, 2, -1)\n    u = y_1 - 1\n    denom = 1 + torch.abs(u)\n    gate = 0.5 + u / denom\n    gate_prime = 1.0 / (denom * denom)\n    return torch.cat((g * gate_prime * y_2, g * gate), -1)\n\n\n@jit_fuser\ndef bias_sssglu_back(g, y, bias):\n    """Computes the gradient for the biased SSSGLU activation function.\n\n    Args:\n        g (torch.Tensor): Gradient tensor from the subsequent layer.\n        y (torch.Tensor): Input tensor that was used in the forward pass.\n        bias (torch.Tensor): Bias tensor that was added in the forward pass.\n\n    Returns:\n        torch.Tensor: Gradient with respect to the input tensor, computed after\n            applying the bias addition.\n    """\n    y = y + bias\n    return sssglu_back(g, y)\n\n\n@jit_fuser\ndef weighted_sssglu_back(g, y, weights):\n    input_dtype = y.dtype\n    w_dtype = weights.dtype\n    input_grad = sssglu_back(g * weights, y)\n    # precison of w may be higher than y and g, so we need to cast g to w_dtype\n    weights_grad = sssglu(y) * g.to(w_dtype)\n    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)\n    return input_grad.to(input_dtype), weights_grad.to(w_dtype)\n\n\nclass BiasSSSGLUFunction(torch.autograd.Function):\n    """Custom autograd function for SSSGLU activation with bias support."""\n\n    @staticmethod\n    @nvtx_decorator()\n    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):\n        """Forward pass of biased SSSGLU activation."""\n        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input\n        if cpu_offload_input:\n            input_for_backward.activation_offloading = True\n            bias.activation_offloading = True\n        ctx.save_for_backward(input_for_backward, bias)\n        ctx.ori_input_dtype = input.dtype\n        ctx.fp8_input_store = fp8_input_store\n        return bias_sssglu(input, bias)\n\n    @staticmethod\n    @nvtx_decorator()\n    def backward(ctx, grad_output):\n        """Backward pass of biased SSSGLU activation."""\n        input, bias = ctx.saved_tensors\n        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input\n        tmp = bias_sssglu_back(grad_output, input, bias)\n        return tmp, tmp, None, None\n\n\nclass SSSGLUFunction(torch.autograd.Function):\n    """Custom autograd function for SSSGLU activation without bias."""\n\n    @staticmethod\n    @nvtx_decorator()\n    def forward(ctx, input, fp8_input_store, cpu_offload_input):\n        """Forward pass of SSSGLU activation."""\n        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input\n        if cpu_offload_input:\n            input_for_backward.activation_offloading = True\n        ctx.save_for_backward(input_for_backward)\n        ctx.ori_input_dtype = input.dtype\n        ctx.fp8_input_store = fp8_input_store\n        return sssglu(input)\n\n    @staticmethod\n    @nvtx_decorator()\n    def backward(ctx, grad_output):\n        """Backward pass of SSSGLU activation."""\n        input = ctx.saved_tensors[0]\n        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input\n        tmp = sssglu_back(grad_output, input)\n        return tmp, None, None\n\n\nclass WeightedSSSGLUFunction(torch.autograd.Function):\n    @staticmethod\n    # bias is an optional argument\n    def forward(ctx, input, weights, fp8_input_store):\n        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input\n        ctx.save_for_backward(input_for_backward, weights)\n        ctx.ori_input_dtype = input.dtype\n        ctx.fp8_input_store = fp8_input_store\n        return weighted_sssglu(input, weights)\n\n    @staticmethod\n    def backward(ctx, grad_output):\n        input, weights = ctx.saved_tensors\n        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input\n        tmp, wgrad = weighted_sssglu_back(grad_output, input, weights)\n        return tmp, wgrad, None\n\n\ndef bias_sssglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):\n    """Implementation of biased SSSGLU that handles different input shapes.\n\n    This function reshapes the input if necessary, applies the SSSGLU activation\n    (with or without bias), and restores the original shape.\n\n    Args:\n        input (torch.Tensor): Input tensor to apply SSSGLU activation.\n        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,\n            uses the bias-free SSSGLU variant.\n        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.\n            Defaults to False.\n\n    Returns:\n        torch.Tensor: Result of biased SSSGLU activation.\n\n    Raises:\n        AssertionError: If input tensor does not have 2 or 3 dimensions.\n    """\n    ori_shape = input.shape\n    assert len(ori_shape) in [2, 3]\n    input = input.view(-1, ori_shape[-1])\n    if bias is not None:\n        output = BiasSSSGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)\n    else:\n        output = SSSGLUFunction.apply(input, fp8_input_store, cpu_offload_input)\n\n    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)\n\n\ndef weighted_bias_sssglu_impl(input, bias, weights, fp8_input_store=False):\n    """\n    Token-wise-weighted bias sssglu fusion.\n    """\n    ori_shape = input.shape\n    assert len(ori_shape) in [2, 3]\n    input = input.view(-1, ori_shape[-1])\n    if bias is not None:\n        raise NotImplementedError("Bias is not supported for weighted sssglu fusion")\n    else:\n        output = WeightedSSSGLUFunction.apply(input, weights, fp8_input_store)\n\n    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)\n'
APPLIED = '# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.\n\n\n# pylint: disable=missing-function-docstring, missing-class-docstring\n\nimport torch\n\nfrom megatron.core.jit import jit_fuser\nfrom megatron.core.utils import nvtx_decorator\n\nimport os as _os\n\n_BEVERIN_FUSED_SSSGLU = _os.environ.get("BEVERIN_FUSED_SSSGLU", "0") == "1"\n\n\ndef _beverin_sssglu_eligible(input, weights=None):\n    """Admit only ordinary contiguous BF16 D1920 rows and optional FP32 row weights."""\n    if not (_BEVERIN_FUSED_SSSGLU and type(input) is torch.Tensor\n            and input.is_cuda and input.dtype == torch.bfloat16\n            and input.layout == torch.strided and not input.is_nested\n            and input.ndim == 2 and input.shape[1] == 3840\n            and input.is_contiguous() and input.storage_offset() == 0):\n        return False\n    if weights is None:\n        return True\n    return (type(weights) is torch.Tensor and weights.is_cuda\n            and weights.dtype == torch.float32 and weights.device == input.device\n            and weights.layout == torch.strided and not weights.is_nested\n            and tuple(weights.shape) == (input.shape[0], 1)\n            and weights.is_contiguous() and weights.storage_offset() == 0)\n\n\n###### BIAS SSSGLU FUSION/ NO AUTOGRAD ################\n# SSSGLU is a gated linear unit whose gate is a shifted, scaled softsign:\n#   gate(x)  = softsign(x - 1) + 0.5 = (x - 1) / (1 + |x - 1|) + 0.5   (== sssglu_act in activations.py)\n#   SSSGLU(y_1, y_2) = gate(y_1) * y_2\n# Like RLGLU (and unlike SwiGLU/SSGLU) the gate is NOT of the form ``x * squash(x)`` -- it is the\n# gate itself, applied directly, so there is no separate SiLU-style helper. The gate derivative is\n#   gate\'(x) = 1 / (1 + |x - 1|)**2\n# which is cheap and needs nothing beyond ``|x - 1|`` (already computed for the gate). Built exactly\n# like the SwiGLU/SSGLU/RLGLU fusions (fused_bias_swiglu.py / fused_bias_ssglu.py /\n# fused_bias_rlglu.py): @jit_fuser forward/backward pairs wrapped in torch.autograd.Function, since\n# the gate has no cross-feature reduction. The gate math is inlined here (rather than importing\n# sssglu_act) to keep the jit_fuser scripting self-contained; it must stay in sync with sssglu_act\n# in activations.py.\n\n\n@jit_fuser\ndef sssglu(y):\n    """Performs SSSGLU (Shifted-Scaled-Softsign-Gated Linear Unit) activation function.\n\n    Args:\n        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.\n\n    Returns:\n        torch.Tensor: Result of SSSGLU activation: gate(y1) * y2, where y1, y2 are the split\n            halves and gate(x) = softsign(x - 1) + 0.5.\n    """\n    y_1, y_2 = torch.chunk(y, 2, -1)\n    u = y_1 - 1\n    return (0.5 + u / (1 + torch.abs(u))) * y_2\n\n\n@jit_fuser\ndef bias_sssglu(y, bias):\n    """Performs SSSGLU activation with bias addition.\n\n    Args:\n        y (torch.Tensor): Input tensor.\n        bias (torch.Tensor): Bias tensor to be added to input.\n\n    Returns:\n        torch.Tensor: Result of bias addition followed by SSSGLU activation.\n    """\n    y = y + bias\n    return sssglu(y)\n\n\n@jit_fuser\ndef weighted_sssglu(y, weights):\n    dtype = y.dtype\n    res = sssglu(y) * weights\n    return res.to(dtype)\n\n\n@jit_fuser\ndef sssglu_back(g, y):\n    """Computes the gradient for the SSSGLU activation function.\n\n    With gate(x) = softsign(x - 1) + 0.5, the gate derivative is\n        gate\'(x) = 1 / (1 + |x - 1|)**2.\n    So d/dy1 [gate(y1) * y2] = gate\'(y1) * y2 and d/dy2 [gate(y1) * y2] = gate(y1).\n\n    Args:\n        g (torch.Tensor): Gradient tensor from the subsequent layer.\n        y (torch.Tensor): Input tensor that was used in the forward pass.\n\n    Returns:\n        torch.Tensor: Gradient with respect to the input tensor.\n    """\n    y_1, y_2 = torch.chunk(y, 2, -1)\n    u = y_1 - 1\n    denom = 1 + torch.abs(u)\n    gate = 0.5 + u / denom\n    gate_prime = 1.0 / (denom * denom)\n    return torch.cat((g * gate_prime * y_2, g * gate), -1)\n\n\n@jit_fuser\ndef bias_sssglu_back(g, y, bias):\n    """Computes the gradient for the biased SSSGLU activation function.\n\n    Args:\n        g (torch.Tensor): Gradient tensor from the subsequent layer.\n        y (torch.Tensor): Input tensor that was used in the forward pass.\n        bias (torch.Tensor): Bias tensor that was added in the forward pass.\n\n    Returns:\n        torch.Tensor: Gradient with respect to the input tensor, computed after\n            applying the bias addition.\n    """\n    y = y + bias\n    return sssglu_back(g, y)\n\n\n@jit_fuser\ndef weighted_sssglu_back(g, y, weights):\n    input_dtype = y.dtype\n    w_dtype = weights.dtype\n    input_grad = sssglu_back(g * weights, y)\n    # precison of w may be higher than y and g, so we need to cast g to w_dtype\n    weights_grad = sssglu(y) * g.to(w_dtype)\n    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)\n    return input_grad.to(input_dtype), weights_grad.to(w_dtype)\n\n\nclass BiasSSSGLUFunction(torch.autograd.Function):\n    """Custom autograd function for SSSGLU activation with bias support."""\n\n    @staticmethod\n    @nvtx_decorator()\n    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):\n        """Forward pass of biased SSSGLU activation."""\n        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input\n        if cpu_offload_input:\n            input_for_backward.activation_offloading = True\n            bias.activation_offloading = True\n        ctx.save_for_backward(input_for_backward, bias)\n        ctx.ori_input_dtype = input.dtype\n        ctx.fp8_input_store = fp8_input_store\n        return bias_sssglu(input, bias)\n\n    @staticmethod\n    @nvtx_decorator()\n    def backward(ctx, grad_output):\n        """Backward pass of biased SSSGLU activation."""\n        input, bias = ctx.saved_tensors\n        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input\n        tmp = bias_sssglu_back(grad_output, input, bias)\n        return tmp, tmp, None, None\n\n\nclass SSSGLUFunction(torch.autograd.Function):\n    """Custom autograd function for SSSGLU activation without bias."""\n\n    @staticmethod\n    @nvtx_decorator()\n    def forward(ctx, input, fp8_input_store, cpu_offload_input):\n        """Forward pass of SSSGLU activation."""\n        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input\n        if cpu_offload_input:\n            input_for_backward.activation_offloading = True\n        ctx.save_for_backward(input_for_backward)\n        ctx.ori_input_dtype = input.dtype\n        ctx.fp8_input_store = fp8_input_store\n        return sssglu(input)\n\n    @staticmethod\n    @nvtx_decorator()\n    def backward(ctx, grad_output):\n        """Backward pass of SSSGLU activation."""\n        input = ctx.saved_tensors[0]\n        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input\n        tmp = sssglu_back(grad_output, input)\n        return tmp, None, None\n\n\nclass WeightedSSSGLUFunction(torch.autograd.Function):\n    @staticmethod\n    # bias is an optional argument\n    def forward(ctx, input, weights, fp8_input_store):\n        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input\n        ctx.save_for_backward(input_for_backward, weights)\n        ctx.ori_input_dtype = input.dtype\n        ctx.fp8_input_store = fp8_input_store\n        return weighted_sssglu(input, weights)\n\n    @staticmethod\n    def backward(ctx, grad_output):\n        input, weights = ctx.saved_tensors\n        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input\n        tmp, wgrad = weighted_sssglu_back(grad_output, input, weights)\n        return tmp, wgrad, None\n\n\ndef bias_sssglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):\n    """Implementation of biased SSSGLU that handles different input shapes.\n\n    This function reshapes the input if necessary, applies the SSSGLU activation\n    (with or without bias), and restores the original shape.\n\n    Args:\n        input (torch.Tensor): Input tensor to apply SSSGLU activation.\n        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,\n            uses the bias-free SSSGLU variant.\n        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.\n            Defaults to False.\n\n    Returns:\n        torch.Tensor: Result of biased SSSGLU activation.\n\n    Raises:\n        AssertionError: If input tensor does not have 2 or 3 dimensions.\n    """\n    ori_shape = input.shape\n    assert len(ori_shape) in [2, 3]\n    input = input.view(-1, ori_shape[-1])\n    if bias is None and not fp8_input_store and not cpu_offload_input and _beverin_sssglu_eligible(input):\n        from megatron.core.fusions.beverin_sssglu import TritonSSSGLU\n\n        output = TritonSSSGLU.apply(input)\n    elif bias is not None:\n        output = BiasSSSGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)\n    else:\n        output = SSSGLUFunction.apply(input, fp8_input_store, cpu_offload_input)\n\n    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)\n\n\ndef weighted_bias_sssglu_impl(input, bias, weights, fp8_input_store=False):\n    """\n    Token-wise-weighted bias sssglu fusion.\n    """\n    ori_shape = input.shape\n    assert len(ori_shape) in [2, 3]\n    input = input.view(-1, ori_shape[-1])\n    if bias is not None:\n        raise NotImplementedError("Bias is not supported for weighted sssglu fusion")\n    elif not fp8_input_store and weights is not None and _beverin_sssglu_eligible(input, weights):\n        from megatron.core.fusions.beverin_sssglu import TritonWeightedSSSGLU\n\n        output = TritonWeightedSSSGLU.apply(input, weights)\n    else:\n        output = WeightedSSSGLUFunction.apply(input, weights, fp8_input_store)\n\n    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)\n'
HELPER = '# Adapted from Imanol Schlag, 0007-t4-triton-sssglu.patch (lab T4).\n# Same real SSSGLU equations; fused rounding is not bitwise baseline equivalence.\n# Only the caller-admitted BF16 D1920 / optional FP32 row weights are staged.\nimport torch\nimport triton\nimport triton.language as tl\n\n\n@triton.jit\ndef _fwd(x_ptr, p_ptr, y_ptr, M, D, HAS_P: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):\n    pm = tl.program_id(0)\n    pd = tl.program_id(1)\n    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)\n    rd = pd * BD + tl.arange(0, BD)\n    mask = (rm[:, None] < M) & (rd[None, :] < D)\n    xo = rm[:, None] * (2 * D) + rd[None, :]\n    a = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)\n    b = tl.load(x_ptr + xo + D, mask=mask, other=0.0).to(tl.float32)\n    u = a - 1.0\n    y = (0.5 + u / (1.0 + tl.abs(u))) * b\n    if HAS_P:\n        p = tl.load(p_ptr + rm, mask=rm < M, other=0.0)\n        y = y * p[:, None]\n    tl.store(y_ptr + rm[:, None] * D + rd[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)\n\n\n@triton.jit\ndef _bwd(x_ptr, p_ptr, g_ptr, gx_ptr, gp_ptr, M, D, HAS_P: tl.constexpr, BM: tl.constexpr,\n         BD: tl.constexpr):\n    pm = tl.program_id(0)\n    rm = (pm * BM + tl.arange(0, BM)).to(tl.int64)\n    mrow = rm < M\n    if HAS_P:\n        p = tl.load(p_ptr + rm, mask=mrow, other=0.0)\n    acc = tl.zeros([BM], dtype=tl.float32)\n    for d0 in range(0, D, BD):\n        rd = d0 + tl.arange(0, BD)\n        mask = mrow[:, None] & (rd[None, :] < D)\n        xo = rm[:, None] * (2 * D) + rd[None, :]\n        go = rm[:, None] * D + rd[None, :]\n        a = tl.load(x_ptr + xo, mask=mask, other=0.0).to(tl.float32)\n        b = tl.load(x_ptr + xo + D, mask=mask, other=0.0).to(tl.float32)\n        g = tl.load(g_ptr + go, mask=mask, other=0.0).to(tl.float32)\n        u = a - 1.0\n        den = 1.0 + tl.abs(u)\n        gate = 0.5 + u / den\n        if HAS_P:\n            acc += tl.sum(tl.where(mask, gate * b * g, 0.0), axis=1)\n            g = g * p[:, None]\n        ga = g * b / (den * den)\n        gb = g * gate\n        tl.store(gx_ptr + xo, ga.to(gx_ptr.dtype.element_ty), mask=mask)\n        tl.store(gx_ptr + xo + D, gb.to(gx_ptr.dtype.element_ty), mask=mask)\n    if HAS_P:\n        tl.store(gp_ptr + rm, acc, mask=mrow)\n\n\n_FBM, _FBD, _BBM, _BBD = 8, 512, 4, 512\n\n\ndef fwd(x2d, p=None):\n    M, D2 = x2d.shape\n    D = D2 // 2\n    y = torch.empty((M, D), device=x2d.device, dtype=x2d.dtype)\n    if M == 0:\n        return y\n    grid = (triton.cdiv(M, _FBM), triton.cdiv(D, _FBD))\n    _fwd[grid](x2d, p if p is not None else x2d, y, M, D, HAS_P=p is not None, BM=_FBM, BD=_FBD,\n               num_warps=8)\n    return y\n\n\ndef bwd(g2d, x2d, p=None):\n    M, D2 = x2d.shape\n    D = D2 // 2\n    gx = torch.empty_like(x2d)\n    gp = torch.empty((M,), device=x2d.device, dtype=torch.float32) if p is not None else None\n    if M == 0:\n        return gx, gp\n    grid = (triton.cdiv(M, _BBM),)\n    _bwd[grid](x2d, p if p is not None else x2d, g2d, gx, gp if gp is not None else x2d, M, D,\n               HAS_P=p is not None, BM=_BBM, BD=_BBD, num_warps=8)\n    return gx, gp\n\n\nclass TritonWeightedSSSGLU(torch.autograd.Function):\n    """y = gate(x1) * x2 * w, w per row ([M, 1], usually fp32 router probs)."""\n\n    @staticmethod\n    def forward(ctx, x, w):\n        x = x.contiguous()\n        p = w.reshape(-1).float().contiguous()\n        ctx.save_for_backward(x, p)\n        ctx.w_dtype, ctx.w_shape = w.dtype, w.shape\n        return fwd(x, p)\n\n    @staticmethod\n    def backward(ctx, g):\n        x, p = ctx.saved_tensors\n        gx, gp = bwd(g.contiguous(), x, p)\n        return gx, gp.to(ctx.w_dtype).view(ctx.w_shape)\n\n\nclass TritonSSSGLU(torch.autograd.Function):\n    """y = gate(x1) * x2."""\n\n    @staticmethod\n    def forward(ctx, x):\n        x = x.contiguous()\n        ctx.save_for_backward(x)\n        return fwd(x)\n\n    @staticmethod\n    def backward(ctx, g):\n        (x,) = ctx.saved_tensors\n        gx, _ = bwd(g.contiguous(), x)\n        return gx\n'
ASSET_SHA = {'before': '9bda2a2dc053486dfbc556225682a3c45f329af062a12addf3a646e2be7acf94', 'applied': 'ae74a2c1932adb87e3414f029119f90afc0dfa58a0239b4a1824d94d5625f324', 'helper': 'c6b92d1505813db5813fc867e5ff7e95261b66c3b1682199015d2b2ab9ee7036'}

IMAGE = '910bfcf936d3f7a1dfd17dc560481cfeff6c5d9a7f5619c14f43b17348a28490'
SOURCE = 'e242f3c3c1136aeae498529bfde5e06bda9a7cca'
POLICY = dict(element_rtol=.015625, element_atol=.00390625, rms_rtol=.015625, rms_atol=1e-5)
VERSIONS = {'torch':'2.14.0+rocm10.1.0','triton':'3.8.0+git669b31ac.rocm10.1.0'}
FAMILIES = {'plain':(4096,129,1,0), 'weighted':(32768,129,1,0)}
D, TILE = 1920, 64
# CPU/source readiness is not scheduler authorization or native numerical acceptance.
IMPLEMENTATION_READY = True
BLOCKERS = ()
GIT, HIP = '11ef10939935619f2c723ba326e3808a750a924e', '7.16.26385'
PACKAGE = Path('/opt/venv/lib/python3.12/site-packages')
COUNTER_PINS = {'torch/_dynamo/utils.py':(214231,'e550c68c924e1c0babaf31fc629e900c1d3b6ecbcf10aa88442d39b405b7102a'),
 'torch/_inductor/metrics.py':(15211,'d3cb2560945d4e284c24c2728621a0a4c95eae70a36a31801c9292a2180c3e2e'),
 'triton/runtime/jit.py':(44747,'38b8151fcceec1a0210a696775b7c64d7fac5e9b69eee2dda7e93c1e2a377509')}
HELPER_MODULE = 'megatron.core.fusions.beverin_sssglu'
ADAPTER = 'import torch\njit_fuser = torch.compile\ndef nvtx_decorator(message=None, color=None):\n    return lambda function: function\n'


def sha(data):
    """Hash component bytes, not the image or caches."""
    return hashlib.sha256(data).hexdigest()


def save(path, record):
    """Exclusive evidence writes preserve an active native exception."""
    primary = sys.exception()
    try:
        with path.open('x') as stream: json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
    except BaseException as error:
        if primary is None: raise
        primary.add_note(f'secondary evidence failure: {error}')


def need(ok, message):
    if not ok: raise ValueError(message)


def validate_policy(policy):
    """Bind the exact reviewed scalar values/types, including the FP32 dp budget."""
    expected=dict(element_rtol=.015625,element_atol=.00390625,rms_rtol=.015625,rms_atol=1e-5)
    need(type(policy) is dict and set(policy)==set(expected) and all(type(policy[k]) is float and policy[k]==v for k,v in expected.items()),'numerical policy drift/type')


SNAPSHOTS = ('before_baseline','after_baseline','before_candidate','after_candidate')


def keys(value, expected):
    """Require an exact plain JSON object schema."""
    need(type(value) is dict and set(value)==set(expected),'object schema')


def integer(value, minimum=0):
    """Reject boolean/inexact count aliases."""
    need(type(value) is int and value>=minimum,'integer type/range')


def validate_sha(value):
    """Validate a record SHA word without shadowing the byte digest helper."""
    need(type(value) is str and len(value)==64 and not set(value)-set('0123456789abcdef') and value!='0'*64,'SHA256')


def same(a,b):
    """Compare JSON values without numeric/boolean aliasing."""
    need(type(a) is type(b),'JSON type mismatch')
    if type(a) is dict:
        keys(a,b)
        for k in b: same(a[k],b[k])
    elif type(a) is list:
        need(len(a)==len(b),'JSON length mismatch')
        for x,y in zip(a,b): same(x,y)
    else: need(a==b,'JSON value mismatch')


def cache_snapshot(snapshot, rank):
    """Validate compiler snapshots without inventing warm stability."""
    keys(snapshot,('dynamo','generated_kernel_count','triton_keys'))
    keys(snapshot['dynamo'],('frames','stats','inductor','graph_break')); integer(snapshot['generated_kernel_count'])
    for group in snapshot['dynamo'].values():
        need(type(group) is dict and all(type(k) is str for k in group),'Dynamo mapping')
        for count in group.values(): integer(count)
    keys(snapshot['triton_keys'],('_fwd','_bwd'))
    for mapping in snapshot['triton_keys'].values():
        need(type(mapping) is dict,'helper cache mapping')
        for device,hashes in mapping.items():
            need(type(device) is str and device==str(rank) and type(hashes) is list,'helper device/list')
            for h in hashes: validate_sha(h)
            need(hashes==sorted(set(hashes)),'unsorted/duplicate helper keys')


def validate_metric(metric, count):
    """Require complete finite producer element gates and the unchanged RMS gate."""
    keys(metric,('finite','numel','checked','mismatches','max_abs','rms_error','reference_rms','raw_mismatches','max_ulp','passed'))
    need(metric['passed'] is True and metric_pass(metric),'failed metric')
    need(metric['numel']==metric['checked']==count,'metric element coverage')


def validate_case(entry, *, family, rank):
    """Return the identical fixed case record after strict proof/numerical validation."""
    need(type(family) is str and family in FAMILIES,'family'); integer(rank); need(rank<4,'rank')
    keys(entry,('rows','strided_upstream','metrics','host_hashes','device_pre','device_post','upstream','layouts','dispatch','compiler','launches'))
    rows=entry['rows']; integer(rows); need(rows in FAMILIES[family],'case rows')
    same(entry['strided_upstream'],family=='weighted' and rows==129)
    components=('y','dx_first','dx_second')+(('dp',) if family=='weighted' else ())
    keys(entry['metrics'],{f'{comparison}/{name}' for comparison in ('baseline/oracle','candidate/oracle','candidate/baseline') for name in components})
    for name,m in entry['metrics'].items(): validate_metric(m,rows if name.endswith('/dp') else rows*D)
    for field in ('host_hashes','device_pre','device_post'):
        need(type(entry[field]) is list and len(entry[field])==3,'three operand hashes')
        for h in entry[field]: validate_sha(h)
    same(entry['host_hashes'],entry['device_pre']); same(entry['host_hashes'],entry['device_post'])
    nominal=family=='plain' and rows==4096; strided=family=='weighted' and rows==129
    spec=lambda shape,stride,offset,dtype:dict(shape=shape,stride=stride,offset=offset,dtype=dtype,device=f'cuda:{rank}')
    source=spec([rows,1,3840] if nominal else [rows,3840],[3840,3840,1] if nominal else [3840,1],0,'torch.bfloat16')
    probability=spec([rows,1],[1,1],0,'torch.float32')
    upstream=spec([rows,1,D] if nominal else [rows,D],[D,D,1] if nominal else [3841,2] if strided else [D,1],1 if strided else 0,'torch.bfloat16')
    same(entry['layouts'],[source,probability,upstream]); same(entry['upstream'],{k:upstream[k] for k in ('shape','stride','offset')})
    same(entry['dispatch'],dict(baseline_original=1,candidate_original=0,candidate_helper=1))
    same(entry['launches'],dict.fromkeys(('_fwd','_bwd'),int(rows>0)))
    keys(entry['compiler'],SNAPSHOTS)
    for snapshot in entry['compiler'].values(): cache_snapshot(snapshot,rank)
    c=entry['compiler']; same(c['before_baseline']['triton_keys'],c['after_baseline']['triton_keys'])
    for arm in ('baseline','candidate'): same(c['before_'+arm]['dynamo']['graph_break'],c['after_'+arm]['dynamo']['graph_break'])
    if rows==0: same(c['before_candidate']['triton_keys'],c['after_candidate']['triton_keys'])
    else:
        for mapping in c['after_candidate']['triton_keys'].values(): need(set(mapping)=={str(rank)} and bool(mapping[str(rank)]),'missing candidate helper proof')
    return entry


def expected_adaptation(root):
    """Recompute export identity from embedded sources, without reading exported files."""
    exports={}
    for name,source in (('baseline',BEFORE),('default_off',APPLIED),('candidate',APPLIED)):
        adapted=source.replace('from megatron.core.jit import jit_fuser','from component_adapters import jit_fuser').replace('from megatron.core.utils import nvtx_decorator','from component_adapters import nvtx_decorator')
        exports[name]=dict(path=str(root/f'{name}.py'),sha256=sha(adapted.encode()))
    return dict(exports=exports,helper_path=str(root/'megatron/core/fusions/beverin_sssglu.py'),helper_sha256=ASSET_SHA['helper'],adapter=ADAPTER,scope='owned Torch2.14 default-compile and disabled-NVTX adaptation; not production MCore/checkpoint import proof')


def validate_worker(r, *, family, rank, expected_sha, scratch=None):
    """Close one successful worker including its source, dispatch and compiler evidence."""
    keys(r,'status proof_status rank family pid pgid driver_sha256 policy image source asset_sha256 started_ns finished_ns versions affinity local_device torch_git torch_hip adaptation compiler_suppression baseline_wrappers counter_sources default_off helper_observer cases fallbacks'.split())
    same(r['status'],'passed'); same(r['proof_status'],'passed'); same(r['rank'],rank); same(r['family'],family)
    for k in ('pid','pgid','started_ns','finished_ns'): integer(r[k],1)
    need(r['started_ns']<r['finished_ns'] and r['finished_ns']-r['started_ns']<=480*10**9,'worker interval')
    validate_sha(expected_sha); same(r['driver_sha256'],expected_sha); validate_policy(r['policy'])
    for k,value in dict(image=IMAGE,source=SOURCE,asset_sha256=ASSET_SHA,versions=VERSIONS,local_device=rank,torch_git=GIT,torch_hip=HIP,compiler_suppression=False).items(): same(r[k],value)
    a=r['affinity']; keys(a,'initial post_gpu_init final restoration_attempted restoration_applied errors'.split())
    need(type(a['initial']) is list and len(a['initial'])==48,'affinity width')
    for cpu in a['initial']: integer(cpu)
    need(a['initial']==sorted(set(a['initial'])),'affinity ordering'); same(a['final'],a['initial']); same(a['errors'],[])
    need(type(a['post_gpu_init']) is list and len(a['post_gpu_init'])>0,'post-init affinity')
    for cpu in a['post_gpu_init']: integer(cpu)
    need(a['post_gpu_init']==sorted(set(a['post_gpu_init'])),'post-init ordering')
    changed=a['post_gpu_init']!=a['initial']; same(a['restoration_attempted'],changed); same(a['restoration_applied'],changed)
    helper=Path(r['adaptation']['helper_path']); owned=helper.parents[3]
    need(owned.is_absolute() and '..' not in helper.parts and str(helper)==str(owned/'megatron/core/fusions/beverin_sssglu.py'),'absolute source path')
    if scratch is not None: same(str(owned),str(Path(scratch)/family/'component-source'))
    same(r['adaptation'],expected_adaptation(owned))
    same(r['helper_observer'],dict(installation='explicit owned import after default-off proof to install observers; not production lazy-import attestation',path=str(helper)))
    names=['sssglu','sssglu_back']+(['weighted_sssglu','weighted_sssglu_back'] if family=='weighted' else [])
    same(r['baseline_wrappers'],{name:dict(wrapper_type='builtins.function',original_path=str(owned/'baseline.py')) for name in names})
    same(r['counter_sources'],{name:dict(path=str(PACKAGE/name),bytes=size,sha256=checksum) for name,(size,checksum) in COUNTER_PINS.items()})
    off=r['default_off']; keys(off,'flag helper_loaded_before compiler_baseline_before compiler_baseline_after baseline_original_calls default_off_original_calls helper_loaded_after metric'.split())
    for name in ('flag','helper_loaded_before','helper_loaded_after'): same(off[name],False)
    for name in ('baseline_original_calls','default_off_original_calls'): same(off[name],1)
    before,after=off['compiler_baseline_before'],off['compiler_baseline_after']
    for snap in (before,after): cache_snapshot(snap,rank); same(snap['triton_keys'],{'_fwd':{},'_bwd':{}})
    for name in ('unique_graphs','calls_captured'): need(after['dynamo']['stats'].get(name,0)>before['dynamo']['stats'].get(name,0),'missing compiled baseline')
    need(after['generated_kernel_count']>before['generated_kernel_count'],'missing generated kernels'); same(before['dynamo']['graph_break'],after['dynamo']['graph_break']); validate_metric(off['metric'],7*D)
    need(type(r['cases']) is list and len(r['cases'])==4,'four ordered cases')
    same([c['rows'] for c in r['cases']],list(FAMILIES[family]))
    for c in r['cases']: validate_case(c,family=family,rank=rank)
    names=['input_stride','input_offset','dense']+(['prob_bf16','prob_offset'] if family=='weighted' else [])
    need(type(r['fallbacks']) is list,'fallback list'); same([f['name'] for f in r['fallbacks']],names)
    for f in r['fallbacks']:
        keys(f,'name original_calls helper_calls guard rows width metric'.split())
        for name,value in dict(original_calls=1,helper_calls=0,guard=False,rows=7,width=17280 if f['name']=='dense' else D).items(): same(f[name],value)
        validate_metric(f['metric'],7*f['width'])
    return r


def validate_rank(r, *, rank, expected_sha, expected_scratch=None):
    """Close ordered workers and bounded supervisor evidence without live state reads."""
    keys(r,'status rank pgid driver_sha256 image source policy started_ns finished_ns scratch cleanup workers logs'.split())
    same(r['status'],'passed'); same(r['rank'],rank); same(r['driver_sha256'],expected_sha); same(r['image'],IMAGE); same(r['source'],SOURCE); validate_policy(r['policy'])
    for name in ('pgid','started_ns','finished_ns'): integer(r[name],1)
    need(r['started_ns']<r['finished_ns'] and r['finished_ns']-r['started_ns']<=480*10**9,'rank deadline')
    scratch=r['scratch']; need(type(scratch) is str and scratch.startswith('/dev/shm/apertus-amd/') and '..' not in Path(scratch).parts and str(Path(scratch))==scratch,'dedicated scratch')
    need(re.fullmatch(r'sssglu-[0-9]+-'+str(rank)+r'-[^/]+',Path(scratch).name) is not None,'job/rank scratch anchor')
    if expected_scratch is not None: same(scratch,expected_scratch)
    same(r['cleanup'],'retained'); keys(r['logs'],FAMILIES); need(type(r['workers']) is list and len(r['workers'])==2,'two workers')
    previous=r['started_ns']; pids=set()
    for family,w in zip(FAMILIES,r['workers']):
        validate_worker(w,family=family,rank=rank,expected_sha=expected_sha,scratch=scratch)
        same(w['pgid'],r['pgid']); need(w['pid'] not in pids and previous<=w['started_ns'],'worker PID/order'); pids.add(w['pid']); previous=w['finished_ns']
        log=r['logs'][family]; keys(log,('sha256','bytes','warnings')); validate_sha(log['sha256']); integer(log['bytes']); need(log['bytes']<=8*2**20,'log size'); same(log['warnings'],[])
    need(previous<=r['finished_ns'],'worker outside supervisor interval')
    return r


def quiescent():
    """Boundedly reject live same-PGID descendants; do not delete their evidence."""
    for index,entry in enumerate(Path('/proc').iterdir()):
        if index>=8192: return False
        if not entry.name.isdigit() or int(entry.name)==os.getpid(): continue
        try: fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError): continue
        if int(fields[2])==os.getpgrp() and fields[0] not in {'Z','X'}: return False
    return True


def export_sources(directory):
    """Adapt exactly two imports; preserve all original math/class ASTs."""
    need(not any(n == 'megatron' or n.startswith('megatron.') for n in sys.modules), 'preexisting megatron modules')
    need('component_adapters' not in sys.modules, 'preexisting component adapter')
    directory.mkdir()
    for name in ('megatron','megatron/core','megatron/core/fusions'):
        path=directory/name; path.mkdir(); (path/'__init__.py').write_text('')
    result={}
    for name, source in (('baseline',BEFORE),('default_off',APPLIED),('candidate',APPLIED)):
        need(sha(source.encode())==ASSET_SHA['before' if name=='baseline' else 'applied'], 'embedded source hash')
        adapted=source.replace('from megatron.core.jit import jit_fuser','from component_adapters import jit_fuser').replace('from megatron.core.utils import nvtx_decorator','from component_adapters import nvtx_decorator')
        old,new=ast.parse(source),ast.parse(adapted)
        need([ast.dump(n) for n in old.body if not isinstance(n,ast.ImportFrom)]==[ast.dump(n) for n in new.body if not isinstance(n,ast.ImportFrom)],'source adaptation changed AST')
        path=directory/f'{name}.py'; path.write_text(adapted); result[name]=dict(path=str(path),sha256=sha(adapted.encode()))
    need(sha(HELPER.encode())==ASSET_SHA['helper'],'helper source hash')
    helper=directory/'megatron/core/fusions/beverin_sssglu.py'; helper.write_text(HELPER)
    (directory/'component_adapters.py').write_text(ADAPTER)
    return dict(exports=result,helper_path=str(helper),helper_sha256=ASSET_SHA['helper'],adapter=ADAPTER,
        scope='owned Torch2.14 default-compile and disabled-NVTX adaptation; not production MCore/checkpoint import proof')


def load_export(name, path, enabled):
    """Load unique code filenames with an explicit import-time flag, restoring its owner."""
    need(name not in sys.modules,'preexisting component module owner')
    previous=os.environ.get('BEVERIN_FUSED_SSSGLU')
    os.environ['BEVERIN_FUSED_SSSGLU']='1' if enabled else '0'
    module=None
    try:
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
        # Dynamo and source inspection must resolve the actual defining module.
        sys.modules[name]=module
        spec.loader.exec_module(module)
        need(Path(module.__file__).resolve()==Path(path).resolve(),'export module origin')
        return module
    except BaseException:
        if module is not None and sys.modules.get(name) is module: del sys.modules[name]
        raise
    finally:
        if previous is None: os.environ.pop('BEVERIN_FUSED_SSSGLU',None)
        else: os.environ['BEVERIN_FUSED_SSSGLU']=previous


@contextmanager
def observe_class(module, name, calls):
    """Observe one actual public Function.apply delegate; never replace arithmetic."""
    original=getattr(module,name)
    class Proxy:
        @staticmethod
        def apply(*args):
            calls[name]=calls.get(name,0)+1
            return original.apply(*args)
    setattr(module,name,Proxy)
    try: yield
    finally:
        if getattr(module,name) is Proxy: setattr(module,name,original)


def tensor_hash(torch, value):
    return sha(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def operands(torch, family, device):
    """Allocate all cases once with a fixed CPU RNG; preserve upstream strided storage."""
    generator=torch.Generator(device='cpu').manual_seed(123)
    result={}
    for rows in FAMILIES[family]:
        x=torch.randn((rows,2*D),dtype=torch.bfloat16,generator=generator)
        for col,value in enumerate((-80.,-1.,-0.,0.,1.,1.0078125,.99609375,80.,.001,-.001)):
            x[:,col::32]=value
        p=torch.rand((rows,1),dtype=torch.float32,generator=generator)
        p[::8]=0.; p[1::8]=-0.; p[2::8]=.25; p[3::8]=1.; p[4::8]=2.5
        strided=family=='weighted' and rows==129
        backing=torch.randn((rows,2*D+1 if strided else D),dtype=torch.bfloat16,generator=generator)
        g=backing[:,1::2] if strided else backing
        gpu_x=x.to(device); gpu_p=p.to(device); gpu_back=backing.to(device)
        gpu_g=gpu_back[:,1::2] if strided else gpu_back
        if family=='plain' and rows==4096:
            gpu_x=gpu_x.view(rows,1,2*D); gpu_g=gpu_g.view(rows,1,D)
        result[rows]=dict(host=(x,p,g),device=(gpu_x,gpu_p,gpu_g),strided_upstream=strided)
    return result


def analytic(torch, x, p, g, weighted):
    """Independent FP64 tile equations with final BF16 y/dx and FP32 dp stores."""
    a,b=x.double().split(D,dim=-1); upstream=g.double(); u=a-1; den=1+u.abs(); gate=.5+u/den
    probability=p.double() if weighted else 1
    result=dict(y=(gate*b*probability).to(torch.bfloat16),
        dx_first=(upstream*probability*b/(den*den)).to(torch.bfloat16),
        dx_second=(upstream*probability*gate).to(torch.bfloat16))
    if weighted: result['dp']=(gate*b*upstream).sum(-1,keepdim=True).to(torch.float32)
    return result


def metric_pass(metric):
    """Use the reviewed finite/full-element and independent RMS gates, including empties."""
    try:
        fields={'finite','numel','checked','mismatches','max_abs','rms_error','reference_rms','raw_mismatches','max_ulp','passed'}
        need(type(metric) is dict and set(metric)==fields,'metric fields')
        for key in ('numel','checked','mismatches','raw_mismatches','max_ulp'):
            need(type(metric[key]) is int and metric[key]>=0,'metric count type')
        for key in ('max_abs','rms_error','reference_rms'):
            need(type(metric[key]) in (int,float) and math.isfinite(metric[key]) and metric[key]>=0,'metric scalar')
        need(type(metric['passed']) is bool,'metric pass type')
        return (metric['finite'] is True and metric['checked']==metric['numel'] and metric['mismatches']==0
            and metric['raw_mismatches']<=metric['numel'] and metric['rms_error']<=POLICY['rms_atol']+POLICY['rms_rtol']*metric['reference_rms']
            and (metric['numel']>0 or metric['max_abs']==metric['rms_error']==metric['reference_rms']==metric['raw_mismatches']==metric['max_ulp']==0))
    except (KeyError,TypeError,ValueError): return False


def compare(torch, actual, reference, rows):
    """Reduce full CPU outputs in64-row tiles; raw bits/ordered ULP are diagnostic."""
    m=dict(finite=True,numel=actual.numel(),checked=0,mismatches=0,max_abs=0.,raw_mismatches=0,max_ulp=0,passed=False)
    se=sr=0.
    for start in range(0,rows,TILE):
        value=actual[start:start+TILE]; expected=reference(start,start+TILE)
        need(value.shape==expected.shape and value.dtype==expected.dtype,'comparison shape/dtype')
        a,b=value.double(),expected.double(); count=a.numel(); m['checked']+=count
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): m['finite']=False; continue
        error=(a-b).abs(); m['mismatches']+=int((error>POLICY['element_atol']+POLICY['element_rtol']*b.abs()).sum().item())
        m['max_abs']=max(m['max_abs'],error.max().item()); se+=error.square().sum().item(); sr+=b.square().sum().item()
        dtype=torch.int16 if value.dtype==torch.bfloat16 else torch.int32; sign=-(2**15 if dtype==torch.int16 else 2**31)
        words=[t.contiguous().view(dtype).long() for t in (value,expected)]
        m['raw_mismatches']+=int((words[0]!=words[1]).sum().item())
        ordered=[torch.where(w<0,sign-w,w) for w in words]; m['max_ulp']=max(m['max_ulp'],int((ordered[0]-ordered[1]).abs().max().item()))
    m.update(rms_error=math.sqrt(se/m['numel']) if m['numel'] else 0.,reference_rms=math.sqrt(sr/m['numel']) if m['numel'] else 0.)
    m['passed']=metric_pass(m); return m


def evaluate(torch, module, family, values):
    """Use the actual applied public call site and native backward, retaining complete CPU outputs."""
    source,probability,upstream=values
    need(type(source) is torch.Tensor and type(probability) is torch.Tensor and type(upstream) is torch.Tensor,'ordinary operand tensors')
    shape=tuple(source.shape[:-1])+(D,)
    need(source.dtype==upstream.dtype==torch.bfloat16 and probability.dtype==torch.float32,'operand dtypes')
    need(source.shape[-1]==2*D and tuple(upstream.shape)==shape and probability.shape==(source.numel()//(2*D),1),'operand shapes')
    need(source.is_cuda and probability.device==source.device==upstream.device and source.device.index==torch.cuda.current_device(),'operand device ownership')
    need(source.is_contiguous() and source.storage_offset()==0 and probability.is_contiguous() and probability.storage_offset()==0,'admitted storage')
    need(bool(torch.isfinite(source).all() and torch.isfinite(probability).all() and torch.isfinite(upstream).all()),'finite operands')
    x=source.detach().requires_grad_(); p=probability.detach().requires_grad_()
    if family=='weighted': y=module.weighted_bias_sssglu_impl(x,None,p)
    else: y=module.bias_sssglu_impl(x,None)
    need(tuple(y.shape)==shape and y.dtype==torch.bfloat16 and y.device==source.device,'native output shape/dtype/device')
    y.backward(upstream)
    need(x.grad is not None and (family=='plain' or p.grad is not None),'missing native gradients')
    need(x.grad.shape==x.shape and x.grad.dtype==torch.bfloat16 and x.grad.device==source.device,'native dx contract')
    if family=='weighted': need(p.grad.shape==p.shape and p.grad.dtype==torch.float32 and p.grad.device==source.device,'native dp contract')
    dx=x.grad.detach().reshape(-1,2*D).cpu()
    result=dict(y=y.detach().reshape(-1,D).cpu(),dx_first=dx[:,:D],dx_second=dx[:,D:])
    if family=='weighted': result['dp']=p.grad.detach().cpu()
    return result


def numerical_cases(torch, family, cases, baseline, candidate, record, proof=None):
    """Compare fixed native operands across two arms and independent full tiled references."""
    validate_policy(POLICY); record['cases']=[]
    for rows,case in cases.items():
        entry=dict(rows=rows,strided_upstream=case['strided_upstream'],metrics={}); record['cases'].append(entry)
        entry['host_hashes']=[tensor_hash(torch,v) for v in case['host']]
        entry['device_pre']=[tensor_hash(torch,v) for v in case['device']]
        need(entry['host_hashes']==entry['device_pre'],'resident pre operand identity')
        entry['upstream']=dict(shape=list(case['device'][2].shape),stride=list(case['device'][2].stride()),offset=case['device'][2].storage_offset())
        entry['layouts']=[dict(shape=list(v.shape),stride=list(v.stride()),offset=v.storage_offset(),dtype=str(v.dtype),device=str(v.device)) for v in case['device']]
        results={arm:(evaluate(torch,module,family,case['device']) if proof is None else proof(module,case['device'],entry,arm)) for arm,module in (('baseline',baseline),('candidate',candidate))}
        host_x,host_p,host_g=case['host']
        for name in results['baseline']:
            for arm in ('baseline','candidate'):
                def oracle(start,end,key=name):
                    return analytic(torch,host_x[start:end],host_p[start:end],host_g[start:end],family=='weighted')[key]
                entry['metrics'][arm+'/oracle/'+name]=compare(torch,results[arm][name],oracle,rows)
            entry['metrics']['candidate/baseline/'+name]=compare(torch,results['candidate'][name],lambda s,e:results['baseline'][name][s:e],rows)
        entry['device_post']=[tensor_hash(torch,v) for v in case['device']]
        need(entry['device_post']==entry['device_pre'],'native operands mutated')
        need(all(m['passed'] is True and metric_pass(m) for m in entry['metrics'].values()),'full numerical rejection')


def compiler_reader(record):
    """Verify exact installed source origins and read existing counters without resets."""
    modules=[importlib.import_module(name) for name in ('torch._dynamo.utils','torch._inductor.metrics','triton.runtime.jit')]
    record['counter_sources']={}
    for module,(relative,(size,checksum)) in zip(modules,COUNTER_PINS.items()):
        path=PACKAGE/relative
        need(Path(module.__file__).resolve()==path and path.resolve()==path and path.stat().st_size==size and sha(path.read_bytes())==checksum,'counter source identity')
        record['counter_sources'][relative]=dict(path=str(path),bytes=size,sha256=checksum)
    du,im,jit=modules
    def snapshot():
        need(isinstance(du.counters,Mapping) and type(im.generated_kernel_count) is int and im.generated_kernel_count>=0,'compiler counter ABI')
        groups={}
        for name in ('frames','stats','inductor','graph_break'):
            values=du.counters.get(name,{})
            need(isinstance(values,Mapping) and all(type(k) is str and type(v) is int and v>=0 for k,v in values.items()),'compiler words')
            groups[name]=dict(values)
        return dict(dynamo=groups,generated_kernel_count=im.generated_kernel_count,triton_keys={'_fwd':{},'_bwd':{}})
    return snapshot,jit


def bind_helper(snapshot, jit, helper, path):
    """Bind exact owned JIT functions once; snapshots remain valid during launch observers."""
    need(Path(helper.__file__).resolve()==path.resolve() and sha(path.read_bytes())==ASSET_SHA['helper'],'owned helper origin')
    need(Path(inspect.getfile(jit.JITFunction)).resolve()==PACKAGE/'triton/runtime/jit.py','JIT class origin')
    functions={name:getattr(helper,name) for name in ('_fwd','_bwd')}
    for function in functions.values():
        need(type(function) is jit.JITFunction and Path(inspect.getfile(function.fn)).resolve()==path.resolve(),'helper JIT ABI/origin')
    def observed():
        result=snapshot()
        for name,function in functions.items():
            caches=vars(function).get('device_caches'); need(isinstance(caches,Mapping),'helper cache mapping')
            for device,value in caches.items():
                need(isinstance(value,tuple) and value and isinstance(value[0],Mapping),'helper cache tuple')
                result['triton_keys'][name][str(device)]=sorted(sha(repr(key).encode()) for key in value[0])
        return result
    return observed


def baseline_wrappers(module, family):
    """Require default torch.compile wrappers with original callables in the owned export."""
    names=['sssglu','sssglu_back']+(['weighted_sssglu','weighted_sssglu_back'] if family=='weighted' else [])
    result={}
    for name in names:
        function=getattr(module,name); original=getattr(function,'_torchdynamo_orig_callable',None)
        need(callable(original) and Path(inspect.getfile(original)).resolve()==Path(module.__file__).resolve(),'baseline not default compiled owned source')
        result[name]=dict(wrapper_type=f'{type(function).__module__}.{type(function).__qualname__}',original_path=inspect.getfile(original))
    return result


def public_forward(module, family, x, p):
    """Call the actual unchanged/applied module entry, not a helper-only substitute."""
    return module.weighted_bias_sssglu_impl(x,None,p) if family=='weighted' else module.bias_sssglu_impl(x,None)


def tiny_inputs(torch, rank, name='ordinary'):
    """Build tiny unsupported storage as views; never copy it into candidate eligibility."""
    width=17280 if name=='dense' else D
    generator=torch.Generator(device='cpu').manual_seed(7)
    columns=4*width if name=='input_stride' else 2*width+(name=='input_offset')
    x=torch.randn((7,columns),dtype=torch.bfloat16,generator=generator).to(f'cuda:{rank}')
    if name=='input_stride': x=x[:,::2]
    if name=='input_offset': x=x[:,1:]
    p=torch.ones((7,2 if name=='prob_offset' else 1),dtype=torch.bfloat16 if name=='prob_bf16' else torch.float32,device=x.device)
    if name=='prob_offset': p=p[:,1:]
    return x,p,width


def prove_default_off(torch, family, baseline, default_off, snapshot, record, rank):
    """Prove original public default-off selection before importing the helper observer."""
    proof=dict(flag=default_off._BEVERIN_FUSED_SSSGLU,helper_loaded_before=HELPER_MODULE in sys.modules)
    record['default_off']=proof
    need(proof['flag'] is False and not proof['helper_loaded_before'],'default-off state/helper already loaded')
    x,p,width=tiny_inputs(torch,rank); name='WeightedSSSGLUFunction' if family=='weighted' else 'SSSGLUFunction'
    baseline_calls={}; off_calls={}; proof['compiler_baseline_before']=snapshot()
    with observe_class(baseline,name,baseline_calls): reference=public_forward(baseline,family,x,p)
    proof['compiler_baseline_after']=snapshot()
    with observe_class(default_off,name,off_calls): actual=public_forward(default_off,family,x,p)
    proof.update(baseline_original_calls=baseline_calls.get(name,0),default_off_original_calls=off_calls.get(name,0),helper_loaded_after=HELPER_MODULE in sys.modules)
    before,after=proof['compiler_baseline_before'],proof['compiler_baseline_after']
    need(all(after['dynamo']['stats'].get(k,0)>before['dynamo']['stats'].get(k,0) for k in ('unique_graphs','calls_captured')) and after['generated_kernel_count']>before['generated_kernel_count'],'missing compiled baseline work')
    need(before['dynamo']['graph_break']==after['dynamo']['graph_break'],'baseline graph break')
    proof['metric']=compare(torch,actual.detach().cpu(),lambda s,e:reference.detach().cpu()[s:e],7)
    need(proof['baseline_original_calls']==proof['default_off_original_calls']==1 and proof['helper_loaded_after'] is False and proof['metric']['passed'],'default-off proof')


@contextmanager
def observe_launches(helper, counts):
    """Count public JIT launch subscriptions without changing functions, configs or caches."""
    originals={name:getattr(helper,name) for name in ('_fwd','_bwd')}; proxies={}
    try:
        for name,original in originals.items():
            class Proxy:
                def __init__(self, key, function): self.key,self.function=key,function
                def __getitem__(self, grid):
                    counts[self.key]=counts.get(self.key,0)+1
                    return self.function[grid]
            proxies[name]=Proxy(name,original); setattr(helper,name,proxies[name])
        yield
    finally:
        for name,proxy in proxies.items():
            if getattr(helper,name) is proxy: setattr(helper,name,originals[name])


def evaluate_proven(torch,module,family,values,entry,arm,snapshot,helper,rank):
    """Attach per-arm actual dispatch/cache evidence; accuracy may compile new shapes."""
    original='WeightedSSSGLUFunction' if family=='weighted' else 'SSSGLUFunction'
    selected='TritonWeightedSSSGLU' if family=='weighted' else 'TritonSSSGLU'
    dispatch=entry.setdefault('dispatch',dict(baseline_original=0,candidate_original=0,candidate_helper=0))
    compiler=entry.setdefault('compiler',{}); compiler['before_'+arm]=snapshot()
    ordinary_calls={}; helper_calls={}; launches={}
    try:
        with observe_class(module,original,ordinary_calls),observe_class(helper,selected,helper_calls),observe_launches(helper,launches):
            result=evaluate(torch,module,family,values)
    except BaseException as error:
        dispatch[arm+'_original']=ordinary_calls.get(original,0)
        if arm=='candidate': dispatch['candidate_helper']=helper_calls.get(selected,0)
        entry['proof_error']=f'{type(error).__name__}: {error}'
        raise
    compiler['after_'+arm]=snapshot()
    dispatch[arm+'_original']=ordinary_calls.get(original,0)
    if arm=='candidate': dispatch['candidate_helper']=helper_calls.get(selected,0); entry['launches']={k:launches.get(k,0) for k in ('_fwd','_bwd')}
    before,after=compiler['before_'+arm],compiler['after_'+arm]
    need(before['dynamo']['graph_break']==after['dynamo']['graph_break'],'native graph break')
    if arm=='baseline': need(dispatch['baseline_original']==1 and not helper_calls and not launches and before['triton_keys']==after['triton_keys'],'baseline dispatched helper')
    else:
        need(dispatch['candidate_original']==0 and dispatch['candidate_helper']==1,'candidate public guard missed')
        if entry['rows']==0: need(not launches and before['triton_keys']==after['triton_keys'],'empty helper launched or keys grew')
        else:
            need(all(set(v)=={str(rank)} and v[str(rank)] for v in after['triton_keys'].values()),'nonempty helper current-device cache')
            need(entry['launches']=={'_fwd':1,'_bwd':1},'candidate launch coverage')
    return result


def prove_fallbacks(torch,family,baseline,candidate,helper,rank,record):
    """Observe unsupported public forward paths without forcing eligibility or backward."""
    record['fallbacks']=[]; original='WeightedSSSGLUFunction' if family=='weighted' else 'SSSGLUFunction'; selected='TritonWeightedSSSGLU' if family=='weighted' else 'TritonSSSGLU'
    for name in ('input_stride','input_offset','dense')+(('prob_bf16','prob_offset') if family=='weighted' else ()):
        x,p,width=tiny_inputs(torch,rank,name); ordinary_calls={}; helper_calls={}
        entry=dict(name=name,original_calls=0,helper_calls=0,guard=candidate._beverin_sssglu_eligible(x,p) if family=='weighted' else candidate._beverin_sssglu_eligible(x),rows=7,width=width)
        record['fallbacks'].append(entry); need(entry['guard'] is False,'unsupported case admitted')
        with observe_class(candidate,original,ordinary_calls),observe_class(helper,selected,helper_calls): actual=public_forward(candidate,family,x,p)
        reference=public_forward(baseline,family,x,p)
        entry.update(original_calls=ordinary_calls.get(original,0),helper_calls=helper_calls.get(selected,0),metric=compare(torch,actual.detach().cpu(),lambda s,e:reference.detach().cpu()[s:e],7))
        need(entry['original_calls']==1 and entry['helper_calls']==0 and entry['metric']['passed'],'fallback dispatch/numerics')


def audit_log(path):
    """Read only a closed family log after reap; fail on compiler fallback/limit warnings."""
    need(path.stat().st_size<=8*2**20,'native log exceeds8MiB')
    with path.open('rb') as stream: raw=stream.read(8*2**20+1)
    need(len(raw)<=8*2**20,'native log exceeds8MiB')
    warnings=re.findall(r'recompile_limit|cache_size_limit|won\x27t\s*convert|falling back to eager',raw.decode(errors='replace'),re.I)
    return dict(sha256=sha(raw),bytes=len(raw),warnings=warnings)


def validate_launch(settings, env):
    """Admit only the proposed four-rank600/510/480 envelope; no allocation is authorized."""
    resource=settings['resources']; procedure=settings['procedure']; rank=int(env['SLURM_PROCID'])
    wanted=dict(nodes=1,gpus_per_node=4,cpus_per_node=192,mpi='pmix',time='00:10:00',reservation='cscs',cpu_bind='map_ldom:0,1,2,3',distribution='block:block')
    need(settings['image']['sha256']==IMAGE and all(resource.get(k)==v for k,v in wanted.items()) and resource.get('cpus_per_task') is None,'resource contract')
    need(procedure['kind']=='commands' and procedure['timeout']==510 and procedure['commands']==[['python3','-u','/apertus/validation.py','run']],'command contract')
    need(0<=rank==int(env['SLURM_LOCALID'])<4 and env['SLURM_NODEID']=='0' and env['SLURM_NTASKS']=='4' and len(os.sched_getaffinity(0))==48,'rank/affinity')
    need(all(env.get(k)==v for k,v in dict(OMP_NUM_THREADS='1',MAX_JOBS='2',TORCH_COMPILE_DISABLE='0').items()) and env.get('TORCHDYNAMO_DISABLE','0')=='0','default compilation contract')
    return rank


def wait_worker(child, deadline):
    """Reserve ten seconds inside480for direct stop/reap; outer runtime owns descendants."""
    try:
        remaining=deadline-time.monotonic()-10; need(remaining>0,'no shutdown reserve')
        return child.wait(timeout=remaining)
    except BaseException as primary:
        try:
            try: child.terminate()
            except ProcessLookupError: pass
            try: child.wait(timeout=max(0,min(5,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                try: child.kill()
                except ProcessLookupError: pass
                child.wait(timeout=max(0,min(5,deadline-time.monotonic())))
        except BaseException as secondary: primary.add_note(f'child teardown: {secondary}')
        raise


def worker(directory, family, rank):
    """Execute retained component numerical work, never claim missing selection/compiler proof."""
    from apertus_amd.runtime import preserve_affinity,require_affinity
    record=dict(status='unqualified',proof_status='incomplete',rank=rank,family=family,pid=os.getpid(),pgid=os.getpgrp(),driver_sha256=sha(Path(__file__).read_bytes()),policy=POLICY,image=IMAGE,source=SOURCE,asset_sha256=dict(ASSET_SHA),started_ns=time.monotonic_ns(),cases=[])
    try:
        record['versions']={k:importlib.metadata.version(k) for k in VERSIONS}; need(record['versions']==VERSIONS,'native versions')
        with preserve_affinity(record.setdefault('affinity',{})) as initial:
            import torch
            need(torch.cuda.device_count()==4 and bool(torch.version.hip),'four ROCm devices')
            torch.cuda.set_device(rank); need(torch.cuda.current_device()==rank and torch.cuda.get_device_properties(rank).gcnArchName.startswith('gfx942'),'local device')
            record.update(local_device=torch.cuda.current_device(),torch_git=torch.version.git_version,torch_hip=torch.version.hip)
            need(record['torch_git']==GIT and record['torch_hip']==HIP,'Torch source/HIP identity')
            torch.set_num_threads(1)
        require_affinity(initial,'SSSGLU initialization')
        owned=Path(os.environ['TMPDIR']).parent/'component-source'; record['adaptation']=export_sources(owned); sys.path.insert(0,str(owned))
        modules={name:load_export(f'sssglu_{name}_{family}',owned/f'{name}.py',name=='candidate') for name in ('baseline','default_off','candidate')}
        record['compiler_suppression']=torch._dynamo.config.suppress_errors
        need(record['compiler_suppression'] is False,'compiler error suppression')
        record['baseline_wrappers']=baseline_wrappers(modules['baseline'],family)
        snapshot,jit=compiler_reader(record)
        prove_default_off(torch,family,modules['baseline'],modules['default_off'],snapshot,record,rank)
        helper=importlib.import_module(HELPER_MODULE)
        record['helper_observer']=dict(installation='explicit owned import after default-off proof to install observers; not production lazy-import attestation',path=str(Path(helper.__file__).resolve()))
        snapshot=bind_helper(snapshot,jit,helper,Path(record['adaptation']['helper_path']))
        values=operands(torch,family,f'cuda:{rank}')
        def proof(module,values,entry,arm): return evaluate_proven(torch,module,family,values,entry,arm,snapshot,helper,rank)
        numerical_cases(torch,family,values,modules['baseline'],modules['candidate'],record,proof=proof)
        prove_fallbacks(torch,family,modules['baseline'],modules['candidate'],helper,rank,record)
        require_affinity(initial,'SSSGLU completion')
        record.update(status='passed',proof_status='passed',finished_ns=time.monotonic_ns())
        validate_worker(record,family=family,rank=rank,expected_sha=sha(Path(__file__).read_bytes()),scratch=str(owned.parent.parent))
    except BaseException as error:
        record.update(status='unqualified',proof_status='failed',error=f'{type(error).__name__}: {error}'); raise
    finally:
        if record.get('status')!='passed': record['finished_ns']=time.monotonic_ns()
        save(directory/'worker.json',record)


def run(settings):
    """Retain two fresh family workers/caches in the single outer PGID and deadline."""
    from apertus_amd.config import CACHE_PATHS
    validate_settings(settings,sha(Path(__file__).read_bytes()))
    rank=validate_launch(settings,os.environ); output=Path(f'sssglu-{rank}').resolve(); output.mkdir()
    root=Path(settings['scratch_root']); root.mkdir(parents=True,exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix=f'sssglu-{os.environ["SLURM_JOB_ID"]}-{rank}-',dir=root))
    record=dict(status='unqualified',rank=rank,pgid=os.getpgrp(),driver_sha256=sha(Path(__file__).read_bytes()),image=IMAGE,source=SOURCE,policy=POLICY,started_ns=time.monotonic_ns(),scratch=str(scratch),cleanup='retained',workers=[],logs={}); deadline=time.monotonic()+480
    try:
        for family in FAMILIES:
            directory=output/family; directory.mkdir(); env=dict(os.environ)
            for key,relative in CACHE_PATHS.items():
                path=scratch/family/relative; path.mkdir(parents=True,exist_ok=True); env[key]=str(path)
            need(deadline-time.monotonic()>10,'no worker shutdown reserve')
            with (directory/'native.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--directory',str(directory),'--family',family],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
                status=wait_worker(child,deadline)
            value=json.loads((directory/'worker.json').read_text()); record['workers'].append(value)
            same(value.get('pid'),child.pid); same(value.get('pgid'),os.getpgrp()); same(value.get('rank'),rank)
            need(type(status) is int and status==0 and value.get('status')=='passed' and value.get('family')==family and value.get('driver_sha256')==record['driver_sha256'],'closed family worker failed')
            need(time.monotonic()<=deadline and quiescent(),'worker descendants/deadline; scratch retained')
            record['logs'][family]=audit_log(directory/'native.log')
            need(not record['logs'][family]['warnings'],'native compiler fallback/recompile warning')
            validate_worker(value,family=family,rank=rank,expected_sha=record['driver_sha256'],scratch=str(scratch))
        need(time.monotonic()<=deadline and quiescent(),'deadline/quiescence; scratch retained')
        record.update(status='passed',finished_ns=time.monotonic_ns())
        validate_rank(record,rank=rank,expected_sha=record['driver_sha256'],expected_scratch=str(scratch))
    except BaseException as error: record.update(status='unqualified',error=f'{type(error).__name__}: {error}'); raise
    finally:
        if record.get('status')!='passed': record['finished_ns']=time.monotonic_ns()
        save(output/'summary.json',record)


def check(directory, target=None):
    """Audit closed four-rank evidence; optionally write an exclusive external derivative."""
    directory=Path(directory).resolve(); actual_sha=sha((directory/'validation.py').read_bytes())
    same(actual_sha,sha(Path(__file__).read_bytes()))
    settings=json.loads((directory/'resolved.json').read_text()); validate_settings(settings,actual_sha)
    pids=set(); hashes={}; job=None
    for rank in range(4):
        base=directory/f'output/0/sssglu-{rank}'; r=json.loads((base/'summary.json').read_text())
        validate_rank(r,rank=rank,expected_sha=actual_sha)
        current=Path(r['scratch']).name.split('-')[1]
        if job is None: job=current
        same(current,job)
        for family,w in zip(FAMILIES,r['workers']):
            same(json.loads((base/family/'worker.json').read_text()),w)
            same(audit_log(base/family/'native.log'),r['logs'][family])
            need(w['pid'] not in pids,'duplicate global PID'); pids.add(w['pid'])
            for c in w['cases']:
                key=(family,c['rows'])
                if key not in hashes: hashes[key]=c['host_hashes']
                same(c['host_hashes'],hashes[key])
        command=json.loads((directory/f'output/0/rank-{rank}.json').read_text())
        same(command['status'],'passed'); same(command['rank'],rank); same(command['argv'],['python3','-u','/apertus/validation.py','run'])
    result=json.loads((directory/'result.json').read_text())
    same(result['status'],'passed'); same(result['execution'],'passed')
    need(type(result['commands']) is list and len(result['commands'])==1,'one runtime command')
    step=result['commands'][0]; same(step['argv'],['python3','-u','/apertus/validation.py','run']); same(step['returncode'],0)
    need(type(step['ranks']) is list and len(step['ranks'])==4,'four runtime ranks')
    for rank,command in enumerate(step['ranks']):
        same(command,json.loads((directory/f'output/0/rank-{rank}.json').read_text()))
    need(len(pids)==8,'eight unique PIDs')
    output=dict(status='qualified',driver_sha256=actual_sha,ranks=4,workers=8,cases=32,metrics=336,scope='isolated SSSGLU component record closure; not production/performance qualification')
    if target is not None:
        target=Path(target).resolve(); need(not target.is_relative_to(directory),'derived target inside authoritative run'); save(target,output)
    return output


def validate_settings(settings, expected_sha):
    """Bind frozen source and exact resource/command contract without Slurm state."""
    same(settings['validation']['sha256'],expected_sha); same(settings['image']['sha256'],IMAGE)
    wanted=dict(nodes=1,gpus_per_node=4,cpus_per_node=192,mpi='pmix',time='00:10:00',reservation='cscs',cpu_bind='map_ldom:0,1,2,3',distribution='block:block')
    for k,v in wanted.items(): same(settings['resources'][k],v)
    same(settings['resources'].get('cpus_per_task'),None)
    for k,v in dict(kind='commands',timeout=510,commands=[['python3','-u','/apertus/validation.py','run']]).items(): same(settings['procedure'][k],v)
    for k,v in dict(OMP_NUM_THREADS='1',MAX_JOBS='2',TORCH_COMPILE_DISABLE='0').items(): same(settings['environment'][k],v)
    same(settings['environment'].get('TORCHDYNAMO_DISABLE','0'),'0')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=('run','worker','check')); parser.add_argument('--directory',type=Path); parser.add_argument('--family',choices=tuple(FAMILIES)); parser.add_argument('--derived-destination',type=Path); args=parser.parse_args()
    if args.mode=='check':
        if args.directory is None or args.family is not None: parser.error('check requires only a run directory')
        print(json.dumps(check(args.directory,args.derived_destination),indent=2)); return
    if not IMPLEMENTATION_READY: raise SystemExit('PARTIAL CPU IMPLEMENTATION; native entry disabled: '+ '; '.join(BLOCKERS))
    if args.derived_destination is not None: parser.error('derived destination is check-only')
    if args.mode=='worker' and (args.directory is None or args.family is None): parser.error('worker requires directory/family')
    if args.mode=='run' and (args.directory is not None or args.family is not None): parser.error('run takes no worker arguments')
    settings=json.loads(Path('/apertus/resolved.json').read_text()); rank=validate_launch(settings,os.environ)
    need(settings['validation']['sha256']==sha(Path(__file__).read_bytes()),'executed validation source SHA')
    if args.mode=='run': run(settings)
    elif args.mode=='worker': worker(args.directory,args.family,rank)
    else: check(args.directory)


if __name__=='__main__': main()
