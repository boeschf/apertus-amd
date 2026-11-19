"""Owned git-apply fixtures and extracted AST only; never import vendor code."""

import ast
import builtins
import hashlib
import inspect
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "experiments/patches/chonk-te-gemm-workspace-e242.patch"
TARGET = Path("megatron/core/extensions/transformer_engine.py")
REFERENCE_SHA256 = "fca82c235734d6dcd967db00db34100ab23b29a4d88aa08d60f42be3afdafd71"
# Exact e242 lines2762–2809, not executable vendor imports or an alternate implementation.
BASE_EXCERPT = '''try:
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    try:
        from transformer_engine.pytorch.module.base import get_workspace

        _get_workspace = get_workspace
    except ImportError:
        _get_workspace = None

    def te_general_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: Optional[torch.dtype] = None,
        layout: str = "TN",
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        grad: bool = False,
    ) -> List[torch.Tensor]:
        """
        Wrapper for TE's general_gemm function.
        It supports fp32, bf16, fp16, and fp8 GEMMs with TN, NN, and NT layouts.
        The output dtype can be specified by `out_dtype`.
        Note: not all combinations of these settings are supported. If not supported,
        cublaslt will throw an error.
        """
        kwargs = dict(
            out_dtype=out_dtype,
            quantization_params=None,
            gelu=None,
            gelu_in=None,
            accumulate=False,
            layout=layout,
            out=out,
            bias=bias,
            use_split_accumulator=False,
            grad=grad,
            ub=None,
            ub_type=None,
            extra_output=None,
            bulk_overlap=False,
        )
        if _get_workspace is not None:
            kwargs["workspace"] = _get_workspace()
        return general_gemm(A, B, **kwargs)

except ImportError:
    te_general_gemm = None  # type: ignore[assignment, misc]
'''


@pytest.fixture
def patched(tmp_path):
    """Apply solely inside pytest's newly owned, non-checkout source fixture."""
    target = tmp_path / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(BASE_EXCERPT)
    for argv in (["git", "apply", "--check", str(PATCH)], ["git", "apply", str(PATCH)]):
        subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, check=True)
    return ast.parse(target.read_text())


def wrapper(tree, native, workspace):
    """Execute only the patched function plus stdlib inspect, with deferred annotations."""
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "te_general_gemm")
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
    assert len(imports) == 1 and ast.unparse(imports[0]) == "import inspect as _inspect"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    selected = ast.fix_missing_locations(ast.Module(body=[future, imports[0], function], type_ignores=[]))
    namespace = {"general_gemm": native, "_get_workspace": workspace}
    exec(compile(selected, "<owned-patched-wrapper>", "exec"), namespace)
    return namespace["te_general_gemm"], namespace


def expected(dtype, out, bias, *, layout="NN", grad=True):
    return dict(out_dtype=dtype, quantization_params=None, gelu=None, gelu_in=None,
                accumulate=False, layout=layout, out=out, bias=bias, use_split_accumulator=False,
                grad=grad, ub=None, ub_type=None, extra_output=None, bulk_overlap=False)


def test_reference_bytes_and_patch_scope(patched):
    assert hashlib.sha256(PATCH.read_bytes()).hexdigest() == REFERENCE_SHA256
    assert PATCH.read_text().count("diff --git ") == 1
    function = next(n for n in ast.walk(patched) if isinstance(n, ast.FunctionDef))
    condition = next(n for n in function.body if isinstance(n, ast.If))
    assert ast.unparse(condition.test) == (
        "_get_workspace is not None and 'workspace' in _inspect.signature(general_gemm).parameters")


@pytest.mark.parametrize("has_parameter", [True, False])
@pytest.mark.parametrize("has_getter", [True, False])
def test_forwarding_once_supported_and_unsupported_workspace(patched, has_parameter, has_getter):
    calls, gets, inspections = [], [], []
    result, ws = object(), object()
    def with_workspace(A, B, *, workspace=None, **kwargs):
        calls.append((A, B, kwargs, workspace))
        return result
    def without_workspace(A, B, **kwargs):
        calls.append((A, B, kwargs, None))
        return result
    def getter():
        gets.append(1)
        return ws
    native = with_workspace if has_parameter else without_workspace
    call, namespace = wrapper(patched, native, getter if has_getter else None)
    def signature(value):
        inspections.append(value)
        return inspect.signature(value)
    namespace["_inspect"] = SimpleNamespace(signature=signature)
    A, B, dtype, out, bias = (object() for _ in range(5))
    assert call(A, B, out_dtype=dtype, layout="NN", out=out, bias=bias, grad=True) is result
    assert calls == [(A, B, expected(dtype, out, bias), ws if has_getter and has_parameter else None)]
    assert gets == ([1] if has_getter and has_parameter else [])
    assert inspections == ([native] if has_getter else [])


def test_defaults_and_identity_no_vendor_import(patched, monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformer_engine", "triton", "megatron"}:
            pytest.fail("vendor import attempted")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    calls, result = [], object()
    def native(A, B, **kwargs):
        calls.append(kwargs)
        return result
    call, _ = wrapper(patched, native, None)
    assert call(object(), object()) is result
    assert calls == [expected(None, None, None, layout="TN", grad=False)]


@pytest.mark.parametrize("stage", ["native", "getter", "signature"])
def test_original_errors_not_swallowed_or_retried(patched, stage):
    error, events = RuntimeError("original failure " + stage), []
    def native(A, B, *, workspace=None, **kwargs):
        events.append("native")
        raise error
    def getter():
        events.append("getter")
        if stage == "getter":
            raise error
        return object()
    call, namespace = wrapper(patched, native, getter)
    def signature(value):
        events.append("signature")
        if stage == "signature":
            raise error
        return inspect.signature(value)
    namespace["_inspect"] = SimpleNamespace(signature=signature)
    with pytest.raises(RuntimeError) as caught:
        call(object(), object())
    assert caught.value is error
    assert events == {"native": ["signature", "getter", "native"],
                      "getter": ["signature", "getter"], "signature": ["signature"]}[stage]
