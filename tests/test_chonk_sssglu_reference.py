"""Exercise actual oracle/reducer logic with tiny stdlib-only tensor stand-ins.

These fixtures verify equations, casts and complete tiling, not native floating
point behavior or GPU correctness.
"""
import importlib.util
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest


@pytest.fixture
def p():
    path = Path(__file__).resolve().parents[1] / "experiments/validation/chonk_sssglu_qualification.py"
    spec = importlib.util.spec_from_file_location("sssglu_reference_fixture", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Matrix:
    """Small unrounded algebra, recording only the requested final store dtype."""
    def __init__(self, rows, dtype=None): self.rows, self.dtype = rows, dtype
    def double(self): return self
    def to(self, dtype): return Matrix(self.rows, dtype)
    def binary(self, other, operation):
        return Matrix([[operation(x, other.rows[i][j if len(other.rows[i]) > 1 else 0] if isinstance(other, Matrix) else other)
                        for j, x in enumerate(row)] for i, row in enumerate(self.rows)])
    def __add__(self, other): return self.binary(other, lambda a,b:a+b)
    __radd__ = __add__
    def __sub__(self, other): return self.binary(other, lambda a,b:a-b)
    def __mul__(self, other): return self.binary(other, lambda a,b:a*b)
    def __truediv__(self, other): return self.binary(other, lambda a,b:a/b)
    def abs(self): return Matrix([[abs(x) for x in row] for row in self.rows])
    def split(self, width, dim):
        assert dim == -1 and all(len(row) == 2*width for row in self.rows)
        return Matrix([row[:width] for row in self.rows]), Matrix([row[width:] for row in self.rows])
    def sum(self, dim, keepdim):
        assert dim == -1 and keepdim
        return Matrix([[sum(row)] for row in self.rows])


@pytest.mark.parametrize("weighted", [False, True])
def test_actual_analytic_equations_and_final_store_contract(p, monkeypatch, weighted):
    monkeypatch.setattr(p, "D", 2)
    torch = SimpleNamespace(bfloat16="bf16", float32="fp32")
    x, g, probability = [0.25, 1.8, 0.3, -0.6], [0.4, -0.2], .7
    result = p.analytic(torch, Matrix([x]), Matrix([[probability]]), Matrix([g]), weighted)
    effective = probability if weighted else 1.
    def loss(values, weight):
        return sum((.5+(a-1)/(1+abs(a-1)))*b*dy*weight for a,b,dy in zip(values[:2],values[2:],g))
    step = 1e-6
    for index in range(4):
        plus, minus = x.copy(), x.copy(); plus[index] += step; minus[index] -= step
        expected = (loss(plus,effective)-loss(minus,effective))/(2*step)
        component = "dx_first" if index < 2 else "dx_second"
        assert result[component].rows[0][index%2] == pytest.approx(expected, abs=1e-9)
        assert result[component].dtype == "bf16"
    assert result["y"].dtype == "bf16"
    if weighted:
        expected = (loss(x,probability+step)-loss(x,probability-step))/(2*step)
        assert result["dp"].rows[0][0] == pytest.approx(expected, abs=1e-9)
        assert result["dp"].dtype == "fp32"
    else:
        assert "dp" not in result


class Constant:
    """Constant-valued two-dimensional fixture; no native tensor semantics implied."""
    def __init__(self, rows, columns, value, dtype="bf16"):
        self.shape, self.value, self.dtype = (rows,columns), value, dtype
    def numel(self): return math.prod(self.shape)
    def double(self): return Constant(*self.shape, self.value, "f64")
    def __getitem__(self, selected):
        start, stop, _ = selected.indices(self.shape[0])
        return Constant(max(0,stop-start), self.shape[1], self.value, self.dtype)
    def binary(self, other, operation):
        return Constant(*self.shape, operation(self.value, other.value if isinstance(other,Constant) else other), self.dtype)
    def __add__(self, other): return self.binary(other, lambda a,b:a+b)
    __radd__ = __add__
    def __sub__(self, other): return self.binary(other, lambda a,b:a-b)
    def __rsub__(self, other): return self.binary(other, lambda a,b:b-a)
    def __mul__(self, other): return self.binary(other, lambda a,b:a*b)
    __rmul__ = __mul__
    def __gt__(self, other): return self.binary(other, lambda a,b:a>b)
    def __lt__(self, other): return self.binary(other, lambda a,b:a<b)
    def __ne__(self, other): return self.binary(other, lambda a,b:a!=b)
    def abs(self): return Constant(*self.shape, abs(self.value), self.dtype)
    def square(self): return Constant(*self.shape, self.value*self.value, self.dtype)
    def max(self): return SimpleNamespace(item=lambda:self.value)
    def sum(self): return SimpleNamespace(item=lambda:self.value*self.numel())
    def all(self): return bool(self.value)
    def contiguous(self): return self
    def long(self): return Constant(*self.shape, int(self.value), "i64")
    def view(self, dtype):
        bits = struct.unpack("<I",struct.pack("<f",self.value))[0]
        if self.dtype == "bf16":
            bits >>= 16; bits = bits-2**16 if bits >= 2**15 else bits
        else:
            bits = bits-2**32 if bits >= 2**31 else bits
        return Constant(*self.shape, bits, dtype)


def fake_torch():
    def where(condition, a, b):
        selected = a if condition.value else b
        return selected if isinstance(selected,Constant) else Constant(*condition.shape, selected)
    return SimpleNamespace(bfloat16="bf16",int16="i16",int32="i32",where=where,
        isfinite=lambda value:Constant(*value.shape,math.isfinite(value.value),"bool"))


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("actual,reference", [(1.,1.),(2.,1.),(0.,-0.),(math.nan,1.)])
def test_actual_full_tiled_reducer_and_raw_diagnostics(p, dtype, actual, reference):
    visits = []
    def oracle(start, end):
        visits.append((start,end))
        return Constant(min(130,end)-start,3,reference,dtype)
    metric = p.compare(fake_torch(), Constant(130,3,actual,dtype), oracle,130)
    assert visits == [(0,64),(64,128),(128,192)]
    assert metric["numel"] == metric["checked"] == 390
    if math.isnan(actual):
        assert metric["finite"] is False and metric["passed"] is False
    else:
        difference = abs(actual-reference)
        assert metric["max_abs"] == metric["rms_error"] == difference
        assert metric["reference_rms"] == abs(reference)
        assert metric["passed"] is (difference == 0)
        assert metric["mismatches"] == (390 if difference else 0)
        assert metric["raw_mismatches"] == (0 if actual == reference and math.copysign(1,actual) == math.copysign(1,reference) else 390)


def test_actual_empty_reducer_has_no_oracle_work(p):
    metric = p.compare(fake_torch(), Constant(0,1920,0), lambda *a:pytest.fail("empty oracle call"),0)
    assert metric["passed"] is True and metric["numel"] == metric["checked"] == 0
