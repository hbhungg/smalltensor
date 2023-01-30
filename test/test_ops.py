import random
import math
from inspect import signature

import pytest
from pytest import approx
import numpy as np

from smalltensor.tensor import Tensor

def central_difference(f, *vals, arg=0, epsilon=1e-6):
  # https://en.wikipedia.org/wiki/Finite_difference for more details.
  def _diff(vals, arg, epsilon):
    # Change vals[arg] by eps
    return [i+epsilon if idx == arg else i for idx, i in enumerate(vals)]

  # delta_h_f(x) = f(x+h/2) - f(x-h/2)
  # f'(x) = delta_h_f(x)/h
  return (f(*_diff(vals, arg, epsilon/2)) - f(*_diff(vals, arg, -epsilon/2)))/epsilon

# FIX: Should we test forward and backward seperately?
def util_test_ops(py_fxn, smt_fxn, atol=1e-5, rtol=1e-5):
  nparams = len(signature(py_fxn).parameters)
  pyi = [np.random.rand(3, 3) for n in range(nparams)]
  sti = [Tensor(d, requires_grad=True) for d in pyi]
  py_val, st_val  = py_fxn(*pyi), smt_fxn(*sti)
  # Compare forward result from python ops and smalltensor ops
  assert st_val.item == approx(py_val, abs=atol, rel=rtol)

  if nparams > 1:
    sti_1 = (2, sti[1])
    sti_2 = (sti[0], 2)
    # Mix Tensor with Python number
    st_val1, py_val1 = smt_fxn(*sti_1), py_fxn(2, pyi[1])
    st_val2, py_val2 = smt_fxn(*sti_2), py_fxn(pyi[0], 2)
    # Compare forward result from python ops and smalltensor ops
    assert st_val1.item == approx(py_val1, abs=atol, rel=rtol), f"const idx 0 of {smt_fxn} fail"
    assert st_val2.item == approx(py_val2, abs=atol, rel=rtol), f"const idx 1 of {smt_fxn} fail"
    # Mix Tensor with np.ndarray, but only if using Tensor's method
    sti_3 = (sti[0], pyi[1])
    st_val3 = smt_fxn(*sti_3)
    assert st_val3.item == approx(py_val, abs=atol, rel=rtol), "tensor ops on numpy fail"

  # Compare backward result from python ops and smalltensor ops
  st_val.backward()
  for idx in range(nparams):
    grad = central_difference(py_fxn, *pyi, arg=idx)
    assert grad == approx(sti[idx].grad.item, abs=atol, rel=rtol), "backward fail"


def test_add():
  util_test_ops(lambda x,y: x+y, Tensor.add)
def test_sub():
  util_test_ops(lambda x,y: x-y, Tensor.sub)
def test_mul():
  util_test_ops(lambda x,y: x*y, Tensor.mul)
def test_div():
  util_test_ops(lambda x,y: x/y, Tensor.div)
  util_test_ops(lambda x,y: x/y, Tensor.div)
def test_rdiv():
  assert (2/Tensor(1)).item == 2

# Numpy treat div 0 as warning and return inf, not exception
np.seterr(divide='raise')
def test_div_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(1)/0
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(1)/Tensor(0)
def test_inv_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(0).inv()

def test_inv():
  util_test_ops(lambda x: 1/x, Tensor.inv)
def test_pow():
  util_test_ops(lambda x,y: x**y, Tensor.pow)
def test_log():
  util_test_ops(lambda x: np.log(x), Tensor.log)
def test_relu():
  util_test_ops(lambda x: np.maximum(x, 0), Tensor.relu)
def test_exp():
  util_test_ops(lambda x: np.exp(x), Tensor.exp)
def test_neg():
  util_test_ops(lambda x: -x, Tensor.__neg__)


def util_test_reduce(pyfxn, stfxn):
  shape = [random.randint(1, 5) for i in range(random.randint(1, 5))]
  a = np.random.rand(*shape)
  assert pyfxn(a) == approx(stfxn(Tensor(a)).item), f"fails with input {a}"
  for i in range(len(shape)):
    assert pyfxn(a, i) == approx(stfxn(Tensor(a), i).item), f"fails with input {a}"

def test_sum():
  util_test_reduce(np.sum, Tensor.sum)
def test_max():
  util_test_reduce(np.max, Tensor.max)
def test_min():
  util_test_reduce(np.min, Tensor.min)
def test_mean():
  util_test_reduce(np.mean, Tensor.mean)
