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

def util_test_ops(py_fxn, smt_fxn, s=-10.0, e=10.0, atol=1e-3, rtol=1e-3):
  nparams = len(signature(py_fxn).parameters)
  pyi = [np.random.rand(3, 3) for n in range(nparams)]
  sti = [Tensor(d, requires_grad=True) for d in pyi]
  py_val, st_val  = py_fxn(*pyi), smt_fxn(*sti)
  # Compare forward result from python ops and smalltensor ops
  assert py_val == approx(st_val.item, abs=atol, rel=rtol)

  if nparams > 1:
    sti_1 = (2, sti[1])
    sti_2 = (sti[0], 2)
    # Mix Tensor with Python number
    st_val1, py_val1 = smt_fxn(*sti_1), py_fxn(2, pyi[1])
    st_val2, py_val2 = smt_fxn(*sti_2), py_fxn(pyi[0], 2)
    # Compare forward result from python ops and smalltensor ops
    assert py_val1 == approx(st_val1.item, abs=atol, rel=rtol), f"const idx 0 of {smt_fxn} fail"
    assert py_val2 == approx(st_val2.item, abs=atol, rel=rtol), f"const idx 1 of {smt_fxn} fail"
    # Mix Tensor with np.ndarray, but only if using Tensor's method
    sti_3 = (sti[0], pyi[1])
    st_val3 = smt_fxn(*sti_3)
    assert py_val == approx(st_val3.item, abs=atol, rel=rtol), "tensor ops on numpy fail"

  # Compare backward result from python ops and smalltensor ops
  st_val.backward()
  for idx in range(nparams):
    grad = central_difference(py_fxn, *pyi, arg=idx)
    assert grad == approx(sti[idx].grad.item, abs=atol, rel=rtol), "backward fail"


def test_add():
  util_test_ops(lambda x,y: x+y, Tensor.__add__)
def test_sub():
  util_test_ops(lambda x,y: x-y, Tensor.__sub__)
def test_mul():
  util_test_ops(lambda x,y: x*y, Tensor.__mul__)
def test_div():
  util_test_ops(lambda x,y: x/y, Tensor.__truediv__, 0.1, 10)
  util_test_ops(lambda x,y: x/y, Tensor.__truediv__, -10, -0.1)

import numpy as np
# Numpy treat div 0 as warning and return inf, not exception
np.seterr(divide='raise')
def test_div_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): c = Tensor(1)/0
  with pytest.raises((ZeroDivisionError, FloatingPointError)): c = Tensor(1)/Tensor(0)
def test_inv_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): c = Tensor(0).inv()

def test_inv():
  util_test_ops(lambda x: 1/x, Tensor.inv)
#def test_pow():
#  util_test_ops(lambda x,y: x**y, Tensor.__pow__, -10, -1)
def test_log():
  util_test_ops(lambda x: np.log(x), Tensor.log, 0.1, 10)
def test_relu():
  util_test_ops(lambda x: np.maximum(x, 0), Tensor.relu)
def test_exp():
  util_test_ops(lambda x: np.exp(x), Tensor.exp)
def test_neg():
  util_test_ops(lambda x: -x, Tensor.__neg__)

