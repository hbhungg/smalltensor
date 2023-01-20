import random
import math
from inspect import signature

import pytest
from pytest import approx

from smalltensor.tensor import Tensor
from smalltensor import set_printoptions

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
  pyi = [random.uniform(s, e) for n in range(nparams)]
  sti = [Tensor(d, requires_grad=True) for d in pyi]
  py_val = py_fxn(*pyi)
  st_val = smt_fxn(*sti)
  # Compare forward result from python ops and smalltensor ops
  assert py_val == approx(st_val.data, abs=atol, rel=rtol)

  # Mix Tensor with Python number
  if nparams > 1:
    sti_1 = (sti[0], pyi[1])
    sti_2 = (pyi[0], sti[1])
    st_val1 = smt_fxn(*sti_1)
    st_val2 = smt_fxn(*sti_2)
    # Compare forward result from python ops and smalltensor ops
    assert py_val == approx(st_val1.data, abs=atol, rel=rtol), f"const idx 1 of {smt_fxn} fail"
    assert py_val == approx(st_val2.data, abs=atol, rel=rtol), f"const idx 2 if {smt_fxn} fail"

  # Compare backward result from python ops and smalltensor ops
  st_val.backward()
  for idx in range(nparams):
    grad = central_difference(py_fxn, *pyi, arg=idx)
    assert grad == approx(sti[idx].grad.data, abs=atol, rel=rtol)

set_printoptions(0)

def test_add():
  util_test_ops(lambda x,y: x+y, Tensor.__add__)

def test_sub():
  util_test_ops(lambda x,y: x-y, Tensor.__sub__)

def test_mul():
  util_test_ops(lambda x,y: x*y, Tensor.__mul__)

def test_inv_raise_zero():
  with pytest.raises(ZeroDivisionError): c = Tensor(0).inv()

def test_div():
  util_test_ops(lambda x,y: x/y, Tensor.__truediv__, 0.1, 10)
  util_test_ops(lambda x,y: x/y, Tensor.__truediv__, -10, -0.1)

def test_div_raise_zero():
  with pytest.raises(ZeroDivisionError): c = Tensor(1)/0
  with pytest.raises(ZeroDivisionError): c = Tensor(1)/Tensor(0)

def test_inv():
  util_test_ops(lambda x: 1/x, Tensor.inv)

#def test_pow():
#  util_test_ops(lambda x,y: x**y, Tensor.__pow__, -10, -1)

def test_log():
  util_test_ops(lambda x: math.log(x), Tensor.log, 0.1, 10)

def test_relu():
  util_test_ops(lambda x: max(x, 0.0), Tensor.relu)

def test_exp():
  util_test_ops(lambda x: math.exp(x), Tensor.exp)

