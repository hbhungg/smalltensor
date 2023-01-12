import random
import math
from inspect import signature

import pytest
from pytest import approx

from smalltensor.tensor import Tensor
from smalltensor import set_printoptions


def util_test_ops(py_fxn, smt_fxn, s=-10.0, e=10.0, atol=1e-3, rtol=1e-3):
  nparams = len(signature(py_fxn).parameters)
  pyi = [random.uniform(s, e) for n in range(nparams)]
  sti = [Tensor(d) for d in pyi]
  py_val = py_fxn(*pyi)
  st_val = smt_fxn(*sti)
  # Compare result from python ops and smalltensor ops
  assert py_val == approx(st_val.data, abs=atol, rel=rtol)

set_printoptions(0)

def test_add():
  util_test_ops(lambda x,y: x+y, Tensor.__add__)

def test_add_const():
  util_test_ops(lambda x,y: 1+y, lambda x,y: 1+y) 
  util_test_ops(lambda x,y: x+1, lambda x,y: x+1) 

def test_sub():
  util_test_ops(lambda x,y: x-y, Tensor.__sub__)

def test_sub_const():
  util_test_ops(lambda x,y: 1-y, lambda x,y: 1-y)
  util_test_ops(lambda x,y: x-1, lambda x,y: x-1)

def test_mul():
  util_test_ops(lambda x,y: x*y, Tensor.__mul__)

def test_mul_const():
  util_test_ops(lambda x,y: 1*y, lambda x,y: 1*y)
  util_test_ops(lambda x,y: x*1, lambda x,y:x*1)

def test_inv():
  util_test_ops(lambda x: 1/x, Tensor.inv)

def test_inv_raise_zero():
  with pytest.raises(ZeroDivisionError): c = Tensor(0).inv()

def test_div():
  util_test_ops(lambda x,y: x/y, Tensor.__truediv__)

def test_div_const():
  util_test_ops(lambda x,y: x/20, lambda x,y: x/20)
  util_test_ops(lambda x,y: 20/y, lambda x,y: 20/y)

def test_div_raise_zero():
  with pytest.raises(ZeroDivisionError): c = Tensor(1)/0
  with pytest.raises(ZeroDivisionError): c = Tensor(1)/Tensor(0)

def test_pow():
  util_test_ops(lambda x,y: x**y, Tensor.__pow__)

def test_log():
  util_test_ops(lambda x: math.log(x), Tensor.log, 0.1, 10)

