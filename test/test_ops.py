from typing import List, Tuple
import pytest
import torch
import numpy as np
from smalltensor import Tensor

def compare(m, x, y, atol, rtol):
  assert x.shape == y.shape
  if not np.allclose(x, y, atol, rtol):
    raise Exception(f"{m} failed with shape {x.shape} {y.shape}")

def util_test_ops(shapes: List[Tuple[int]], torch_fxn, smallt_fxn, 
                  backward=True, a=-1.0, b=1.0,
                  atol=1e-3, rtol=1e-3):
  """
  shapes: list of shapes of Tensor
  torch_fxn: PyTorch's ops
  smallt_fxn: smalltensor's ops
  backward: test for backward pass
  atol: absolute difference
  rtol: relative defference

  https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html#numpy.random.random_sample
  a, b: lower and upper bound of numpy random sample in Uniform[a, b), a < b """
  sts = [Tensor(b*(a+np.random.random(size=s).astype(np.float32)), requires_grad=backward) for s in shapes]
  tts = [torch.tensor(a.numpy(), requires_grad=backward) for a in sts]

  # Test forward
  st_val = smallt_fxn(*sts)
  tt_val = torch_fxn(*tts)
  compare("forward pass", st_val.numpy(), tt_val.detach().numpy(), atol, rtol)

  if backward:
    # Test backward
    st_val = st_val.mean().backward()
    tt_val = tt_val.mean().backward()
    for i, (st, tt) in enumerate(zip(sts, tts)):
      compare(f"backward pass of tensor idx {i}", st.grad.numpy(), tt.grad.detach().numpy(), atol, rtol)

# Binary ops
def test_add():
  util_test_ops([(10, 10), (10, 10)], lambda x,y: x+y, Tensor.add)
def test_sub():
  util_test_ops([(10, 10), (10, 10)], lambda x,y: x-y, Tensor.sub)
def test_mul():
  util_test_ops([(10, 10), (10, 10)], lambda x,y: x*y, Tensor.mul)
def test_div():
  util_test_ops([(10, 10), (10, 10)], lambda x,y: x/y, Tensor.div)
def test_pow():
  util_test_ops([(10, 10)], lambda x: x**2, lambda x: Tensor.pow(x,2), a=0)
  # util_test_ops([(10, 10), (10, 10)], lambda x,y: torch.pow(x,y), Tensor.pow, a=0)
def test_matmul():
  util_test_ops([(10, 10), (10, 10)], lambda x,y: torch.matmul(x,y), Tensor.matmul)

np.seterr(divide='raise')   # Numpy treat div 0 as warning and return inf, not exception
def test_div_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(1)/0
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(1)/Tensor(0)
def test_inv_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(0).inv()

def test_neg():
  util_test_ops([(10, 10)], lambda x: -x, Tensor.__neg__)
def test_inv():
  util_test_ops([(10, 10)], lambda x: 1/x, Tensor.inv)
def test_log():
  util_test_ops([(10, 10)], lambda x: torch.log(x), Tensor.log, a=0)
def test_relu():
  util_test_ops([(10, 10)], lambda x: torch.relu(x), Tensor.relu)
def test_exp():
  util_test_ops([(10, 10)], lambda x: torch.exp(x), Tensor.exp)

@pytest.mark.skip(reason="not written")
def test_sum():
  pass
@pytest.mark.skip(reason="not written")
def test_max():
  pass
@pytest.mark.skip(reason="not written")
def test_min():
  pass
@pytest.mark.skip(reason="not written")
def test_mean():
  pass
@pytest.mark.skip(reason="not written")
def test_expand():
  pass
@pytest.mark.skip(reason="not written")
def test_permute():
  pass
