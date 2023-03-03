from typing import List, Tuple
import pytest
import torch
import numpy as np
from smalltensor import Tensor

def compare(m, st, tt, atol, rtol):
  assert st.shape == tt.shape
  if not np.allclose(st, tt, atol, rtol):
    raise Exception(f"{m} failed with shape {st.shape} {tt.shape}")

def util_test_ops(shapes: List[Tuple[int]], torch_fxn, smallt_fxn, 
                  backward=True, a=-1.0, b=3.0,
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
  sts = [Tensor((b-a)*(a+np.random.random(size=s).astype(np.float32)), requires_grad=backward) for s in shapes]
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

# Unary ops
def test_neg():
  util_test_ops([(10,10)], lambda x: -x, Tensor.__neg__)
def test_inv():
  util_test_ops([(10,10)], lambda x: 1/x, lambda x: x.inv())
def test_log():
  util_test_ops([(10,10)], lambda x: torch.log(x), lambda x: x.log(), a=0)
def test_relu():
  util_test_ops([(10,10)], lambda x: torch.relu(x), lambda x: x.relu())
def test_exp():
  util_test_ops([(10,10)], lambda x: torch.exp(x), lambda x: x.exp())
def test_square():
  util_test_ops([(10,10)], lambda x: torch.square(x), lambda x: x.square())
def test_sigmoid():
  util_test_ops([(10,10)], lambda x: torch.sigmoid(x), lambda x: x.sigmoid())

# Binary ops
def test_add():
  util_test_ops([(10,10), (10,10)], lambda x,y: x+y, lambda x,y: x.add(y))
def test_sub():
  util_test_ops([(10,10), (10,10)], lambda x,y: x-y, lambda x,y: x.sub(y))
def test_mul():
  util_test_ops([(10,10), (10,10)], lambda x,y: x*y, lambda x,y: x.mul(y))
def test_div():
  util_test_ops([(10,10), (10,10)], lambda x,y: x/y, lambda x,y: x.div(y))
def test_pow():
  util_test_ops([(10,10), (10,10)], lambda x,y: torch.pow(x,y), lambda x,y: x.pow(y), a=0)
def test_matmul():
  util_test_ops([(20,10), (10,20)], lambda x,y: torch.matmul(x,y), lambda x,y: x.matmul(y))
  # Incorrect shape
  with pytest.raises(ValueError):
    util_test_ops([(20,20), (10,20)], lambda x,y: torch.matmul(x,y), lambda x,y: x.matmul(y))
  util_test_ops([(10,10,20), (10,20,10)], lambda x,y: torch.matmul(x,y), lambda x,y: x.matmul(y))

np.seterr(divide='raise')   # Numpy treat div 0 as warning and return inf, not exception
def test_div_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(1)/0
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(1)/Tensor(0)
def test_inv_raise_zero():
  with pytest.raises((ZeroDivisionError, FloatingPointError)): Tensor(0).inv()

sops = [Tensor.add, Tensor.sub, Tensor.mul, Tensor.div, Tensor.pow]
tops = [torch.add, torch.sub, torch.mul, torch.div, torch.pow]
@pytest.mark.parametrize("tfxn, sfxn", zip(tops, sops))
def test_broadcasted(tfxn, sfxn):
  shapes = [[(10, 10), (10, 1)], [(1, 10), (10, 1)]]
  for s in shapes:
    util_test_ops(s, tfxn, sfxn, a=0 if sfxn==Tensor.pow else -1.0)


# Reduce ops
def test_sum():
  util_test_ops([(10,10)], lambda x: torch.sum(x), Tensor.sum)
def test_max():
  util_test_ops([(10,10)], lambda x: torch.max(x), Tensor.max)
def test_min():
  util_test_ops([(10,10)], lambda x: torch.min(x), Tensor.min)
def test_mean():
  util_test_ops([(10,10)], lambda x: torch.mean(x), Tensor.mean)

# Movement ops
def test_expand():
  util_test_ops([(1,2,3,4)], lambda x: x.expand(2,2,3,4), lambda x: x.expand(2,2,3,4))
def test_expand_new_dim():
  util_test_ops([(1,2,3,4)], lambda x: x.expand(1,1,2,3,4), lambda x: x.expand(1,1,2,3,4))
  util_test_ops([(10,2,3,4)], lambda x: x.expand(1,1,10,2,3,4), lambda x: x.expand(1,1,10,2,3,4))
def test_permute():
  util_test_ops([(1,2,3,4)], lambda x: x.permute(1,0,3,2), lambda x: x.permute(1,0,3,2))
def test_reshape():
  util_test_ops([(1,2,3,4)], lambda x: x.reshape(2,1,3,4), lambda x: x.reshape(2,1,3,4))
def test_reshape_combine():
  util_test_ops([(1,2,3,4)], lambda x: x.reshape(2,3,4), lambda x: x.reshape(2,3,4))
  util_test_ops([(1,2,3,4)], lambda x: x.reshape(2,12), lambda x: x.reshape(2,12))
