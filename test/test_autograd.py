import random

import pytest
from pytest import approx

import torch

from smalltensor.tensor import Tensor
from smalltensor import set_printoptions

set_printoptions(0)

def test_diamond_backward():
  """
  Test multiple ops in backward pass.
  In diamond config to test the accumulate
  """
  ax = random.uniform(-10, 10)
  bx = random.uniform(-10, 10)
  cx = random.uniform(-10, 10)

  def small_tensor_diamond_backward():
    a = Tensor(ax, requires_grad=True)
    b = Tensor(bx, requires_grad=True)
    c = Tensor(cx, requires_grad=True)
    x = a*b
    y = a*c
    out = (x+y)*y
    out.backward()
    return out, a.grad, b.grad, c.grad

  # Trust PyTorch!
  def torch_diamond_backward():
    a = torch.tensor(ax, requires_grad=True)
    b = torch.tensor(bx, requires_grad=True)
    c = torch.tensor(cx, requires_grad=True)
    x = a*b
    y = a*c
    out = (x+y)*y
    out.backward()
    return out.detach(), a.grad, b.grad, c.grad
  
  ll = zip(small_tensor_diamond_backward(), torch_diamond_backward())
  for x, y in ll:
    assert x.item == approx(y.item())

