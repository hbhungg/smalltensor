import pytest
from smalltensor.tensor import Tensor
from smalltensor.ops import Add
from smalltensor.buffer import Buffer
import numpy as np

def test1():
  a = [0]
  b = Buffer(a, (1,))
  assert isinstance(b._buffer, np.ndarray)

  a = np.array([0])
  b = Buffer(a, (1,))
  assert isinstance(b._buffer, np.ndarray)

def test_add():
  a = Tensor(1)
  b = Tensor(2)
  c = (a + b)
  assert c.data == 3
  assert isinstance(c._ctx, Add)
  assert len(c._ctx.parent) == 2

def test_const():
  a = Tensor(1)
  assert (a+1).data == 2
  assert (1+a).data == 2

def test_neg():
  a = Tensor(1)
  assert (-a).data == -1

def test_sub():
  a = Tensor(1)
  b = Tensor(2)
  assert (a-b).data == -1
  assert (b-a).data == 1
  assert (a-1).data == 0
  assert (1-a).data == 0

def test_mul():
  a = Tensor(2)
  b = Tensor(3)
  assert (a*b).data == 6
  assert (b*a).data == 6
  assert (a*1).data == 2
  assert (1*b).data == 3

