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
  assert len(c._ctx.needs_input_grad) == 2

