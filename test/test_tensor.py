import pytest
from smalltensor.tensor import Buffer, Variable
import numpy as np

def test1():
  a = [0]
  b = Buffer(a, (1,))
  assert isinstance(b._buffer, np.ndarray)

  a = np.array([0])
  b = Buffer(a, (1,))
  assert isinstance(b._buffer, np.ndarray)

def test_add():
  a = Variable(1)
  b = Variable(2)
  assert (a + b).data == 3
