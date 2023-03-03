import pytest
from smalltensor.nn import Linear, Module
from smalltensor import Tensor

@pytest.mark.skip(reason="not written")
def test_linear():
  a = Linear(10, 1)
  v = a(Tensor.randn(10))

def test_module_parameters():
  class A(Module):
    def __init__(self):
      self.a = 1
      self.b = 2

  class B(Module):
    def __init__(self):
      self.a = 3
      self.b = 4
      self.c = A()

  b = B()
  assert len(b.parameters()) == 4
  assert b.parameters() ==[3, 4, 1, 2]


