import numpy as np
from smalltensor.backend import NumpyBuffer, UnaryOps

# Right now, most of this test feels redundant since we are testing numpy backend using numpy
def test_unary_op():
  data = np.random.rand(2, 2)
  a = NumpyBuffer((2,2), buffer=data)
  assert np.allclose(a.unary_op(UnaryOps.NEG), -data)
  assert np.allclose(a.unary_op(UnaryOps.EXP), np.exp(data))

def test_binary_op():
  d1 = np.random.rand(2, 2)
  d2 = np.random.rand(2, 2)
  assert np.allclose(NumpyBuffer((2, 2), buffer=d1)+NumpyBuffer((2, 2), buffer=d2), d1+d2)
