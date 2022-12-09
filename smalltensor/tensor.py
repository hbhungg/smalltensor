import numpy as np
from typing import Optional, Tuple, Union, List, Dict


class Variable:
  def __init__(self, data):
    self.data = data
    self._grad = None
    self._ctx: Optional[Function] = None

  def __add__(self, x):
    r = Add()
    return r.apply(self, x)


class Buffer:
  def __init__(self, buffer: Union[np.ndarray, List], shape: Tuple, strides: Optional[Tuple]=None):
    self._buffer = buffer if isinstance(buffer, np.ndarray) else np.ndarray(buffer)
    self.shape = shape
    self.strides = strides


class Function:
  def __init__(self):
    self.saved_tensor: List[Variable] = []
    self._needs_input_grad: List[Variable] = []

  def saved_for_backward(self, *x):
    self.saved_tensor.extend(x)

  def apply(self, *x):
    val = self.forward(*[i.data for i in x])
    ret = Variable(val)
    ret._ctx = self
    return ret


class Add(Function):
  def forward(self, a, b):
    self.saved_for_backward(a, b)
    return a + b

  def backward(self, grad_output):
    a, b = self.saved_tensor
    return a, b

