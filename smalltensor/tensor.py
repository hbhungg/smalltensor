from typing import Optional, Tuple, Union, List, Dict
import inspect, importlib, functools

import numpy as np


class Tensor:
  def __init__(self, data):
    self.data = data
    self._grad: Optional[Tensor] = None
    self.requires_grad: Optional[Bool] = None

    # Context for autograph construction
    self._ctx: Optional[Function] = None

  def backward(self, x):
    pass

  def __repr__(self):
    return f"Tensor({self.data})"

  def deepwalk(self):
    def _deepwalk(x: Tensor):
      print(x)
      if x._ctx is not None:
        for t in x._ctx.parent:
          _deepwalk(t)
    _deepwalk(self)
  
  # This should be broadcasted
  # For now, this will turn all python number into Tensor
  @staticmethod
  def ensure_tensor(fxn, x, y):
    x, y = [Tensor(t) if not isinstance(t, Tensor) else t for t in [x, y]]
    return fxn(x, y)

  def __add__(self, x): return Tensor.ensure_tensor(Tensor._add, self, x)
  def __radd__(self, x): return Tensor.ensure_tensor(Tensor._add, x, self)
  def __neg__(self): return Tensor._neg(self)
  def __sub__(self, x): return Tensor.ensure_tensor(Tensor._add, self, -x)
  def __rsub__(self, x): return Tensor.ensure_tensor(Tensor._add, -x, self)


class Function:
  def __init__(self, *tensors: Tensor):
    self.parent = tensors
    self.saved_tensor: List[Tensor] = []
    # self.needs_input_grad: List[Tensor] = [t.requires_grad for t in self.parent]

  def saved_for_backward(self, *x):
    self.saved_tensor.extend(x)

  @classmethod
  def apply(cls, *x):
    # Create an instance of the Function
    ctx = cls(*x) 
    # Every ops create a new Tensor
    ret = Tensor(data=ctx.forward(*[v.data for v in x]))
    ret._ctx = ctx
    return ret


# Meta programming
# For some reason this get pass the circular import problem.
for name, cls in inspect.getmembers(importlib.import_module('smalltensor.ops'), inspect.isclass):
  if name != "Function": 
    setattr(Tensor, f"_{name.lower()}", functools.partialmethod(cls.apply))
