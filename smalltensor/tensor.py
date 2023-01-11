from typing import Optional, Tuple, Union, List, Dict
import inspect, importlib, functools

import numpy as np


class Tensor:
  def __init__(self, data, requires_grad=True):
    self.data = data
    self._grad: Optional[Tensor] = None
    self.requires_grad: Optional[Bool] = requires_grad

    # Context for autograph construction
    self._ctx: Optional[Function] = None

  def __repr__(self):
    return f"Tensor({self.data})"

  @staticmethod
  def ensure_tensor(fxn, x, y):
    """
    Turn all Python number into Tensor
    """
    x, y = [Tensor(t, requires_grad=False) if not isinstance(t, Tensor) else t for t in [x, y]]
    return fxn(x, y)

  def deepwalk(self):
    # Toposort
    def _deepwalk(node, visited, nodes):
      visited.add(node)
      if node._ctx is not None:
        [_deepwalk(i, visited, nodes) for i in node._ctx.parents if i not in visited]
      nodes.append(node)
      return nodes
    return _deepwalk(self, set(), [])
  
  def backward(self):
    self._grad = Tensor(1, requires_grad=False)
    return self._ctx.backward(self._grad)


  # Unary ops
  def __neg__(self): return Tensor._neg(self)

  # Binary ops
  def __add__(self, x): return Tensor.ensure_tensor(Tensor._add, self, x)
  def __radd__(self, x): return Tensor.ensure_tensor(Tensor._add, x, self)
  def __sub__(self, x): return Tensor.ensure_tensor(Tensor._add, self, -x)
  def __rsub__(self, x): return Tensor.ensure_tensor(Tensor._add, -x, self)
  def __mul__(self, x): return Tensor.ensure_tensor(Tensor._mul, self, x)
  def __rmul__(self, x): return Tensor.ensure_tensor(Tensor._mul, x, self)

  # TODO:
  # Reduce ops
  # Movement ops


class Function:
  def __init__(self, *tensors: Tensor):
    self.parents = tensors
    self.saved_tensor: List[Tensor] = []
    self.needs_input_grad: List[Tensor] = [t.requires_grad for t in self.parents]

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
