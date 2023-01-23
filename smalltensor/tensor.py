import os
from typing import Optional, Tuple, Union, List
import inspect, importlib, functools

import numpy as np

#from .buffer import Buffer


class Tensor:
  def __init__(self, item, requires_grad=False):
    self.item = np.array(item) if not isinstance(item, np.ndarray) else item
    self.grad: Optional[Tensor] = None
    self.requires_grad: Bool = requires_grad

    # Context for autograph construction
    self._ctx: Optional[Function] = None

  @property
  def shape(self): return self.item.shape

  @property
  def dtype(self): return np.float32

  def __repr__(self):
    if self.requires_grad and self._ctx is not None:
      return f"Tensor({self.item}, requires_grad={self.requires_grad}, grad_fn={self._ctx})"
    elif self.requires_grad and self._ctx is None:
      return f"Tensor({self.item}, requires_grad={self.requires_grad})"
    else:
      return f"Tensor({self.item})"

  @staticmethod
  def ensure_tensor(fxn, x, y):
    """
    Turn all Python number into Tensor
    """
    x, y = [Tensor(t, requires_grad=False) if not isinstance(t, Tensor) else t for t in [x, y]]
    return fxn(x, y)

  @classmethod
  def ones(cls, *shape): return cls(np.ones(shape, dtype=np.float32))

  def toposort(self):
    """
    Toposort for backward pass
    """
    def _toposort(node, visited, nodes):
      visited.add(node)
      if node._ctx is not None:
        [_toposort(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _toposort(self, set(), [])
  
  def backward(self):
    self.grad = Tensor.ones(*self.shape)
    for t0 in reversed(self.toposort()):
      grads = t0._ctx.backward(t0.grad)
      grads = [grads] if not isinstance(grads, Tuple) else grads
      grads = [Tensor(g, requires_grad=False) if not isinstance(g, Tensor) else g for g in grads]
      for t, g in zip(t0._ctx.parents, grads):
        if t.requires_grad:
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx
    return self.grad

  # Unary ops
  # Should neg be a standalone function or using sub?
  def __neg__(self): return Tensor._neg(self)
  def inv(self): return Tensor._inv(self)
  def relu(self): return Tensor._relu(self)
  def log(self): return Tensor._log(self)
  def exp(self): return Tensor._exp(self)

  # Binary ops
  def __add__(self, x): return Tensor.ensure_tensor(Tensor._add, self, x)
  def __radd__(self, x): return Tensor.ensure_tensor(Tensor._add, x, self)
  def __sub__(self, x): return Tensor.ensure_tensor(Tensor._sub, self, x)
  def __rsub__(self, x): return Tensor.ensure_tensor(Tensor._sub, x, self)
  def __mul__(self, x): return Tensor.ensure_tensor(Tensor._mul, self, x)
  def __rmul__(self, x): return Tensor.ensure_tensor(Tensor._mul, x, self)
  def __truediv__(self, x): return self * (x.inv() if isinstance(x, Tensor) else (1/x))

  def __rtruediv__(self, x): return self.inv() * x
  #def __pow__(self, x): return Tensor.ensure_tensor(Tensor._pow, self, x)
  def eq(self, x): return Tensor._eq(self, x)

  # TODO:
  # Reduce ops
  def sum(self, x): raise NotImplementedError("will implement")
  def max(self, x): raise NotImplementedError("will implement")
  # Movement ops
  # Processing ops


class Function:
  def __init__(self, *tensors: Tensor):
    self.parents = tensors
    self.saved_tensor: List[Tensor] = []
    self.needs_input_grad: List[Bool] = [t.requires_grad for t in self.parents]
    self.requires_grad: Bool = any(self.needs_input_grad)

  def saved_for_backward(self, *x):
    self.saved_tensor.extend(x)

  @classmethod
  def apply(cls, *x):
    # Create an instance of the Function
    ctx = cls(*x) 
    # Every ops create a new Tensor
    ret = Tensor(item=ctx.forward(*[v.item for v in x]), requires_grad=ctx.requires_grad)
    ret._ctx = ctx
    return ret
  
  def __repr__(self):
    return f"<{self.__class__.__name__}Backward>"


# Meta programming
# For some reason this get pass the circular import problem.
for name, cls in inspect.getmembers(importlib.import_module('smalltensor.ops'), inspect.isclass):
  if name != "Function": 
    setattr(Tensor, f"_{name.lower()}", functools.partialmethod(cls.apply))
