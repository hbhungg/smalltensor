import os
from typing import Optional, Tuple, Union, List
import inspect, importlib, functools

import numpy as np

#from .buffer import Buffer


class Tensor:
  def __init__(self, item: Union[List, int, float, np.ndarray], requires_grad=False):
    self.item = np.array(item, dtype=np.float32) if not isinstance(item, np.ndarray) else item.astype(np.float32)
    self.grad: Optional[Tensor] = None
    self.requires_grad: bool = requires_grad

    # Context for autograph construction
    self._ctx: Optional[Function] = None

  @property
  def shape(self): return self.item.shape

  @property
  def dtype(self): return np.float32

  # Detach Tensor out of the autodiff graph
  def detach(self): return Tensor(self.item, requires_grad=False)

  def __repr__(self):
    if self.requires_grad and self._ctx is not None:
      return f"Tensor({self.prettify()}, requires_grad={self.requires_grad}, ctx={self._ctx})"
    elif self.requires_grad and self._ctx is None:
      return f"Tensor({self.prettify()}, requires_grad={self.requires_grad})"
    else:
      return f"Tensor({self.prettify()})"

  # HACK: Make numpy print with correct indent when printing Tensor. Is there a better way?
  def prettify(self):
    return str(self.item).replace("\n", "\n" + 7*' ')

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
    if self.requires_grad is False:
      raise RuntimeError("Tensor does not require grad and does not have ctx. Make sure all of the tensors have requires_grad=True")
    self.grad = Tensor.ones(*self.shape)
    for t0 in reversed(self.toposort()):
      grads = t0._ctx.backward(t0.grad.item)
      grads = [grads] if not isinstance(grads, Tuple) else grads
      grads = [Tensor(g, requires_grad=False) if not isinstance(g, Tensor) else g for g in grads]
      for t, g in zip(t0._ctx.parents, grads):
        if t.requires_grad:
          t.grad = g if t.grad is None else (t.grad + g)
    return self.grad

  @staticmethod
  def ensure_tensor(fxn, x, y):
    """
    Turn all Python number into Tensor
    """
    x, y = [Tensor(t, requires_grad=False) if not isinstance(t, Tensor) else t for t in [x, y]]
    return fxn(x, y)

  # Unary ops
  # Should neg be a standalone function or using sub?
  def __neg__(self): return Tensor._neg(self)
  def inv(self): return Tensor._inv(self)
  def relu(self): return Tensor._relu(self)
  def log(self): return Tensor._log(self)
  def exp(self): return Tensor._exp(self)

  # Binary ops
  def add(self, x): return Tensor.ensure_tensor(Tensor._add, self, x)
  def sub(self, x): return Tensor.ensure_tensor(Tensor._sub, self, x)
  def mul(self, x): return Tensor.ensure_tensor(Tensor._mul, self, x)
  def div(self, x): return self * (x.inv() if isinstance(x, Tensor) else (1/x))
  def eq(self, x): return Tensor._eq(self, x)
  #def pow(self, x): return Tensor.ensure_tensor(Tensor._pow, self, x)

  # NOTES: This all could be generate using setattr, but im keeping this for clarity.
  __add__ = add
  __sub__ = sub
  __mul__ = mul
  __truediv__ = div
  def __radd__(self, x): return Tensor.add(x, self)
  def __rsub__(self, x): return Tensor.sub(x, self)
  def __rmul__(self, x): return Tensor.mul(x, self)
  def __rtruediv__(self, x): return Tensor.div(x, self)

  # Reduce ops
  def sum(self, dim=None): return Tensor._sum(self, dim=dim)
  def max(self, dim=None): return Tensor._max(self, dim=dim)

  # TODO:
  # Movement ops
  # Processing ops


class Function:
  """
  Base class for all of Tensor ops.
  All of the operations on the Tensor will be handle by Function.apply
  An instance of the Function act as the Context that work on Tensor.
  A Function remember the Tensors that it operate on, and create a new Tensor as result.
  The resulting Tensor remember that Function, which creating the autodiff DAG graph.
  """
  def __init__(self, *tensors: Tensor):
    self.parents = tensors
    self.saved_tensor: List[Tensor] = []
    self.needs_input_grad: List[bool] = [t.requires_grad for t in self.parents]
    self.requires_grad: bool = any(self.needs_input_grad)

  def saved_for_backward(self, *x):
    self.saved_tensor.extend(x)

  @classmethod
  def apply(cls, *x: Tensor, **kwargs):
    # Create an instance of the Function
    ctx = cls(*x) 
    # Every ops create a new Tensor
    ret = Tensor(item=ctx.forward(*[t.item for t in x], **kwargs), requires_grad=ctx.requires_grad)
    ret._ctx = ctx if ctx.requires_grad else None
    return ret

  # In case any function derived have not implement these.
  def forward(self, *args, **kwargs): raise NotImplementedError(f"Not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise NotImplementedError(f"Not implemented for {type(self)}")
  
  def __repr__(self):
    return f"<{self.__class__.__name__}Backward>"


# HACK: Since smalltensor/ops.py import this file, we could not import the ops due to circular.
# HACK: We instead use importlib to late import ops.py after tensor.py.
for name, cls in inspect.getmembers(importlib.import_module('smalltensor.ops'), inspect.isclass):
  if name != "Function": 
    setattr(Tensor, f"_{name.lower()}", functools.partialmethod(cls.apply))

