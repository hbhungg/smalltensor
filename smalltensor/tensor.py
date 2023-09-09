from __future__ import annotations
from typing import Optional, List
import math
from .backend import Buffer, NumpyBuffer
import numpy as np
from .utils import broadcast_shapes

class Function:
  """
  Base class for all of Tensor ops.
  All of the operations on the Tensor will be handle by Function.apply
  An instance of the Function act as the Context that work on Tensor.
  A Function remember the Tensors that it operate on, and create a new Tensor as result.
  The resulting Tensor remember that Function, which creating the autodiff DAG graph. """
  def __init__(self, *tensors: Tensor):
    self.parents = tensors
    self.saved_tensor: List[Tensor] = []
    self.needs_input_grad: List[bool] = [t.requires_grad for t in self.parents]
    self.requires_grad: bool = any(self.needs_input_grad)

  def saved_for_backward(self, *x) -> None:
    self.saved_tensor.extend(x)

  @classmethod
  def apply(cls, *x: Tensor, **kwargs) -> Tensor:
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

from . import ops

class Tensor:
  def __init__(self, item, requires_grad: bool=False):
    self.item: np.ndarray = np.array(item, dtype=np.float32) if not isinstance(item, np.ndarray) else item.astype(np.float32)
    # self.item: Buffer = NumpyBuffer(item, dtype=np.float32)
    self.grad: Optional[Tensor] = None
    self.requires_grad: bool = requires_grad

    # Context for autograph construction
    self._ctx: Optional[Function] = None

  def __repr__(self):
    return f"Tensor({np.array2string(self.item, prefix=7*' ', precision=4, separator=', ')}" + (f", requires_grad={self.requires_grad})" if self.requires_grad is True else ")")

  @property
  def shape(self): return self.item.shape

  @property
  def dtype(self): return np.float32

  @classmethod
  def zeros(cls, *shape, **kwargs) -> Tensor: return cls(np.zeros(shape, dtype=np.float32), **kwargs)

  @classmethod
  def ones(cls, *shape, **kwargs) -> Tensor: return cls(np.ones(shape, dtype=np.float32), **kwargs)

  @classmethod
  def randn(cls, *shape, **kwargs) -> Tensor: 
    """ Random number from normal distribution with mean 0 and var 1 """
    return cls(np.random.default_rng().standard_normal(size=shape, dtype=np.float32), **kwargs)

  @classmethod
  def uniform(cls, *shape, **kwargs) -> Tensor: 
    """ Random number from continuous uniform distribution"""
    return cls((np.random.default_rng().random(size=shape, dtype=np.float32)*2-1), **kwargs)

  @classmethod
  def scale_uniform(cls, *shape, **kwargs) -> Tensor: 
    return cls((np.random.default_rng().random(size=shape, dtype=np.float32)*2-1) * (math.sqrt(math.prod(shape))), **kwargs)

  @classmethod
  def xavier_uniform(cls, *shape, **kwargs) -> Tensor:
    return cls((np.random.default_rng().random(size=shape, dtype=np.float32)*2-1) * (math.sqrt(6/(shape[0]+math.prod(shape[1:])))), **kwargs)


  def detach(self) -> Tensor: return Tensor(self.item, requires_grad=False)
  def numpy(self) -> np.ndarray: return np.array(self.item)

  def toposort(self) -> List[Tensor]:
    """
    Toposort the graph """
    def _toposort(node, visited, nodes):
      visited.add(node)
      if node._ctx is not None:
        [_toposort(i, visited, nodes) for i in node._ctx.parents if i not in visited]
        nodes.append(node)
      return nodes
    return _toposort(self, set(), [])
  
  def backward(self) -> None:
    """
    Compute the gradient of the current Tensor w.r.t graph leaves. The graph is differentiated using the chain rule. """
    if self.requires_grad is False:
      raise RuntimeError("Tensor does not require grad and does not have ctx. Make sure all of the tensors have requires_grad=True")
    if self.shape != (1,) and self.shape != ():
      raise RuntimeError(f"grad can be implicitly created only for scalar outputs, while Tensor is of shape {self.shape}")
    self.grad = Tensor.ones(*self.shape)
    for t0 in reversed(self.toposort()):
      assert t0._ctx is not None
      assert t0.grad is not None
      grads = t0._ctx.backward(t0.grad.item)
      grads = [grads] if not isinstance(grads, tuple) else grads
      grads = [Tensor(g, requires_grad=False) if not isinstance(g, Tensor) else g for g in grads]
      for t, g in zip(t0._ctx.parents, grads):
        if t.requires_grad:
          assert t.shape == g.shape, f"grad shape does not match tensor shape, {g.shape} != {t.shape}"
          t.grad = g if t.grad is None else (t.grad + g)

  @staticmethod
  def broadcasted_tensor(fxn, x, y):
    """
    Turn all Python number into Tensor. 
    Perform broadcasting using expand."""
    x, y = [Tensor(t, requires_grad=False) if not isinstance(t, Tensor) else t for t in [x, y]]
    # No need to broadcast if similar shape
    if x.shape == y.shape:
      return fxn(x, y)
    # Manually perform broadcast, in order to backward.
    bshape = broadcast_shapes(x.shape, y.shape)
    return fxn(x.expand(*bshape), y.expand(*bshape))

  # Unary ops
  # Should neg be a standalone function or using sub?
  def neg(self): return ops.Neg.apply(self)
  def inv(self): return ops.Inv.apply(self)
  def relu(self): return ops.ReLU.apply(self)
  def log(self): return ops.Log.apply(self)
  def exp(self): return ops.Exp.apply(self)
  def square(self): return self*self
  def sigmoid(self): return (self.neg().exp() + 1).inv()

  # Binary ops
  def add(self, x): return Tensor.broadcasted_tensor(ops.Add.apply, self, x)
  def sub(self, x): return Tensor.broadcasted_tensor(ops.Sub.apply, self, x)
  def mul(self, x): return Tensor.broadcasted_tensor(ops.Mul.apply, self, x)
  def div(self, x): return self * (x.inv() if isinstance(x, Tensor) else (1/x))
  def pow(self, x): return Tensor.broadcasted_tensor(ops.Pow.apply, self, x)
  def eq(self, x): return ops.Eq.apply(self, x)
  # NOTES: Numpy allow broadcasting on batch dim (not the last 2 dims)
  # NOTES: However, we have not implement it for backward, should we?
  def matmul(self, x): return ops.Matmul.apply(self, x)

  # Reduce ops
  def sum(self, dim=None, keepdims=False): return ops.Sum.apply(self, dim=dim, keepdims=keepdims)
  def max(self, dim=None, keepdims=False): return ops.Max.apply(self, dim=dim, keepdims=keepdims)
  def min(self, dim=None, keepdims=False): return -ops.Max.apply(-self, dim=dim, keepdims=keepdims)
  def mean(self, dim=None, keepdims=False): 
    out = ops.Sum.apply(self, dim=dim, keepdims=keepdims)
    return out * math.prod(out.shape)/math.prod(self.shape)

  # Movement ops
  def reshape(self, *shape): return ops.Reshape.apply(self, shape=shape)
  def permute(self, *order): return ops.Permute.apply(self, order=order)
  # NOTES: Should we mirror PyTorch where passing -1 means keeping that dim size the same?
  # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
  def expand(self, *shape): return ops.Expand.apply(self, shape=shape)

  # TODO:
  # Processing ops
  # conv2d?

  # NOTES: This all could be generate using setattr, but im keeping this for clarity.
  __neg__ = neg
  __add__ = add
  __sub__ = sub
  __mul__ = mul
  __truediv__ = div
  __pow__ = pow
  __matmul__ = matmul
  # NOTES: Are there any cool tricks to golf this?
  def __radd__(self, x): return Tensor.add(x, self)
  def __rsub__(self, x): return Tensor.sub(x, self)
  def __rmul__(self, x): return Tensor.mul(x, self)
  def __rtruediv__(self, x): return Tensor.div(x, self)
