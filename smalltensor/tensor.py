from typing import Optional, Tuple, Union, List, Dict
import inspect, importlib, functools
import numpy as np


class Tensor:
  def __init__(self, data):
    self.data = data
    self._grad: Optional[Tensor] = None
    self.requires_grad: Optional[Bool] = None


class Function:
  def __init__(self, *variables: Tensor):
    self.saved_tensor: List[Tensor] = []
    self.needs_input_grad: List[Tensor] = variables

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
  if name != "Function": setattr(Tensor, f"__{name.lower()}__", functools.partialmethod(cls.apply))
# setattr(Tensor, f"__{name}__", 
