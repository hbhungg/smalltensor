from typing import List
from smalltensor import Tensor
import numpy as np

class Optimizer:
  def __init__(self, params: List[Tensor]):
    for param in params:
      param.requires_grad = True
    self.params: List[Tensor] = [t for t in params if t.requires_grad]

  def zero_grad(self):
    for param in self.params:
      param.grad = None

  def clipnorm(self, g):
    pass
  
  def step(self) -> None:
    raise NotImplementedError(f"Not implemented for {type(self)}")

class SGD(Optimizer):
  def __init__(self, params: List[Tensor], lr:float, momentum:float=0, nesterov:bool=False,
               weight_decay:float=0, dampening:float=0):
    super().__init__(params)
    self.lr, self.momentum = lr, momentum 
    self.weight_decay, self.dampening = weight_decay, dampening
    self.nesterov = nesterov
    self.b = [np.zeros(t.shape) for t in self.params] if self.momentum != 0 else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
  def step(self) -> None:
    for idx, param in enumerate(self.params):
      assert param.grad is not None
      g = param.grad.numpy()
      if self.weight_decay != 0:
        g = g + self.weight_decay * param.item
      if self.momentum != 0:
        self.b[idx] = self.momentum * self.b[idx] + g 
        g = (g + self.momentum * self.b[idx]) if self.nesterov else self.b[idx]
      param.item = param.item - self.lr * g


class Adam(Optimizer):
  def __init__(self, params: List[Tensor], lr:float, b1:float=0.9, b2:float=0.999, esp:float=1e-8):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.esp = lr, b1, b2, esp
    self.m = [np.zeros(t.shape) for t in self.params]
    self.v = [np.zeros(t.shape) for t in self.params]

  # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
  def step(self) -> None:
    for idx, param in enumerate(self.params):
      assert param.grad is not None
      g = param.grad.numpy()
      self.m[idx] = self.b1 + (1- self.b1)*g
      self.v[idx] = self.b2 + (1- self.b2)*g*g
      mh = self.m[idx]/(1-self.b1)
      vh = self.v[idx]/(1-self.b2)
      param.item = param.item - self.lr * self.m[idx]/(np.sqrt(self.v[idx]) + self.esp)


class RMSprop(Optimizer):
  def __init__(self, params: List[Tensor]):
    super().__init__(params)
