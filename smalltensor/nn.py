from smalltensor import Tensor

class Module:
  def _parameters(self):
    for i in self.__dict__.values():
      if isinstance(i, Module):
        yield from i._parameters()
      else:
        yield i

  def parameters(self):
    return list(self._parameters())

  def forward(self, x: Tensor):
    raise NotImplementedError(f"Not implemented for {type(self)}")

  def __call__(self, x: Tensor):
    return self.forward(x)

class Linear(Module):
  def __init__(self, in_features: int, out_features: int, bias: bool=True):
    self.w = Tensor.scale_uniform(out_features, in_features)
    self.b = Tensor.scale_uniform(out_features) if bias else Tensor.zeros(out_features)

  def forward(self, x):
    ret = x.matmul(self.w.permute(1, 0)).add(self.b)
    return ret
