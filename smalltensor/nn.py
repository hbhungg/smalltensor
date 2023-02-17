from smalltensor import Tensor

class Module:
  def __init__(self):
    pass

  def parameters(self):
    for i in self.__dict__.values():
      yield i

  def forward(self, x: Tensor):
    raise NotImplementedError(f"Not implemented for {type(self)}")

  def __call__(self, x: Tensor):
    return self.forward(x)

class Linear:
  pass
