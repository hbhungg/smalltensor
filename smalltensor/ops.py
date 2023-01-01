from smalltensor.tensor import Function

class Add(Function):
  def forward(self, a, b):
    self.saved_for_backward(a, b)
    return a + b

  def backward(self, grad_output):
    a, b = self.saved_tensor
    return a*grad_output, b
