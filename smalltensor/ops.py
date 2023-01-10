from smalltensor.tensor import Function

#  ********** Unary ops **********

class Neg(Function):
  def forward(self, a):
    return -a
  def backward(self, grad_output):
    return -grad_output

# *********** Binary ops **********

class Add(Function):
  def forward(self, a, b):
    return a + b

  def backward(self, grad_output):
    return grad_output, grad_output


class Mul(Function):
  def forward(self, a, b):
    self.saved_for_backward(a, b)
    return a*b

  def backward(self, grad_output):
    a, b = self.saved_tensor
    return a*grad_output, b*grad_output
