import math
from smalltensor.tensor import Function

# *********** Unary ops **********

class Neg(Function):
  def forward(self, a):
    return -a

  def backward(self, grad_output):
    return -grad_output

class Inv(Function):
  def forward(self, a):
    self.saved_for_backward(a)
    return 1/a
  
  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output*(-1/(a*a))

class ReLU(Function):
  def forward(self, a):
    self.saved_for_backward(a)
    return max(a, 0.0)

  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output if a > 0 else 0.0

class Log(Function):
  def forward(self, a):
    self.saved_for_backward(a)
    return math.log(a)

  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output * 1/a

class Exp(Function):
  def forward(self, a):
    self.saved_for_backward(a)
    return math.exp(a)

  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output * math.exp(a)

# *********** Reduce ops **********

class Sum(Function):
  def forward(self, a):
    raise NotImplementedError("will implement")

  def backward(self, grad_output):
    raise NotImplementedError("will implement")

class Max(Function):
  def forward(self, a):
    raise NotImplementedError("will implement")

  def backward(self, grad_output):
    raise NotImplementedError("will implement")

# *********** Binary ops **********

class Add(Function):
  def forward(self, a, b):
    return a+b

  def backward(self, grad_output):
    return grad_output, grad_output

class Sub(Function):
  def forward(self, a, b):
    return a-b

  def backward(self, grad_output):
    return grad_output, -grad_output

class Mul(Function):
  def forward(self, a, b):
    self.saved_for_backward(a, b)
    return a*b

  def backward(self, grad_output):
    a, b = self.saved_tensor
    return b*grad_output, a*grad_output

#class Pow(Function):
#  def forward(self, a, b):
#    self.saved_for_backward(a, b)
#    return a**b
#
#  def backward(self, grad_output):
#    a, b = self.saved_tensor
#    return grad_output*b*(a**(b-1)), grad_output*(a**b)*math.log(abs(a))

class Eq(Function):
  def forward(self, a, b):
    return a == b

  def backward(self, grad_output):
    return 0.0, 0.0

# TODO: Do we need this 2 ?
# *********** Movement ops **********
# *********** Processing ops **********
