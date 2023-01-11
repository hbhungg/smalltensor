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
    return -1/a*a

class ReLU(Function):
  def forward(self, a):
    raise NotImplementedError("will implement")

  def backward(self, grad_output):
    raise NotImplementedError("will implement")

class Log(Function):
  def forward(self, a):
    raise NotImplementedError("will implement")

  def backward(self, grad_output):
    raise NotImplementedError("will implement")

class Exp(Function):
  def forward(self, a):
    raise NotImplementedError("will implement")

  def backward(self, grad_output):
    raise NotImplementedError("will implement")

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

class Mul(Function):
  def forward(self, a, b):
    self.saved_for_backward(a, b)
    return a*b

  def backward(self, grad_output):
    a, b = self.saved_tensor
    return b*grad_output, a*grad_output

class Pow(Function):
  def forward(self, a, b):
    raise NotImplementedError("will implement")

  def backward(self, grad_output):
    raise NotImplementedError("will implement")

# TODO: Do we need this 2 ?
# *********** Movement ops **********
# *********** Processing ops **********
