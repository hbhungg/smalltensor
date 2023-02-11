import numpy as np
from .tensor import Function
from .utils import broadcast_indices

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
    return np.maximum(a, 0.0)

  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output * np.greater(a, 0)

class Log(Function):
  def forward(self, a):
    self.saved_for_backward(a)
    return np.log(a)

  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output * 1/a

class Exp(Function):
  def forward(self, a):
    self.saved_for_backward(a)
    return np.exp(a)

  def backward(self, grad_output):
    a = self.saved_tensor[0]
    return grad_output * np.exp(a)

# *********** Reduce ops **********

class Sum(Function):
  def forward(self, a, dim, keepdims):
    self.in_shape = a.shape
    return np.sum(a, dim, keepdims=keepdims)

  def backward(self, grad_output):
    return np.broadcast_to(grad_output, self.in_shape)

class Max(Function):
  def forward(self, a, dim, keepdims):
    self.saved_for_backward(a, dim)
    return np.amax(a, dim, keepdims=keepdims)

  def backward(self, grad_output):
    a, dim = self.saved_tensor
    ret = np.zeros(a.shape)
    argmax = a.argmax(axis=dim)
    if dim is not None:
      ret[np.arange(len(a)), argmax] = 1
    else:
      ret[np.unravel_index(argmax, a.shape)] = 1
    return grad_output*ret

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
    return grad_output*b, grad_output*a

class Pow(Function):
 def forward(self, a, b):
   ret = np.power(a, b)
   self.saved_for_backward(a, b, ret)
   return ret

 def backward(self, grad_output):
   a, b, ret = self.saved_tensor
   return grad_output*b*ret/a, grad_output*ret*np.log(a)

class Eq(Function):
  def forward(self, a, b):
    return a == b

  def backward(self, grad_output):
    return 0.0, 0.0

class Matmul(Function):
  def forward(self, a, b):
    self.saved_for_backward(a, b)
    return np.matmul(a, b)

  def backward(self, grad_output):
    a, b = self.saved_tensor
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    # N-D, N>2 ndarray are treated as stacks of 2-D matrices
    # Transposing it means swapping the last 2 dims (-1 and -2)
    return np.matmul(grad_output, np.swapaxes(b, -1, -2)), \
           np.matmul(np.swapaxes(a, -1, -2), grad_output)

# *********** Movement ops **********

class Reshape(Function):
  def forward(self, a, shape):
    self.in_shape = a.shape
    return np.reshape(a, shape)

  def backward(self, grad_output):
    return np.reshape(grad_output, self.in_shape)

class Expand(Function):
  def forward(self, a, shape):
    self.in_shape = a.shape
    return np.broadcast_to(a, shape)

  def backward(self, grad_output):
    axs = broadcast_indices(self.in_shape, grad_output.shape)
    return np.sum(grad_output, axs, keepdims=True)

class Permute(Function):
  def forward(self, a, order):
    self.order = order
    return np.transpose(a, order)

  def backward(self, grad_output):
    return np.transpose(grad_output, np.argsort(self.order))

# NOTES: Not important?
# class Slice(Function):
#   pass
#
# class Flip(Function):
#   pass

# *********** Processing ops **********

# TODO: Conv2d?
