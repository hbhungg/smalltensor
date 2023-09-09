import numpy as np
from smalltensor import Tensor

def test_tensor_creation_numpy():
  shape = tuple(np.random.randint(1, 6) for _ in range(np.random.randint(1, 5)))  # Random shape
  arr = np.random.rand(*shape)
  assert Tensor(arr).shape == shape
  assert Tensor(arr) * arr