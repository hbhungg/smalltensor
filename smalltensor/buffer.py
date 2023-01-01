from typing import Optional, Tuple, Union, List, Dict
import numpy as np

class Buffer:
  def __init__(self, buffer: Union[np.ndarray, List], shape: Tuple, strides: Optional[Tuple]=None):
    self._buffer = buffer if isinstance(buffer, np.ndarray) else np.ndarray(buffer)
    self.shape = shape
    self.strides = strides
