from typing import Optional, Tuple, Union, List, Dict
import numpy as np

class Buffer:
  def __init__(self, buffer: Union[np.ndarray, List]):
    self._buffer = buffer if isinstance(buffer, np.ndarray) else np.array(buffer)

  @property
  def shape(self):
    return self._buffer.shape

  @property
  def strides(self):
    return self._buffer.strides
