import os
from smalltensor import tensor

def set_printoptions(lvl: int):
  os.environ['VERBOSE'] = f"{lvl}"

