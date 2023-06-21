from enum import Enum, auto
from typing import Union

class UnaryOps(Enum):
  NEG = auto()
  EXP = auto()
  LOG = auto()

class BinaryOps(Enum): 
  ADD = auto()
  SUB = auto()
  MUL = auto()
  DIV = auto()
  POW = auto()
  CMPEQ = auto()

class ReduceOps(Enum):
  SUM = auto()
  MAX = auto() # max(a) = a[0]; max element

class MovementOps(Enum): 
  RESHAPE = auto()
  PERMUTE = auto()
  EXPAND = auto()
  # PAD = auto()
  # SHRINK = auto()
  # STRIDE = auto()

Op = Union[UnaryOps, BinaryOps, ReduceOps, MovementOps]

class Buffer:
  def unary_op(self, op: UnaryOps):
    raise NotImplementedError(f"Not implemented for {self}")

  def binary_op(self, op: BinaryOps, val):
    raise NotImplementedError(f"Not implemented for {self}")

  def reduce_op(self, op: ReduceOps, shape, keepdims):
    raise NotImplementedError(f"Not implemented for {self}")

  def movement_op(self, op: MovementOps, shape):
    raise NotImplementedError(f"Not implemented for {self}")




