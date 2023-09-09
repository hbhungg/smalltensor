from __future__ import annotations
from .buffer import Buffer, Op, UnaryOps, BinaryOps, ReduceOps, MovementOps
import numpy as np

class NumpyBuffer(Buffer, np.ndarray):
  def unary_op(self, op: UnaryOps):
    match op:
      case UnaryOps.NEG: return np.negative(self)
      case UnaryOps.EXP: return np.exp(self)
      case UnaryOps.LOG: return np.log(self)

  def binary_op(self, op: BinaryOps, val: NumpyBuffer):
    match op:
      case BinaryOps.ADD: return np.add(self, val)
      case BinaryOps.SUB: return np.subtract(self, val)
      case BinaryOps.MUL: return np.multiply(self, val)
      case BinaryOps.POW: return np.power(self, val)
      case BinaryOps.CMPEQ: return np.array_equal(self, val)
  
  def reduce_op(self, op: ReduceOps, shape, keepdims):
    match op:
      case ReduceOps.SUM: return np.sum(self, shape, keepdims=keepdims)
      case ReduceOps.MAX: return np.amax(self, shape, keepdims=keepdims)

  def movement_op(self, op: MovementOps, arg):
     match op:
      case MovementOps.RESHAPE: return np.reshape(self, arg)
      case MovementOps.PERMUTE: return np.transpose(self, arg)
      case MovementOps.EXPAND: return np.broadcast_to(self, arg)