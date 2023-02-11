import pytest

from smalltensor.utils import broadcast_shapes, broadcast_indices

def test_broadcast_shapes_equal():
  assert broadcast_shapes((1, 2), (3, 2)) == (3, 2)
  assert broadcast_shapes((1, 2), (3, 1)) == (3, 2)

def test_broadcast_shapes_unequal():
  assert broadcast_shapes((7,), (5, 1, 7)) == (5, 1, 7)
  assert broadcast_shapes((6, 7), (5, 6, 1)) == (5, 6, 7)

def test_broascast_shapes_fail():
  with pytest.raises(ValueError):
    broadcast_shapes((3, 4), (5,))

def test_broadcast_indices_equal():
  assert broadcast_indices((1, 2), (3, 2)) == (0,)
  assert broadcast_indices((1, 2), (3, 1)) == (0, 1)

def test_broadcast_indices_unequal():
  assert broadcast_indices((7,), (5, 1, 7)) == (0, 1)
  assert broadcast_indices((6, 7), (5, 6, 1)) == (0, 2)
  assert broadcast_indices((1, 1), (1,)) == (0,)
