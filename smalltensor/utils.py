import os
import networkx as nx

# BUG: This func is broken
def create_autodiff_graph(node):
  def walk(node, G):
    if node._ctx is not None:
      for i in node._ctx.parents:
        G.add_edge(hash(i), hash(node))
        walk(i, G)
    G.nodes[hash(node)]['label'] = str(node)

  G = nx.DiGraph()
  walk(node, G)

  nx.drawing.nx_pydot.write_dot(G, '/tmp/net.dot')
  os.system('dot -Tsvg /tmp/net.dot -o /tmp/net.svg')
  os.system('open /tmp/net.svg')

from itertools import zip_longest
from typing import Tuple

# NOTES: Should these 2 function be combine? They share pretty much the same functionality
def broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]):
  """
  Compute the broadcasted shape between 2 shapes. 
  broadcast_shapes((1, 2), (3, 2)) == (3, 2)
  broadcast_shapes((7,), (5, 1, 7)) == (5, 1, 7) """
  ret = []
  for idx, dims in enumerate(zip_longest(*[reversed(s) for s in [s1, s2]], fillvalue=1)):
    if min(dims) != 1 and (min(dims) != max(dims)):
      raise ValueError(f"Cannot broadcast shapes of {s1} with {s2} at idx:{idx} ({dims[0]} and {dims[1]})")
    ret.append(max(dims))
  return tuple(reversed(ret))

def broadcast_indices(s1: Tuple[int, ...], s2: Tuple[int, ...]):
  """
  Compute the indices that are broadcasted. 
  broadcast_indices((1, 2), (3, 2)) == (0,)
  broadcast_indices((6, 7), (5, 6, 1)) == (0, 2) """
  ret, mlen = [], max(len(s1), len(s2))
  for idx, dims in enumerate(zip_longest(*[reversed(s) for s in [s1, s2]])):
    dims = tuple(s for s in dims if s is not None)
    if min(dims) != 1 and (min(dims) != max(dims)):
      raise ValueError(f"Cannot broadcast shapes of {s1} with {s2} at idx:{idx} ({dims[0]} and {dims[1]})")
    elif min(dims) == 1 and (min(dims) != max(dims)) or len(dims) == 1:
      ret.append(mlen-idx-1)
  return tuple(reversed(ret))
