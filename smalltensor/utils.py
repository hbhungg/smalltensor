import os
import networkx as nx

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

def broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]):
  """
  Find our the broadcasted shape between 2 shapes. 
  broadcast_shapes((1, 2), (3, 2)) == (3, 2)
  broadcast_shapes((7,), (5, 1, 7)) == (5, 1, 7) """
  ret = []
  for idx, dims in enumerate(zip_longest(*[reversed(s) for s in [s1, s2]], fillvalue=1)):
    v = set(dims)
    if min(v) != 1 and len(v) == 2:
      raise ValueError(f"Cannot broadcast shapes of {s1} with {s2} at idx:{idx} ({dims[0]} and {dims[1]})")
    ret.append(max(v))
  return tuple(reversed(ret))
