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
def broadcast_shapes(*shapes):
  ret = []
  for dims in zip_longest(*[reversed(shape) for shape in shapes]):
    v = set(dims) - {None, 1}
    if len(v) < 2: 
      ret.append(next(iter(v)))
    else:
      raise ValueError(f"Cannot broadcast, mismatch shape.")
  return tuple(reversed(ret))
