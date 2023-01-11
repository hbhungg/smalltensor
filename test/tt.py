from smalltensor.tensor import Tensor
import random

g = []
for i in range(1, 4):
  g.append(Tensor(i))

def ssum(lst):
  ret = lst[0]
  for i in range(1, len(lst)):
      ret /= lst[i]
  return ret

#dw = ssum(g)
#print(dw.backward())
#print(g[0].grad)
a = Tensor(10)
c = 2/a
print(c.backward())

from smalltensor.utils import create_autodiff_graph
create_autodiff_graph(c)

