from itertools import combinations, chain
from functools import reduce
import random

# https://docs.python.org/2/library/itertools.html#recipes
def powerset(iterable):
  """
  powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
  s = list(iterable)
  return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def random_reduce(fxn, iter, l):
  """
  Like reduce, but only on random number of elements in iter
  If l==len(iter), it is just a normal reduce
  Eg: random_reduce(lambda x,y:x+y, [1, 2, 3], 2) = [4, 2]
  Used for generating reshape reshape for testing"""
  z = random.sample(range(len(iter)), l) 
  s = reduce(fxn, [iter[i] for i in z])
  return [s] + [j for i,j in enumerate(iter) if i not in z]
