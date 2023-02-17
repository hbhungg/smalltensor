from smalltensor import Tensor
import smalltensor.optim as optim
import smalltensor.nn as nn

from itertools import product
import random

class XorModel(nn.Module):
  def __init__(self):
    self.w1 = Tensor.scale_uniform(10, 2)
    self.b1 = Tensor.scale_uniform(10)
    self.w2 = Tensor.scale_uniform(1, 10)
    self.b2 = Tensor.scale_uniform(1)

  def forward(self, x):
    r = x.matmul(self.w1.permute(1, 0)).add(self.b1).sigmoid()
    r = r.matmul(self.w2.permute(1, 0)).add(self.b2).sigmoid()
    return r

if __name__ == "__main__":
  model = XorModel()
  opt = optim.SGD([model.w1, model.b1, model.w2, model.b2], lr=0.3)

  b4 = model.w1.detach().numpy()
  x_train = list(product([0, 1], [0, 1]))
  for epoch in range(1000):
    print(epoch)
    al = 0.0
    random.shuffle(x_train)
    for x, y in x_train:
      inp = Tensor([[x, y]])
      # [class 0, class 1]
      true = Tensor([[x^y]])
      out = model(inp)

      loss = (true - out).square().mean()
      al += loss.numpy()

      loss.backward()
      opt.step()
      opt.zero_grad()

      print(f"Loss: {loss.numpy():.3f}, True: {true.numpy()}, Pred: {out.numpy()}")
    print(f"Avg loss: {al/4}")

  # print(b4)
  # print(model.w1)
