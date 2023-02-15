import pytest
from smalltensor import Tensor
import smalltensor.optim as optim
import torch

def util_test_optim(smt_optim, torch_optim):
  sts = [Tensor.randn(3, 3, requires_grad=True)]
  tts = [torch.tensor(i.numpy(), requires_grad=True) for i in sts]

  cst = sum(sts).sum()
  cst.backward()

  ctt = sum(tts).sum()
  ctt.backward()

  smt_optim(params=sts+[cst], lr=0.1)
  torch_optim(params=tts+[ctt], lr=0.1)

  smt_optim.step()
  torch_optim.step()

  for s, t in zip(sts, tts):
    assert np.assert_close(s.numpy(), t.detach().numpy())

@pytest.mark.skip()
def test_adam():
  util_test_optim(optim.Adam, torch.optim.Adam)
