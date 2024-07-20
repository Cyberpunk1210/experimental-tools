import torch
import torch.nn.functional as F


x = torch.randn(2, 2)
y = torch.randn(2, 2)

loss = ((x - y)**2).mean()

_loss = F.mse_loss(x, y)

print(f"allclose: {torch.allclose(loss, _loss)}")

