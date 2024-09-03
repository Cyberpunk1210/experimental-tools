import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


x = torch.randn(10, 1)

# softplus formula: f(x) = ln(1+exp(x))
# dy = dx * torch.exp(x) / torch.exp(x) + 1
sp = torch.log(1.0 + torch.exp(x))
_x = F.softplus(x)
print(f"sp and _x weather to allclose: {torch.allclose(_x, sp)}")

plt.plot(sp, _x)
plt.show()

