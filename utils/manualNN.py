import torch
import torch.nn.functional as F


max_step = 100000

g = torch.Generator().manual_seed(2147483647)
x = torch.randn(8, 6, requires_grad=True, generator=g)
y = torch.randn(8, 4, generator=g)

W1 = torch.randn(6, 4, requires_grad=True, generator=g) * 0.1
b1 = torch.randn(4, requires_grad=True, generator=g) * 0.1

parameters = [x, W1, b1]
lossi = []

B, C = x.size()
# for p in parameters:
#     p.requires_grad = True

with torch.no_grad():
    for i in range(max_step):
        yhat = x @ W1 + b1
        logit = torch.log(1 + torch.exp(yhat))
        loss = F.mse_loss(logit, y)
        # for t in parameters:
        #     t.retain_grad()
        # # loss.backward()
        for p in parameters:
            p.grad = None

        dlogit = torch.ones_like(logit)
        dlogit = 2 * (logit - y) / dlogit.nelement()
        dyhat = torch.exp(yhat) / (torch.exp(yhat) + 1.0) * dlogit
        dx = dyhat @ W1.T
        dW1 = x.T @ dyhat
        db1 = dyhat.sum(0)

        grads = [dx, dW1, db1]
        lr = 0.1 if i < 50000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad
        if i%2000 == 0:
            print(f"{i:7d}/{max_step:7d}: {loss.item():.4f}")
        lossi.append(loss.log10().item())

# TODO Adam
