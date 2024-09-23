import inspect
import torch

torch.manual_seed(114514)
precision = torch.float16

x = torch.randn(3, dtype=precision)
y = torch.randn(3, dtype=precision)

# example 1: y = x * x
def simple_predict_first(x):
    return x * x

Jvf = torch.eye(x.numel(), dtype=precision)
for i in range(x.numel()):
    Jvf[i][i] = x[i] * 2

Jacf = torch.autograd.functional.jacobian(simple_predict_first, x)
print(torch.allclose(Jacf, Jvf))  # True


# example 2: z = x * x + 3 * y
def simple_predict_second(x, y):
    return x * x + 3 * y

sign = inspect.signature(simple_predict_second).parameters.keys().__len__()

Jvsx = tuple(torch.eye(x.numel(), dtype=precision) for _ in range(sign))

for i in range(x.shape[0]):
    Jvsx[0][i][i] = x[i] * 2
for i in range(y.shape[0]):
    Jvsx[1][i][i] = y[i] * 0 + 3

Jacs = torch.autograd.functional.jacobian(simple_predict_second, (x, y))

print(torch.allclose(Jacs[0], Jvsx[0]))  # True
print(torch.allclose(Jacs[1], Jvsx[1]))  # True
