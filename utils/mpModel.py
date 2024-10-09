import torch
import torch.amp
from torch.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

N, D_in, H, D_out = 640, 4096, 2048, 1024
use_amp = True
dtype = torch.float16
scaler = GradScaler(enabled=use_amp)


model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.1))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def train(model):
    data, target = torch.randn(10000, D_in), torch.randn(10000, D_out)
    loader = DataLoader(TensorDataset(data, target), shuffle=True, batch_size=32)
    for data, label in loader:
        optimizer.zero_grad()
        yhat = model(data)
        loss_fn(yhat, label).backward()
        optimizer.step()


if __name__ == "__main__":
    num_processes = 8
    model.share_memory()  # use fork method to work
    process = []

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        process.append(p)
        for p in process:
            p.join()


# cuda stream demonstration
# static_input = torch.randn(N, D_in, device="cuda")
# static_target = torch.randn(N, D_out, device="cuda")

# s = torch.cuda.Stream()
# s.wait_stream(torch.cuda.current_stream())
# with torch.cuda.Stream(s):
#     for i in range(3):
#         optimizer.zero_grad(set_to_none=True)
#         with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
#             y_pred = model(static_input)
#             loss = loss_fn(y_pred, static_target)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
# torch.cuda.current_stream().wait_stream(s)

# g = torch.cuda.CUDAGraph()
# optimizer.zero_grad(set_to_none=True)
# with torch.cuda.graph(g):
#     with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
#         static_y_pred = model(static_input)
#         static_loss = loss_fn(static_y_pred)
#     scaler.scale(static_loss).backward()

# real_inputs = [torch.rand_like(static_input) for _ in range(10)]
# real_targets = [torch.rand_like(static_target) for _ in range(10)]

# for data, target in zip(real_inputs, real_targets):
#     static_input.copy_(data)
#     static_target.copy_(target)
#     g.replay()
#     # Runs scaler.step and scaler.update eagerly
#     scaler.step(optimizer)
#     scaler.update()

