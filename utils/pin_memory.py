import time
import torch
from torch.utils.data import DataLoader, Dataset
from bitsandbytes.nn import Linear8bitLt

x = torch.randn(100000, 32).half()

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


Net = torch.nn.Sequential(Linear8bitLt(32, 128, has_fp16_weights=True), Linear8bitLt(128, 32, has_fp16_weights=True)).to("cuda")
dataset = MyDataset(x)

# pinned_memory
pin_dataset = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
output = []

# train loop
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for batch in pin_dataset:
    batch = batch.to("cuda:0", non_blocking=True)
    yhat = Net(batch)
    output.append(yhat)
end_event.record()
torch.cuda.synchronize()
out_tensor = torch.cat(output, dim=0)
del output
print(f"Pinned Memory elapsed time(s): {(start_event.elapsed_time(end_event)):.4f}, output size is: {out_tensor.size()}")

# pageable_memory
pageable_dataset = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
output = []
# train loop
start_event.record()
for batch in pageable_dataset:
    batch = batch.to("cuda:0")
    yhat = Net(batch)
    output.append(yhat)
end_event.record()
torch.cuda.synchronize()
out_tensor = torch.cat(output, dim=0)
del output
print(f"Unpinned Memory elapsed time(s): {(start_event.elapsed_time(end_event)):.4f}, output size is: {out_tensor.size()}")