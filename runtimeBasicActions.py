import torch
from torch.nn import functional as F
from tools import model_size, tensor_size

torch.cuda.memory._record_memory_history(stacks="all")


# t1 = torch.rand(100, 1000, 1000, device="cuda")
# t2 = torch.rand(100, 1000, 1000, device="cuda")
# t3 = torch.rand(100, 1000, 1000, device="cuda")
# print(tensor_size(t1))
# t4 = t1 * t2
# t5 = t4 + t1

# del t1, t2, t3, t4, t5
# torch.cuda.empty_cache()

# t1 = torch.rand(100, 1000, 1000, device="cuda") * 2
# t2 = torch.rand(100, 1000, 1000, device="cuda") * 2
# t3 = F.relu(t1)
# t4 = F.softmax(t2, dim=0)
# t5 = F.tanh(t3)

# del t1, t2, t3, t4, t5
# torch.cuda.empty_cache()

# t1 = torch.rand(100, 1000, 1000, device="cuda") * 2
# t2 = torch.rand(100, 1000, 1000, device="cuda") * 2
# t3 = F.relu_(t1)
# t4 = F.softmax(t2, dim=0)
# t5 = F.tanh(t3)

# del t1, t2, t3, t4, t5
# torch.cuda.empty_cache()

l1 = torch.nn.Sequential(torch.nn.Linear(1000, 1000, device="cuda"), torch.nn.ReLU())
t1 = torch.rand(100, 1000, 1000, device="cuda")
loss_f = torch.nn.L1Loss()
with torch.no_grad():
    y1 = l1(t1)
ls = loss_f(t1,y1)
# ls.backward()

del l1, t1, y1, ls
torch.cuda.empty_cache()

torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
