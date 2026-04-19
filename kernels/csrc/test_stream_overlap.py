import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import time

dev = 'cuda:0'
n = 256 * 1024 * 1024
a = torch.randint(-128, 127, (n,), dtype=torch.int8, device=dev)
b = torch.empty_like(a)
c = torch.empty_like(a)
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    b.copy_(a)
    c.copy_(a)
torch.cuda.synchronize()
t_serial = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(20):
    with torch.cuda.stream(s1):
        b.copy_(a)
    with torch.cuda.stream(s2):
        c.copy_(a)
torch.cuda.synchronize()
t_parallel = time.perf_counter() - t0

print(f'serial={t_serial*1000:.2f}ms  parallel={t_parallel*1000:.2f}ms  ratio={t_serial/t_parallel:.2f}')
