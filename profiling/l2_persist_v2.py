"""
L2 Cache Persistence v2: More realistic tests
Focus on scenarios where L2 persistence should matter:
1. Multiple tensors competing for L2 - pin one, measure if it stays hot
2. Gather from large buffer where working set < 60MB but total > L2
3. cuBLAS with streaming reads vs persistent weights
"""
import torch
import ctypes
import time

def setup_l2(size_mb):
    lib = ctypes.cdll.LoadLibrary('libcudart.so')
    lib.cudaDeviceSetLimit(0x06, size_mb * 1024 * 1024)

def set_persist(tensor, persist=True):
    lib = ctypes.cdll.LoadLibrary('libcudart.so')
    
    class AccessPolicyWindow(ctypes.Structure):
        _fields_ = [
            ("base_ptr", ctypes.c_void_p),
            ("num_bytes", ctypes.c_size_t),
            ("hitRatio", ctypes.c_float),
            ("hitProp", ctypes.c_int),
            ("missProp", ctypes.c_int),
        ]
    
    class StreamAttrValue(ctypes.Union):
        _fields_ = [("accessPolicyWindow", AccessPolicyWindow)]
    
    val = StreamAttrValue()
    w = val.accessPolicyWindow
    w.base_ptr = tensor.data_ptr()
    w.num_bytes = min(tensor.nelement() * tensor.element_size(), 60 * 1024 * 1024)
    if persist:
        w.hitRatio = 1.0
        w.hitProp = 2  # PERSISTING
        w.missProp = 1  # STREAMING (evict quickly)
    else:
        w.hitRatio = 0.0
        w.hitProp = 0
        w.missProp = 0
    
    return lib.cudaStreamSetAttribute(ctypes.c_void_p(0), ctypes.c_int(1), ctypes.byref(val), ctypes.c_size_t(ctypes.sizeof(val)))

print("=" * 70)
print("L2 Persistence v2: Cache Thrashing Scenarios")
print("=" * 70)

# Test: Weight stays hot while activations stream through
# Scenario: Expert weight (8MB) stays pinned while we process many batches
# This simulates repeated expert dispatch in MoE
print()
print("Test: Expert weight pinned while activations stream through")
print("-" * 70)

expert_weight = torch.randn(1024, 4096, dtype=torch.bfloat16, device='cuda')  # 8MB
weight_mb = expert_weight.nelement() * 2 / 1e6

# Create many different activation batches to thrash cache
num_batches = 50
batches = [torch.randn(64, 4096, dtype=torch.bfloat16, device='cuda') for _ in range(num_batches)]
# These batches total 64*4096*2*50 = 25MB - enough to thrash L2

def run_sequential_expert_dispatch(persist_weight, iters=200, warmup=50):
    """Simulate: for each batch, do matmul with the SAME expert weight."""
    if persist_weight:
        setup_l2(60)
        set_persist(expert_weight, True)
        # Mark activations as streaming (low priority for L2)
        for b in batches[:5]:
            set_persist(b, False)
    else:
        setup_l2(0)
    
    torch.cuda.synchronize()
    
    outputs = [torch.empty(64, 1024, dtype=torch.bfloat16, device='cuda') for _ in range(num_batches)]
    
    # Warmup
    for i in range(warmup):
        torch.mm(batches[i % num_batches], expert_weight.t(), out=outputs[i % num_batches])
    torch.cuda.synchronize()
    
    # Timed: cycle through all batches, always using same weight
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for i in range(iters):
        torch.mm(batches[i % num_batches], expert_weight.t(), out=outputs[i % num_batches])
    end.record()
    torch.cuda.synchronize()
    
    total_ms = start.elapsed_time(end)
    return total_ms / iters

t_no = run_sequential_expert_dispatch(persist_weight=False)
t_yes = run_sequential_expert_dispatch(persist_weight=True)
print(f"  Weight: {weight_mb:.0f} MB, {num_batches} batches cycling")
print(f"  NO persistence:   {t_no:.4f} ms/iter")
print(f"  WITH persistence: {t_yes:.4f} ms/iter")
print(f"  Speedup: {t_no/t_yes:.3f}x")

# Test: Gather from huge buffer, only accessing hot subset
print()
print("Test: Sparse gather from large buffer (hot subset fits in L2)")
print("-" * 70)

# 512MB buffer, but we only access ~32MB worth of pages repeatedly
total_elements = 256 * 1024 * 1024 // 2  # 256MB in bf16
big_buffer = torch.randn(total_elements, dtype=torch.bfloat16, device='cuda')

# Hot region: first 16M elements (32MB)
hot_size = 16 * 1024 * 1024  # elements
hot_region = big_buffer[:hot_size]

# Create gather indices: 80% from hot region, 20% from cold
import random
random.seed(42)
gather_size = 64 * 1024  # gather 128KB per op
hot_indices = torch.randint(0, hot_size, (gather_size,), dtype=torch.long, device='cuda')
cold_indices = torch.randint(hot_size, total_elements, (gather_size,), dtype=torch.long, device='cuda')

# Mix: 80% hot, 20% cold
num_hot = int(gather_size * 0.8)
mixed_indices = torch.cat([hot_indices[:num_hot], cold_indices[:gather_size - num_hot]])
mixed_indices = mixed_indices[torch.randperm(gather_size, device='cuda')]

def bench_gather(persist, iters=1000, warmup=200):
    if persist:
        setup_l2(60)
        set_persist(hot_region, True)
    else:
        setup_l2(0)
    torch.cuda.synchronize()
    
    out = torch.empty(gather_size, dtype=torch.bfloat16, device='cuda')
    
    for _ in range(warmup):
        out = big_buffer[mixed_indices]
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = big_buffer[mixed_indices]
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

t_no = bench_gather(persist=False)
t_yes = bench_gather(persist=True)
print(f"  Buffer: 256MB, hot region: 32MB, gather: {gather_size*2/1024:.0f} KB")
print(f"  NO persistence:   {t_no:.4f} ms")
print(f"  WITH persistence: {t_yes:.4f} ms")
print(f"  Speedup: {t_no/t_yes:.3f}x")

# Test: Multiple competing streams, one pinned
print()
print("Test: Competing kernels - pinned weight survives interference")
print("-" * 70)

weight_a = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')  # 8MB "hot" weight
weight_b = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')  # 8MB interference
weight_c = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda')  # 8MB interference
x = torch.randn(16, 2048, dtype=torch.bfloat16, device='cuda')

def bench_competing(persist_a, iters=500, warmup=100):
    if persist_a:
        setup_l2(60)
        set_persist(weight_a, True)
        # Mark interference as streaming
        set_persist(weight_b, False)
        set_persist(weight_c, False)
    else:
        setup_l2(0)
    torch.cuda.synchronize()
    
    # Pattern: A, B, C, A, B, C, ... (A should be faster if pinned)
    for _ in range(warmup):
        torch.mm(x, weight_a.t())
        torch.mm(x, weight_b.t())
        torch.mm(x, weight_c.t())
    torch.cuda.synchronize()
    
    times_a = []
    times_bc = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.mm(x, weight_a.t())
        e.record()
        torch.cuda.synchronize()
        times_a.append(s.elapsed_time(e))
        
        # Interference
        torch.mm(x, weight_b.t())
        torch.mm(x, weight_c.t())
    
    times_a.sort()
    return times_a[len(times_a) // 2]

t_no = bench_competing(persist_a=False)
t_yes = bench_competing(persist_a=True)
print(f"  Weight A: 8MB (pinned), B+C: 16MB (interference)")
print(f"  Pattern: A -> B -> C -> A -> B -> C -> ...")
print(f"  NO persistence (A median):   {t_no:.4f} ms")
print(f"  WITH persistence (A median): {t_yes:.4f} ms")
print(f"  Speedup for weight A: {t_no/t_yes:.3f}x")

# Cleanup
setup_l2(0)
del big_buffer
torch.cuda.empty_cache()

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
