"""Check shared memory attributes via torch (which already loaded libcuda)."""
import torch
torch.cuda.init()
import ctypes

RTLD_NOLOAD = 4
libcuda = ctypes.CDLL("libcuda.so.1", mode=RTLD_NOLOAD)
dev = ctypes.c_int(0)
val = ctypes.c_int(0)

attrs = {
    8: "MAX_SHARED_MEMORY_PER_BLOCK",
    18: "SHARED_MEMORY_PER_BLOCK",
    81: "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
    97: "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
}
for attr_id, name in sorted(attrs.items()):
    ret = libcuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, dev)
    if ret == 0:
        print(f"Attr {attr_id:3d} ({name}): {val.value} bytes = {val.value/1024:.1f} KB")
