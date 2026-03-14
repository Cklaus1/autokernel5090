import torch
props = torch.cuda.get_device_properties(0)
print(f"Shared memory per block (static): {props.shared_memory_per_block}")
# CUDAAttribute for max dynamic shared memory
import ctypes
RTLD_NOLOAD = 4
libcuda = ctypes.CDLL("libcuda.so.1", mode=RTLD_NOLOAD)
dev = ctypes.c_int(0)
val = ctypes.c_int(0)
# CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
ret = libcuda.cuDeviceGetAttribute(ctypes.byref(val), 97, dev)
print(f"Max shared memory per block (optin, attr 97): {val.value} (ret={ret})")
# CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
ret = libcuda.cuDeviceGetAttribute(ctypes.byref(val), 81, dev)
print(f"Max shared memory per SM (attr 81): {val.value} (ret={ret})")

import triton
print(f"Triton {triton.__version__}")
# Look at Triton's shared memory limit detection
import triton.runtime.driver as trd
