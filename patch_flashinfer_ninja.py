#!/usr/bin/env python3
"""Patch FlashInfer's JIT ninja build files to use CUDA 12.9 + gcc-12.

FlashInfer JIT generates ninja files with cuda_home=/usr which uses
system nvcc. On our system, /usr/bin/nvcc is 12.9 but CCCL headers at
/usr/include/cccl are incompatible (--host-stub-linkage-explicit error).

Fix: point cuda_home to /usr/local/cuda-12.9 which has matching CCCL,
and use gcc-12 as the C++ compiler.
"""
import os, sys

# Monkey-patch FlashInfer's build_ninja_file to fix cuda_home
import flashinfer.jit.cpp_ext as ext

_original_build_ninja = ext.build_ninja_file

def patched_build_ninja(*args, **kwargs):
    result = _original_build_ninja(*args, **kwargs)
    # The result is the content of the ninja file as a string
    if isinstance(result, str):
        result = result.replace('cuda_home = /usr\n', 'cuda_home = /usr/local/cuda-12.9\n')
        result = result.replace('cxx = c++\n', 'cxx = /usr/bin/g++-12\n')
    return result

ext.build_ninja_file = patched_build_ninja

# Also patch already-generated ninja files on disk
cache_dir = os.path.expanduser('~/.cache/flashinfer/0.6.4/120a/cached_ops')
if os.path.isdir(cache_dir):
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f == 'build.ninja':
                path = os.path.join(root, f)
                with open(path) as fh:
                    content = fh.read()
                if 'cuda_home = /usr\n' in content:
                    content = content.replace('cuda_home = /usr\n', 'cuda_home = /usr/local/cuda-12.9\n')
                    content = content.replace('cxx = c++\n', 'cxx = /usr/bin/g++-12\n')
                    with open(path, 'w') as fh:
                        fh.write(content)
                    print(f'Patched: {path}')

print('FlashInfer ninja files patched for CUDA 12.9 + gcc-12')
