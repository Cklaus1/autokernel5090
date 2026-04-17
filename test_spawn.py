import multiprocessing

def test():
    from flashinfer.jit.cpp_ext import get_cuda_path
    print(f'Child CUDA path: {get_cuda_path()}')
    import torch
    print(f'Child torch: {torch.__version__}')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    p = multiprocessing.Process(target=test)
    p.start()
    p.join()
    print(f'Child exit code: {p.exitcode}')
