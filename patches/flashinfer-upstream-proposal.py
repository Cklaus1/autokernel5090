"""
Proposed upstream fix for FlashInfer JIT compilation on Blackwell (SM120+) GPUs.

Problem:
  FlashInfer's JIT compiler uses the system default nvcc and gcc. On systems
  where multiple CUDA toolkits are installed (e.g., CUDA 12.8 as system default,
  CUDA 12.9 for SM120 FP4 support), the JIT compiler picks the wrong nvcc and
  fails with cryptic errors. Additionally, newer gcc versions (13+) are rejected
  by nvcc 12.9.

Fix:
  1. Add FLASHINFER_CUDA_HOME env var for explicit override (highest priority)
  2. Auto-detect the newest compatible CUDA toolkit for the current GPU
  3. Auto-detect a compatible gcc version for the selected nvcc
  All changes are opt-in via env vars or auto-detected only when needed.
  Existing behavior is preserved on systems where the defaults work.

Author: Chris Klaus <cklaus@fusen.world>
Found during RTX 5090 (SM120) kernel optimization with AutoKernel.
"""

# === Change 1: get_cuda_path() in flashinfer/jit/cpp_ext.py ===
# BEFORE:
#   @functools.cache
#   def get_cuda_path() -> str:
#       cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
#       ...

# AFTER:
import functools
import glob
import logging
import os
import subprocess

logger = logging.getLogger("flashinfer.jit")


@functools.cache
def _get_gpu_compute_capability() -> tuple[int, int] | None:
    """Detect the compute capability of the first CUDA GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            cap = result.stdout.strip().split("\n")[0]
            major, minor = cap.split(".")
            return (int(major), int(minor))
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


@functools.cache
def _find_best_cuda_home() -> str | None:
    """Find the newest CUDA toolkit that supports the current GPU.

    Searches standard install locations: /usr/local/cuda-*
    Returns None if no suitable toolkit is found (falls back to defaults).
    """
    gpu_cc = _get_gpu_compute_capability()
    if gpu_cc is None:
        return None

    # SM120 (Blackwell) needs CUDA 12.9+
    # SM100 needs CUDA 12.8+
    # SM90 (Hopper) needs CUDA 12.0+
    min_cuda = {12: "12.9", 10: "12.8", 9: "12.0"}.get(gpu_cc[0])
    if min_cuda is None:
        return None

    # Search for installed CUDA toolkits
    candidates = sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    for path in candidates:
        nvcc = os.path.join(path, "bin/nvcc")
        if not os.path.exists(nvcc):
            continue
        # Extract version from path (e.g., /usr/local/cuda-12.9 → 12.9)
        version_str = os.path.basename(path).replace("cuda-", "")
        try:
            parts = version_str.split(".")
            version = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
            min_parts = min_cuda.split(".")
            min_version = (int(min_parts[0]), int(min_parts[1]))
            if version >= min_version:
                logger.info(
                    f"Auto-detected CUDA {version_str} at {path} for GPU SM{gpu_cc[0]}{gpu_cc[1]}"
                )
                return path
        except (ValueError, IndexError):
            continue

    return None


@functools.cache
def _find_compatible_gcc(nvcc_path: str) -> str | None:
    """Find a gcc version compatible with the given nvcc.

    nvcc has maximum supported gcc version constraints:
      CUDA 12.9: gcc <= 12
      CUDA 12.8: gcc <= 12
      CUDA 12.0-12.7: gcc <= 12

    Returns the path to a compatible gcc, or None to use system default.
    """
    # Check if system gcc works with this nvcc (try a trivial compile)
    try:
        result = subprocess.run(
            [nvcc_path, "--compiler-bindir", "/usr/bin", "-x", "cu", "/dev/null",
             "-o", "/dev/null", "--dry-run"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return None  # System gcc is fine
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # System gcc failed — search for older versions
    for version in [12, 11, 10]:
        gcc_path = f"/usr/bin/gcc-{version}"
        if os.path.exists(gcc_path):
            logger.info(f"Using {gcc_path} for nvcc compatibility")
            return gcc_path

    return None


@functools.cache
def get_cuda_path() -> str:
    """Get the CUDA toolkit path, with auto-detection for newer GPUs.

    Priority:
      1. FLASHINFER_CUDA_HOME env var (explicit override)
      2. CUDA_HOME / CUDA_PATH env vars (standard)
      3. Auto-detected best toolkit for current GPU
      4. System nvcc location
      5. /usr/local/cuda default
    """
    # Priority 1: explicit FlashInfer override
    flashinfer_home = os.environ.get("FLASHINFER_CUDA_HOME")
    if flashinfer_home is not None:
        return flashinfer_home

    # Priority 2: standard env vars
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home

    # Priority 3: auto-detect for current GPU
    auto_home = _find_best_cuda_home()
    if auto_home is not None:
        return auto_home

    # Priority 4-5: existing logic (find nvcc in PATH, fallback to /usr/local/cuda)
    try:
        nvcc_path = subprocess.run(
            ["which", "nvcc"], capture_output=True, check=True
        )
        return os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    except subprocess.CalledProcessError:
        cuda_home = "/usr/local/cuda"
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
        return cuda_home


# === Change 2: _build_cuda_cflags() — auto-detect compatible gcc ===
# Add after cc_env = os.environ.get("CC"):

def _build_cuda_cflags_patch():
    """Patch: auto-detect compatible gcc for nvcc."""
    cc_env = os.environ.get("CC")
    if cc_env is None:
        cuda_home = get_cuda_path()
        nvcc_path = os.path.join(cuda_home, "bin/nvcc")
        compatible_gcc = _find_compatible_gcc(nvcc_path)
        if compatible_gcc is not None:
            cc_env = compatible_gcc
    return cc_env


# === Change 3: CXX default — match gcc version ===
# Replace: cxx = os.environ.get("CXX", "c++")
# With:

def _get_cxx_default():
    """Get CXX compiler matching the CC compiler for consistency."""
    cxx = os.environ.get("CXX")
    if cxx is not None:
        return cxx
    # If we auto-detected gcc-N, use g++-N
    cc = os.environ.get("CC")
    if cc and "gcc-" in cc:
        gxx = cc.replace("gcc-", "g++-")
        if os.path.exists(gxx):
            return gxx
    return "c++"
