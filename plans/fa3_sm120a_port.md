# FlashAttention-3 → SM120a (RTX PRO 6000 Blackwell) Port Attempt

Date: 2026-04-17
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (compute capability 12.0, `sm_120`)
Toolchain: `nvcc 12.8.93`, PyTorch `2.11.0+cu130`, Python 3.12
Upstream: `github.com/Dao-AILab/flash-attention@main`, `hopper/` subdirectory
Clone path: `/tmp/flash-attention` (shallow, + `csrc/cutlass` shallow submodule)

Verdict: **KILL** (feasibility blocker: WGMMA + warp-spec pipeline are not emulable on SM120).

## 1. What FA3 actually requires from the GPU

FA3's forward is implemented in `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` +
`hopper/flash_fwd_kernel_sm90.h`. It is a **warp-specialized** producer/consumer
pipeline built on four SM90-only hardware features:

| FA3 requirement | PTX/CUTE primitive | SM90a | SM120a |
| --- | --- | --- | --- |
| Async Tensor-Core GEMM in producer-consumer pipeline | `wgmma.mma_async.sync.aligned.*` (CUTE `cute::GMMA::ss_op_selector` → SM90 `GMMA_64x*x*_F*_SS`) | YES | **NO** |
| GMMA descriptor + swizzled shared-mem layouts | `cute::GMMA::Major::K / MN`, swizzle descriptors tied to WGMMA | YES | **NO** |
| Warp-group register reallocation | `setmaxnreg.inc.sync.aligned.u32` | YES | **NO** |
| TMA (bulk async copy G→S) | `cp.async.bulk.tensor.*` | YES | **YES** |
| Cluster barriers (CGA) | `barrier.cluster.*`, `fence.proxy.async.shared::cta` | YES | **YES** |
| STSM / LDSM (shared-mem matrix moves) | `stmatrix.sync.aligned.*` | YES | **YES** |
| mbarrier (async pipeline) | `mbarrier.*` | YES | **YES** |

Source of truth in the vendored CUTLASS:
- `csrc/cutlass/include/cutlass/arch/config.h:48` — `CUTLASS_ARCH_MMA_SM90A_ENABLED`
  only fires when `__CUDA_ARCH__ == 900`, never for SM120.
- `csrc/cutlass/include/cute/arch/config.hpp:56-62` — SM100a/SM120a **do** enable
  `CUTE_ARCH_TMA_SM90_ENABLED`, `CUTE_ARCH_DEVICE_MODIFIABLE_TMA_SM90_ENABLED`,
  `CUTE_ARCH_STSM_SM90_ENABLED`. But they do **not** enable the WGMMA macro.
- `csrc/cutlass/include/cute/arch/mma_sm90_gmma.hpp` — every WGMMA op is wrapped
  in `#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) ... #else CUTE_INVALID_CONTROL_PATH(...)`.
- SM120 has its own tensor-core MMAs in `cute/arch/mma_sm120.hpp`
  (`SM120_16x8x32_TN` — plain `mma.sync.aligned.kind::f8f6f4.m16n8k32.*`, same
  form as SM89/SM80, NOT WGMMA). These are synchronous and not producer-consumer.

## 2. SM120a compute-capability reference (per PTX ISA 8.x + CUTLASS config.h)

SM120 on consumer Blackwell inherits **Hopper TMA / STSM / mbarrier / async
pipeline** but **NOT** WGMMA (`wgmma.*`) or TMEM (`tcgen05.*`). Datacenter
Blackwell (SM100) has TMEM but still no WGMMA — it replaces WGMMA with
`tcgen05.mma.*` which addresses the tensor-memory register file. SM120 has
neither: it uses traditional warpgroup-synchronous `mma.sync.aligned.m16n8k*`
tensor cores, same as SM89 but widened to FP8/FP6/FP4 (`kind::f8f6f4`).

Discovery #24 confirmed: SM90 and SM100 cubins are **binary-incompatible with
SM120** — runtime loader refuses. So "just build FA3 for sm_90a" does not work
on this GPU: `cudaLaunchKernel` returns `no kernel image available for device`.

## 3. Build attempt

### 3.1. Shallow clone + cutlass submodule

```
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
git -C /tmp/flash-attention submodule update --init --depth 1 csrc/cutlass
```

Both succeed. Cutlass is pinned at `7127592069c2fe01b041e174ba4345ef9b279671`
(has `mma_sm120.hpp`).

### 3.2. First compile — single TU, `-gencode arch=compute_120a,code=sm_120a`

Driver: `plans/fa3_probe.py` compiles
`hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu` with the same flags
`hopper/setup.py` would use, but with `sm_120a` instead of `sm_90a`. Full log:
`/tmp/fa3_build_attempt.log`.

Result: `RC=0`, 139760-byte object file. **But it is a false success.**

`cuobjdump --dump-sass /tmp/fa3_probe_120a.o`:
```
ELF: flash_fwd_hdim128_bf16_sm90.sm_120a.cubin
arch = sm_120a
Function : _ZN7cutlass13device_kernelIN5flash11enable_sm90INS1_16FlashAttnFwdSm90I...
  /*0000*/  LDC R1, c[0x0][0x37c];
  /*0010*/  EXIT;
  /*0020*/  BRA 0x20;
  (rest: NOPs)
```

The kernel body is completely stripped. Root cause: `hopper/utils.h:36-44`:

```cpp
template <typename Kernel>
struct enable_sm90 : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};
```

The launcher wraps every SM90 kernel in `enable_sm90<...>`
(`flash_fwd_launch_template.h:79`). For `__CUDA_ARCH__ == 1200` the body is
empty, so we get a valid cubin containing `LDC;EXIT` — launches fine, produces
garbage output, never writes the result tensor. Silent correctness bug.

Dispatch code in `hopper/static_switch.h:138-150` (`ARCH_SWITCH` macro) also
maps every `arch >= 90` runtime-detected SM to the `Arch=90` template — so on
SM120 hardware, the device-side kernel body being empty means the whole forward
is a no-op at runtime.

### 3.3. Second compile — patched `enable_sm90` to allow `__CUDA_ARCH__ == 1200`

Driver: `plans/fa3_patch_and_probe.py`. Same TU; cubin grew to 452080 bytes,
~12261 SASS lines. RC=0 still.

But counting opcodes in the SASS (see log, `--- opcode counts (patched) ---`):

```
WGMMA: 0     HMMA: 0      IMMA: 0      BMMA: 0      MMA: 0    TCGEN: 0    TMEM: 0
UTMA: 54     STS: 17      STSM: 16     BAR: 31      FENCE: 14
```

**Zero tensor-core MMAs were emitted.** TMA / STSM / barriers / fences compile
fine, but every `cute::GMMA::ss_op_selector<...>()::fma(...)` call expanded into
`CUTE_INVALID_CONTROL_PATH("... without CUTE_ARCH_MMA_SM90A_ENABLED")`, which
`cute/config.hpp:128-133` defines as `assert(0 && x); printf(x); __brkpt()`.
With `-DNDEBUG` the assert is a no-op and `printf`+`__brkpt` get dead-coded out
when the compiler sees the returned `DRegisters` are unused. Net effect: the Q·K
and P·V matmuls are silently omitted. Launching this kernel would touch memory
via TMA and STSM but never compute anything — correctness failure #2.

So there is **no compile error** — FA3 relies on preprocessor-guarded intrinsics
that degrade to runtime traps / dead code on non-SM90 archs. A real port has to
replace the MMA dispatch, not fix a compiler error.

## 4. What would a real port require?

Four layered changes, in order of difficulty:

1. **Trivial (5 min)**: change `hopper/setup.py:491` from
   `"arch=compute_90a,code=sm_90a"` to `sm_120a`, and relax `enable_sm90` +
   `ARCH_SWITCH`. Produces a compileable-but-broken kernel. Does not work.

2. **Non-trivial (days)**: rewrite the mainloop to drop warp-specialization.
   FA3's `mainloop_fwd_sm90_tma_gmma_ws.hpp` issues WGMMAs from the consumer
   warpgroup while the producer warps do TMA loads; the pipeline is 2-stage
   (`kStages = 2`) keyed on WGMMA's async-completion semantics
   (`wgmma.commit_group.sync / wait_group`). SM120 has no WGMMA and no
   warp-group async, so the producer/consumer model collapses. You have to:
   - Replace `TiledMMA<GMMA_64x*x*>` with `TiledMMA<SM120_16x8x32_TN>` (from
     `cute/arch/mma_sm120.hpp`). Tile shapes shrink from 64×N×K to 16×N×K per
     warp-issue, so the register accumulator layout, softmax rescaling, and
     epilogue indexing all have to change.
   - Replace WGMMA commit/wait with `__syncwarp()` + a plain pipelined
     `mma.sync`. This kills FA3's biggest win over FA2 (producer/consumer
     overlap), dropping measured Hopper 1.5-2× speedup over FA2 closer to 1.0×.
   - Re-tune `kBlockM/kBlockN/kStages` for SM120's 228KB smem, 32 register files
     of 64KB, and 4 warp-schedulers per SM. Values in `hopper/tile_size.h` are
     hand-tuned for H100.

3. **Hard (weeks)**: re-derive the FP8 cache layout. FA3's FP8 forward relies on
   `GMMA::Major::MN` + swizzled layouts so WGMMA's native descriptor can read V
   directly. SM120's `mma.sync.aligned.kind::f8f6f4.m16n8k32` needs register-file
   A/B operands via LDSM with a different swizzle. Porting means redoing the
   smem layout atom, the TMA descriptor, and the V-transpose handling.

4. **Harder still**: FA3 backward (`mainloop_bwd_sm90_tma_gmma_ws.hpp`) uses
   WGMMA with F32 accumulator + ScaleOut::Zero tricks and dV/dK double-buffered
   WGMMAs. Would need the same full port.

Effort estimate:
- Forward BF16 non-causal working on SM120: **1–2 weeks** of focused CUTLASS
  work. Expected speedup over FA2 BF16 on SM120: **0–10%**, because once you
  remove WGMMA + warp-spec, the remaining wins (TMA, 2CTA cluster, intra-WG
  overlap) are small on consumer Blackwell (no 2CTA gain on 1-GPC workloads).
- Forward FP8 working: **3–4 weeks** incl. layout rework.
- Full FA3 backward: **6–8+ weeks**. Not justified.

Better path, almost certainly: use **FlashInfer's native SM120 kernels** (which
they built around `mma_sm120.hpp` from the start), or wait for Dao-AILab to add
an `_sm120.cu` kernel family — they explicitly plumbed `_sm100.cu` in the build
system already (`setup.py:236-252`) so an `_sm120.cu` addition would be a
natural extension.

## 5. Microbench

Not attempted. Compile "succeeds" but produces kernels that either launch empty
(unpatched) or compute nothing (patched). A microbench would either return
zeros or trap. A real benchmark requires all the Section-4 work first.

## 6. Files produced

- `plans/fa3_sm120a_port.md` — this document.
- `plans/fa3_probe.py` — compile driver for one FA3 TU.
- `plans/fa3_patch_and_probe.py` — same, with `enable_sm90` relaxed.
- `plans/fa3_disasm.py`, `plans/fa3_dump.py` — SASS inspection helpers.
- `plans/fa3_revert.py` — restore `utils.h` from backup.
- `/tmp/fa3_build_attempt.log` — full compile + disasm log.
- `/tmp/fa3_probe_120a.o` (empty stub), `/tmp/fa3_probe_120a_patched.o`
  (no-MMA stub) — artifacts.

## 7. Risk assessment (citations)

- NVIDIA PTX ISA 8.5, §9.7.13 `wgmma` — "Supported on devices with `sm_90a` or
  `sm_90`. The behavior is undefined for other targets." SM120 is not in the
  list.
- NVIDIA CUDA compute-capability table (CUDA 12.8 release notes) — SM120
  arch-family = `Blackwell (RTX 50 / RTX PRO 6000)`, does not include datacenter
  WGMMA. SM100 (B100/B200/GB200) has TMEM but no WGMMA either.
- CUTLASS 3.8 `include/cutlass/arch/config.h:48` — `CUTLASS_ARCH_MMA_SM90A_ENABLED`
  is gated on `__CUDA_ARCH__ == 900` only; SM120 path (`:160-173`) enables
  `_SM120A_ENABLED`, not `_SM90A_`.
- Discovery #24 (`DEEP_DIVE.md`) — SM100 cubins don't run on SM120; arch
  families are disjoint binaries.
- Discovery #30 (`DEEP_DIVE.md`) — Triton FP8 decode can't beat FA2 BF16 on
  consumer Blackwell; FA3's FP8 wins depend on WGMMA layout that SM120 lacks.

## 8. Recommendation

KILL this port in favor of:
- Waiting for upstream FA3 SM120 support (setup.py already has SM100 hooks;
  SM120 addition is a natural follow-on). Track Dao-AILab/flash-attention issues.
- Using FlashInfer for attention on SM120 — they ship SM120-native kernels.
- If FP8 attention on SM120 is really the goal: write a purpose-built kernel
  from `mma_sm120.hpp` (the `kind::f8f6f4` 16×8×32 MMA) directly, not by
  down-porting FA3. This is roughly a "FA2 written in CUTLASS 3.8 for SM120"
  exercise and should take 1–2 weeks if starting fresh.
