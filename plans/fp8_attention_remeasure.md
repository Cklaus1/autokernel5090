# FP8 Attention Re-measure on FlashInfer 0.6.7 TRTLLM SM120 Backend

**Date:** 2026-04-17
**Hardware:** RTX PRO 6000 Blackwell Max-Q (SM120, HBM peak 1792 GB/s), GPU 1
**Environment:** `vllm-fusencache:latest`, FlashInfer 0.6.7, torch 2.11.0+cu130
**Shape:** Gemma4 26B decode, num_q=8, num_kv=2, seq_len=2048 (+ sweep 4k/8k/16k), B=1
**Script:** `/home/cklaus/projects/autokernel/fp8_attn_remeasure.py`, `fp8_attn_longseq.py`

## Verdict: **FAIL — Discovery #37 still holds**

FP8 attention on FlashInfer 0.6.7's new TRTLLM SM120 backend delivers at best **1.11× vs BF16** (at hd=256, seq=8192) — far below the 1.3× PASS gate and the Discovery #23 theoretical 1.44× target. At the nominal Gemma4 shape (hd=256, seq=2048) the ratio is **1.01×** — statistical noise.

## Numbers (median of 200 iters, CUDA events)

### Nominal shape (Discovery #37 shape)
| kernel | kv_dtype | latency (µs) | GB/s | % of 1792 peak |
|---|---|---:|---:|---:|
| `single_decode_with_kv_cache` | bfloat16 | 28.2 | 149 | 8.3% |
| `single_decode_with_kv_cache` | float8_e4m3fn | 36.4 | 58 | 3.2% |
| `trtllm_batch_decode_with_kv_cache` | bfloat16 | **18.2** | 230 | 12.8% |
| `trtllm_batch_decode_with_kv_cache` | float8_e4m3fn | **18.0** | 116 | 6.5% |

**Ratio (BF16 / FP8) = 18.2 / 18.0 = 1.01×**

### hd=128 sliding layer (seq=2048)
| kernel | bf16 | fp8 | ratio |
|---|---:|---:|---:|
| trtllm | 12.9 µs | 13.1 µs | **0.99×** |

### Long-seq sweep (TRTLLM only)
| hd | seq | BF16 µs / GB/s / %peak | FP8 µs / GB/s / %peak | ratio |
|---:|---:|---|---|---:|
| 256 | 4096 | 18.1 / 463 / 25.8% | 22.8 / 184 / 10.3% | 0.79× |
| 128 | 4096 | 18.0 / 234 / 13.0% | 20.3 / 103 / 5.8% | 0.88× |
| 256 | 8192 | 21.8 / 770 / 43.0% | 19.6 / 427 / 23.8% | **1.11×** (best) |
| 128 | 8192 | 17.8 / 472 / 26.4% | 17.1 / 246 / 13.7% | 1.04× |
| 256 | 16384 | 27.2 / 1235 / 68.9% | 25.4 / 659 / 36.8% | 1.07× |
| 128 | 16384 | 18.5 / 906 / 50.5% | 26.2 / 320 / 17.9% | 0.71× |

## Why FP8 fails to deliver

The smoking gun is the % bandwidth column. At seq=16384 / hd=256 the **BF16** kernel hits **69% peak** (1235 GB/s) — near-ceiling, clearly bandwidth-bound. The **FP8** kernel with *half* the bytes to move only reaches **37% peak** (659 GB/s). If FP8 were bandwidth-bound, it would push 1500+ GB/s; it pushes 659 GB/s. **The FP8 path leaves ~800 GB/s of bandwidth unused**, meaning it is compute- or layout-bound inside the kernel (dequant inline, non-vectorized loads, or scale-broadcast serialization — exactly Discovery #37's diagnosis).

The new `trtllm_batch_decode_with_kv_cache` TRTLLM SM120 backend (the new thing in 0.6.7) is faster than the generic `single_decode_with_kv_cache` (18 µs vs 28 µs for BF16), but it has **not** fixed the FP8-specific compute bottleneck. FP8 absolute latency roughly matches BF16 because both converge on the same compute ceiling.

## Integration recommendation

**Do not integrate FP8 KV attention** for Gemma4 26B decode on SM120. The KV-halving benefit from Discovery #23 does not materialize in this kernel. Options if FP8 KV memory savings are still needed:
1. Accept the memory win without the latency win (FP8 KV cache + unchanged decode latency → double effective context, no TTFT improvement).
2. Wait for FlashInfer to ship a properly vectorized SM120 FP8 decode kernel (next release cadence).
3. Revisit if upstream adds FA3-on-SM120 — that path has native FP8 WGMMA-style compute on Blackwell.

## Files produced

- `/home/cklaus/projects/autokernel/fp8_attn_remeasure.py` — nominal-shape microbench
- `/home/cklaus/projects/autokernel/fp8_attn_longseq.py` — long-seq sweep
- `results.tsv` row `fp8_attn_flashinfer_0_6_7` (FAIL, 1.01×)
