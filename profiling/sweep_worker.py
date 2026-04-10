#!/usr/bin/env python3
"""Worker: sweep num_warps x num_stages for ONE (H, BLOCK_H) pair across multiple B values."""
import sys, json, gc, torch, triton, triton.language as tl

@triton.jit
def _qfp4(v):
    c = (v * 0).to(tl.int32)
    c = tl.where(v > 0.25, 1, c); c = tl.where(v > 0.75, 2, c)
    c = tl.where(v > 1.25, 3, c); c = tl.where(v > 1.75, 4, c)
    c = tl.where(v > 2.5, 5, c); c = tl.where(v > 3.5, 6, c)
    c = tl.where(v > 5.0, 7, c)
    return c

@triton.jit
def kern(X, W, OF, OS, GS, B, H: tl.constexpr, sx, so, ss, eps,
         BH: tl.constexpr, QBS: tl.constexpr, HW: tl.constexpr, VS: tl.constexpr):
    row = tl.program_id(0)
    gs = tl.load(GS).to(tl.float32)
    VD: tl.constexpr = H if VS == 0 else VS
    NV: tl.constexpr = (VD + BH - 1) // BH
    NI: tl.constexpr = (H + BH - 1) // BH
    QPI: tl.constexpr = BH // QBS; HB: tl.constexpr = BH // 2; HQ: tl.constexpr = QBS // 2
    ssq = tl.zeros([1], dtype=tl.float32)
    for _i in range(NV):
        o = _i * BH + tl.arange(0, BH); m = o < VD
        x = tl.load(X + row * sx + o, mask=m, other=0.0).to(tl.float32)
        ssq += tl.sum(x * x, axis=0)
    rr = 1.0 / tl.sqrt(ssq / VD + eps)
    for _i in range(NI):
        b = _i * BH
        eo = b + tl.arange(0, HB) * 2; oo = eo + 1
        em = eo < H; om = oo < H
        xe = tl.load(X + row * sx + eo, mask=em, other=0.0).to(tl.float32)
        xo = tl.load(X + row * sx + oo, mask=om, other=0.0).to(tl.float32)
        ne = xe * rr; no = xo * rr
        if HW:
            we = tl.load(W + eo, mask=em, other=1.0).to(tl.float32)
            wo = tl.load(W + oo, mask=om, other=1.0).to(tl.float32)
            ne = ne * we; no = no * wo
        ae = tl.abs(tl.reshape(ne, [QPI, HQ])); ao = tl.abs(tl.reshape(no, [QPI, HQ]))
        bm = tl.maximum(tl.max(ae, axis=1), tl.max(ao, axis=1))
        bs = tl.minimum(bm / (6.0 * gs), 448.0)
        si = _i * QPI + tl.arange(0, QPI); sm = si < (H // QBS)
        tl.store(OS + row * ss + si, bs, mask=sm)
        bf = tl.load(OS + row * ss + si, mask=sm, other=0.0).to(tl.float32)
        bd = tl.reshape(tl.broadcast_to(tl.reshape(bf, [QPI, 1]), [QPI, HQ]), [HB]) * gs
        bd = tl.where(bd > 0.0, bd, 1.0)
        ve = tl.abs(ne) / bd; vo = tl.abs(no) / bd
        ce = _qfp4(ve); co = _qfp4(vo)
        se = tl.where(ne < 0.0, 8, 0).to(tl.int32); so2 = tl.where(no < 0.0, 8, 0).to(tl.int32)
        pk = ((ce | se) & 0xF) | (((co | so2) & 0xF) << 4)
        bi = _i * HB + tl.arange(0, HB)
        tl.store(OF + row * so + bi, pk.to(tl.uint8), mask=bi < (H // 2))

# Args: H BLOCK_H B1,B2,...
H = int(sys.argv[1])
BH = int(sys.argv[2])
B_values = [int(b) for b in sys.argv[3].split(",")]

results = []
for nw in [1, 2, 4, 8, 16, 32]:
    for ns in [1, 2, 3, 4, 5, 6, 7, 8]:
        for B in B_values:
            x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(H, device="cuda", dtype=torch.bfloat16).abs() + 0.1
            gst = torch.tensor([1.0], device="cuda", dtype=torch.float32)
            fo = torch.empty((B, H // 2), device="cuda", dtype=torch.uint8)
            so = torch.empty((B, H // 16), device="cuda", dtype=torch.float8_e4m3fn)
            grid = (B,)
            def make_run(B=B, nw=nw, ns=ns):
                def run():
                    kern[grid](x, w, fo, so, gst, B, H, x.stride(0), fo.stride(0), so.stride(0),
                               1e-6, BH=BH, QBS=16, HW=True, VS=0, num_warps=nw, num_stages=ns)
                return run
            try:
                fn = make_run()
                fn(); torch.cuda.synchronize()
                lat_ms = triton.testing.do_bench(fn, warmup=15, rep=50)
                results.append({"B": B, "H": H, "BLOCK_H": BH, "num_warps": nw,
                                "num_stages": ns, "latency_us": round(lat_ms * 1000, 3)})
            except Exception:
                pass
            del x, w, gst, fo, so
            torch.cuda.empty_cache()

print(json.dumps(results))
