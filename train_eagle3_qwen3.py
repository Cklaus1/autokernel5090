#!/usr/bin/env python3
# train_eagle3_qwen3.py -- EAGLE3 draft-head training for Qwen3-30B-A3B (NVFP4)
#
# Two-phase pipeline:
#   phase=teacher : spin up vLLM offline with the NVFP4 teacher on GPU 1,
#                   generate (input_ids, topk_ids, topk_logprobs) shards to /tmp/eagle3_labels/
#   phase=train   : load shards, train a minimal EAGLE3-style draft head, log loss +
#                   top-1 match rate, checkpoint every 15 minutes.
#
# Expected to run inside the vllm-fusencache-gemma4fix:latest container bound with
#     -v /root/models:/models -v /home/cklaus/projects/autokernel:/work
# on GPU 1 ONLY.
#
# This is a pipeline shake-out (~1 h), not a full training run.

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ----------------------------------------------------------------------------
# shared config
# ----------------------------------------------------------------------------

TARGET_MODEL = "/models/Qwen3-30B-A3B-NVFP4"
LABELS_DIR = Path("/tmp/eagle3_labels")
CKPT_DIR = Path("/tmp/eagle3_ckpts")
LOG_FILE = Path("/tmp/eagle3_train.log")

HIDDEN = 2048
VOCAB = 151_936
NUM_HEADS = 32
NUM_KV_HEADS = 4
INTERMEDIATE = 6144
TOPK = 20

PROMPT_LEN = 512
GEN_LEN = 128
SEQ_LEN = PROMPT_LEN + GEN_LEN


# ----------------------------------------------------------------------------
# logging helper
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# phase A: teacher label generation
# ----------------------------------------------------------------------------

def load_prompts(n: int, seed: int = 0) -> list[str]:
    """Load prompts from an open-licensed public dataset via the HF datasets-server REST API.

    We avoid a hard dep on `datasets` (not installed in the vLLM container by default).
    Falls back to a built-in prompt list if the network is unreachable.
    """
    import urllib.request
    import urllib.parse

    datasets = [
        # (repo, config, split)
        ("allenai/tulu-3-sft-personas-instruction-following", "default", "train"),
        ("HuggingFaceH4/ultrachat_200k", "default", "train_sft"),
    ]
    prompts: list[str] = []
    for ds, cfg, split in datasets:
        url = (
            "https://datasets-server.huggingface.co/rows?"
            + urllib.parse.urlencode(
                dict(dataset=ds, config=cfg, split=split, offset=0, length=100)
            )
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read())
            for row in data.get("rows", []):
                rec = row.get("row", {})
                # try common fields in priority order
                for key in ("prompt", "messages", "input", "instruction", "text"):
                    val = rec.get(key)
                    if not val:
                        continue
                    if isinstance(val, list):
                        # messages-style → take first user content
                        user = next((m for m in val if m.get("role") == "user"), None)
                        if user:
                            prompts.append(str(user.get("content", ""))[:2000])
                            break
                    elif isinstance(val, str):
                        prompts.append(val[:2000])
                        break
            if prompts:
                break
        except Exception as exc:  # noqa: BLE001
            log(f"  dataset fetch {ds} failed: {exc!r}")
            continue

    # Offline fallback — synthesised instruction seeds (public domain content).
    if not prompts:
        log("  dataset fetch failed; using built-in fallback prompts")
        fallback = [
            "Explain transformer self-attention in one paragraph.",
            "Write a haiku about a cat chasing a red laser dot.",
            "List five culturally significant foods from Vietnam and why.",
            "Describe the difference between TCP and UDP for a junior developer.",
            "Summarise the plot of The Odyssey in 150 words.",
            "Give me Python code that detects palindromes ignoring punctuation.",
            "Compare electric induction stoves and gas stoves for a home cook.",
            "Write a professional apology email for a late project delivery.",
            "Explain why the sky appears blue using Rayleigh scattering.",
            "Outline a two-week beginner strength-training plan, 3 days/week.",
        ]
        prompts = [fallback[i % len(fallback)] for i in range(n)]

    random.Random(seed).shuffle(prompts)
    return prompts[:n]


def phase_teacher(args) -> None:
    """Use vLLM offline to generate top-k teacher logprobs per token, write to shards."""
    import torch  # noqa: F401  (used for tensors via vllm)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)

    log(f"phase=teacher | model={TARGET_MODEL} | n_prompts={args.n_examples}")
    t0 = time.time()

    # The NVFP4 checkpoint ships hf_quant_config.kv_cache_scheme=fp8_e4m3.
    # On this vLLM build (c0c98b8b9) the fp8 KV cache reshape is buggy:
    # `shape [..] invalid for input of size ..` because the allocator sizes
    # for bf16 while reshape uses fp8 strides. We try to sidestep by
    # overriding kv cache dtype to auto via kwargs and by supplying
    # an explicit kv_cache_memory_bytes that is a clean multiple of the
    # per-block size.
    llm = LLM(
        model=TARGET_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        dtype="bfloat16",
        max_model_len=SEQ_LEN + 64,
        max_num_seqs=16,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=True,
        kv_cache_memory_bytes=20 * (1 << 30),  # 20 GiB, block-aligned enough
    )
    log(f"  vLLM loaded in {time.time() - t0:0.1f}s")

    sp = SamplingParams(
        max_tokens=GEN_LEN,
        temperature=0.0,
        logprobs=TOPK,
        prompt_logprobs=TOPK,
    )

    prompts = load_prompts(args.n_examples)
    log(f"  got {len(prompts)} prompts")

    BATCH = 32
    shard_idx = 0
    total_tokens = 0
    t_gen = time.time()

    for batch_start in range(0, len(prompts), BATCH):
        batch = prompts[batch_start : batch_start + BATCH]
        outs = llm.generate(batch, sp, use_tqdm=False)
        shard_rows = []
        for out in outs:
            try:
                prompt_tok_ids = list(out.prompt_token_ids or [])
                gen_tok_ids = list(out.outputs[0].token_ids)
                tok_ids = prompt_tok_ids + gen_tok_ids
                if len(tok_ids) < 16:
                    continue

                # Build top-k distribution per position.
                # prompt_logprobs: list[Optional[dict[int, Logprob]]] length T_prompt
                # outputs[0].logprobs: list[dict[int, Logprob]] length T_gen
                pl = out.prompt_logprobs or []
                gl = out.outputs[0].logprobs or []
                all_lp = pl + gl

                topk_ids_rows: list[list[int]] = []
                topk_lp_rows: list[list[float]] = []
                for pos, entry in enumerate(all_lp):
                    if not entry:
                        topk_ids_rows.append([0] * TOPK)
                        topk_lp_rows.append([-20.0] * TOPK)
                        continue
                    items = sorted(entry.items(), key=lambda kv: -kv[1].logprob)[:TOPK]
                    ids = [int(k) for k, _ in items]
                    lps = [float(v.logprob) for _, v in items]
                    # pad if necessary
                    while len(ids) < TOPK:
                        ids.append(0)
                        lps.append(-20.0)
                    topk_ids_rows.append(ids)
                    topk_lp_rows.append(lps)

                shard_rows.append(
                    dict(
                        input_ids=tok_ids,
                        topk_ids=topk_ids_rows,
                        topk_logprobs=topk_lp_rows,
                    )
                )
                total_tokens += len(tok_ids)
            except Exception as exc:  # noqa: BLE001
                log(f"    skip example: {exc!r}")
                continue

        # write shard
        if shard_rows:
            import torch

            max_T = max(len(r["input_ids"]) for r in shard_rows)
            N = len(shard_rows)
            ids_t = torch.zeros((N, max_T), dtype=torch.int32)
            tk_ids = torch.zeros((N, max_T, TOPK), dtype=torch.int32)
            tk_lp = torch.full((N, max_T, TOPK), -20.0, dtype=torch.float16)
            mask = torch.zeros((N, max_T), dtype=torch.bool)
            for i, r in enumerate(shard_rows):
                T = len(r["input_ids"])
                ids_t[i, :T] = torch.tensor(r["input_ids"], dtype=torch.int32)
                tk_ids[i, :T] = torch.tensor(r["topk_ids"], dtype=torch.int32)
                tk_lp[i, :T] = torch.tensor(r["topk_logprobs"], dtype=torch.float16)
                mask[i, :T] = True
            out_path = LABELS_DIR / f"shard_{shard_idx:05d}.pt"
            torch.save(
                dict(
                    input_ids=ids_t,
                    topk_ids=tk_ids,
                    topk_logprobs=tk_lp,
                    mask=mask,
                ),
                out_path,
            )
            shard_idx += 1
        dt = time.time() - t_gen
        log(
            f"  batch {batch_start // BATCH} / {math.ceil(len(prompts)/BATCH)} "
            f"tokens={total_tokens} elapsed={dt:0.1f}s throughput={total_tokens/max(dt,1e-3):0.0f}tok/s"
        )

    log(f"phase=teacher done. shards={shard_idx} tokens={total_tokens}")


# ----------------------------------------------------------------------------
# phase B: draft-head training
# ----------------------------------------------------------------------------

@dataclass
class DraftConfig:
    hidden_size: int = HIDDEN
    intermediate_size: int = INTERMEDIATE
    num_attention_heads: int = NUM_HEADS
    num_key_value_heads: int = NUM_KV_HEADS
    vocab_size: int = VOCAB
    num_layers: int = 1
    rms_eps: float = 1e-6
    max_position_embeddings: int = SEQ_LEN + 32


def build_draft(cfg: DraftConfig):
    """Minimal EAGLE3-style draft: 1 Qwen-style decoder layer + lm_head.

    NB: v1 omits the (embed, target_hidden) concat since we only have logprob
    labels, not target hidden states. The first layer consumes token embeddings
    directly. This trains an auto-regressive LM distilled from the teacher.
    """
    import torch
    import torch.nn as nn
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    )

    qcfg = Qwen3Config(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        num_hidden_layers=cfg.num_layers,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_eps,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_theta=1_000_000.0,
        head_dim=128,
        attention_bias=False,
        tie_word_embeddings=False,
    )

    class DraftHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = qcfg
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList(
                [Qwen3DecoderLayer(qcfg, layer_idx=i) for i in range(cfg.num_layers)]
            )
            self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
            self.rotary_emb = Qwen3RotaryEmbedding(config=qcfg)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            # tie for parameter economy
            self.lm_head.weight = self.embed_tokens.weight

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B,T) -> (B,T,V)
            B, T = input_ids.shape
            h = self.embed_tokens(input_ids)
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            # precompute RoPE cos/sin
            cos, sin = self.rotary_emb(h, positions)
            pos_embeds = (cos, sin)
            # build causal mask
            mask = torch.full((T, T), float("-inf"), device=h.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            for layer in self.layers:
                out = layer(
                    hidden_states=h,
                    attention_mask=mask,
                    position_ids=positions,
                    position_embeddings=pos_embeds,
                )
                h = out[0] if isinstance(out, tuple) else out
            h = self.norm(h)
            return self.lm_head(h)

    return DraftHead()


class ShardDataset:
    def __init__(self, shard_paths: list[Path]):
        self.paths = shard_paths
        self._cache: dict[Path, dict] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, path: Path) -> dict:
        import torch

        if path not in self._cache:
            self._cache[path] = torch.load(path, map_location="cpu", weights_only=False)
            # keep cache bounded
            if len(self._cache) > 4:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[path]


def phase_train(args) -> None:
    import torch

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(LABELS_DIR.glob("shard_*.pt"))
    if not shards:
        log("phase=train ABORT: no shards found in /tmp/eagle3_labels")
        return

    log(f"phase=train | shards={len(shards)} device=cuda:0 (GPU 1 mapped)")

    cfg = DraftConfig()
    dtype = torch.bfloat16
    model = build_draft(cfg).to(device="cuda", dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  draft params: {n_params/1e6:0.1f}M (tied embed/lm_head)")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    warmup = 200

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * step / max(1, warmup)
        # cosine to 10% of lr over args.max_steps
        pct = (step - warmup) / max(1, args.max_steps - warmup)
        pct = min(1.0, max(0.0, pct))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pct)))

    ds = ShardDataset(shards)
    start_time = time.time()
    last_ckpt = start_time
    step = 0
    running_loss = 0.0
    running_match = 0.0
    running_n = 0

    while step < args.max_steps and (time.time() - start_time) < args.max_seconds:
        shard = ds.load(random.choice(shards))
        input_ids = shard["input_ids"].to("cuda", dtype=torch.long)
        topk_ids = shard["topk_ids"].to("cuda", dtype=torch.long)
        topk_lp = shard["topk_logprobs"].to("cuda", dtype=torch.float32)
        mask = shard["mask"].to("cuda")
        N, T = input_ids.shape

        # sample a mini-batch of sequences from the shard
        idx = torch.randint(0, N, (args.batch_size,), device="cuda")
        ids = input_ids[idx]
        tk_ids = topk_ids[idx]
        tk_lp = topk_lp[idx]
        m = mask[idx]

        # teacher-forcing: predict next token. Shift inputs/labels.
        if T < 2:
            continue
        x = ids[:, :-1]
        tk_ids_y = tk_ids[:, 1:]
        tk_lp_y = tk_lp[:, 1:]
        m_y = m[:, 1:]

        logits = model(x)  # (B,T-1,V)
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # gather top-k log-probs at teacher top-k ids
        #   draft_lp_at_topk: (B,T-1,K)
        draft_lp_at_topk = torch.gather(log_probs, dim=-1, index=tk_ids_y)
        teacher_probs = torch.softmax(tk_lp_y, dim=-1)  # normalise teacher topk to a prob
        per_tok = -(teacher_probs * draft_lp_at_topk).sum(dim=-1)  # (B,T-1)
        loss = (per_tok * m_y.float()).sum() / m_y.float().sum().clamp_min(1.0)

        # top-1 match proxy
        with torch.no_grad():
            draft_top1 = logits.argmax(dim=-1)
            teacher_top1 = tk_ids_y[..., 0]
            match = ((draft_top1 == teacher_top1) & m_y).float().sum() / m_y.float().sum().clamp_min(1.0)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for pg in optim.param_groups:
            pg["lr"] = lr_at(step)
        optim.step()

        running_loss += float(loss.detach())
        running_match += float(match.detach())
        running_n += 1
        step += 1

        if step % args.log_every == 0 or step == 1:
            mean_loss = running_loss / running_n
            mean_match = running_match / running_n
            running_loss = running_match = 0.0
            running_n = 0
            elapsed = time.time() - start_time
            log(
                f"  step={step:5d} lr={lr_at(step):.2e} loss={mean_loss:6.3f} "
                f"top1_match={mean_match*100:5.1f}% elapsed={elapsed:0.0f}s"
            )

        if time.time() - last_ckpt > args.ckpt_every_s:
            ckpt_path = CKPT_DIR / f"eagle3_draft_step{step:05d}.pt"
            torch.save(
                dict(
                    step=step,
                    model=model.state_dict(),
                    cfg=cfg.__dict__,
                ),
                ckpt_path,
            )
            log(f"  checkpoint -> {ckpt_path}")
            last_ckpt = time.time()

    ckpt_path = CKPT_DIR / f"eagle3_draft_final_step{step:05d}.pt"
    torch.save(
        dict(step=step, model=model.state_dict(), cfg=cfg.__dict__),
        ckpt_path,
    )
    log(f"phase=train done. final ckpt={ckpt_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def phase_pseudo_labels(args) -> None:
    """Build 'pseudo' teacher shards from a public text corpus using next-token identity
    as the teacher top-1 label. This validates the Phase B training loop end-to-end when
    the real vLLM teacher path is blocked (e.g. vLLM kv-cache reshape bug on this build).

    Label semantics:
      topk_ids[t, 0]     = input_ids[t+1]  (the actual next token)
      topk_ids[t, 1..19] = random sample of other vocab ids
      topk_logprobs[t, 0] = log(0.9), others share the remaining 0.1 mass.
    Draft that learns next-token LM will hit top-1 trivially once the loop works.
    """
    import math as _math

    import torch
    from transformers import AutoTokenizer

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    prompts = load_prompts(args.n_examples)
    log(f"phase=pseudo | prompts={len(prompts)}")

    target_len = SEQ_LEN  # 512

    shard_rows = []
    rng = random.Random(0)
    for p in prompts:
        # encode with truncation
        ids = tok.encode(p, add_special_tokens=True)[:target_len]
        if len(ids) < 32:
            continue
        # top-1 is actual next token; others random
        T = len(ids)
        tk_ids = [[ids[min(t + 1, T - 1)]] + [rng.randrange(VOCAB) for _ in range(TOPK - 1)] for t in range(T)]
        top1_logprob = _math.log(0.9)
        other_logprob = _math.log(0.1 / (TOPK - 1))
        tk_lp = [[top1_logprob] + [other_logprob] * (TOPK - 1) for _ in range(T)]
        shard_rows.append(dict(input_ids=ids, topk_ids=tk_ids, topk_logprobs=tk_lp))

    # write in 100-row shards
    BATCH = 50
    shard_idx = 0
    for start in range(0, len(shard_rows), BATCH):
        batch = shard_rows[start : start + BATCH]
        max_T = max(len(r["input_ids"]) for r in batch)
        N = len(batch)
        ids_t = torch.zeros((N, max_T), dtype=torch.int32)
        tk_ids = torch.zeros((N, max_T, TOPK), dtype=torch.int32)
        tk_lp = torch.full((N, max_T, TOPK), -20.0, dtype=torch.float16)
        mask = torch.zeros((N, max_T), dtype=torch.bool)
        for i, r in enumerate(batch):
            T = len(r["input_ids"])
            ids_t[i, :T] = torch.tensor(r["input_ids"], dtype=torch.int32)
            tk_ids[i, :T] = torch.tensor(r["topk_ids"], dtype=torch.int32)
            tk_lp[i, :T] = torch.tensor(r["topk_logprobs"], dtype=torch.float16)
            mask[i, :T] = True
        out_path = LABELS_DIR / f"shard_{shard_idx:05d}.pt"
        torch.save(dict(input_ids=ids_t, topk_ids=tk_ids, topk_logprobs=tk_lp, mask=mask), out_path)
        shard_idx += 1
    log(f"phase=pseudo done. shards={shard_idx} rows={len(shard_rows)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        choices=("teacher", "pseudo", "train", "both", "pseudo_train"),
        default="both",
    )
    p.add_argument("--n-examples", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ckpt-every-s", type=int, default=900)
    args = p.parse_args()

    log(f"START phase={args.phase}")
    if args.phase in ("teacher", "both"):
        phase_teacher(args)
    if args.phase in ("pseudo", "pseudo_train"):
        phase_pseudo_labels(args)
    if args.phase in ("train", "both", "pseudo_train"):
        phase_train(args)
    log("DONE")


if __name__ == "__main__":
    main()
