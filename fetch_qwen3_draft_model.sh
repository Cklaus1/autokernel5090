#!/bin/bash
# fetch_qwen3_draft_model.sh
# Downloads Qwen3-1.7B from Hugging Face as the same-family draft model
# for Qwen3-30B-A3B speculative decoding.
#
# Tag: W7_qwen3_samefamily_spec_prep
# Ref: kill_audit §16 — same-family spec decode, projected +98% aggregate (29.6k→~58k tok/s)
#
# Architecture: Qwen3ForCausalLM (dense, NOT MoE).
# Vocab:        151,669 — identical to Qwen3-30B-A3B tokenizer.
# Disk:         ~3.5 GB (BF16 weights, 7 shards typical).
#
# DO NOT RUN IN AUTOKERNEL AGENT — parent shell only (needs HF network + disk space).

set -euo pipefail

HF_REPO="Qwen/Qwen3-1.7B"
TARGET_DIR="/root/models/Qwen3-1.7B"
MIN_FREE_GB=5

# ── 1. Pre-flight: disk space ──────────────────────────────────────────────────
echo "[preflight] Checking available disk space on /root/models ..."
MODELS_DIR="/root/models"
if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "[error] ${MODELS_DIR} does not exist. Create it first: mkdir -p ${MODELS_DIR}"
  exit 1
fi

AVAIL_KB=$(df -k "${MODELS_DIR}" | awk 'NR==2 {print $4}')
AVAIL_GB=$(echo "scale=1; ${AVAIL_KB}/1048576" | bc)
echo "[preflight] Available: ${AVAIL_GB} GB  (need >= ${MIN_FREE_GB} GB)"

if (( AVAIL_KB < MIN_FREE_GB * 1048576 )); then
  echo "[error] Insufficient disk space. Free at least ${MIN_FREE_GB} GB before downloading."
  exit 1
fi

# ── 2. Pre-flight: huggingface-cli ────────────────────────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
  echo "[error] huggingface-cli not found. Install with: pip install huggingface_hub"
  exit 1
fi
echo "[preflight] huggingface-cli found: $(huggingface-cli --version 2>&1 | head -1)"

# ── 3. Pre-flight: tokenizer compat check (CPU, no GPU needed) ────────────────
echo "[preflight] Checking tokenizer vocab_size match (Qwen3-1.7B vs Qwen3-30B-A3B-NVFP4) ..."
python3 - <<'PYEOF'
import sys
try:
    from transformers import AutoTokenizer
except ImportError:
    print("[skip] transformers not available in this env — skipping tokenizer pre-check.")
    sys.exit(0)

DRAFT_REPO = "Qwen/Qwen3-1.7B"
TARGET_CONFIG = "/root/models/Qwen3-30B-A3B-NVFP4/tokenizer_config.json"

import json, os

# Read target vocab_size from local tokenizer_config
if not os.path.exists(TARGET_CONFIG):
    print(f"[warn] Target tokenizer_config not found at {TARGET_CONFIG} — skipping local check.")
    sys.exit(0)

with open(TARGET_CONFIG) as f:
    tgt_cfg = json.load(f)

# Qwen3 tokenizer_config encodes vocab_size via the model_max_length / tokenizer_class
# The actual vocab size is in the model's config.json
tgt_model_config = "/root/models/Qwen3-30B-A3B-NVFP4/config.json"
if os.path.exists(tgt_model_config):
    with open(tgt_model_config) as f:
        tgt_mcfg = json.load(f)
    tgt_vocab = tgt_mcfg.get("vocab_size", "unknown")
    print(f"[tokenizer] Qwen3-30B-A3B vocab_size (local): {tgt_vocab}")
    EXPECTED_VOCAB = 151669
    if tgt_vocab == EXPECTED_VOCAB:
        print(f"[tokenizer] PASS: target vocab_size matches expected {EXPECTED_VOCAB}")
    else:
        print(f"[tokenizer] WARN: target vocab_size={tgt_vocab} != expected {EXPECTED_VOCAB}")
else:
    print(f"[warn] Target config.json not found at {tgt_model_config}")

print(f"[tokenizer] After download, verify draft vocab_size == {tgt_vocab if os.path.exists(tgt_model_config) else 151669}")
print("[tokenizer] Run: python3 -c \"from transformers import AutoConfig; c=AutoConfig.from_pretrained('/root/models/Qwen3-1.7B'); print(c.vocab_size)\"")
PYEOF

# ── 4. Download ───────────────────────────────────────────────────────────────
echo ""
echo "[download] Pulling ${HF_REPO} → ${TARGET_DIR}"
echo "[download] This is ~3.5 GB; expect 2–10 min depending on bandwidth."
echo ""

huggingface-cli download \
  "${HF_REPO}" \
  --local-dir "${TARGET_DIR}" \
  --local-dir-use-symlinks False \
  --exclude "*.msgpack" "*.h5" "flax_model*" "tf_model*" "rust_model.ot"

# ── 5. Post-download verification ─────────────────────────────────────────────
echo ""
echo "[verify] Checking downloaded files ..."
ls -lh "${TARGET_DIR}/"

python3 - <<'PYEOF'
import os, json

cfg_path = "/root/models/Qwen3-1.7B/config.json"
if not os.path.exists(cfg_path):
    print("[verify] ERROR: config.json not found — download may be incomplete.")
    exit(1)

with open(cfg_path) as f:
    cfg = json.load(f)

arch = cfg.get("architectures", ["unknown"])[0]
vocab = cfg.get("vocab_size", "unknown")
hidden = cfg.get("hidden_size", "unknown")
layers = cfg.get("num_hidden_layers", "unknown")
rope_theta = cfg.get("rope_theta", "unknown")

print(f"[verify] architecture  : {arch}")
print(f"[verify] vocab_size    : {vocab}")
print(f"[verify] hidden_size   : {hidden}")
print(f"[verify] num_layers    : {layers}")
print(f"[verify] rope_theta    : {rope_theta}")

EXPECTED_ARCH  = "Qwen3ForCausalLM"
EXPECTED_VOCAB = 151669

ok = True
if arch != EXPECTED_ARCH:
    print(f"[verify] FAIL: expected arch {EXPECTED_ARCH}, got {arch}")
    ok = False
else:
    print(f"[verify] PASS: architecture is {EXPECTED_ARCH}")

if vocab != EXPECTED_VOCAB:
    print(f"[verify] FAIL: expected vocab_size {EXPECTED_VOCAB}, got {vocab}")
    ok = False
else:
    print(f"[verify] PASS: vocab_size matches target ({vocab})")

if ok:
    print("\n[verify] ALL CHECKS PASSED — Qwen3-1.7B ready for speculative decoding.")
    print("[verify] Next step: run ./launch_qwen3_speculative.sh with AUTOKERNEL_QWEN3_SPEC=1")
else:
    print("\n[verify] CHECKS FAILED — review above before launching.")
    exit(1)
PYEOF
