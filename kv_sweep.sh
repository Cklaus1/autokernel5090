#!/bin/bash
# KV dtype sweep: BF16 (auto) vs FP8 vs FusenCache k4v4b64
# Tests throughput at various concurrency levels with 128-token outputs
set -e

RESULTS=/root/projects/autokernel/kv_sweep_results.tsv
SERVER_IP=172.17.0.2
PORT=8001
MODEL=/models/gemma-4-26B-A4B-it-NVFP4-modelopt

echo -e "kv_dtype\tconc\tok\ttotal_tokens\twall_s\taggregate_tok_s\tpeak_gen_tok_s" > $RESULTS

run_sweep_for_kv() {
  local KV=$1
  local EXTRA_ARGS=$2
  local LOG=/root/projects/autokernel/sweep_${KV//\//_}.log

  echo "=== Restarting container for KV=$KV ==="
  docker exec vllm-gemma4 bash -c "pkill -9 python3 2>/dev/null; sleep 2" 2>/dev/null || true
  docker restart vllm-gemma4 >/dev/null 2>&1
  sleep 3
  docker exec vllm-gemma4 bash -c "rm -rf /fusen; ln -sf /workspace /fusen" 2>/dev/null || true

  echo "=== Launching with KV=$KV ==="
  if [ "$KV" = "k4v4b64" ]; then
    # FusenCache path
    docker exec -d vllm-gemma4 bash -c "PYTHONPATH=/fusen:\${PYTHONPATH:-} python3 /fusen/fusen_kv/launch_vllm.py \
      --model $MODEL --quantization modelopt --max-model-len 4096 --max-num-seqs 64 \
      --trust-remote-code --port $PORT --kv-cache-dtype k4v4b64 \
      -cc.mode none -cc.cudagraph_mode full \
      -cc.cudagraph_capture_sizes '[1,2,4,8,16,24,32,48,64]' -cc.max_cudagraph_capture_size 64 \
      > /workspace/${LOG##*/} 2>&1"
  else
    # Stock path (auto/fp8)
    docker exec -d vllm-gemma4 bash -c "python3 -m vllm.entrypoints.openai.api_server \
      --model $MODEL --quantization modelopt --max-model-len 4096 --max-num-seqs 256 \
      --trust-remote-code --port $PORT $EXTRA_ARGS \
      -cc.cudagraph_mode full \
      > /workspace/${LOG##*/} 2>&1"
  fi

  # Wait up to 4 min for server ready
  for i in $(seq 1 24); do
    if curl -s http://${SERVER_IP}:${PORT}/v1/models 2>/dev/null | grep -q "model"; then
      echo "   Server ready after ${i}0s"
      break
    fi
    if [ $i -eq 24 ]; then
      echo "   SERVER TIMEOUT for KV=$KV"
      return 1
    fi
    sleep 10
  done

  # Let graph capture settle
  sleep 3

  # Run concurrency sweep
  python3 - <<PYEOF
import asyncio, aiohttp, time, subprocess, re, os

KV = "$KV"
MODEL = "$MODEL"
URL = "http://${SERVER_IP}:${PORT}/v1/chat/completions"
HEALTH = "http://${SERVER_IP}:${PORT}/v1/models"
LOG = "$LOG"
RESULTS = "$RESULTS"

async def send(s, p, mt):
    data = {'model': MODEL, 'messages': [{'role': 'user', 'content': p}], 'max_tokens': mt, 'temperature': 0}
    try:
        async with s.post(URL, json=data, timeout=aiohttp.ClientTimeout(total=300)) as r:
            j = await r.json()
            if 'error' in j: return (False, 0)
            return (True, j.get('usage',{}).get('completion_tokens',0))
    except: return (False, 0)

async def bench(C, mt):
    async with aiohttp.ClientSession() as s:
        tasks = []
        t0 = time.time()
        for i in range(C):
            prompts = f'Write a short paragraph about the number {i+5}.'
            tasks.append(asyncio.create_task(send(s, prompts, mt)))
            if C > 256 and i < C-1: await asyncio.sleep(0.003)
        res = await asyncio.gather(*tasks)
        dt = time.time() - t0
        ok = sum(1 for r,_ in res if r)
        tok = sum(t for r,t in res if r)
        return ok, tok, dt

def read_peak_gen_from_log(log_path):
    if not os.path.exists(log_path):
        return 0.0
    peak = 0.0
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = re.search(r'Avg generation throughput:\s+([\d.]+)', line)
            if m:
                val = float(m.group(1))
                if val > peak: peak = val
    return peak

async def main():
    for C in [1, 32, 128, 256, 512, 1024]:
        try:
            async with aiohttp.ClientSession() as s2:
                async with s2.get(HEALTH, timeout=aiohttp.ClientTimeout(total=5)) as r: pass
        except:
            print(f'  [{KV} C={C}] server down'); break
        ok, tok, dt = await bench(C, 128)
        agg = tok/dt if dt>0 else 0
        peak_gen = read_peak_gen_from_log(LOG)
        line = f"{KV}\t{C}\t{ok}\t{tok}\t{dt:.1f}\t{agg:.0f}\t{peak_gen:.0f}"
        print(f"  [{KV} C={C}] ok={ok}/{C}, tok={tok}, wall={dt:.1f}s, agg={agg:.0f} tok/s, peak_gen={peak_gen:.0f} tok/s")
        with open(RESULTS, 'a') as rf: rf.write(line + "\n")
        if ok < C:
            # Check server still alive
            try:
                async with aiohttp.ClientSession() as s2:
                    async with s2.get(HEALTH, timeout=aiohttp.ClientTimeout(total=5)) as r: pass
            except:
                print(f'  [{KV} C={C}] server crashed!'); break

asyncio.run(main())
PYEOF
}

echo "==========================================="
echo "KV dtype sweep: BF16 vs FP8 vs FusenCache"
echo "==========================================="

run_sweep_for_kv "auto" ""
run_sweep_for_kv "fp8" "--kv-cache-dtype fp8"
run_sweep_for_kv "k4v4b64" ""

echo ""
echo "=========== FINAL RESULTS ==========="
column -t -s $'\t' $RESULTS
