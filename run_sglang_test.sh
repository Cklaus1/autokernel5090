#!/bin/bash
set +e

export PATH=/usr/local/cuda-12.8/bin:$PATH
NVIDIA_LIBS=$(python -c "import nvidia, os; base=os.path.dirname(nvidia.__file__); libs=[os.path.join(base,d,'lib') for d in os.listdir(base) if os.path.isdir(os.path.join(base,d,'lib'))]; print(':'.join(libs))")
export LD_LIBRARY_PATH=$NVIDIA_LIBS:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

MODEL=${1:-Qwen/Qwen3.5-9B}
EXTRA_ARGS="${@:2}"

echo "Launching: $MODEL $EXTRA_ARGS"

/root/sglang_env/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --port 30000 --dtype bfloat16 \
    --trust-remote-code --attention-backend triton \
    --max-mamba-cache-size 128 \
    $EXTRA_ARGS \
    > /tmp/sglang_test.log 2>&1 &
PID=$!

for i in $(seq 1 60); do
    sleep 5
    if ! kill -0 $PID 2>/dev/null; then
        echo "DIED at $((i*5))s"
        tail -15 /tmp/sglang_test.log
        exit 1
    fi
    HTTP=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:30000/health 2>/dev/null)
    if [ "$HTTP" = "200" ]; then
        echo "READY at $((i*5))s"
        break
    fi
done

echo "=== Quality Check ==="
curl -s http://127.0.0.1:30000/generate \
    -d '{"text": "<|im_start|>user\nExplain how TCP works in 50 words.<|im_end|>\n<|im_start|>assistant\n", "sampling_params": {"temperature": 0.0, "max_new_tokens": 100}}' \
    -H "Content-Type: application/json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('text','NO TEXT')[:500])"

echo ""
echo "=== Speed Check ==="
python3 -c "
import requests, time
for i in range(3):
    t0 = time.perf_counter()
    r = requests.post('http://127.0.0.1:30000/generate', json={
        'text': '<|im_start|>user\nWrite quicksort in Python.<|im_end|>\n<|im_start|>assistant\n',
        'sampling_params': {'temperature': 0.0, 'max_new_tokens': 256}
    }, timeout=60).json()
    elapsed = time.perf_counter() - t0
    tokens = r.get('meta_info',{}).get('completion_tokens',256)
    print(f'  Run {i+1}: {tokens/elapsed:.1f} tok/s ({tokens} tok)')
"

kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "Done."
