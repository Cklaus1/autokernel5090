#!/bin/bash
# Poll health endpoint until ready, output periodic status
URL="http://localhost:8400/health"
MAX_ATTEMPTS=120  # 120 x 5s = 10 minutes

for i in $(seq 1 $MAX_ATTEMPTS); do
  STATUS=$(curl -sf -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null || echo "000")
  TIMESTAMP=$(date +%H:%M:%S)
  LOG_LINES=$(docker logs vllm-lmcache-smoke 2>&1 | wc -l)
  if [ "$STATUS" = "200" ]; then
    echo "[$TIMESTAMP] READY after ${i} polls (~$((i*5))s) | log_lines=$LOG_LINES"
    exit 0
  fi
  echo "[$TIMESTAMP] poll=$i/$MAX_ATTEMPTS status=$STATUS log_lines=$LOG_LINES"
  sleep 5
done
echo "TIMEOUT: server not ready after ${MAX_ATTEMPTS} polls"
exit 1
