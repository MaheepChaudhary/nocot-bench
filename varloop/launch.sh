#!/usr/bin/env bash
# launch.sh <passes> <port>: one SGLang server per loop setting.
# Recipe: SGLang cookbook RTX PRO 6000 "Balanced". --reasoning-parser is left out
# but SGLang forces "inkling" for this model anyway. The installed SGLang carries
# sglang_counter.patch (prefilled answers are not counted as reasoning).
# Usage: launch.sh <passes> <port> [mem_fraction]  (p3 needs 0.5 next to p1)
set -euo pipefail
P=$1
PORT=$2
MEM=${3:-0.30}
cd /workspace/varloop_work
export PATH=$PWD/sgenv/bin:$PATH
exec sgenv/bin/python -m sglang.launch_server \
  --model-path /workspace/varloop_work/weights \
  --served-model-name "tml-varloop@p$P" --inkling2-loop-num-passes "$P" \
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
  --grammar-backend xgrammar --enable-multimodal --tool-call-parser inkling2 \
  --mem-fraction-static "$MEM" --host 127.0.0.1 --port "$PORT"
