#!/usr/bin/env bash
# run_passes.sh <passes> <port>
# Two draws for one extra pass count, with the same nocot.run flags as the
# p1/p3 runs (run_all.sh step 1, plus --cache-salt draw2 on the second draw).
# Each setting gets its own model label, cache dir and output dirs.
# Grading goes to graded/p<N> and graded_draw2/p<N>, like p1 and p3.
set -euo pipefail
P=$1
PORT=$2
cd "$(dirname "$0")/.."
export PATH=/workspace/varloop_work/sgenv/bin:$PATH
export OPENROUTER_API_KEY=dummy
export NOCOT_API=http://127.0.0.1:$PORT/v1/chat/completions
MODEL="tml-varloop@p$P"

python -m nocot.run --model "$MODEL" --all-ncri --all-knowledge --workers 8 \
  --cache-dir ".cache_p$P" --out "runs_p$P"
python -m nocot.run --model "$MODEL" --all-ncri --all-knowledge --workers 8 \
  --cache-dir ".cache_p$P" --cache-salt draw2 --out "runs_p${P}_draw2"

python -m nocot.grade --rows "runs_p$P/${MODEL}__*.jsonl" --out "graded/p$P/"
python -m nocot.grade --rows "runs_p${P}_draw2/${MODEL}__*.jsonl" --out "graded_draw2/p$P/"
echo "done p$P"
