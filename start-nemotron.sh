#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mobrienv/code/nemotron35-bench
LLAMA=/home/mobrienv/src/llama.cpp/build-cuvk/bin/llama-server
MODEL="$ROOT/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf"
PORT="${PORT:-8080}"
CTX="${CTX:-65536}"
NP="${NP:-1}"
ALIAS="${ALIAS:-nemotron35-lightning-30b-a3b-nvfp4}"

# Default placement: early10 (10 routed-expert blocks on RTX). MoE blocks
# 24..51 stay in host RAM. Override with OT=... for other placements
# (e.g. OT='^blk\.(20|22|...)\...=CPU' for early8).
OT_DEFAULT='^blk\.(24|27|29|31|34|36|38|40|43|45|47|49|51)\.ffn_(up|down)_exps\.weight$=CPU'
OT="${OT:-$OT_DEFAULT}"

# Eight routed-expert blocks remain on the RTX (1,3,6,8,10,13,15,17).
# All later routed-expert matrices stay in host RAM. Dense/Mamba/attention/
# shared-expert tensors remain on CUDA. Keep fit disabled so placement is stable.
# Context is split across slots: NP=4 CTX=262144 => 4 slots x 64K each.
exec "$LLAMA" \
  -m "$MODEL" --alias "$ALIAS" \
  --host 127.0.0.1 --port "$PORT" --no-webui --jinja \
  --device CUDA0 -ngl all -fit off \
  -t 12 -tb 12 -c "$CTX" -np "$NP" \
  -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 \
  -ot "$OT"
