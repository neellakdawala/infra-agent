#!/usr/bin/env bash
# Launch Hermes 3 on vLLM with an OpenAI-compatible API and native tool calling.
set -euo pipefail

# Load .env if present
if [ -f .env ]; then set -a; source .env; set +a; fi

MODEL_ID="${MODEL_ID:-NousResearch/Hermes-3-Llama-3.1-8B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
VLLM_API_KEY="${VLLM_API_KEY:-local-dev-key}"

QUANT_ARGS=()
if [ -n "${QUANTIZATION:-}" ]; then QUANT_ARGS=(--quantization "$QUANTIZATION"); fi

# Flag notes:
#   --enable-auto-tool-choice   lets the model decide when to call a tool (tool_choice="auto")
#   --tool-call-parser hermes   parses Hermes' <tool_call>{...}</tool_call> output into OpenAI tool_calls
#   --max-model-len             caps context length; smaller = less KV-cache memory reserved
#   --gpu-memory-utilization    fraction of VRAM vLLM may claim (weights + KV cache)
exec vllm serve "$MODEL_ID" \
  --host 0.0.0.0 --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  "${QUANT_ARGS[@]}"
