# Phase 1 — Serving Hermes 3 on vLLM

## 1. Will it fit? VRAM math

An 8B-parameter model in fp16/bf16 stores 2 bytes per parameter:
8 × 10⁹ × 2 bytes ≈ **16 GB just for weights**. Whatever VRAM remains goes to the **KV cache**.

KV cache per token = 2 (K and V) × layers × kv_heads × head_dim × bytes.
For Llama 3.1 8B: 2 × 32 × 8 × 128 × 2 bytes = **128 KB per token**.
So 8,192 tokens of context for one request is about 1 GB, and serving 16 such requests at once needs about 16 GB of cache.

That is why:
- a 24 GB card (A10G, L4, 4090) fits fp16 weights plus a modest cache;
- `--max-model-len` matters, because it caps how much cache one request can claim;
- `--quantization fp8` (on supported GPUs) roughly halves weight memory, leaving more room for concurrent requests. You give up a small amount of quality. Measure it, don't assume it.

Llama 3.1 uses **grouped-query attention** (8 KV heads shared across 32 query heads), which is why its KV cache is 4× smaller than it would otherwise be. It's a good interview detail.

## 2. Prefill vs decode

Generation has two phases:
- **Prefill:** the whole prompt is processed in one parallel pass. It is compute-bound and fast per token. It determines **time to first token (TTFT)**.
- **Decode:** one token at a time, each step reading all the weights from memory. It is memory-bandwidth-bound. It determines **time per output token (TPOT)**.

E2E latency ≈ TTFT + (output_tokens − 1) × TPOT.
So "650 ms latency" means nothing until you say which one, at what output length, and at what concurrency. The benchmark reports all three.

## 3. PagedAttention

Naive servers reserve one contiguous KV-cache slab per request, sized for the maximum length. Most of it sits empty, so memory fills up after a few requests.
vLLM splits the cache into fixed-size **blocks** (like OS memory pages) and allocates them as the sequence grows, tracked by a block table. Waste drops to almost nothing, so far more requests fit at once. More requests in flight means higher throughput.

## 4. Continuous batching

Static batching waits for every sequence in a batch to finish before starting new ones, so short answers sit idle behind long ones.
**Continuous (iteration-level) batching** re-forms the batch at every decode step: finished sequences leave and waiting ones join immediately. The GPU stays busy, and throughput scales with concurrency until compute or cache memory saturates.

What to look for in your benchmark: going from concurrency 1 to 16, **throughput should multiply several times while per-request E2E latency rises only modestly**. That curve is the evidence that continuous batching works, and it's the chart for your README.

## 5. Tool calling with Hermes

Hermes 3 was fine-tuned to emit tool calls as JSON inside `<tool_call>...</tool_call>` tags, using tool definitions placed in its system prompt. vLLM's `--tool-call-parser hermes` does two things:
1. renders your OpenAI-format `tools=[...]` into the prompt via the model's chat template;
2. parses the tags out of the generated text and returns them as structured `tool_calls`.

Your client code therefore looks exactly like OpenAI's API, which is what makes the model swappable later.
Failure modes to watch for (and to count in Phase 5): the model answers in prose instead of calling a tool, picks the wrong tool, emits invalid JSON, or invents argument values.

## 6. Checklist

- [ ] `start_vllm.sh` serving; note GPU type, VRAM, vLLM version
- [ ] `smoke_test.py` passes 4/4
- [ ] `benchmark.py` run; JSON saved
- [ ] README results table filled with real numbers and the GPU named
- [ ] Optional: rerun with `QUANTIZATION=fp8` and compare

## Questions you should be able to answer afterwards

1. Why does throughput rise with concurrency but per-request latency rises too?
2. Your TTFT spikes at concurrency 32 but TPOT doesn't. What's happening? (Hint: queueing and prefill contention.)
3. Why is decode memory-bandwidth-bound rather than compute-bound?
4. What breaks if you set `--max-model-len 32768` on a 24 GB card?
