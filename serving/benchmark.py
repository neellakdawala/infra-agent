"""Latency and throughput vs concurrency for any OpenAI-compatible server.

Metrics per concurrency level:
  TTFT  time to first token (queueing + prefill)
  TPOT  time per output token after the first (decode speed)
  E2E   end-to-end request latency
  throughput  total output tokens / wall-clock time, across all requests

    python serving/benchmark.py --concurrency 1 2 4 --requests 16
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tabulate import tabulate

load_dotenv()
MODEL = os.getenv("MODEL_ID", "hermes3:8b")

PROMPTS = [  # triage-style, similar length to what the agent will send
    "A Kubernetes pod is in CrashLoopBackOff. List the first three things you would check and why.",
    "Explain the difference between a liveness probe and a readiness probe in Kubernetes.",
    "A Docker container exits immediately with code 137. What are the likely causes?",
    "A GitHub Actions workflow fails with 'permission denied' when pushing an image. How do you debug it?",
    "An EC2 instance is unreachable over SSH but shows as running. Walk through the triage steps.",
    "What does ImagePullBackOff mean and how do you resolve it?",
    "How would you find which process is using the most memory inside a running container?",
    "A pod is stuck in Pending. What scheduler-related causes should you investigate?",
]


def pct(values, p):
    """Percentile with linear interpolation (same method as numpy's default)."""
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_request(client, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    usage_tokens = 0
    chunks = 0
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,                              # needed to observe the FIRST token
        stream_options={"include_usage": True},   # ask for exact token counts in the last chunk
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
        if chunk.usage:
            usage_tokens = chunk.usage.completion_tokens
    e2e = time.perf_counter() - t0
    tokens = usage_tokens or chunks               # fall back to chunk count if usage is missing
    ttft = ttft if ttft is not None else e2e
    tpot = (e2e - ttft) / max(tokens - 1, 1)      # decode time spread over the tokens after the first
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot, "tokens": tokens}


async def run_level(client, concurrency, n_requests, max_tokens):
    sem = asyncio.Semaphore(concurrency)          # at most `concurrency` requests in flight

    async def bounded(i):
        async with sem:
            return await one_request(client, PROMPTS[i % len(PROMPTS)], max_tokens)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(bounded(i) for i in range(n_requests)))
    wall = time.perf_counter() - t0

    ms = lambda key: [r[key] * 1000 for r in results]
    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "ttft_p50_ms": pct(ms("ttft"), 50),
        "ttft_p95_ms": pct(ms("ttft"), 95),
        "e2e_p50_ms": pct(ms("e2e"), 50),
        "e2e_p95_ms": pct(ms("e2e"), 95),
        "tpot_mean_ms": statistics.mean(ms("tpot")),
        "throughput_tok_s": sum(r["tokens"] for r in results) / wall,
        "avg_output_tokens": statistics.mean(r["tokens"] for r in results),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--requests", type=int, default=16, help="requests per concurrency level")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    client = AsyncOpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("VLLM_API_KEY", "ollama"),
        timeout=300,
    )

    print("Warming up (loads the model, fills caches)...")
    await run_level(client, 2, 4, 16)

    rows = []
    for c in args.concurrency:
        print(f"concurrency={c}: {args.requests} requests...")
        rows.append(await run_level(client, c, args.requests, args.max_tokens))

    print()
    print(tabulate(
        [[r["concurrency"],
          f'{r["ttft_p50_ms"]:.0f} / {r["ttft_p95_ms"]:.0f}',
          f'{r["e2e_p50_ms"]:.0f} / {r["e2e_p95_ms"]:.0f}',
          f'{r["tpot_mean_ms"]:.1f}',
          f'{r["throughput_tok_s"]:.1f}',
          f'{r["avg_output_tokens"]:.0f}'] for r in rows],
        headers=["conc", "TTFT p50/p95 ms", "E2E p50/p95 ms", "TPOT ms", "tok/s", "avg out tok"],
    ))

    out = Path("results")
    out.mkdir(exist_ok=True)
    path = out / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({
        "model": MODEL,
        "base_url": os.getenv("VLLM_BASE_URL"),
        "max_tokens": args.max_tokens,
        "levels": rows,
    }, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    asyncio.run(main())