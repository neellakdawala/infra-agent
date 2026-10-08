"""Phase 1 smoke test: is vLLM up, does chat work, and does tool calling parse?

Run after `bash serving/start_vllm.sh` is serving:
    python serving/smoke_test.py
"""
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("VLLM_API_KEY", "local-dev-key"),
)
MODEL = os.getenv("MODEL_ID", "NousResearch/Hermes-3-Llama-3.1-8B")

# A fake tool in OpenAI's schema. In Phase 3 these will be generated from MCP servers.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pod_logs",
            "description": "Fetch recent logs for a Kubernetes pod. Use when asked why a pod is failing or restarting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    "pod": {"type": "string", "description": "Pod name"},
                    "tail_lines": {"type": "integer", "description": "Number of lines", "default": 100},
                },
                "required": ["namespace", "pod"],
            },
        },
    }
]


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def main():
    results = []

    # 1. Server reachable and serving the expected model
    try:
        models = [m.id for m in client.models.list().data]
        results.append(check("server reachable", MODEL in models, f"models={models}"))
    except Exception as e:
        check("server reachable", False, str(e))
        sys.exit(1)

    # 2. Plain chat
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "In one sentence, what is a Kubernetes pod?"}],
        max_tokens=60,
        temperature=0,
    )
    dt = (time.perf_counter() - t0) * 1000
    text = r.choices[0].message.content or ""
    results.append(check("plain chat", len(text) > 0, f"{dt:.0f} ms, {r.usage.completion_tokens} tokens: {text.strip()[:100]}"))

    # 3. Tool calling: the model should emit a structured call, not prose
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an infrastructure triage assistant. Use tools to gather evidence before answering."},
            {"role": "user", "content": "The pod payments-7f9c in namespace prod keeps restarting. Why?"},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    msg = r.choices[0].message
    calls = msg.tool_calls or []
    ok = len(calls) > 0 and calls[0].function.name == "get_pod_logs"
    detail = "no tool call returned; content=" + repr((msg.content or "")[:150])
    if calls:
        args = json.loads(calls[0].function.arguments)
        detail = f"{calls[0].function.name}({args})"
        ok = ok and args.get("namespace") == "prod" and args.get("pod") == "payments-7f9c"
    results.append(check("tool call parsed", ok, detail))

    # 4. Round trip: feed a fake tool result back and get a final answer
    if calls:
        followup = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an infrastructure triage assistant."},
                {"role": "user", "content": "The pod payments-7f9c in namespace prod keeps restarting. Why?"},
                msg.model_dump(exclude_none=True),
                {
                    "role": "tool",
                    "tool_call_id": calls[0].id,
                    "content": "Error: connect ECONNREFUSED 10.0.3.14:5432\nprocess exited with code 1",
                },
            ],
            tools=TOOLS,
            temperature=0,
        )
        answer = followup.choices[0].message.content or ""
        results.append(check("tool result -> answer", "5432" in answer or "database" in answer.lower() or "postgres" in answer.lower(), answer.strip()[:160]))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
