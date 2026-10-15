"""Phase 1 smoke test: server up, chat works, tool call parses, tool result is used."""
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # read .env into os.environ; real env vars take precedence
client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("VLLM_API_KEY", "ollama"),
)
MODEL = os.getenv("MODEL_ID", "hermes3:8b")

SYSTEM = "You are an infrastructure triage assistant. Use tools to gather evidence before answering."
USER = "The pod payments-7f9c in namespace prod keeps restarting. Why?"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_pod_logs",
        "description": "Fetch recent logs for a Kubernetes pod. Use when asked why a pod is failing or restarting.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Kubernetes namespace"},
                "pod": {"type": "string", "description": "Pod name"},
            },
            "required": ["namespace", "pod"],
        },
    },
}]


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main():
    results = []

    # 1. Is the server up and serving the model we expect?
    try:
        models = [m.id for m in client.models.list().data]
    except Exception as e:
        check("server reachable", False, f"{type(e).__name__}: {e}")
        sys.exit(1)
    results.append(check("server reachable", MODEL in models, f"models={models}"))

    # 2. Plain chat, no tools
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "In one sentence, what is a Kubernetes pod?"}],
        max_tokens=60,
        temperature=0,
    )
    ms = (time.perf_counter() - t0) * 1000
    text = (r.choices[0].message.content or "").strip()
    results.append(check("plain chat", bool(text),
                         f"{ms:.0f} ms, {r.usage.completion_tokens} tokens: {text[:100]}"))

    # 3. Does the model emit a tool call, and does the server parse it?
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    r = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto", temperature=0,
    )
    msg = r.choices[0].message
    calls = msg.tool_calls or []
    if not calls:
        # Leaked "<tool_call>" text in content means the model tried but the parser failed
        results.append(check("tool call parsed", False, f"no tool_calls; content={msg.content!r:.200}"))
    else:
        call = calls[0]
        try:
            args = json.loads(call.function.arguments)  # arguments arrive as a JSON *string*
        except json.JSONDecodeError as e:
            args = {}
            print(f"       malformed arguments: {call.function.arguments!r} ({e})")
        ok = (call.function.name == "get_pod_logs"
              and args.get("namespace") == "prod"
              and args.get("pod") == "payments-7f9c")
        results.append(check("tool call parsed", ok,
                             f"{call.function.name}({args}), finish_reason={r.choices[0].finish_reason}"))

        # 4. Round trip: return a fake tool result and see if the model reasons over it
        messages.append(msg.model_dump(exclude_none=True))  # the assistant turn WITH its tool_calls
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,  # links this result to the call that requested it
            "content": "Error: connect ECONNREFUSED 10.0.3.14:5432\nprocess exited with code 1",
        })
        r = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0)
        answer = (r.choices[0].message.content or "").strip()
        grounded = any(k in answer.lower() for k in ("5432", "postgres", "database", "connection refused", "econnrefused"))
        results.append(check("tool result -> answer", grounded, answer[:200]))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()