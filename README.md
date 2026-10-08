# Self-Hosted Infrastructure Agent

An infrastructure-triage agent that runs entirely on a self-hosted open model. Hermes 3 (Llama 3.1 8B) is served by vLLM, and the agent reaches Docker, Kubernetes, AWS and GitHub through MCP servers. No external LLM APIs.

> Status: Phase 1 (model serving) in progress. Metrics below are filled in only from runs of `serving/benchmark.py` and `evals/`.

## Architecture (target)

```
user ──► agent loop (MCP client) ──► vLLM (Hermes 3 8B, OpenAI-compatible API)
              │
              ├──► mcp_servers/docker
              ├──► mcp_servers/k8s
              ├──► mcp_servers/aws
              └──► mcp_servers/github
```

## Repo layout

| Path | Purpose |
|---|---|
| `serving/` | vLLM launch script, smoke test, latency/throughput benchmark |
| `mcp_servers/` | One MCP server per platform (Phase 2, 4) |
| `agent/` | Hand-written agent loop and MCP client (Phase 3) |
| `evals/` | Task suite, setup scripts, scoring harness (Phase 5) |
| `deploy/` | Dockerfiles, docker-compose, Kubernetes manifests (Phase 6) |
| `docs/` | Learning notes per phase |

## Roadmap

- [ ] **Phase 0:** GPU environment, repo, Python env
- [ ] **Phase 1:** Serve Hermes 3 on vLLM; smoke-test tool calling; benchmark latency vs concurrency
- [ ] **Phase 2:** Docker MCP server, tested with MCP Inspector
- [ ] **Phase 3:** Agent loop: MCP tools → OpenAI tool schemas → vLLM → execute → repeat
- [ ] **Phase 4:** Kubernetes (kind), AWS (LocalStack + read-only IAM), GitHub (sandbox repo) servers
- [ ] **Phase 5:** Eval harness: task completion, tool-selection accuracy, steps, latency, failure analysis
- [ ] **Phase 6:** Containerize, Helm/K8s deploy with RBAC, CI, demo GIF

## Quickstart (Phase 1)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

bash serving/start_vllm.sh          # terminal 1: start the model server
python serving/smoke_test.py        # terminal 2: verify chat + tool calling
python serving/benchmark.py         # measure latency at 1/4/16/32 concurrent requests
```

## Results

_To be filled from real runs._

| Metric | Value | How measured |
|---|---|---|
| p50 / p95 end-to-end latency | – | `serving/benchmark.py` |
| Time to first token | – | `serving/benchmark.py` |
| Task completion | – | `evals/` |
| Tool-selection accuracy | – | `evals/` |
