"""MCP server exposing Docker triage tools over stdio.

Run directly (the client launches it as a subprocess):  python mcp_servers/docker/server.py
Debug with the Inspector:                              mcp dev mcp_servers/docker/server.py
"""
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Annotated

import docker
from docker.errors import DockerException, NotFound
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

# stdout carries the JSON-RPC protocol over stdio, so ALL logging goes to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s docker-mcp %(levelname)s %(message)s")
log = logging.getLogger("docker-mcp")

mcp = MCPServer("docker")

MAX_LOG_CHARS = 6000  # ~1,500 tokens; keeps tool output inside the 8K context budget
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

ContainerName = Annotated[str, Field(description="Container name or ID, e.g. 'payments'")]

_docker = None


def _client() -> docker.DockerClient:
    """Connect lazily so the server starts (and lists tools) even if Docker is down."""
    global _docker
    if _docker is None:
        try:
            _docker = docker.from_env()
            _docker.ping()
        except DockerException as e:
            _docker = None
            raise ToolError(f"Cannot reach the Docker daemon. Is Docker Desktop running? ({e})") from None
    return _docker


def _get(name: str):
    """Look up a container; on a miss, tell the model which names DO exist so it can retry."""
    try:
        return _client().containers.get(name)
    except NotFound:
        existing = [c.name for c in _client().containers.list(all=True)]
        raise ToolError(f"No container named '{name}'. Existing containers: {existing}") from None


@mcp.tool(annotations=READ_ONLY)
def list_containers(
    include_stopped: Annotated[bool, Field(description="Also include exited/stopped containers")] = True,
) -> list[dict]:
    """List Docker containers with status, exit code, restart count and health.

    Start here when you don't know which containers exist or the user gives only a partial name.
    """
    out = []
    for c in _client().containers.list(all=include_stopped):
        state = c.attrs["State"]
        out.append({
            "name": c.name,
            "image": c.attrs["Config"]["Image"],
            "status": state["Status"],
            "exit_code": state["ExitCode"],
            "restart_count": c.attrs.get("RestartCount", 0),
            "health": state.get("Health", {}).get("Status"),
        })
    return out


@mcp.tool(annotations=READ_ONLY)
def inspect_container(name: ContainerName) -> dict:
    """Explain WHY a container is in its current state: exit code, OOM-killed flag, error, restart policy
    and count, memory limit, health, ports.

    Use this FIRST to diagnose a crash or restart loop. Exit code 137 with oom_killed=true means it ran
    out of memory. A non-zero exit with oom_killed=false means the application itself failed; read its
    logs next. Environment variable values are hidden; only their names are returned.
    """
    c = _get(name)
    a, state, host = c.attrs, c.attrs["State"], c.attrs["HostConfig"]
    mem_limit = host.get("Memory") or 0
    return {
        "name": c.name,
        "image": a["Config"]["Image"],
        "command": a["Config"].get("Cmd"),
        "status": state["Status"],
        "exit_code": state["ExitCode"],
        "oom_killed": state["OOMKilled"],
        "error": state.get("Error") or None,
        "started_at": state["StartedAt"],
        "finished_at": state["FinishedAt"],
        "restart_count": a.get("RestartCount", 0),
        "restart_policy": host.get("RestartPolicy", {}).get("Name") or "no",
        "memory_limit_mb": round(mem_limit / 2**20) if mem_limit else "unlimited",
        "health": state.get("Health", {}).get("Status"),
        "ports": a["NetworkSettings"].get("Ports"),
        "env_var_names": sorted(e.split("=", 1)[0] for e in a["Config"].get("Env") or []),
    }


@mcp.tool(annotations=READ_ONLY)
def get_container_logs(
    name: ContainerName,
    tail: Annotated[int, Field(ge=1, le=500, description="How many of the most recent lines to return")] = 100,
    since_minutes: Annotated[int | None, Field(ge=1, description="Only return lines from the last N minutes")] = None,
) -> str:
    """Read what the application printed (stdout and stderr): stack traces, error messages, connection failures.

    Use after inspect_container shows a non-zero exit code, or when asked what a container is doing.
    Very long output is truncated, keeping the most recent lines.
    """
    c = _get(name)
    kwargs = {"tail": tail, "timestamps": True}
    if since_minutes:
        kwargs["since"] = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    text = c.logs(**kwargs).decode("utf-8", errors="replace")
    if not text.strip():
        return f"(no log output from '{c.name}' in the requested window)"
    if len(text) > MAX_LOG_CHARS:
        dropped = len(text) - MAX_LOG_CHARS
        text = f"...[{dropped} earlier characters truncated]...\n" + text[-MAX_LOG_CHARS:]
    return text


@mcp.tool(annotations=READ_ONLY)
def get_container_stats(name: ContainerName) -> dict:
    """Current CPU and memory usage of a RUNNING container, compared with its memory limit.

    Use for slowness or suspected memory pressure on a container that is still up. For a container
    that has already died, use inspect_container (its oom_killed field) instead.
    """
    c = _get(name)
    if c.status != "running":
        raise ToolError(f"'{c.name}' is {c.status}; stats exist only for running containers. "
                        f"Use inspect_container instead.")
    st = c.stats(stream=False)  # waits for two samples so CPU can be computed as a delta
    cpu, pre = st["cpu_stats"], st["precpu_stats"]
    cpu_delta = cpu["cpu_usage"]["total_usage"] - pre.get("cpu_usage", {}).get("total_usage", 0)
    sys_delta = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    cpu_pct = cpu_delta / sys_delta * (cpu.get("online_cpus") or 1) * 100 if sys_delta > 0 else 0.0
    mem = st["memory_stats"]
    used = mem.get("usage", 0) - mem.get("stats", {}).get("inactive_file", 0)  # exclude reclaimable page cache
    limit = mem.get("limit", 0)
    return {
        "name": c.name,
        "cpu_percent": round(cpu_pct, 1),
        "memory_used_mb": round(used / 2**20, 1),
        "memory_limit_mb": round(limit / 2**20, 1),
        "memory_percent": round(used / limit * 100, 1) if limit else None,
    }


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                                      idempotentHint=False, openWorldHint=False))
def restart_container(
    name: ContainerName,
    reason: Annotated[str, Field(description="One sentence: the evidence that a restart will fix the problem")],
) -> str:
    """Restart a container. MUTATING: interrupts the running service.

    Use only after diagnosing with inspect_container and get_container_logs, and only when a restart is
    likely to fix the cause (for example, a dependency that was down has recovered). A restart does NOT
    fix out-of-memory kills or configuration errors. Never restart just to see what happens.
    """
    c = _get(name)
    log.warning("RESTART %s requested. Reason: %s", c.name, reason)  # audit trail on stderr
    c.restart(timeout=10)
    c.reload()
    return f"Restarted '{c.name}'; status is now '{c.status}'. Recorded reason: {reason}"


if __name__ == "__main__":
    mcp.run(transport="stdio")