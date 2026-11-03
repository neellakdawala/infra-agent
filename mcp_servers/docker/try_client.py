"""Minimal MCP client: launch the Docker server over stdio, list tools, call one."""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    server = StdioServerParameters(
        command=sys.executable,  # the venv's python, so the server sees the same packages
        args=["mcp_servers/docker/server.py"],
    )
    async with stdio_client(server) as (read, write):          # spawn the subprocess, wire up stdin/stdout
        async with ClientSession(read, write) as session:       # JSON-RPC session over those streams
            await session.initialize()                          # handshake

            tools = (await session.list_tools()).tools
            print(f"{len(tools)} tools:")
            for t in tools:
                ro = t.annotations.read_only_hint if t.annotations else None
                print(f"  - {t.name:22} read_only={ro}")
            print("\ninspect_container schema:")
            print(json.dumps(next(t for t in tools if t.name == "inspect_container").input_schema, indent=2))

            for name in ["worker", "paymnts"]:
                result = await session.call_tool("inspect_container", {"name": name})
                print(f"\ninspect_container({name!r})  is_error={result.is_error}")
                print(result.content[0].text[:600])


if __name__ == "__main__":
    asyncio.run(main())