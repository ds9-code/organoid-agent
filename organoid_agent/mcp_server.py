"""
MCP server — exposes the organoid-agent tool space (HNOCAModel + PubMedTool)
to any MCP-compatible client (Hermes-CLI, Claude Desktop, Cursor, Zed, etc.).

Runs as a stdio subprocess. Hermes (or another client) starts this process
on demand, talks to it via JSON-RPC on stdin/stdout, and discovers all our
tools automatically.

Usage (manual smoke test):
    PYTHONPATH=. python -m organoid_agent.mcp_server          # waits for JSON-RPC on stdin

Usage (via Hermes — add to ~/.hermes/config.yaml):
    mcp_servers:
      organoid_agent:
        command: "python"
        args: ["-m", "organoid_agent.mcp_server"]
        env:
          PYTHONPATH: "/path/to/organoid-agent"

Then `hermes -z 'What cells dominate Velasco D100?'` will route through
our query_composition / classify_cell_type / pubmed_search tools and get
real answers from real HNOCA cells, no hallucination.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from organoid_agent.tool_space.hnoca import HNOCAModel
from organoid_agent.tool_space.pubmed import PubMedTool
from organoid_agent.tool_space.instructions import TOOL_REGISTRY


# -- module-level singletons (loaded once when the server starts) --
_hnoca: HNOCAModel | None = None
_pubmed: PubMedTool | None = None


def _ensure_loaded() -> tuple[HNOCAModel, PubMedTool]:
    """Lazy-load HNOCA + PubMed on the first tool call."""
    global _hnoca, _pubmed
    if _hnoca is None:
        # Print init progress to stderr so it doesn't corrupt MCP stdio
        print("[organoid-agent MCP] loading HNOCA subset ...", file=sys.stderr)
        _hnoca = HNOCAModel(verbose=False)
        print(f"[organoid-agent MCP] {_hnoca.adata.n_obs:,} cells × "
              f"{_hnoca.adata.n_vars:,} genes loaded", file=sys.stderr)
    if _pubmed is None:
        _pubmed = PubMedTool()
    return _hnoca, _pubmed


def _build_mcp_tool(entry: dict[str, Any]) -> Tool:
    """Convert one entry from tool_config.json into an MCP Tool spec."""
    return Tool(
        name=entry["name"],
        description=entry["description"],
        inputSchema=entry["parameters"],
    )


def _run_tool(name: str, args: dict[str, Any]) -> Any:
    """Dispatch a tool call to the right backend instance."""
    hnoca, pubmed = _ensure_loaded()
    # Tools defined on HNOCAModel
    if hasattr(hnoca, name):
        fn = getattr(hnoca, name)
        return fn(**(args or {}))
    # Tools defined on PubMedTool
    if hasattr(pubmed, name):
        fn = getattr(pubmed, name)
        return fn(**(args or {}))
    raise ValueError(f"Unknown tool: {name!r}")


# ----------------------------------------------------------------------- #
# MCP server setup                                                        #
# ----------------------------------------------------------------------- #
server: Server = Server("organoid-agent")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """MCP protocol: tell the client which tools we expose."""
    return [_build_mcp_tool(t) for t in TOOL_REGISTRY]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[TextContent]:
    """MCP protocol: execute one tool, return its result as JSON text."""
    try:
        result = _run_tool(name, arguments or {})
        payload = json.dumps(result, default=str, indent=2)
    except Exception as exc:
        payload = json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "tool": name},
            indent=2,
        )
    return [TextContent(type="text", text=payload)]


async def _amain() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
