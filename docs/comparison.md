# Compared to other tools

| Tool | Branch-aware | Structured handoffs | Per-section embeddings | Local-only | Storage |
|------|:---:|:---:|:---:|:---:|---|
| **Engram** | ✓ | ✓ | ✓ | ✓ | SQLite + ONNX |
| [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) | — | — | — | ✓ | SQLite-vec |
| [Mem0](https://mem0.ai) | — | — | — | partial | Vector DB + LLM extraction |
| [Zep / Graphiti](https://getzep.com) | — | — | — | — | Neo4j |
| [Letta (MemGPT)](https://letta.com) | — | OS-style blocks | — | partial | Pluggable |

Comparison as of 2026-05. Engram is built for coding agents working in git; the others target broader assistant memory.

## Mem0

Mem0 defaults to the cloud, runs LLM extraction on every write, and has no branch model. It suits chat personalization more than an agent switching git contexts during a day.

## mcp-memory-service

mcp-memory-service is a general memory server with no handoff structure and no notion of branches. Engram gives up breadth for a workflow built around sessions on branches.

## A vector database

A vector database is one part of this. Decay, deduplication, clustering, todos and the MCP layer are what Engram adds on top, and Qdrant or Chroma on their own provide none of them.

## SQLite

SQLite rather than Postgres or libSQL because there is nothing to operate, and at 256 dimensions the embeddings are small enough that SQLite handles what one developer accumulates. For several users sharing a store, this is the wrong tool.
