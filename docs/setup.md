# Setup

## Install

```bash
cargo install engram_mcp
```

Prebuilt archives for Linux x86_64, macOS arm64 and Windows x86_64 are on the [releases page](https://github.com/edg-l/engram-mcp/releases); each holds both binaries.

Or from source:

```bash
git clone https://github.com/edg-l/engram-mcp.git
cd engram-mcp
cargo build --release   # target/release/engram, target/release/engram-cli
```

`engram` speaks MCP over stdio and takes no arguments; your MCP client starts it. Run in a terminal, it prints a pointer to `engram-cli`, which is where every interactive command lives.

The store is one SQLite file, `~/.local/share/engram/memories.db` by default (`ENGRAM_DB` moves it).

## Claude Code

```bash
claude mcp add -s user engram $(which engram)
```

To skip the permission prompt on every Engram tool, add this to `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["mcp__engram__*"]
  }
}
```

Two prompts ship with the server: `/mcp__engram__handoff` writes a handoff, `/mcp__engram__resume` restores one.

## Claude Desktop

Add the server to the config file, `~/.config/Claude/claude_desktop_config.json` on Linux or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS:

```json
{
  "mcpServers": {
    "engram": {
      "command": "/path/to/engram"
    }
  }
}
```

## Load context when a session starts

`scripts/engram-hook.sh` builds a query from the current branch and recent commits, runs `engram-cli context`, and prints the result for Claude Code to put in the session's context. The agent starts with the relevant memories without calling `memory_context` itself.

```bash
cp scripts/engram-hook.sh ~/.claude/hooks/engram-hook.sh
```

Then add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/engram-hook.sh" }
        ]
      }
    ]
  }
}
```

It works outside git repositories, and exits silently when `engram-cli` is not on `PATH`.

## Capture session summaries automatically

```bash
engram-cli hooks install
```

This registers Engram for three Claude Code events. `SessionEnd` stores the session's last assistant message as a summary. `UserPromptSubmit` and `SubagentStop` capture prompts and subagent results, and both are off until you enable them. Captures are deduplicated, redacted for common secret formats, and capped at 50 per project per day. [hooks/README.md](../hooks/README.md) has the per-event details; `engram-cli hooks status` and `engram-cli hooks uninstall` inspect and remove the entries.

## Handoff skills

Two skills wrap the handoff tools for Claude Code: `handoff` gathers the session's state and calls `handoff_create`, and `read-handoffs` calls `handoff_resume` and pairs it with `memory_context`.

```bash
scripts/install-skills.sh
```

The bundled prompts cover the same flow if you prefer not to install skills.

## Sync two machines

```bash
engram-cli sync user@host --dry-run   # payload sizes in each direction, nothing written
engram-cli sync user@host
```

`sync` runs `engram-cli` on the other machine over `ssh`, so the remote needs `engram-cli` on the `PATH` of a non-interactive shell (or pass `--remote-bin`). Every project moves in both directions, and only what changed since the last sync with that host. The rules for conflicts are in [how it works](how-it-works.md#sync).

## Import markdown handoffs

Handoffs written as markdown files under `.claude/handoff/` import with:

```bash
scripts/port_md_handoffs.py ~/.claude/handoff /path/to/repo/.claude/handoff          # dry run
scripts/port_md_handoffs.py --apply ~/.claude/handoff /path/to/repo/.claude/handoff
```

Old headings map onto the current sections and `Continues from:` lines become chain links. The mapping is lossy (`Dead ends` becomes `blockers`); the original files are left in place.

## Upgrading

Restart every running `engram` server after installing a new version. A server resolves its project once, at startup.
