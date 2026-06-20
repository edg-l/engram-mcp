# Engram Claude Code Lifecycle Hooks

Engram passively captures decisions and session summaries from
Claude Code lifecycle events, storing them as typed memories without any
explicit `memory_store` calls. Each event handler filters, redacts, and
dispatches to the appropriate storage path (handoff or Fact).
For loading context at the start of a session, see `scripts/engram-hook.sh`,
which handles the `SessionStart` event separately.

## Install

```bash
engram-cli hooks install
```

This writes 6 managed hook entries to `~/.claude/settings.json`:
`UserPromptSubmit`, `PostToolUse`, `Stop`, `PreCompact`, `SessionEnd`,
`SubagentStop`. Each entry carries `"_source": "engram-cli"` so that
`hooks uninstall` can remove only the managed entries without touching any
user-authored hooks. If `~/.claude/settings.json` already exists, a backup
is created at `settings.json.bak.<timestamp>` before writing.

`SessionStart` is intentionally not managed here; see the
[SessionStart loader](#sessionstart-loader) section.

## Per-event reference

| Event | Memory type | Trigger condition | Notes |
|---|---|---|---|
| `SessionStart` | (context load, no store) | Every session start | Handled by `scripts/engram-hook.sh`, not by `hooks install`. See [SessionStart loader](#sessionstart-loader). |
| `UserPromptSubmit` | `Decision` | Prompt contains a cue word and is >= `ENGRAM_HOOK_MIN_CHARS` chars | Cue regex default: `let's`, `decided`, `we'll`, `going to`, `switched to`, `use`, `prefer`, `chose`, `choose`, `reason`, `because`. |
| `PostToolUse` | (none, no-op) | Never | Tool-call outcomes are deliberately not captured — low-signal noise that bloats the store. The dispatch handler validates the payload and returns immediately. |
| `Stop` | `handoff` (or `Fact` fallback when no git branch) | `last_assistant_message` >= 200 chars and `stop_hook_active` is not `true` | `stop_hook_active=true` means a Stop hook is already running; skipped to prevent reentrance. |
| `PreCompact` | `handoff` (or `Fact` fallback when no git branch) | Always fires when Claude Code compacts context | Uses `custom_instructions` as the summary if present, otherwise stores a timestamp marker. |
| `SessionEnd` | `Fact` (importance 0.3) | Session ends for any reason except `clear` or `resume` | Low importance. Skipped on `reason: "clear"` and `reason: "resume"`. |
| `SubagentStop` | `Fact` (importance 0.55) | Subagent completes; `last_assistant_message` >= 200 chars | Off by default. Requires `ENGRAM_HOOK_SUBAGENTSTOP_ENABLED=true`. |

## Env vars

| Variable | Default | Meaning |
|---|---|---|
| `ENGRAM_HOOK_USERPROMPTSUBMIT_ENABLED` | `true` | Enable/disable the UserPromptSubmit handler. |
| `ENGRAM_HOOK_STOP_ENABLED` | `true` | Enable/disable the Stop handler. |
| `ENGRAM_HOOK_PRECOMPACT_ENABLED` | `true` | Enable/disable the PreCompact handler. |
| `ENGRAM_HOOK_SESSIONEND_ENABLED` | `true` | Enable/disable the SessionEnd handler. |
| `ENGRAM_HOOK_SUBAGENTSTOP_ENABLED` | `false` | Enable/disable the SubagentStop handler. Must be explicitly set to `true`. |
| `ENGRAM_HOOK_SESSIONSTART_ENABLED` | `true` | Controls the SessionStart handler in dispatch. No effect on `hooks install` (SessionStart is never managed). |
| `ENGRAM_HOOK_MIN_CHARS` | `40` | Minimum byte length for prompt content to be processed by UserPromptSubmit. |
| `ENGRAM_HOOK_MIN_IMPORTANCE` | `0.5` | Minimum importance score for hook-stored memories. |
| `ENGRAM_HOOK_PROMPT_CUE_REGEX` | (built-in regex) | Override the cue word pattern for UserPromptSubmit. Must be a valid Rust `regex` crate pattern. |

Set any per-event toggle to `0`, `false`, or `no` to disable; `1`, `true`, or
`yes` to enable.

## Secret redaction

Before any content is stored, it passes through a redaction filter. The
following six pattern classes are matched and replaced with `[REDACTED]`:

- **AWS access key ID**: patterns matching `AKIA` followed by 16 uppercase
  alphanumeric characters.
- **AWS secret access key (context-gated)**: patterns matching
  `aws_secret_access_key` (case-insensitive) followed by a 40-char value.
- **GitHub tokens**: fine-grained PATs, OAuth tokens, and server tokens
  matching the `gh[pousr]_` prefix followed by 36+ alphanumeric characters.
- **Generic env assignment**: variable names ending in `_KEY`, `_SECRET`,
  `_TOKEN`, or `PASSWORD` followed by `=value`; only the value is replaced,
  the variable name remains visible.
- **JWTs**: three Base64url segments matching the `eyJ...eyJ...` structure.
- **OpenAI secret keys**: strings matching `sk-` followed by 20+ alphanumeric
  characters.

This is a best-effort filter, not a complete DLP solution. Do not rely on it
as your sole protection for secrets in prompts or tool output.

## Testing without Claude Code

Process a canned payload directly:

```bash
echo '{"prompt":"we decided to use Postgres because writes are contended"}' \
  | engram-cli hook-event UserPromptSubmit
```

Use `--dry-run` to preview what would be stored without writing anything:

```bash
echo '{"reason":"other"}' | engram-cli hook-event SessionEnd --dry-run
```

Canned payloads for all six managed events are in `hooks/fixtures/`. Pipe any
of them to test a specific handler:

```bash
engram-cli hook-event Stop --dry-run < hooks/fixtures/stop.json
```

## Uninstall

```bash
engram-cli hooks uninstall
```

Removes only entries tagged `_source: "engram-cli"`. Any hooks you added
manually are preserved. Events whose matcher list becomes empty after removal
are deleted from the hooks object.

## Status

```bash
engram-cli hooks status
```

Lists all events currently managed by `engram-cli` and flags any events where
a user-authored hook is also registered alongside the managed entry (shadowed
events).

## SessionStart loader

`scripts/engram-hook.sh` is the canonical handler for `SessionStart`. It
builds a context query from the current git branch and recent commits, calls
`engram-cli context`, and prints the results as a markdown block so Claude
Code injects them into the system prompt. This gives the LLM project context
without requiring an explicit `memory_context` call at the start of each
session.

`engram-cli hooks install` deliberately skips `SessionStart` to avoid
conflicting with this script or duplicating the context-load behavior.

To wire the script manually, copy it somewhere stable and add to
`~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/engram-hook.sh"
          }
        ]
      }
    ]
  }
}
```

The script header comment (`scripts/engram-hook.sh` lines 1-24) shows the
exact JSON structure expected by Claude Code.

## Troubleshooting

**Hook times out.** The default timeout for managed hooks is 4 seconds
(5 seconds for `SessionEnd`). If embedding model cold-start latency on first
invocation exceeds this (typically ~500ms after the ONNX runtime is warm, but
up to a few seconds on a cold process), increase the `timeout` field for the
affected event in `~/.claude/settings.json`. There is no CLI option for this;
edit the JSON directly after running `hooks install`.

**Permission error writing settings.json.** `~/.claude/` must be writable by
the current user. If it was created by Claude Code with restricted permissions,
run `chmod u+w ~/.claude/settings.json` or re-run `hooks install` after fixing
permissions.

**Hook fires but nothing is stored.** Run the payload through `--dry-run` to
see which skip reason is returned (`prompt_too_short`, `prompt_no_cue_match`,
`posttooluse_noop`, `stop_noop`, `reason_filtered`, etc.).
The dispatch logic lives in `src/hooks/dispatch.rs`.
