#!/usr/bin/env bash
# Claude Code SessionStart hook for Engram memory context.
#
# Loads relevant memories at the start of each conversation so the LLM
# has project context without needing to call memory_context explicitly.
#
# Install:
#   Add to ~/.claude/settings.json (or .claude/settings.json per-project):
#
#   {
#     "hooks": {
#       "SessionStart": [
#         {
#           "hooks": [
#             {
#               "type": "command",
#               "command": "/path/to/engram-hook.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJECT_NAME=$(basename "$PROJECT_DIR")

# Build context from recent git activity if available
CONTEXT_PARTS="working on $PROJECT_NAME"

if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    BRANCH=$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null)
    if [ -n "$BRANCH" ]; then
        CONTEXT_PARTS="$CONTEXT_PARTS, branch: $BRANCH"
    fi

    RECENT=$(git -C "$PROJECT_DIR" log --oneline -5 2>/dev/null)
    if [ -n "$RECENT" ]; then
        CONTEXT_PARTS="$CONTEXT_PARTS, recent commits: $RECENT"
    fi
fi

# Query engram for relevant context (silent fail if engram-cli not found)
MEMORIES=$(engram-cli context "$CONTEXT_PARTS" -l 10 2>/dev/null)

if [ -n "$MEMORIES" ]; then
    echo "## Engram Memory Context"
    echo ""
    echo "The following memories were loaded from Engram for this project:"
    echo ""
    echo "$MEMORIES"
fi

exit 0
