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

# Use project directory name as context hint
PROJECT_NAME=$(basename "${CLAUDE_PROJECT_DIR:-$(pwd)}")

# Query engram for relevant context (silent fail if engram-cli not found)
CONTEXT=$(engram-cli context "working on $PROJECT_NAME project" -l 10 2>/dev/null)

if [ -n "$CONTEXT" ]; then
    echo "## Engram Memory Context"
    echo ""
    echo "The following memories were loaded from Engram for this project:"
    echo ""
    echo "$CONTEXT"
fi

exit 0
