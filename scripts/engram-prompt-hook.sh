#!/usr/bin/env bash
# Claude Code UserPromptSubmit hook for Engram memory context.
#
# Queries relevant memories for each user prompt so the LLM has targeted
# context without needing to call memory_query explicitly.
#
# Install:
#   Add to ~/.claude/settings.json (or .claude/settings.json per-project):
#
#   {
#     "hooks": {
#       "UserPromptSubmit": [
#         {
#           "hooks": [
#             {
#               "type": "command",
#               "command": "/path/to/engram-prompt-hook.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }

# Read the user's prompt from stdin JSON
PROMPT=$(cat | jq -r '.user_prompt // empty' 2>/dev/null)

# If no prompt or jq not available, exit silently
if [ -z "$PROMPT" ]; then
    exit 0
fi

# Truncate long prompts to avoid poor embeddings from huge text
PROMPT=$(echo "$PROMPT" | head -c 500)

# Query engram for relevant memories (silent fail if engram-cli not found)
MEMORIES=$(engram-cli query "$PROMPT" -l 5 2>/dev/null)

if [ -n "$MEMORIES" ]; then
    echo "## Relevant Engram Memories"
    echo ""
    echo "$MEMORIES"
fi

exit 0
