#!/usr/bin/env bash
# Install Engram-aware handoff skills into ~/.claude/skills/.
#
# Copies each skill in ./skills/<name>/SKILL.md into ~/.claude/skills/<name>/SKILL.md.
# If a destination skill already exists and differs, the old version is backed up
# to SKILL.md.bak.<timestamp> alongside the new one.
#
# Usage:
#   scripts/install-skills.sh                # install to $HOME/.claude/skills
#   CLAUDE_HOME=/custom/.claude scripts/install-skills.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="$REPO_DIR/skills"
DEST_DIR="${CLAUDE_HOME:-$HOME/.claude}/skills"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "error: $SRC_DIR not found (run from a checkout of engram-mcp)" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
ts="$(date +%Y%m%d-%H%M%S)"

installed=0
skipped=0

for skill_dir in "$SRC_DIR"/*/; do
    name="$(basename "$skill_dir")"
    src_file="$skill_dir/SKILL.md"
    dest_skill_dir="$DEST_DIR/$name"
    dest_file="$dest_skill_dir/SKILL.md"

    [[ -f "$src_file" ]] || continue

    mkdir -p "$dest_skill_dir"

    if [[ -f "$dest_file" ]]; then
        if cmp -s "$src_file" "$dest_file"; then
            echo "  skip: $name (up to date)"
            skipped=$((skipped + 1))
            continue
        fi
        backup="$dest_file.bak.$ts"
        cp "$dest_file" "$backup"
        echo "  backup: $name -> $(basename "$backup")"
    fi

    cp "$src_file" "$dest_file"
    echo "  install: $name"
    installed=$((installed + 1))
done

echo "done. installed=$installed skipped=$skipped"
