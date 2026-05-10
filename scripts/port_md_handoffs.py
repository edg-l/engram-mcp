#!/usr/bin/env python3
"""
Port legacy markdown handoffs into Engram.

Walks one or more directories of `YYYY-MM-DD-N.md` handoff files written by the
old skill template, translates each into the new Handoff schema, and calls
`engram-cli handoff create --from-file`.

Old skill sections      -> new schema field
------------------------  ------------------
Accomplished              summary (first paragraph) + notes (full)
Not done / next steps     next_steps
Key decisions             decisions
Dead ends                 blockers
Open questions ...        notes (appended)
Important context         mental_model
Working state at handoff  notes (appended)
## Update HH:MM blocks    notes (appended verbatim)

Frontmatter lines (`**Branch:** ...`, `**Continues from:** filename.md`) are
parsed and used to set --branch and --continues-from.

Files are processed in chronological order (lexicographic on filename) so chain
links can resolve via a {filename -> new_id} map built as we go.

Usage:
    python3 port_md_handoffs.py [--apply] DIR [DIR ...]

Without --apply, prints what would happen and writes nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Section name mapping (case-insensitive, trailing colon stripped)
# ---------------------------------------------------------------------------

CANONICAL_HEADINGS = [
    "summary",
    "decisions",
    "todos",
    "blockers",
    "mental_model",
    "next_steps",
    "notes",
]

# Old-skill heading -> (new field, mode)
#   mode = "set"   replace target field
#          "append"  append (separated by blank line) to target field
#          "summary_split"  first paragraph -> summary, full -> notes
ALIAS_MAP = {
    "accomplished": ("__accomplished__", "summary_split"),
    "not done / next steps": ("next_steps", "set"),
    "not done": ("next_steps", "set"),
    "next steps": ("next_steps", "set"),
    "key decisions": ("decisions", "set"),
    "decisions": ("decisions", "set"),
    "dead ends": ("blockers", "set"),
    "blockers": ("blockers", "set"),
    "open questions": ("notes", "append_q"),
    "open questions for the user": ("notes", "append_q"),
    "important context": ("mental_model", "set"),
    "mental model": ("mental_model", "set"),
    "working state at handoff": ("notes", "append_ws"),
    "working state": ("notes", "append_ws"),
    "summary": ("summary", "set"),
    "todos": ("todos", "set"),
    "notes": ("notes", "set"),
}

CANONICAL_TITLES = {
    "summary": "Summary",
    "decisions": "Decisions",
    "todos": "Todos",
    "blockers": "Blockers",
    "mental_model": "Mental Model",
    "next_steps": "Next Steps",
    "notes": "Notes",
}

H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
UPDATE_RE = re.compile(r"^##\s+Update\s+", re.IGNORECASE)
BRANCH_RE = re.compile(r"^\*\*Branch:\*\*\s*`?([^`\s]+)`?", re.MULTILINE)
CONTINUES_RE = re.compile(r"^\*\*Continues from:\*\*\s*([^\s]+)", re.MULTILINE)


@dataclass
class Handoff:
    path: Path
    branch: str | None = None
    continues_from_filename: str | None = None
    sections: dict[str, str] = field(default_factory=dict)


def parse_handoff(path: Path) -> Handoff:
    text = path.read_text(encoding="utf-8")
    h = Handoff(path=path)

    if m := BRANCH_RE.search(text):
        h.branch = m.group(1).strip()
    if m := CONTINUES_RE.search(text):
        cf = m.group(1).strip().strip("`.,;: ")
        if cf.lower() not in ("none", "n/a", "-"):
            h.continues_from_filename = Path(cf).name

    # Split into H2 chunks. Each chunk: heading + body until next H2 or EOF.
    matches = list(H2_RE.finditer(text))
    chunks: list[tuple[str, str]] = []  # (heading, body)
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip("\n")
        chunks.append((heading, body))

    accumulated: dict[str, list[str]] = {k: [] for k in CANONICAL_HEADINGS}

    for heading, body in chunks:
        if not body.strip():
            continue
        # "## Update HH:MM" appended later -> notes
        if UPDATE_RE.match("## " + heading):
            accumulated["notes"].append(f"### Update: {heading}\n{body}")
            continue
        key = heading.lower().rstrip(":").strip()
        mapping = ALIAS_MAP.get(key)
        if mapping is None:
            # Unknown heading -> dump into notes verbatim
            accumulated["notes"].append(f"### {heading}\n{body}")
            continue
        target, mode = mapping
        if mode == "set":
            accumulated[target].append(body)
        elif mode == "append_q":
            accumulated["notes"].append(f"### Open questions\n{body}")
        elif mode == "append_ws":
            accumulated["notes"].append(f"### Working state\n{body}")
        elif mode == "summary_split":
            # First paragraph -> summary; full body also -> notes for fidelity.
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            first = paragraphs[0] if paragraphs else body.strip()
            accumulated["summary"].append(first)
            accumulated["notes"].append(f"### Accomplished\n{body}")

    h.sections = {k: "\n\n".join(v).strip() for k, v in accumulated.items() if v}
    return h


def render_canonical_md(h: Handoff) -> str:
    """Render a Handoff into the new schema's markdown so parse_markdown accepts it."""
    sections = dict(h.sections)
    if not sections.get("summary"):
        # Summary is required by parse_markdown; synthesize one.
        sections["summary"] = f"Imported handoff from {h.path.name}"

    out: list[str] = []
    for key in CANONICAL_HEADINGS:
        if key not in sections or not sections[key]:
            continue
        out.append(f"## {CANONICAL_TITLES[key]}")
        out.append("")
        out.append(sections[key])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def discover(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".md":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        else:
            print(f"warning: skipping {p} (not a file or directory)", file=sys.stderr)
    # Dedupe and sort lexicographically (filename encodes date, so this
    # produces chronological order within a directory).
    files = sorted(set(files))
    return files


def run_create(
    md_path: Path,
    branch: str | None,
    continues_from_id: str | None,
    cli: list[str],
    apply: bool,
) -> str | None:
    cmd = list(cli) + ["handoff", "create", "--from-file", str(md_path)]
    if branch:
        cmd += ["--branch", branch]
    if continues_from_id:
        cmd += ["--continues-from", continues_from_id]
    if not apply:
        print("  would run:", " ".join(cmd))
        return None
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"engram-cli failed: exit {proc.returncode}")
    print(proc.stdout.rstrip())
    for line in proc.stdout.splitlines():
        if line.startswith("Handoff created:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("could not parse new handoff id from stdout")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="files or directories to import")
    ap.add_argument("--apply", action="store_true", help="actually invoke engram-cli (default: dry run)")
    ap.add_argument("--engram-cli", default=os.environ.get("ENGRAM_CLI", "engram-cli"), help="path to engram-cli binary")
    args = ap.parse_args()

    files = discover(args.paths)
    if not files:
        print("no .md files found")
        return 1

    print(f"found {len(files)} handoff file(s)")
    if not args.apply:
        print("(dry run; pass --apply to commit)")

    cli = [args.engram_cli]
    file_to_id: dict[str, str] = {}
    skipped: list[tuple[Path, str]] = []

    for path in files:
        print(f"\n{path}")
        try:
            h = parse_handoff(path)
        except Exception as e:
            print(f"  parse failed: {e}")
            skipped.append((path, str(e)))
            continue

        if not any(h.sections.values()):
            print("  empty after parse, skipping")
            skipped.append((path, "empty"))
            continue

        cf_id = None
        if h.continues_from_filename:
            cf_id = file_to_id.get(h.continues_from_filename)
            if not cf_id:
                print(f"  note: continues_from {h.continues_from_filename!r} not yet imported; chain link skipped")

        new_md = render_canonical_md(h)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(new_md)
            tmp_path = Path(tmp.name)

        try:
            new_id = run_create(tmp_path, h.branch, cf_id, cli, args.apply)
        finally:
            tmp_path.unlink(missing_ok=True)

        if new_id:
            file_to_id[path.name] = new_id

    if skipped:
        print(f"\nskipped {len(skipped)} file(s):")
        for p, why in skipped:
            print(f"  {p}: {why}")

    print(f"\ndone: {len(file_to_id)} imported, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
