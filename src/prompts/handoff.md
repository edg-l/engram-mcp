You are writing a session handoff. Gather the state of this session across the seven sections below, then call `handoff_create` with the structured data. Do NOT write the handoff to a file — call the tool directly.

## Shape: summaries, not transcripts

Each section is a SHORT SUMMARY meant to fit on a screen or two. Hard guidance:
- Keep each section under ~5000 characters; individual list items under ~1000.
- Do NOT paste verbatim tool output, full agent reports, file dumps, or chat logs into sections.
- If long context matters, store it as a separate memory with `memory_store` (type `debug` / `pattern` / `decision`) — it will auto-link and surface in `handoff_resume` as `linked_memories`. Reference it from the section by a one-line description.
- Oversized sections trigger an advisory warning in the response. The handoff is still saved, but treat the warning as a signal to split content out next time.

## Section guidance

**summary** (required)
One to three sentences covering both what the user asked for and what actually happened. State the goal even when it was not reached — a session that ended short of its target still needs the target recorded, or the next session inherits the work without the intent behind it. Be concrete: mention files changed, features added, or problems solved. Avoid vague phrases like "made progress".

**decisions**
Each choice made this session: what was decided AND why, including trade-offs weighed. Record the assumptions you made that nobody specified, not only the deliberate architectural calls — an unstated assumption is what the next agent unknowingly contradicts. Omit choices that follow from a convention already written down.

**blockers**
Things preventing forward motion right now (missing access, failing dependency, unanswered question).

**tried**
Approaches attempted this session and abandoned, each with the concrete reason it failed: "X, because Y". This is the section that pays for itself — rediscovering a dead end costs the next session the same time it cost this one. Record what was ruled out even when the replacement worked.

**mental_model**
The architectural understanding needed to continue this work: how the relevant subsystems fit together, invariants the code relies on, non-obvious constraints. Write for an agent with no memory of this session. One to five sentences or a short bulleted list.

**next_steps**
Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup.

**notes**
Freeform notes that don't fit elsewhere: environment quirks, partial workarounds, references, or anything a fresh agent would find useful. Optional — omit if empty.

## Open work is not a section

There is no `todos` section, and passing one is an error. Outstanding work lives in the durable todo list, which has identity and an explicit close: call `todo_write` to add what a later session should pick up, and to mark `done` or `drop` what this session settled. `handoff_resume` reads that list directly and returns it as `open_todos`.

Before writing the handoff, walk the open todos and reconcile them — anything you finished should be closed here, not carried as prose. A handoff is a snapshot of a session; a todo is state that outlives one, and a snapshot cannot tell "finished" from "forgotten".

**blockers** still work the snapshot way: re-emit every blocker that is still unresolved, whether or not you touched it, since `handoff_resume` returns the newest handoff's blockers verbatim and an omitted one reads as resolved.

**tried** is the opposite of both: a dead end is a permanent fact, not open state. Record it once and leave it; `handoff_search` with `section_filter: ["tried"]` reaches it later.

## Sensitive data

Before calling `handoff_create`, scrub: API tokens, passwords, private URLs, customer data, internal hostnames. If unsure whether something is sensitive, omit it.

## Calling the tool

After gathering all sections, call `handoff_create` with:

```json
{
  "sections": {
    "summary": "...",
    "decisions": ["...", "..."],
    "blockers": ["..."],
    "tried": ["...", "..."],
    "mental_model": "...",
    "next_steps": ["...", "..."],
    "notes": "...",
    "continues_from": "<id of the handoff this session resumed from, if any>"
  }
}
```

Omit any section that has nothing real to say. The `summary` field is the only required one.
