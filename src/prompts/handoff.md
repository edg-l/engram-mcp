You are writing a session handoff. Gather the state of this session across the seven sections below, then call `handoff_create` with the structured data. Do NOT write the handoff to a file — call the tool directly.

## Shape: summaries, not transcripts

Each section is a SHORT SUMMARY meant to fit on a screen or two. Hard guidance:
- Keep each section under ~5000 characters; individual list items under ~1000.
- Do NOT paste verbatim tool output, full agent reports, file dumps, or chat logs into sections.
- If long context matters, store it as a separate memory with `memory_store` (type `debug` / `pattern` / `decision`) — it will auto-link and surface in `handoff_resume` as `linked_memories`. Reference it from the section by a one-line description.
- Oversized sections trigger an advisory warning in the response. The handoff is still saved, but treat the warning as a signal to split content out next time.

## Section guidance

**summary** (required)
One to three sentences. What was this session about? What is the single most important thing the next session needs to know? Be concrete: mention files changed, features added, or problems solved. Avoid vague phrases like "made progress".

**decisions**
List each architectural or design choice made this session. Format: what was decided AND why. Include trade-offs that were weighed. Omit trivial choices.

**todos**
Within-session work the next agent should pick up immediately. Concrete, ready-to-execute items.

**blockers**
Things preventing forward motion right now (missing access, failing dependency, unanswered question).

**mental_model**
The architectural understanding needed to continue this work: how the relevant subsystems fit together, invariants the code relies on, non-obvious constraints. Write for an agent with no memory of this session. One to five sentences or a short bulleted list.

**next_steps**
Post-session follow-ups beyond the current thread. Future-facing, not for immediate pickup.

**notes**
Freeform notes that don't fit elsewhere: environment quirks, partial workarounds, references, or anything a fresh agent would find useful. Optional — omit if empty.

## Sensitive data

Before calling `handoff_create`, scrub: API tokens, passwords, private URLs, customer data, internal hostnames. If unsure whether something is sensitive, omit it.

## Topic scoping for parallel work

Handoffs are keyed by (project, branch). If this session is independent/parallel work on a branch that may already have other handoffs (a second task in the same checkout, work that isn't a continuation of whatever `handoff_resume` last returned), pass a top-level `topic` (a short slug, e.g. `"auth-refactor"`) so a later `handoff_resume --topic ...` can find this thread specifically instead of just the branch's latest handoff. Ask the user for a topic name if it isn't obvious, or infer one from the task and state it so they can correct you. Skip `topic` for normal single-threaded work.

## Calling the tool

After gathering all sections, call `handoff_create` with:

```json
{
  "sections": {
    "summary": "...",
    "decisions": ["...", "..."],
    "todos": ["...", "..."],
    "blockers": ["..."],
    "mental_model": "...",
    "next_steps": ["...", "..."],
    "notes": "...",
    "continues_from": "<id of the handoff this session resumed from, if any>"
  },
  "topic": "<short topic slug, only for independent/parallel work on this branch>"
}
```

Omit any section that has nothing real to say. The `summary` field is the only required one. Omit `topic` entirely for normal single-threaded work. If `continues_from` is set and points at a handoff with a different topic, the tool returns a non-blocking warning — the handoff is still saved.
