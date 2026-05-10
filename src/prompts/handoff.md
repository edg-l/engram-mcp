You are writing a session handoff. Gather the state of this session across the seven sections below, then call `handoff_create` with the structured data. Do NOT write the handoff to a file — call the tool directly.

## Section guidance

**summary** (required)
One to three sentences. What was this session about? What is the single most important thing the next session needs to know? Be concrete: mention files changed, features added, or problems solved. Avoid vague phrases like "made progress".

**decisions**
List each architectural or design choice made this session. Format: what was decided AND why. Include trade-offs that were weighed. Omit trivial choices.

**todos**
Unfinished action items that must happen next. Each item should be specific enough that the next session can act on it without additional context. Include file paths or function names where relevant.

**blockers**
Anything that stopped or would stop forward progress. Include: missing information, external dependencies, unresolved errors, or decisions that need user input. If none, omit this section.

**mental_model**
The architectural understanding needed to continue this work: how the relevant subsystems fit together, invariants the code relies on, non-obvious constraints. Write for an agent with no memory of this session. One to five sentences or a short bulleted list.

**next_steps**
The ordered sequence of concrete actions for the next session to take. Item 1 should be the exact first thing to do. Be specific about which file, function, or command.

**notes**
Freeform notes that don't fit elsewhere: environment quirks, partial workarounds, references, or anything a fresh agent would find useful. Optional — omit if empty.

## Sensitive data

Before calling `handoff_create`, scrub: API tokens, passwords, private URLs, customer data, internal hostnames. If unsure whether something is sensitive, omit it.

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
  }
}
```

Omit any section that has nothing real to say. The `summary` field is the only required one.
