# CLI

`engram-cli --help` and `engram-cli <command> --help` list every flag. Two global flags apply to any command: `-p/--project <id>` runs it against another project (short names resolve, as for the MCP `project` argument), and `--current-branch <name>` overrides the detected branch, for a shell that sits in a different checkout than the work.

## Handoffs

```bash
engram-cli handoff create                         # prompts for each section
engram-cli handoff create --non-interactive \
  --summary "Cut the 2.0 branch" --tried "bulk UPDATE, locked the table"
engram-cli handoff create --from-file session.md  # pre-written markdown
engram-cli handoff resume --branch feat/x
engram-cli handoff search "auth refactor" --section blockers,tried
```

`handoff create` prompts only when stdin is a terminal and `--non-interactive` is absent. Otherwise a missing optional section stays empty and a missing `--summary` is an error, so scripts never hang on a prompt.

## Todos

```bash
engram-cli todo add "Tag the 2.0 release" --detail "CI must be green on the release branch first"
engram-cli todo list                     # titles; --full for the whole text
engram-cli todo done mem_abc123
engram-cli todo drop mem_abc123 --reason "superseded by the 2.1 plan"
engram-cli todo reopen mem_abc123
engram-cli todo edit mem_abc123 "New title"
```

## Search

```bash
engram-cli query "how does authentication work"
engram-cli context "working on the auth refactor"
```

## Memories

```bash
engram-cli store "The API uses rate limiting" -t fact --tags api,security
engram-cli store "Always use snake_case" -t preference --pinned --global
engram-cli store "Bench results show BM25 wins" -t fact --artifact benchmarks/longmemeval/RESULTS.md
engram-cli store "Retries now back off exponentially" --supersedes mem_abc123
engram-cli show mem_abc123
engram-cli list --status superseded      # live | superseded | dead | all
engram-cli list --order created          # relevance | created | updated | accessed
engram-cli update mem_abc123 -c "Updated content" --importance 0.9
engram-cli update mem_abc123 --sections-json '{"blockers": []}'   # patch a handoff
engram-cli update mem_abc123 --dead --dead-reason "service retired"
engram-cli pin mem_abc123
engram-cli link mem_abc123 mem_def456 -r relates_to
engram-cli delete mem_abc123
engram-cli trash
engram-cli restore mem_abc123
```

## ADRs

```bash
engram-cli adr create
engram-cli adr list --status accepted
engram-cli adr show 3
engram-cli adr update-status 3 deprecated
engram-cli adr export --write            # docs/adr/NNNN-slug.md
```

## Projects, export and sync

```bash
engram-cli projects                                          # every project in the store
engram-cli projects merge ~/dev/old git:github.com/me/new    # dry run: what would move
engram-cli projects merge ~/dev/old git:github.com/me/new --confirm
engram-cli export -o backup.json
engram-cli export --all-projects --embeddings -o full.json
engram-cli import backup.json                                # merge; --mode replace to replace
engram-cli import -                                          # from stdin
engram-cli sync user@host --dry-run
engram-cli sync user@host
```

## Maintenance

```bash
engram-cli stats
engram-cli decay                 # relevance, auto-prune and trash sweep
engram-cli prune -t 0.2 --confirm
engram-cli dedup -t 0.90         # add --confirm to merge
engram-cli wipe --confirm        # delete every memory of the project
engram-cli insights
engram-cli health
engram-cli hooks install         # see hooks/README.md
```

Commands that change data in bulk (`prune`, `dedup`, `wipe`, `projects merge`) only report what they would do until `--confirm` is passed.

## JSON output

`--json` makes a read command print one JSON document instead of text. It works on `query`, `context`, `stats`, `projects`, `list`, `show`, `handoff resume`/`search`/`show` and `adr list`/`show`. An empty result is still JSON (`count: 0`), and any other command rejects the flag with exit code 2 rather than printing text a script would misparse.

```bash
engram-cli --json query "how does auth work" | jq '.memories[].memory.id'
engram-cli --json handoff resume | jq '.open_todos'
```

| Command | Shape |
|---|---|
| `query`, `context` | `{project, query\|context, count, memories: [{memory, score…}]}` |
| `list` | `{project, count, memories: [memory]}` |
| `show` | `{memory, relationships: {outgoing, incoming}}` |
| `stats` | `{project, memory_count, cluster_count, dead_count, trash_count, …}` |
| `projects` | `{current_project, count, projects: [{id, memory_count, handoff_count, adr_count, latest_activity_at, current}]}` |
| `handoff resume` | `{branch, latest_handoff_id, chain, open_todos, open_todo_count, stale_todo_count, stale_todo_ids, open_blockers, top_sections, linked_memories, message}` |
| `handoff search` | `{query, count, matches: [{handoff_id, section_name, score, section_text}]}` |
| `handoff show` | `{memory, sections}` |
| `adr list` | `{project, count, adrs: [{number, status, title, id}]}` |
| `adr show` | `{id, number, status, title, context, decision, consequences}` |

A `memory` object has the same shape as in an export.
