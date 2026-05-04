# 13 – Outer replan carryover with a large knowledge digest

**Goal:** When the **final consistency check** returns `requires_replan: true`, Agent Hive may start a **new outer cycle** (fresh plan, new workers). If **`HIVE_REPLAN_CARRYOVER_ENABLED`** is on, the previous cycle writes a markdown snapshot plus **`LATEST.md`** under `HIVE_OUTPUT_DIR/run_carryover/<session>/knowledge/`. The next cycle’s planner (and workers) receive a **`[Prior run knowledge base]`** block built from `LATEST.md`, clamped to **`HIVE_REPLAN_KNOWLEDGE_MAX_CHARS`** (head + tail if oversized).

This sample sets a **very large** `HIVE_REPLAN_KNOWLEDGE_MAX_CHARS` and **wider** carryover file excerpt limits so hard, multi-task runs still leave a **usable trace** for the replanned attempt—without pretending the model saw every token of every task output.

## Why tune these knobs (advanced runs)

- **Expensive wrong plans** – on gnarly specs (formal consistency, multi-tenant isolation, quantitative SLOs), the first full attempt may be internally inconsistent while still containing **valuable partial analysis**. Carryover avoids “tabula rasa” amnesia on the next outer cycle.
- **`HIVE_REPLAN_KNOWLEDGE_MAX_CHARS`** – trades **context** vs **cost/latency**. Raising it is for problems where the verdict + merged excerpt + per-task snippets must **coexist** in the injected digest so the replanner can surgically fix gaps instead of re-deriving everything.
- **`HIVE_REPLAN_CARRYOVER_TASK_MAX_CHARS` / `HIVE_REPLAN_CARRYOVER_MERGED_MAX_CHARS`** – control how much raw material is **stored into** `attempt_*.md` / `LATEST.md` **before** the knowledge prompt cap; larger values help when individual tasks produced long structured artifacts.

`HIVE_REPLAN_SHARE_WORKSPACE` stays **false** here so this sample isolates **knowledge carryover** from workspace reuse (see sample **14**).

## Test prompt (very advanced)

```
You are producing a single coherent technical specification (not code) for a hypothetical “multi-region, active-active” transactional service with RPO=0 for user-visible writes and strict serializability per user id.

Deliverables inside the answer:
1) Explicit consistency model (informal but precise): what each user may observe after their own writes vs others’ writes; what happens during regional partition.
2) A concrete data-placement and replication protocol sketch (primary/secondary or quorum-based—your choice) with failure modes.
3) Conflict handling: same user id concurrent writes from two regions—pick a resolution policy and argue it.
4) Migration story from today’s single-region deployment: phases, dual-write risks, rollback.
5) Observability: SLOs, SLIs, and what to alert on.

Hard constraint for the orchestration: the final narrative must not contradict itself on RPO/serializability claims. If you discover while working that RPO=0 + strict per-user serializability + naive active-active is in tension, document the tension honestly and adjust the architecture rather than hand-waving.

Expect that a strict final reviewer may demand a full replan; use prior-attempt knowledge if provided to improve the next version rather than repeating the same mistake.
```

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_REPLAN_ATTEMPTS` | More **outer** replan cycles allowed after `requires_replan`. |
| `HIVE_REPLAN_CARRYOVER_ENABLED` | Write and re-inject carryover snapshots. |
| `HIVE_REPLAN_KNOWLEDGE_MAX_CHARS` | Max size of `LATEST.md` text injected into the next cycle’s prompts. |
| `HIVE_REPLAN_CARRYOVER_*` | Larger excerpts persisted into carryover files. |
| `HIVE_REPLAN_SHARE_WORKSPACE` | `false` — workspace is still per outer attempt (timestamped under `HIVE_WORKSPACE_PARENT_DIR`). |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\13_outer_replan_carryover_large_digest\.env.sample').Path
python agent_hive.py "You are producing a single coherent technical specification..."
```

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/13_outer_replan_carryover_large_digest/.env.sample"
python agent_hive.py "You are producing a single coherent technical specification..."
```

See [AGENT_HIVE.md](../../AGENT_HIVE.md) (environment table) and [`.env.example`](../../.env.example) for defaults and related keys.
