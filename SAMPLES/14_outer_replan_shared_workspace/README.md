# 14 – Outer replan with shared workspace

**Goal:** Same **outer replan** story as sample 13, but with **`HIVE_REPLAN_SHARE_WORKSPACE=true`**. Then `run_hive` keeps a **single workspace directory** under `run_carryover/<session>/workspace` for **every** outer attempt in that session, instead of a new timestamped folder per cycle.

That matters when workers **materialize heavy artifacts** (downloaded corpora, generated CSV/JSON, intermediate benchmarks, multi-step build outputs) that you do **not** want to re-fetch or recompute after a final-check `requires_replan`.

## Why this is useful (advanced)

- **Tool-heavy pipelines** – research or evaluation tasks that `http_fetch` large inputs or `run_script` produce multi-megabyte local files: a naive replan would point workers at an **empty** new workspace; sharing preserves **paths and caches** across outer cycles.
- **Still combine with carryover** – `HIVE_REPLAN_CARRYOVER_ENABLED` remains on so planners/workers also see the **verdict + excerpts** in `[Prior run knowledge base]`; files + narrative context complement each other.
- **Tradeoff** – stale or contradictory files from a “bad” attempt remain visible; the prompt and final check should instruct workers to **re-validate** artifacts after replan, not blindly trust them.

`HIVE_REPLAN_KNOWLEDGE_MAX_CHARS` here is **moderate** (52k) to reflect a typical balance when disk artifacts carry much of the state.

## Test prompt (very advanced)

```
Design and partially implement (in the run workspace only) a reproducible evaluation harness for comparing two approximate nearest neighbor (ANN) libraries on the same synthetic dataset.

Requirements:
1) Create a small Python script (or scripts) that: generates or downloads a fixed-size synthetic embedding dataset (specify dimension N and point count M), builds indexes for library A and library B, runs batch queries with fixed random query vectors, and writes results to CSV/JSON under the workspace (paths documented in the answer text).
2) Report build time, query latency (p50/p95), peak RSS if obtainable without privileged OS APIs, and recall@K against exact brute-force on a held-out query subset (define K).
3) Document failure modes: bad builds, OOM, non-deterministic threading—what you would log and how you would bisect.

Important: if a strict final reviewer forces a full replan, subsequent attempts must reuse existing workspace files when still valid (avoid redundant multi-GB downloads); only regenerate what the verdict invalidated. Explain in prose which files are stable vs must be rebuilt.
```

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_REPLAN_SHARE_WORKSPACE` | **`true`** — reuse `run_carryover/<session>/workspace` across outer replans. |
| `HIVE_REPLAN_CARRYOVER_ENABLED` | Keep markdown/json carryover for planner/worker injection. |
| `HIVE_REPLAN_KNOWLEDGE_MAX_CHARS` | Caps injected `LATEST.md` digest size. |
| `HIVE_MAX_TOOL_ROUNDS` / `HIVE_TOOL_TIMEOUT_SEC` | Headroom for scripts and fetches. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\14_outer_replan_shared_workspace\.env.sample').Path
python agent_hive.py "Design and partially implement a reproducible evaluation harness..."
```

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/14_outer_replan_shared_workspace/.env.sample"
python agent_hive.py "Design and partially implement a reproducible evaluation harness..."
```

See [AGENT_HIVE.md](../../AGENT_HIVE.md) and [`.env.example`](../../.env.example).
