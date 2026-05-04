# 09 – Mandatory deliverables (from env)

**Goal:** `HIVE_REQUIRED_DELIVERABLES` is injected into every planner pass and into the **final check** deliverable coverage logic—forcing predefined sections or phrases.

## Test prompt

```
Design a simple observability stack for a small team (metrics, logs, traces): components, data flow, and a recommended open-source toolchain. Do not mandate a specific commercial product; justify choices.
```

## Why these settings?

- **`HIVE_REQUIRED_DELIVERABLES`** – `parse_deliverables_from_env`: if the value contains `|` and no newlines, it splits on **pipes** into a list. Here three mandatory blocks: executive summary, Mermaid diagram, risks table.
- **`HIVE_FINAL_CHECK_ATTEMPTS`** – more chances for the final reviewer if deliverable coverage is incomplete.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_REQUIRED_DELIVERABLES` | Pipe-separated mandatory items (one line in `.env.sample`). |
| `HIVE_FINAL_CHECK_ATTEMPTS` | Number of final-check passes. |

Multi-line deliverable lists are also supported (one item per line); see `.env.example` / `hive_env._env_raw_preserve`.

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\09_mandatory_deliverables\.env.sample').Path
python agent_hive.py "Design a simple observability stack for a small team..."
```

The answer must include all three deliverables (summary, Mermaid, risks table); the final check looks for them.
