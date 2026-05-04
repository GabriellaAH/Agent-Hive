# 02 – High parallelism

**Goal:** Problems where the planner emits many **independent** subtasks; use the worker pool and auto CPU scaling.

## Test prompt

```
Give 10 separate, mutually independent ideas for REST API endpoints for a small task-management app (CRUD plus one extra operation). Each idea must be at most 4 sentences: endpoint name, HTTP method, short description, one validation rule. Do not make the ideas depend on each other.
```

## Why these settings?

- **`HIVE_AUTO_PARALLEL_WORKERS=true`** plus higher **`HIVE_MAX_PARALLEL_WORKERS`** / **`_CAP`** – independent tasks can run in parallel across cores.
- **`HIVE_MAX_PLAN_TASKS=20`** – the prompt explicitly asks for 10+ subtasks; raise the planner’s upper bound accordingly.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_MAX_PARALLEL_WORKERS` | Default slot count (here 8). |
| `HIVE_MAX_PARALLEL_WORKERS_CAP` | Upper cap when auto mode scales. |
| `HIVE_AUTO_PARALLEL_WORKERS` | Scale worker count toward CPU. |
| `HIVE_MAX_PLAN_TASKS` | Planner task limit matched to the workload. |
| `HIVE_OUTPUT_DIR` | Output under `hive_outputs/samples/02_high_parallelism`. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\02_high_parallelism\.env.sample').Path
python agent_hive.py "Give 10 separate, mutually independent ideas for REST API endpoints..."
```

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/02_high_parallelism/.env.sample"
python agent_hive.py "Give 10 separate, mutually independent ideas for REST API endpoints..."
```
