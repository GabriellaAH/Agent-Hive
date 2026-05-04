# 01 – Local LM, minimal parallelism

**Goal:** A simple, cohesive task on a local (LM Studio / OpenAI-compatible) server with few parallel workers—lower load and more predictable behavior.

## Test prompt

```
Briefly explain what dynamic programming means in computer science: core idea, one classic example (e.g. Fibonacci or knapsack), and when it is not worth using. At most 15 sentences; bullet lists allowed.
```

## Why these settings?

- **`HIVE_MAX_PARALLEL_WORKERS=2`** – the planner may still split into few dependent tasks; two slots are enough without overloading the machine.
- **`HIVE_AUTO_PARALLEL_WORKERS=false`** – worker count does not scale with CPU; stable, reproducible runs.
- **Lower `HIVE_HTTP_TIMEOUT_SEC`** – faster feedback with small local models; increase for large models (see `.env.example`).

## `.env.sample` variables

| Variable | Value / role |
|----------|----------------|
| `HIVE_BASE_URL` / `HIVE_API_KEY` | Local server (LM Studio defaults). |
| `HIVE_MODEL` | Optional: pin a model in CLI; if unset, first model from `/v1/models`. |
| `HIVE_OUTPUT_DIR` | Writes under `hive_outputs/samples/01_local_lm_minimal`. |

## Run

PowerShell (repo root):

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\01_local_lm_minimal\.env.sample').Path
python agent_hive.py "Briefly explain what dynamic programming means in computer science..."
```

bash:

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/01_local_lm_minimal/.env.sample"
python agent_hive.py "Briefly explain what dynamic programming means in computer science..."
```
