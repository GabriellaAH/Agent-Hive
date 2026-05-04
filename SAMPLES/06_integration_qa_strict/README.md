# 06 – Integration QA strict mode

**Goal:** For problems with several **dependent** tasks, an **integration QA** failure along the critical chain **fails the run** (`HIVE_INTEGRATION_QA_STRICT=true`) instead of only logging.

## Test prompt

```
Design and document a three-layer “mini pipeline”: (1) input validation, (2) transformation with business rules, (3) output schema and error handling. Each layer must have a precise interface to the next. The final answer must include: a summary table (layer, responsibility, input, output), one worked example through all three layers, then a consistency check: where could contradictions appear?
```

## Why these settings?

- **`HIVE_INTEGRATION_QA_STRICT=true`** – integration QA failure fails the run (see `.env.example` / AGENT_HIVE.md).
- **`HIVE_INTEGRATION_QA_ENABLED=true`** – explicit on (default is already true; included for clarity).

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_INTEGRATION_QA_STRICT` | Strict integration QA. |
| `HIVE_MAX_PLAN_TASKS` | Multi-step but bounded plan size. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\06_integration_qa_strict\.env.sample').Path
python agent_hive.py "Design and document a three-layer mini pipeline..."
```
