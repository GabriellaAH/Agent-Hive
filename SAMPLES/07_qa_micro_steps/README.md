# 07 – QA micro-steps (fail decompose)

**Goal:** When worker output **fails** QA JSON checks, an extra LLM pass decides whether one more normal attempt is enough or work must be split into **micro-steps** (at most `HIVE_QA_FAIL_DECOMPOSE_MAX_STEPS`), run sequentially, then QA again.

## Test prompt

```
Produce an OpenAPI 3.1 YAML “skeleton” (paths + one GET component + components.schemas) for simple `/health` and `/users/{id}` endpoints. The answer must be a YAML code block only, preceded by at most 3 sentences of explanation. Acceptance: YAML must be syntactically coherent and include required top-level keys: openapi, info, paths.
```

## Why these settings?

- **`HIVE_QA_FAIL_DECOMPOSE_ENABLED=true`** – enables the micro-step path after QA failure.
- **`HIVE_QA_FAIL_DECOMPOSE_MIN_ATTEMPT=1`** – available from the first QA failure (good for demos; in production consider 2–3).
- **`HIVE_QA_RETRY_TOOL_TRACE`** – inject prior tool trace on retry (default true; listed explicitly here).

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_QA_FAIL_DECOMPOSE_ENABLED` | Micro-step pipeline. |
| `HIVE_QA_FAIL_DECOMPOSE_MAX_STEPS` | Upper bound on micro-steps (clamped 2–24 in code). |
| `HIVE_QA_FAIL_DECOMPOSE_MIN_ATTEMPT` | QA attempt index from which decompose is allowed. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\07_qa_micro_steps\.env.sample').Path
python agent_hive.py "Produce an OpenAPI 3.1 YAML skeleton for simple /health and /users/{id} endpoints..."
```
