# 08 – Large merge, hierarchical strategy

**Goal:** Many subtasks with **long** text outputs are merged **hierarchically** / in layers when combined input exceeds the threshold; final synthesis polishes a single answer.

## Test prompt

```
Produce a “market-style” background report in 6 chapters (each chapter at least 12 sentences): (1) market definition, (2) players, (3) trends, (4) risks, (5) regulation, (6) conclusions. Each chapter must read standalone; end with a short executive summary. For the planner: treat each chapter as its own task, then merge.
```

## Why these settings?

- **`HIVE_MERGE_STRATEGY=hierarchical`** – layered summarization for large merge inputs (AGENT_HIVE.md).
- **`HIVE_MERGER_INPUT_THRESHOLD_CHARS`** – tuned so “large” input is detected earlier (here 50000; adjust as needed).
- **`HIVE_MERGE_COMPRESS_MAX_CHUNKS`** – more chunks in the compression phase.
- **`HIVE_FINAL_SYNTHESIS_ENABLED=true`** – explicit final polish pass.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_MERGE_STRATEGY` | `hierarchical` for large combined merges. |
| `HIVE_MERGER_INPUT_THRESHOLD_CHARS` | Merge / compress heuristic threshold. |
| `HIVE_MERGE_COMPRESS_MAX_CHUNKS` | Compression chunk budget. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\08_large_merge_hierarchical\.env.sample').Path
python agent_hive.py "Produce a market-style background report in 6 chapters..."
```
