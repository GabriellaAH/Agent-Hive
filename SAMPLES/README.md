# Agent Hive – sample use cases

This directory contains **15 standalone scenarios**. Each subfolder includes:

- **`.env.sample`** – environment variables tuned to that scenario (minimal set plus typical LM Studio settings).
- **`README.md`** – test prompt, rationale, and run examples.

Full variable reference: [`.env.example`](../.env.example) and [AGENT_HIVE.md](../AGENT_HIVE.md).

## How to run

1. Activate your virtual environment and `cd` to the **repository root** (`CS_Dashboard`).
2. Set **`DOTENV_PATH`** to the **absolute or relative path** of that sample’s `.env.sample`. Agent Hive then loads **only that file** (the root `.env` is not merged unless you copy values there).
3. Run: `python agent_hive.py "…test prompt from the README…"`

### PowerShell (Windows)

```powershell
cd D:\development\python\CS_Dashboard
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\01_local_lm_minimal\.env.sample').Path
python agent_hive.py "Paste the test prompt text from README.md here."
```

### bash / Git Bash

```bash
cd /path/to/CS_Dashboard
export DOTENV_PATH="$(pwd)/SAMPLES/01_local_lm_minimal/.env.sample"
python agent_hive.py "Paste the test prompt text from README.md here."
```

**Note:** `HIVE_SKILLS_ENABLED_FILE` is resolved relative to the **process working directory**, so always run from the repo root as shown in each sample `README.md`.

## Samples overview

| Folder | One-line summary |
|--------|------------------|
| [01_local_lm_minimal](01_local_lm_minimal/README.md) | Fewer parallel workers; simple local LM task |
| [02_high_parallelism](02_high_parallelism/README.md) | Many independent subtasks – auto scaling / more slots |
| [03_web_research_tavily](03_web_research_tavily/README.md) | Tavily + `tavily-search` skill only; more tool rounds |
| [04_pricing_ledger_strict](04_pricing_ledger_strict/README.md) | Claims ledger; strict pricing; URL requirement |
| [05_academic_ledger](05_academic_ledger/README.md) | Ledger + `academic_strict` evidence |
| [06_integration_qa_strict](06_integration_qa_strict/README.md) | Integration QA strict mode |
| [07_qa_micro_steps](07_qa_micro_steps/README.md) | Micro-step decomposition after QA failure |
| [08_large_merge_hierarchical](08_large_merge_hierarchical/README.md) | Hierarchical merge for large combined text |
| [09_mandatory_deliverables](09_mandatory_deliverables/README.md) | Mandatory deliverables from env |
| [10_openai_web_ui](10_openai_web_ui/README.md) | OpenAI key + web UI / hosted profile |
| [11_checkpoint_replan_exploratory](11_checkpoint_replan_exploratory/README.md) | Higher checkpoint replan budget for exploratory / research pivots |
| [12_checkpoint_replan_adaptive_design](12_checkpoint_replan_adaptive_design/README.md) | Phased design with checkpoints + optional final whole-run replan |
| [13_outer_replan_carryover_large_digest](13_outer_replan_carryover_large_digest/README.md) | Final-check outer replan: large `LATEST.md` carryover into planner/workers |
| [14_outer_replan_shared_workspace](14_outer_replan_shared_workspace/README.md) | Outer replan with one shared workspace across cycles (artifacts survive) |
| [15_user_knowledge_base](15_user_knowledge_base/README.md) | `HIVE_KB_DIR` + `kb_list` / `kb_read`; bundled `kb_demo` reference files |

Outputs go under `hive_outputs/samples/<folder_name>/` by default so they stay separate from other runs.
