# 11 – Exploratory work with checkpoint replanning

**Goal:** Problems where **early findings can invalidate the rest of the plan** (research, competitive intel, “figure out what’s true first”). Tasks marked `key_task: true` run a **checkpoint** model after completion; it may choose **`replan`** and replace all **not-yet-started** tasks with a fresh list while keeping finished work. This sample **raises the checkpoint replan budget** so several such pivots are allowed in one run.

## Why this is useful

- **Open-ended discovery** – you cannot reliably pre-specify every downstream step until you know what the landscape looks like.
- **Less wasted work** – without replanning, workers might execute a long tail of tasks based on assumptions the first research phase already disproved.
- **Controlled adaptation** – replans still go through the checkpoint JSON contract (continue / finish_early / replan) and stay within `HIVE_MAX_CHECKPOINT_REPLANS`; the web UI can also reflect DAG updates after checkpoint replans.

The planner must actually emit **`key_task: true`** on sensible milestone tasks; the user prompt below nudges that behavior.

## Test prompt

```
You are planning a multi-phase investigation (not a single essay). Goal: compare three plausible technical approaches for “edge ML deployment” for a mid-size IoT product (constraints: 512MB RAM, intermittent connectivity, OTA updates). Phases: (1) frame evaluation criteria and unknowns, (2) short desk research per approach with sources, (3) comparative scoring, (4) final recommendation and risks.

Important for the execution plan: after each major research phase completes, progress must be reviewed before heavy later work. Mark appropriate milestone tasks as key_task checkpoints so the orchestrator can replan remaining tasks if early findings change which approaches deserve deep analysis or if new sub-questions appear. Prefer fewer parallel branches until criteria are stable.
```

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_MAX_CHECKPOINT_REPLANS` | Upper bound on how many times a checkpoint may **`replan`** pending work (here **6** vs a tighter default). |
| `HIVE_MAX_PLAN_TASKS` / `HIVE_MAX_PLAN_DEPTH_HINT` | Headroom for phased plans plus replan-injected tasks. |
| `HIVE_CHECKPOINT_TASK_OUTPUT_CHARS` | How much of each completed task the checkpoint model sees when deciding. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\11_checkpoint_replan_exploratory\.env.sample').Path
python agent_hive.py "You are planning a multi-phase investigation..."
```

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/11_checkpoint_replan_exploratory/.env.sample"
python agent_hive.py "You are planning a multi-phase investigation..."
```

See [AGENT_HIVE.md](../../AGENT_HIVE.md) (pipeline / checkpoints) and [`.env.example`](../../.env.example) for `HIVE_MAX_CHECKPOINT_REPLANS` and related keys.
