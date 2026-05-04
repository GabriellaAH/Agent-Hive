# 12 – Adaptive phased design (checkpoints + optional full replan)

**Goal:** **Systems or product design** where later phases (API shape, data model, rollout) should **depend on frozen decisions** from earlier phases. Use **`key_task`** milestones after requirements synthesis and after architecture choice so the checkpoint model can **`replan`** remaining implementation-style tasks if stakeholders would implicitly contradict earlier assumptions—or if a simpler architecture makes whole task groups obsolete.

## Why this is useful

- **Dependency on decisions, not just task order** – classical `depends_on` encodes “B after A”, but not “if A’s conclusion is X then do B1 else B2”. Checkpoints approximate that by **rewriting the pending task graph** after milestone outputs exist.
- **Multiple pivots** – real design conversations rarely get the plan right once; `HIVE_MAX_CHECKPOINT_REPLANS` here is **5** so two or three meaningful replans remain realistic without immediately hitting the cap.
- **`HIVE_REPLAN_ATTEMPTS`** (final consistency pass) is nudged up slightly: that is **different** from checkpoint replan—it is when the **final check** demands a **whole-run** replan; both knobs together suit “messy but important” deliverables.

## Test prompt

```
Produce a phased delivery plan for a small internal “feature flag + gradual rollout” service (not full SaaS): requirements and invariants, threat model sketch, API and storage sketch, migration plan from a single JSON config file today, observability, and a rollback story.

Planning requirements: include explicit key_task checkpoint tasks (a) immediately after the requirements + invariants phase is done, and (b) after the architecture/API sketch phase is done, so remaining phases can be replanned if the checkpoint reviewer chooses replan. Later phases must not assume details that those checkpoints have not effectively “locked”. Keep the plan within reasonable task count; batch where possible.
```

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_MAX_CHECKPOINT_REPLANS` | Budget for **checkpoint-triggered** replans of pending tasks. |
| `HIVE_REPLAN_ATTEMPTS` | How many **final-check** whole-run replans are allowed (orthogonal to checkpoints). |
| `HIVE_CHECKPOINT_TASK_OUTPUT_CHARS` | Context window for checkpoint decisions. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\12_checkpoint_replan_adaptive_design\.env.sample').Path
python agent_hive.py "Produce a phased delivery plan for a small internal feature flag service..."
```

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/12_checkpoint_replan_adaptive_design/.env.sample"
python agent_hive.py "Produce a phased delivery plan for a small internal feature flag service..."
```
