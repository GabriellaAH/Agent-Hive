# 05 – Academic / PhD research, ledger + strict evidence

**Goal:** Prioritize literature, papers, DOI / publisher pages; blogs and social feeds **do not** verify academic-style claims under the `academic_strict` preset.

## Test prompt

```
Write a short PhD-style research plan outline (max 800 words) on “federated learning and privacy”: research question, 4–6 key references (prefer DOI or publisher / arXiv URLs), methodological steps, expected contribution. For each major claim note the intended evidence type (e.g. literature inference vs. your own experiment).
```

## Why these settings?

- **`HIVE_EVIDENCE_POLICY=academic_strict`** – stricter source-type rules (e.g. blog/news cannot verify certain academic claim types); see AGENT_HIVE.md.
- **Ledger on** – structured claims + evidence for merge/synthesis.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_POST_QA_ASSERTION_LEDGER` | Claims ledger after QA. |
| `HIVE_EVIDENCE_POLICY` | `academic_strict` for literature-heavy tasks. |
| `HIVE_SAVE_CLAIMS_REPORT` | Optional archival of claims JSON. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\05_academic_ledger\.env.sample').Path
python agent_hive.py "Write a short PhD-style research plan outline on federated learning and privacy..."
```
