# 04 – Pricing comparison, strict ledger

**Goal:** Vendor **prices** and metrics enter the claims ledger only with **verifiable, URL-backed** evidence; merge/synthesis follow the policy; optional `hive_claims_*.json` report.

## Test prompt

```
Compare public pricing on the official websites of three SaaS CI/CD platforms (e.g. GitHub Actions, GitLab CI, CircleCI): free tier, paid tiers, minute bundles / concurrency where applicable. For every numeric claim include the source URL and retrieval date. If something cannot be confirmed officially, label it explicitly as not verifiable.
```

## Why these settings?

- **`HIVE_POST_QA_ASSERTION_LEDGER=true`** – after QA, JSON claims + evidence; merge may use this instead of full prose.
- **`HIVE_EVIDENCE_POLICY=pricing_strict`** – pricing-style claims only verify under strict source types (see AGENT_HIVE.md / claims module).
- **`HIVE_PRICING_REQUIRES_URL=true`** – pricing claims require an `https?://` source.
- **`HIVE_SAVE_CLAIMS_REPORT=true`** – write `hive_claims_*.json` when claim rows exist.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_POST_QA_ASSERTION_LEDGER` | Enables the ledger pipeline. |
| `HIVE_EVIDENCE_POLICY` | `pricing_strict` for pricing research. |
| `HIVE_PRICING_REQUIRES_URL` | URL required for price-type claims. |
| `HIVE_LEDGER_CONFLICT_RESOLUTION` | Resolve cross-task conflicts toward more official sources. |
| `HIVE_SAVE_CLAIMS_REPORT` | Persist claims JSON report. |
| `HIVE_MAX_TOOL_ROUNDS` | More fetches of official pages. |

## Run

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\04_pricing_ledger_strict\.env.sample').Path
python agent_hive.py "Compare public pricing on the official websites of three SaaS CI/CD platforms..."
```
