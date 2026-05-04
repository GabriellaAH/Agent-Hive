# Internal product brief (fictional sample for Agent Hive KB)

## Codename: Northline

Northline is an internal metrics pipeline (not customer-facing). SLA targets:

- Ingest latency p95: **under 90 seconds** from event receipt to queryable row.
- Daily batch reconciliation window: **02:00–04:00 UTC**; no user-facing downtime required during batch.
- Data retention: raw events **90 days**; aggregated rollups **24 months**.

## Out of scope (v1)

- Real-time alerting on per-row anomalies (deferred to v2).
- Cross-region replication (single region only).

## Glossary link

The term **RPU** (rollup processing unit) is defined in `glossary.md`. Do not redefine RPU in downstream docs; reference the glossary.
