# API constraints (KB sample)

## Read endpoints

- `GET /events` is paginated; **max page size 500**.
- Default sort is `received_at desc`.

## Write endpoints

- `POST /events/bulk` accepts at most **10_000** events per request body.
- Idempotency: clients SHOULD send `Idempotency-Key` header; server dedupes within **24 hours**.

## Rate limits (staging)

- Authenticated clients: **120 requests/minute** per API key (burst up to 200).

These values are authoritative for any task that designs client libraries or load tests against staging.
