# 03 – Web research (Tavily + tavily-search skill)

**Goal:** Workers can reach the `tavily-search` skill docs/scripts via the **skill router**; `TAVILY_API_KEY` is required for Tavily API calls. Only one skill is enabled so other skills do not clutter routing.

## Test prompt

```
Compare current public pricing and entry models for three hosted vector database services (e.g. Pinecone, Weaviate Cloud, Zilliz Cloud). Use fresh, official sources (vendor sites). Provide a tabular summary and mark wherever information is uncertain.
```

## Why these settings?

- **`HIVE_SKILLS_ENABLED_FILE`** – points at this folder’s `hive_skills_enabled.json`. **Every discovered skill is explicit** `true`/`false`, because missing keys default to **enabled** in `merge_enabled_with_discovery` (`skills_registry.py`).
- **`HIVE_MAX_TOOL_ROUNDS`** – more `http_fetch` / script rounds when many URLs and searches are needed.
- **`TAVILY_API_KEY`** – Tavily-backed search is not meaningful without a real key.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_SKILLS_ENABLED_FILE` | `SAMPLES/03_web_research_tavily/hive_skills_enabled.json` when cwd is repo root. |
| `HIVE_MAX_TOOL_ROUNDS` | More tool calls per worker attempt. |
| `TAVILY_API_KEY` | Real key from [Tavily](https://tavily.com). |
| `HIVE_OUTPUT_DIR` | `hive_outputs/samples/03_web_research_tavily`. |

## Run

```powershell
cd D:\development\python\CS_Dashboard
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\03_web_research_tavily\.env.sample').Path
python agent_hive.py "Compare current public pricing and entry models for three hosted vector database services..."
```

Set `TAVILY_API_KEY` inside `.env.sample` (or export it under the same name before running).
