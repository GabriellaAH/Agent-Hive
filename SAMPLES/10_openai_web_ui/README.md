# 10 – OpenAI + Web UI profile

**Goal:** The Flask **web UI** (`python agent_hive.py --web`) lists OpenAI models when **`OPENAI_API_KEY`** is set; each run can pick the **OpenAI** provider and model. The local LM Studio profile still uses `HIVE_BASE_URL` / `HIVE_API_KEY`.

## Test prompt (CLI also works with local LM)

```
Draft a developer-style changelog message (semver, breaking / feature / fix sections) for a fictional “1.2.0” release from three commit summaries.
```

You can paste the same prompt in the web UI; there, choose the **OpenAI** profile and model when the key is valid.

## Why these settings?

- **`OPENAI_API_KEY`** – authenticates web `/api/llm-options` and OpenAI-provider runs.
- **`HIVE_OPENAI_BASE_URL`** – defaults to `https://api.openai.com`; override for another compatible host.
- **`HIVE_WEB_HOST` / `HIVE_WEB_PORT`** – where Flask binds (overridable with `--host` / `--port`).

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `OPENAI_API_KEY` | OpenAI (or compatible) key—**do not** commit real values. |
| `HIVE_OPENAI_BASE_URL` | API root (client appends `/v1` if needed). |
| `HIVE_WEB_HOST`, `HIVE_WEB_PORT` | Web UI bind address. |
| `HIVE_BASE_URL` | LM Studio profile for CLI and the web “LM Studio (local)” option. |

## Run – Web UI

```powershell
cd D:\development\python\CS_Dashboard
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\10_openai_web_ui\.env.sample').Path
python agent_hive.py --web
```

Then open the browser at `http://127.0.0.1:5000` (or your host/port). Under **New run**, pick an OpenAI model if the key is valid.

## Run – CLI (one-off)

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\10_openai_web_ui\.env.sample').Path
python agent_hive.py "Draft a developer-style changelog message for a fictional 1.2.0 release..."
```

By default the CLI calls the server at **`HIVE_BASE_URL`** (here LM Studio). Driving OpenAI from the CLI instead is typically done via client overrides or a web run—see [AGENT_HIVE.md](../../AGENT_HIVE.md).
