# 15 – User knowledge base (`HIVE_KB_DIR`)

**Goal:** Point Agent Hive at a **folder of your own reference files**. The orchestrator builds a **capped text index** from allowed extensions and injects it into the **planner** and **worker / skill-router / micro-step** prompts as **`[User knowledge base]`**. Workers can load full (bounded) file contents with host tools **`kb_list`** and **`kb_read`** (path-sandboxed under the same root; extension allowlist applies).

This sample includes a tiny **`kb_demo/`** tree (markdown only) so you can run immediately from the repo root with `DOTENV_PATH`—no extra setup.

## Why use it

- **Grounding in your docs** — SLAs, glossaries, internal APIs, policy snippets stay on disk; the model sees an index first, then can `kb_read` the exact sections it needs.
- **Separation from workspace** — unlike `run_workspace_python` artifacts, the KB is **read-mostly** reference material you control; it is not overwritten by task scripts unless you point `HIVE_KB_DIR` at a writable folder (not recommended for production secrets).
- **Web UI** — you can also pass **`kb_dir` per run** (overrides / supplements env); see [AGENT_HIVE.md](../../AGENT_HIVE.md) “How to run” / knowledge base bullet.

## `.env.sample` variables

| Variable | Role |
|----------|------|
| `HIVE_KB_DIR` | Root directory scanned for indexing and for `kb_list` / `kb_read`. |
| `HIVE_KB_INDEX_MAX_CHARS` | Max total size of the injected index digest (head+tail if over cap). |
| `HIVE_KB_INDEX_PER_FILE_HEAD_CHARS` | Per-file excerpt length inside the digest. |
| `HIVE_KB_READ_MAX_CHARS` | Upper bound for `kb_read` responses per call. |
| `HIVE_KB_FILE_EXTENSIONS` | Allowlist of extensions under the KB root. |
| `HIVE_KB_MAX_FILES` | Cap on files scanned (DoS guard). |
| `HIVE_MAX_TOOL_ROUNDS` | Room for multiple `kb_list` / `kb_read` rounds. |

## Test prompt

```
Using ONLY the files under the configured user knowledge base (use kb_list and kb_read as needed), produce a one-page “engineering handoff” memo for a new hire that covers:

1) What Northline is and the three SLA numbers from the product brief (with exact values).
2) What RPU means and where to look for terminology vs SLA conflicts (per glossary rules).
3) Concrete limits for client implementers: max page size, max bulk batch size, staging rate limit — cite the source filenames.

If something is missing from the KB files, say explicitly what is missing rather than inventing it.
```

## Run

From repository root (so `SAMPLES/15_user_knowledge_base/kb_demo` resolves):

```powershell
$env:DOTENV_PATH = (Resolve-Path '.\SAMPLES\15_user_knowledge_base\.env.sample').Path
python agent_hive.py "Using ONLY the files under the configured user knowledge base..."
```

```bash
export DOTENV_PATH="$(pwd)/SAMPLES/15_user_knowledge_base/.env.sample"
python agent_hive.py "Using ONLY the files under the configured user knowledge base..."
```

**Web UI:** set **Knowledge base directory** to the absolute path of `SAMPLES/15_user_knowledge_base/kb_demo`, or rely on `kb_dir_default` from `.env` when `HIVE_KB_DIR` is set globally.

See [AGENT_HIVE.md](../../AGENT_HIVE.md) (final-check / user knowledge base) and [`.env.example`](../../.env.example) (`HIVE_KB_*`).
