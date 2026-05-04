"""Flask web UI for agent hive: submit problem, poll status."""
from __future__ import annotations

import threading
from typing import Any

import json as json_lib
import time

from flask import Flask, Response, jsonify, request, stream_with_context

from agent_hive import HiveConfig, RunState, run_hive
from hive_env import get_hive_env
from lm_client import filter_openai_chat_completion_model_ids, list_model_ids, make_openai_client
from skills_registry import (
    default_skills_dir,
    discover_skills,
    load_enabled_map,
    merge_enabled_with_discovery,
    save_enabled_map,
)

_run_lock = threading.Lock()
_busy = False
_bg_thread: threading.Thread | None = None
_shared_state = RunState()


def run_web(host: str | None = None, port: int | None = None) -> None:
    we = get_hive_env()
    if host is None:
        host = we.web_host
    if port is None:
        port = we.web_port
    app = create_app()
    app.run(host=host, port=port, threaded=True)


def _skills_snapshot() -> dict[str, Any]:
    root = default_skills_dir()
    discovered = discover_skills(root)
    stored = load_enabled_map()
    merged = merge_enabled_with_discovery(discovered, stored)
    skills: list[dict[str, Any]] = []
    for s in discovered:
        skills.append(
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "has_scripts": s.has_scripts(),
                "enabled": merged[s.id],
            }
        )
    return {"skills": skills}


def _current_enabled_skill_ids() -> set[str]:
    root = default_skills_dir()
    discovered = discover_skills(root)
    stored = load_enabled_map()
    merged = merge_enabled_with_discovery(discovered, stored)
    return {k for k, v in merged.items() if v}


def _list_models_at(base_url: str, api_key: str) -> dict[str, Any]:
    try:
        client = make_openai_client(base_url, api_key)
        ids = list_model_ids(client)
        return {"ok": True, "models": ids, "error": None}
    except Exception as exc:
        return {"ok": False, "models": [], "error": str(exc)}


def _hive_config_for_run(enabled_ids: set[str], data: dict[str, Any]) -> HiveConfig:
    kwargs: dict[str, Any] = {"enabled_skill_ids": enabled_ids}
    raw = data.get("max_parallel_workers")
    if raw is not None:
        try:
            kwargs["max_parallel_workers"] = max(1, int(raw))
        except (TypeError, ValueError):
            pass
    provider = (data.get("llm_provider") or "local").strip().lower()
    model = (data.get("model") or "").strip()
    if model:
        kwargs["model"] = model
    if provider == "openai":
        e = get_hive_env()
        kwargs["base_url"] = e.openai_base_url
        kwargs["api_key"] = e.openai_api_key or ""
    kb_dir = (data.get("kb_dir") or "").strip()
    if kb_dir:
        kwargs["kb_dir"] = kb_dir
    return HiveConfig(**kwargs)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/skills")
    def api_skills_get() -> Any:
        return jsonify(_skills_snapshot())

    @app.put("/api/skills")
    def api_skills_put() -> Any:
        data = request.get_json(silent=True) or {}
        enabled = data.get("enabled")
        if not isinstance(enabled, dict):
            return jsonify({"ok": False, "error": "Missing enabled object"}), 400
        root = default_skills_dir()
        discovered = discover_skills(root)
        valid = {s.id for s in discovered}
        current = merge_enabled_with_discovery(discovered, load_enabled_map())
        for k, v in enabled.items():
            if k in valid and isinstance(v, bool):
                current[k] = v
        save_enabled_map(current)
        return jsonify({"ok": True})

    @app.get("/api/status")
    def api_status() -> Any:
        with _run_lock:
            snap = _shared_state.snapshot()
            snap["busy"] = _busy
        return jsonify(snap)

    @app.get("/api/llm-options")
    def api_llm_options() -> Any:
        e = get_hive_env()
        local = _list_models_at(e.base_url, e.api_key)
        local["base_url"] = e.base_url
        openai_key = e.openai_api_key
        openai_configured = bool(openai_key)
        if not openai_configured:
            openai: dict[str, Any] = {"ok": False, "configured": False, "models": [], "error": None}
        else:
            openai = _list_models_at(e.openai_base_url, openai_key or "")
            openai["configured"] = True
            if openai.get("models"):
                openai["models"] = filter_openai_chat_completion_model_ids(openai["models"])
        return jsonify(
            {
                "local": local,
                "openai": openai,
                "openai_configured": openai_configured,
                "kb_dir_default": e.kb_dir or "",
            }
        )

    @app.post("/api/run")
    def api_run() -> Any:
        global _busy, _bg_thread
        data = request.get_json(silent=True) or {}
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "error": "Missing prompt"}), 400
        provider = (data.get("llm_provider") or "local").strip().lower()
        if provider not in ("local", "openai"):
            return jsonify({"ok": False, "error": "Invalid llm_provider"}), 400
        model = (data.get("model") or "").strip()
        if not model:
            return jsonify({"ok": False, "error": "Missing model"}), 400
        env = get_hive_env()
        if provider == "openai" and not env.openai_api_key:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "OpenAI API key is not configured (set OPENAI_API_KEY in .env).",
                    }
                ),
                400,
            )
        enabled_ids = _current_enabled_skill_ids()
        cfg0 = _hive_config_for_run(enabled_ids, data)
        with _run_lock:
            if _busy:
                return jsonify({"ok": False, "error": "A run is already in progress."}), 409
            _busy = True
            _shared_state.reset(max_worker_slots=cfg0.resolved_worker_count())

        def job() -> None:
            global _busy
            try:
                enabled_ids2 = _current_enabled_skill_ids()
                cfg = _hive_config_for_run(enabled_ids2, data)
                run_hive(
                    prompt,
                    state=_shared_state,
                    config=cfg,
                )
            except Exception as e:
                _shared_state.set_phase("error")
                _shared_state.set_error(str(e))
            finally:
                with _run_lock:
                    _busy = False

        _bg_thread = threading.Thread(target=job, name="hive-run", daemon=True)
        _bg_thread.start()
        return jsonify({"ok": True})

    @app.post("/api/cancel")
    def api_cancel() -> Any:
        _shared_state.request_cancel()
        return jsonify({"ok": True})

    @app.get("/api/stream")
    def api_stream() -> Any:
        """Server-Sent Events: compact snapshot updates for phase, metrics, and log tail."""

        def event_generator():
            last_sig = ""
            while True:
                with _run_lock:
                    snap = _shared_state.snapshot()
                    busy = _busy
                lines = snap.get("log_lines") or []
                tail = lines[-40:] if lines else []
                payload = {
                    "phase": snap.get("phase"),
                    "busy": busy,
                    "run_id": snap.get("run_id"),
                    "metrics": snap.get("metrics"),
                    "error": snap.get("error"),
                    "log_lines_tail": tail,
                }
                sig = json_lib.dumps(payload, sort_keys=True)
                if sig != last_sig:
                    last_sig = sig
                    yield f"data: {sig}\n\n"
                time.sleep(0.45)

        return Response(
            stream_with_context(event_generator()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Agent Hive</title>
  <style>
    :root {
      --bg: #f0f2f5;
      --surface: #ffffff;
      --border: #d8dee9;
      --text: #1c2433;
      --muted: #5c6578;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --danger: #b91c1c;
      --danger-bg: #fef2f2;
      --radius: 10px;
      --shadow: 0 1px 3px rgba(28, 36, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 1.25rem 1rem 2.5rem;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    .page { max-width: 52rem; margin: 0 auto; }
    .site-header { margin-bottom: 1.25rem; }
    .site-header h1 { font-size: 1.5rem; font-weight: 700; margin: 0 0 0.35rem 0; letter-spacing: -0.02em; }
    details.help { margin-top: 0.35rem; font-size: 0.875rem; color: var(--muted); }
    details.help summary { cursor: pointer; user-select: none; color: var(--primary); font-weight: 500; }
    details.help summary:hover { text-decoration: underline; }
    details.help[open] summary { margin-bottom: 0.5rem; }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 1rem 1.1rem;
      margin-bottom: 0.85rem;
    }
    .card h2 { font-size: 1rem; font-weight: 600; margin: 0 0 0.65rem 0; color: var(--text); }
    .card > .muted { margin-top: -0.35rem; margin-bottom: 0.65rem; }
    textarea#prompt {
      width: 100%;
      min-height: 7.5rem;
      font: inherit;
      padding: 0.65rem 0.75rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      resize: vertical;
    }
    textarea#prompt:focus { outline: 2px solid rgba(37, 99, 235, 0.35); outline-offset: 1px; border-color: var(--primary); }
    input#kb-dir {
      width: 100%;
      font: inherit;
      padding: 0.5rem 0.65rem;
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    input#kb-dir:focus { outline: 2px solid rgba(37, 99, 235, 0.35); outline-offset: 1px; border-color: var(--primary); }
    .actions { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin-top: 0.75rem; }
    button {
      font: inherit;
      padding: 0.5rem 1rem;
      border-radius: 8px;
      border: 1px solid transparent;
      cursor: pointer;
      font-weight: 600;
    }
    #start { background: var(--primary); color: #fff; }
    #start:hover { background: var(--primary-hover); }
    #cancel { background: var(--surface); color: var(--text); border-color: var(--border); }
    #cancel:hover { background: #f8fafc; }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      border: 1px solid var(--border);
      background: #f8fafc;
      color: var(--muted);
    }
    .badge.busy { background: #eff6ff; color: var(--primary); border-color: #bfdbfe; }
    .status-grid {
      display: grid;
      gap: 0.65rem 1rem;
      align-items: start;
    }
    @media (min-width: 520px) {
      .status-grid { grid-template-columns: auto 1fr; }
    }
    .phase-pill {
      display: inline-block;
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-weight: 600;
      font-size: 0.9rem;
      background: #e2e8f0;
      color: var(--text);
    }
    .phase-pill[data-phase="error"] { background: var(--danger-bg); color: var(--danger); }
    .phase-pill[data-phase="done"], .phase-pill[data-phase="complete"] { background: #ecfdf5; color: #047857; }
    .phase-pill[data-phase="planning"], .phase-pill[data-phase="workers"], .phase-pill[data-phase="running"] {
      background: #eff6ff; color: #1d4ed8;
    }
    .status-meta { font-size: 0.875rem; color: var(--muted); margin: 0; }
    .status-meta code { font-size: 0.8em; background: #f1f5f9; padding: 0.1rem 0.35rem; border-radius: 4px; }
    .metrics-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.25rem; }
    .chip { font-size: 0.8rem; padding: 0.2rem 0.5rem; background: #f1f5f9; border-radius: 6px; color: var(--muted); }
    .err-box {
      margin: 0.5rem 0 0 0;
      padding: 0.6rem 0.75rem;
      border-radius: 8px;
      background: var(--danger-bg);
      border: 1px solid #fecaca;
      color: var(--danger);
      font-size: 0.9rem;
      font-weight: 500;
    }
    .err-box:empty { display: none; }
    details.panel { border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow); margin-bottom: 0.65rem; overflow: hidden; }
    details.panel > summary {
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      padding: 0.65rem 0.85rem;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.95rem;
      background: #f8fafc;
      border-bottom: 1px solid transparent;
      user-select: none;
    }
    details.panel > summary::-webkit-details-marker { display: none; }
    details.panel > summary::after { content: "▸"; font-size: 0.75rem; color: var(--muted); transition: transform 0.15s; }
    details.panel[open] > summary::after { transform: rotate(90deg); }
    details.panel[open] > summary { border-bottom-color: var(--border); }
    details.panel > summary:hover { background: #f1f5f9; }
    .panel-body { padding: 0.75rem 0.85rem; }
    .muted { color: var(--muted); font-size: 0.875rem; }
    pre.log {
      background: #f8fafc;
      padding: 0.65rem 0.75rem;
      max-height: 18rem;
      overflow: auto;
      font-size: 0.8125rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin: 0;
    }
    #io-log { max-height: 32rem; overflow: auto; }
    details.io-entry {
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-bottom: 0.5rem;
      background: #fafbfc;
    }
    details.io-entry > summary {
      list-style: none;
      padding: 0.5rem 0.65rem;
      cursor: pointer;
      font-size: 0.875rem;
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 0.35rem 0.65rem;
      user-select: none;
    }
    details.io-entry > summary::-webkit-details-marker { display: none; }
    details.io-entry > summary::after { content: "▸"; margin-left: auto; color: var(--muted); font-size: 0.7rem; }
    details.io-entry[open] > summary::after { content: "▾"; }
    .io-sum-title { font-weight: 600; }
    .io-sum-meta { font-size: 0.8rem; color: var(--muted); }
    .io-inner { padding: 0 0.65rem 0.65rem; }
    .io-meta { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.35rem; }
    .io-label { font-weight: 600; font-size: 0.78rem; margin: 0.5rem 0 0.2rem 0; text-transform: uppercase; letter-spacing: 0.03em; color: var(--muted); }
    table.data { border-collapse: collapse; width: 100%; font-size: 0.875rem; }
    table.data th, table.data td { border: 1px solid var(--border); padding: 0.4rem 0.55rem; text-align: left; }
    table.data th { background: #f1f5f9; font-weight: 600; }
    table.data tr:nth-child(even) td { background: #fafbfc; }
    .skills-list {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      max-height: 16rem;
      overflow: auto;
      padding: 0.25rem 0;
    }
    .skill-row { display: flex; gap: 0.55rem; align-items: flex-start; font-size: 0.875rem; cursor: pointer; padding: 0.25rem 0; border-radius: 6px; }
    .skill-row:hover { background: #f8fafc; }
    .skill-row input { margin-top: 0.2rem; flex-shrink: 0; }
    #result {
      white-space: pre-wrap;
      background: #f8fafc;
      padding: 0.75rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      min-height: 2.5rem;
      font-size: 0.9rem;
    }
    .result-meta { font-size: 0.875rem; margin: 0.35rem 0; }
    .live-region { min-height: 1.2em; }
    .field-row { margin-top: 0.65rem; display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }
    .field-row label { font-size: 0.875rem; color: var(--muted); }
    .disabled-block { opacity: 0.55; pointer-events: none; }
    #max-workers { width: 4.5rem; padding: 0.35rem 0.5rem; border: 1px solid var(--border); border-radius: 8px; font: inherit; }
    .plan-diagram-wrap { min-height: 3rem; overflow: auto; max-width: 100%; }
  </style>
</head>
<body>
  <div class="page">
    <header class="site-header">
      <h1>Agent Hive</h1>
      <details class="help">
        <summary>How this run works</summary>
        <p class="muted" style="margin:0">Submit a problem below. The planner runs first; parallel workers process sub-tasks (worker → QA → retry). The UI refreshes with live updates (SSE plus polling).</p>
      </details>
    </header>

    <section class="card" aria-labelledby="run-heading">
      <h2 id="run-heading">New run</h2>
      <fieldset class="llm-fieldset" style="margin:0 0 0.75rem 0;padding:0.65rem 0.75rem;border:1px solid var(--border);border-radius:8px;background:#fafbfc">
        <legend style="font-size:0.875rem;padding:0 0.35rem;color:var(--muted)">LLM provider and model</legend>
        <div class="field-row" style="margin-top:0">
          <label style="color:var(--text)"><input type="radio" name="llm-provider" value="local" checked /> LM Studio (local)</label>
          <span id="openai-provider-wrap" class="disabled-block">
            <label style="color:var(--text)"><input type="radio" name="llm-provider" value="openai" id="openai-provider-radio" disabled /> OpenAI</label>
          </span>
        </div>
        <p id="llm-options-hint" class="muted" style="margin:0.4rem 0 0 0;font-size:0.8125rem"></p>
        <div class="field-row" style="margin-top:0.5rem">
          <label for="model-select" class="muted">Model</label>
          <select id="model-select" aria-label="Model id" style="flex:1;min-width:10rem;max-width:100%;padding:0.35rem 0.5rem;border:1px solid var(--border);border-radius:8px;font:inherit"></select>
        </div>
      </fieldset>
      <label for="prompt" class="muted" style="display:block;margin-bottom:0.35rem">Problem or goal</label>
      <textarea id="prompt" placeholder="Describe your problem…" aria-describedby="prompt-hint"></textarea>
      <p id="prompt-hint" class="muted" style="margin:0.4rem 0 0 0">Use a clear goal; enabled skills (below) are available to the router and workers.</p>
      <label for="kb-dir" class="muted" style="display:block;margin:0.75rem 0 0.35rem 0">Knowledge base directory (optional)</label>
      <input type="text" id="kb-dir" name="kb-dir" autocomplete="off" spellcheck="false" placeholder="e.g. D:\\docs\\articles or ./my_kb" aria-describedby="kb-dir-hint" />
      <p id="kb-dir-hint" class="muted" style="margin:0.35rem 0 0 0">Text files under this folder are indexed into the planner prompt; workers can use kb_list / kb_read. Leave empty to use HIVE_KB_DIR from .env only.</p>
      <div class="field-row">
        <label for="max-workers">Parallel worker slots</label>
        <input type="number" id="max-workers" name="max-workers" min="1" max="64" value="4" step="1" />
        <span class="muted" style="font-size:0.8rem">(clamped by HiveConfig cap)</span>
      </div>
      <div class="actions">
        <button type="button" id="start" disabled>Start run</button>
        <button type="button" id="cancel">Cancel</button>
        <span id="busy" class="badge" aria-live="polite">Idle</span>
      </div>
    </section>

    <section class="card" aria-live="polite" aria-atomic="false">
      <h2>Run status</h2>
      <div class="status-grid">
        <div>
          <span id="phase" class="phase-pill" data-phase="">—</span>
        </div>
        <div>
          <p id="run-meta" class="status-meta"></p>
          <div id="metrics-line" class="metrics-chips live-region"></div>
        </div>
      </div>
      <p id="error" class="err-box" role="alert"></p>
    </section>

    <details class="panel">
      <summary>Plan (dependencies)</summary>
      <div class="panel-body">
        <p class="muted" style="margin-top:0">DAG from the planner (updates after checkpoint replans). Renders when planning finishes.</p>
        <div id="plan-mermaid-host" class="plan-diagram-wrap"></div>
        <pre id="plan-fallback" class="log" style="display:none;margin-top:0.5rem"></pre>
      </div>
    </details>

    <details class="panel">
      <summary><span id="skills-summary-label">Skills</span></summary>
      <div class="panel-body">
        <p class="muted" style="margin-top:0">Toggle skills for the per-task router and worker tools. Saved to <code>hive_skills_enabled.json</code>.</p>
        <div id="skills-list" class="skills-list"></div>
      </div>
    </details>

    <details class="panel">
      <summary>Workers &amp; tasks</summary>
      <div class="panel-body">
        <p class="muted" style="margin-top:0">Thread slots and task queue state.</p>
        <h3 style="font-size:0.9rem;margin:0.75rem 0 0.35rem 0">Threads (worker slots)</h3>
        <table class="data">
          <thead><tr><th>Slot</th><th>Current task</th></tr></thead>
          <tbody id="threads"></tbody>
        </table>
        <h3 style="font-size:0.9rem;margin:0.85rem 0 0.35rem 0">Tasks</h3>
        <table class="data">
          <thead><tr><th>ID</th><th>Status</th><th>Attempt</th><th>Last message</th></tr></thead>
          <tbody id="tasks"></tbody>
        </table>
      </div>
    </details>

    <details class="panel">
      <summary>Process log</summary>
      <div class="panel-body">
        <p class="muted" style="margin-top:0">High-level steps and messages from the hive.</p>
        <pre class="log" id="log"></pre>
      </div>
    </details>

    <details class="panel">
      <summary>LLM I/O (prompts &amp; responses)</summary>
      <div class="panel-body">
        <p class="muted" style="margin-top:0">Full model input and raw reply per call (latest last; capped in memory). Open a row to read content.</p>
        <div id="io-log"></div>
      </div>
    </details>

    <section class="card" aria-labelledby="result-heading">
      <h2 id="result-heading">Final result</h2>
      <p id="saved-to" class="muted result-meta"></p>
      <p id="final-check" class="muted result-meta"></p>
      <div id="result"></div>
    </section>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    function slugPhase(phase) {
      if (!phase || typeof phase !== 'string') return '';
      return phase.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    }
    function setPhaseUi(phase) {
      const el = document.getElementById('phase');
      const t = phase ? String(phase) : '—';
      el.textContent = t;
      el.dataset.phase = slugPhase(phase && String(phase));
    }
    function setBusyUi(busy) {
      const el = document.getElementById('busy');
      el.textContent = busy ? 'Busy' : 'Idle';
      el.classList.toggle('busy', !!busy);
    }
    function renderMetricsChips(metrics) {
      const host = document.getElementById('metrics-line');
      host.innerHTML = '';
      const m = metrics || {};
      const items = [];
      if (m.llm_calls != null) items.push({ k: 'LLM calls', v: String(m.llm_calls) });
      if (m.tokens_estimated_cumulative != null) items.push({ k: 'Est. tokens', v: String(m.tokens_estimated_cumulative) });
      if (m.qa_failures != null) items.push({ k: 'QA fails', v: String(m.qa_failures) });
      items.forEach(function (it) {
        const span = document.createElement('span');
        span.className = 'chip';
        span.textContent = it.k + ': ' + it.v;
        host.appendChild(span);
      });
    }
    function updateSkillsSummaryLabel() {
      const box = document.getElementById('skills-list');
      const cbs = box.querySelectorAll('input[type=checkbox]');
      let n = 0;
      const tot = cbs.length;
      cbs.forEach(function (cb) { if (cb.checked) n++; });
      const label = document.getElementById('skills-summary-label');
      if (!tot) {
        label.textContent = 'Skills (none found)';
        return;
      }
      label.textContent = 'Skills (' + n + ' enabled, ' + tot + ' found)';
    }
    var llmState = { localModels: [], openaiModels: [], openaiConfigured: false, localErr: '', openaiErr: '' };
    function currentLlmProvider() {
      const r = document.querySelector('input[name="llm-provider"]:checked');
      return r ? r.value : 'local';
    }
    function populateModelSelect() {
      const sel = document.getElementById('model-select');
      if (!sel) return;
      const prov = currentLlmProvider();
      const ids = prov === 'openai' ? llmState.openaiModels : llmState.localModels;
      const prev = sel.value;
      sel.innerHTML = '';
      ids.forEach(function (id) {
        const o = document.createElement('option');
        o.value = id;
        o.textContent = id;
        sel.appendChild(o);
      });
      if (prev && ids.indexOf(prev) >= 0) sel.value = prev;
      else if (ids.length) sel.selectedIndex = 0;
      updateLlmStartDisabled();
    }
    function updateLlmStartDisabled() {
      const sel = document.getElementById('model-select');
      const start = document.getElementById('start');
      if (!start || !sel) return;
      const ok = sel.options.length > 0 && sel.value;
      start.disabled = !ok;
    }
    function updateLlmOptionsHint() {
      const hint = document.getElementById('llm-options-hint');
      if (!hint) return;
      const prov = currentLlmProvider();
      const parts = [];
      if (!llmState.openaiConfigured) {
        parts.push('OpenAI is grayed out until OPENAI_API_KEY is set in .env (restart the server after editing).');
      }
      if (prov === 'local' && llmState.localErr) parts.push('Local: ' + llmState.localErr);
      if (prov === 'openai' && llmState.openaiErr) parts.push('OpenAI: ' + llmState.openaiErr);
      hint.textContent = parts.join(' ');
    }
    async function loadLlmOptions() {
      try {
        const r = await fetch('/api/llm-options');
        const d = await r.json();
        llmState.localModels = (d.local && d.local.models) ? d.local.models : [];
        llmState.openaiModels = (d.openai && d.openai.models) ? d.openai.models : [];
        llmState.openaiConfigured = !!d.openai_configured;
        llmState.localErr = (d.local && !d.local.ok && d.local.error) ? d.local.error : '';
        llmState.openaiErr = (d.openai && d.openai.configured && !d.openai.ok && d.openai.error) ? d.openai.error : '';
        const kbEl = document.getElementById('kb-dir');
        if (kbEl && typeof d.kb_dir_default === 'string' && !String(kbEl.value || '').trim()) {
          kbEl.value = d.kb_dir_default;
        }
        const wrap = document.getElementById('openai-provider-wrap');
        const radio = document.getElementById('openai-provider-radio');
        if (llmState.openaiConfigured) {
          if (wrap) wrap.classList.remove('disabled-block');
          if (radio) radio.disabled = false;
        } else {
          if (wrap) wrap.classList.add('disabled-block');
          if (radio) radio.disabled = true;
          if (currentLlmProvider() === 'openai') {
            const loc = document.querySelector('input[name="llm-provider"][value="local"]');
            if (loc) loc.checked = true;
          }
        }
        populateModelSelect();
        updateLlmOptionsHint();
      } catch (e) {
        llmState.localModels = [];
        llmState.openaiModels = [];
        const hint = document.getElementById('llm-options-hint');
        if (hint) hint.textContent = 'Could not load model lists: ' + e;
        updateLlmStartDisabled();
      }
    }
    var _lastPlanSig = '';
    function planTasksSignature(runId, planTasks) {
      try { return String(runId || '') + '|' + JSON.stringify(planTasks || []); }
      catch (e) { return String(runId || ''); }
    }
    function hiveMermaidEscape(s) {
      return String(s == null ? '' : s)
        .replace(/&/g, ' and ')
        .replace(/"/g, "'")
        .replace(/\\n/g, ' ')
        .replace(/\\[/g, ' ')
        .replace(/\\]/g, ' ')
        .replace(/[#$><`|{}]/g, ' ')
        .trim()
        .slice(0, 96);
    }
    var _statusPollAbort = null;
    function updatePlanDiagram(s) {
      const plan = s.plan_tasks || [];
      const sig = planTasksSignature(s.run_id, plan);
      if (sig === _lastPlanSig) return;
      const host = document.getElementById('plan-mermaid-host');
      const fb = document.getElementById('plan-fallback');
      if (!host || !fb) return;
      host.innerHTML = '';
      fb.style.display = 'none';
      fb.textContent = '';
      if (!plan.length) {
        host.innerHTML = '<p class="muted">Plan appears after the planning phase completes.</p>';
        _lastPlanSig = sig;
        return;
      }
      const idToIdx = {};
      plan.forEach(function (t, i) { idToIdx[t.id] = i; });
      const lines = ['flowchart TD'];
      plan.forEach(function (t, i) {
        const kt = t.key_task ? ' [checkpoint]' : '';
        const base = (t.title || t.id || 'task') + ' (' + (t.id || '') + ')' + kt;
        lines.push('  n' + i + '["' + hiveMermaidEscape(base) + '"]');
      });
      plan.forEach(function (t, i) {
        const deps = t.depends_on || [];
        deps.forEach(function (d) {
          const j = idToIdx[d];
          if (typeof j === 'number') lines.push('  n' + j + ' --> n' + i);
        });
      });
      const def = lines.join('\\n');
      if (typeof mermaid === 'undefined' || typeof mermaid.run !== 'function') {
        fb.textContent = def;
        fb.style.display = 'block';
        _lastPlanSig = sig;
        return;
      }
      if (!window.__hiveMermaidInit) {
        try {
          mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: 'neutral' });
        } catch (e1) { /* ignore */ }
        window.__hiveMermaidInit = true;
      }
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = def;
      host.appendChild(div);
      _lastPlanSig = sig;
      try {
        const p = mermaid.run({ nodes: [div] });
        if (p && typeof p.then === 'function') {
          p.catch(function () {
            host.innerHTML = '';
            fb.textContent = def;
            fb.style.display = 'block';
          });
        }
      } catch (e2) {
        host.innerHTML = '';
        fb.textContent = def;
        fb.style.display = 'block';
      }
    }
    async function poll() {
      if (_statusPollAbort) {
        try { _statusPollAbort.abort(); } catch (e0) { /* ignore */ }
      }
      const ac = new AbortController();
      _statusPollAbort = ac;
      let r;
      try {
        r = await fetch('/api/status', { signal: ac.signal });
      } catch (e) {
        if (ac.signal.aborted) return;
        throw e;
      }
      if (_statusPollAbort !== ac) return;
      let s;
      try {
        s = await r.json();
      } catch (e) {
        if (ac.signal.aborted || _statusPollAbort !== ac) return;
        throw e;
      }
      if (_statusPollAbort !== ac) return;
      setBusyUi(!!s.busy);
      setPhaseUi(s.phase);
      document.getElementById('log').textContent = (s.log_lines || []).join('\\n');
      document.getElementById('result').textContent = s.result || '';
      const saved = s.output_file || '';
      document.getElementById('saved-to').textContent = saved ? ('Saved to: ' + saved) : '';
      const fc = s.final_check || null;
      if (fc) {
        const status = fc.pass ? 'PASS' : 'FAIL';
        const conf = (typeof fc.confidence_score === 'number') ? (' · confidence=' + fc.confidence_score.toFixed(2)) : '';
        const replan = fc.requires_replan ? ' · requires_replan=true' : '';
        document.getElementById('final-check').textContent = 'Final consistency: ' + status + conf + replan;
      } else {
        document.getElementById('final-check').textContent = '';
      }
      document.getElementById('error').textContent = s.error || '';
      const rid = s.run_id || '';
      document.getElementById('run-meta').textContent = rid ? ('Run id: ' + rid) : '';
      renderMetricsChips(s.metrics);
      const tt = document.getElementById('threads');
      tt.innerHTML = '';
      const tm = s.thread_task || {};
      const maxSlot = Math.max(1, parseInt(s.max_worker_slots, 10) || 4);
      for (let i = 1; i <= maxSlot; i++) {
        const tr = document.createElement('tr');
        const td1 = document.createElement('td');
        td1.textContent = String(i);
        const td2 = document.createElement('td');
        td2.textContent = (tm[i] != null && tm[i] !== '') ? String(tm[i]) : '—';
        tr.appendChild(td1);
        tr.appendChild(td2);
        tt.appendChild(tr);
      }
      const tb = document.getElementById('tasks');
      tb.innerHTML = '';
      const tasks = s.tasks || {};
      for (const id of Object.keys(tasks)) {
        const t = tasks[id];
        const tr = document.createElement('tr');
        const tdId = document.createElement('td');
        tdId.textContent = id;
        const tdSt = document.createElement('td');
        tdSt.textContent = t.status || '';
        const tdAt = document.createElement('td');
        tdAt.textContent = String(t.attempt || 0);
        const tdLm = document.createElement('td');
        tdLm.textContent = t.last_message || '';
        tr.appendChild(tdId);
        tr.appendChild(tdSt);
        tr.appendChild(tdAt);
        tr.appendChild(tdLm);
        tb.appendChild(tr);
      }
      const ioBox = document.getElementById('io-log');
      const openCallIds = new Set();
      const openIndices = new Set();
      ioBox.querySelectorAll('details.io-entry').forEach(function (d, idx) {
        if (!d.open) return;
        if (d.dataset.callId) openCallIds.add(String(d.dataset.callId));
        else openIndices.add(idx);
      });
      ioBox.innerHTML = '';
      const iolog = s.io_log || [];
      iolog.forEach(function (e, idx) {
        const bits = [];
        if (e.run_id) bits.push('run: ' + e.run_id);
        if (e.call_id != null) bits.push('call: ' + e.call_id);
        if (e.role) bits.push('role: ' + e.role);
        if (e.task_id) bits.push('task: ' + e.task_id);
        if (e.worker_slot != null && e.worker_slot !== '') bits.push('slot: ' + e.worker_slot);
        const det = document.createElement('details');
        det.className = 'io-entry';
        if (e.call_id != null) det.dataset.callId = String(e.call_id);
        const sum = document.createElement('summary');
        const title = document.createElement('span');
        title.className = 'io-sum-title';
        title.textContent = e.label || ('Call ' + (idx + 1));
        sum.appendChild(title);
        if (bits.length) {
          const sm = document.createElement('span');
          sm.className = 'io-sum-meta';
          sm.textContent = bits.join(' · ');
          sum.appendChild(sm);
        }
        det.appendChild(sum);
        const inner = document.createElement('div');
        inner.className = 'io-inner';
        if (bits.length) {
          const meta = document.createElement('div');
          meta.className = 'io-meta';
          meta.textContent = bits.join(' · ');
          inner.appendChild(meta);
        }
        const li = document.createElement('div');
        li.className = 'io-label';
        li.textContent = 'Input prompt';
        inner.appendChild(li);
        const pin = document.createElement('pre');
        pin.className = 'log';
        pin.textContent = e.input || '';
        inner.appendChild(pin);
        const lo = document.createElement('div');
        lo.className = 'io-label';
        lo.textContent = 'Output response';
        inner.appendChild(lo);
        const pout = document.createElement('pre');
        pout.className = 'log';
        pout.textContent = e.output || '';
        inner.appendChild(pout);
        det.appendChild(inner);
        ioBox.appendChild(det);
        const cid = e.call_id != null ? String(e.call_id) : '';
        if (cid && openCallIds.has(cid)) det.open = true;
        else if (openIndices.has(idx)) det.open = true;
      });
      if (iolog.length === 0) {
        const empty = document.createElement('p');
        empty.className = 'muted';
        empty.textContent = 'No LLM calls recorded yet.';
        ioBox.appendChild(empty);
      }
      updatePlanDiagram(s);
    }
    setInterval(poll, 1200);
    poll();
    try {
      const es = new EventSource('/api/stream');
      es.onmessage = function(ev) {
        try {
          const j = JSON.parse(ev.data);
          if (j.phase) setPhaseUi(j.phase);
          if (typeof j.busy === 'boolean') setBusyUi(j.busy);
          if (j.run_id) document.getElementById('run-meta').textContent = 'Run id: ' + j.run_id;
          if (j.metrics != null) renderMetricsChips(j.metrics);
          if (j.log_lines_tail && j.log_lines_tail.length) {
            document.getElementById('log').textContent = j.log_lines_tail.join('\\n');
          }
          if (j.error) document.getElementById('error').textContent = j.error;
        } catch (e) { /* ignore */ }
      };
    } catch (e) { /* EventSource unsupported */ }
    async function loadSkills() {
      const r = await fetch('/api/skills');
      const data = await r.json();
      const box = document.getElementById('skills-list');
      box.innerHTML = '';
      const list = data.skills || [];
      if (list.length === 0) {
        const p = document.createElement('p');
        p.className = 'muted';
        p.textContent = 'No skills found under the skills folder.';
        box.appendChild(p);
        updateSkillsSummaryLabel();
        return;
      }
      list.forEach(function (sk) {
        const row = document.createElement('label');
        row.className = 'skill-row';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = !!sk.enabled;
        cb.dataset.skillId = sk.id;
        cb.addEventListener('change', saveSkillsFromUi);
        row.appendChild(cb);
        const span = document.createElement('span');
        const desc = (sk.description || '').slice(0, 280);
        const scriptsHint = sk.has_scripts ? ' [scripts]' : '';
        span.textContent = sk.name + scriptsHint + ' — ' + desc;
        row.appendChild(span);
        box.appendChild(row);
      });
      updateSkillsSummaryLabel();
    }
    async function saveSkillsFromUi() {
      const box = document.getElementById('skills-list');
      const enabled = {};
      box.querySelectorAll('input[type=checkbox]').forEach(function (cb) {
        enabled[cb.dataset.skillId] = cb.checked;
      });
      const r = await fetch('/api/skills', { method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }) });
      const j = await r.json();
      if (!j.ok) alert(j.error || 'Failed to save skills');
      else updateSkillsSummaryLabel();
    }
    document.querySelectorAll('input[name="llm-provider"]').forEach(function (el) {
      el.addEventListener('change', function () {
        populateModelSelect();
        updateLlmOptionsHint();
      });
    });
    const modelSelectEl = document.getElementById('model-select');
    if (modelSelectEl) modelSelectEl.addEventListener('change', updateLlmStartDisabled);
    loadLlmOptions();
    loadSkills();
    document.getElementById('start').onclick = async () => {
      const prompt = document.getElementById('prompt').value.trim();
      if (!prompt) { alert('Enter a problem.'); return; }
      const sel = document.getElementById('model-select');
      const model = sel ? String(sel.value || '').trim() : '';
      if (!model) { alert('Select a model (check LM Studio is running and reload the page).'); return; }
      const mwEl = document.getElementById('max-workers');
      const mw = mwEl ? parseInt(mwEl.value, 10) : 4;
      const kbDirEl = document.getElementById('kb-dir');
      const kbDir = kbDirEl ? String(kbDirEl.value || '').trim() : '';
      const body = {
        prompt: prompt,
        max_parallel_workers: (isNaN(mw) ? 4 : mw),
        llm_provider: currentLlmProvider(),
        model: model,
        kb_dir: kbDir
      };
      const r = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body) });
      const j = await r.json();
      if (!j.ok) alert(j.error || r.statusText);
    };
    document.getElementById('cancel').onclick = () => fetch('/api/cancel', { method: 'POST' });
  </script>
</body>
</html>
"""
