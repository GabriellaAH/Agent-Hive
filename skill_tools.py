"""Host-side execution of skill tools (http_fetch, run_script, run_workspace_python, kb_list, kb_read). Safe paths and timeouts."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote_plus, urlsplit, urlunsplit

import requests

from skills_registry import SkillInfo

# Refuse to read entire huge files into memory for kb_read.
KB_READ_FILE_MAX_BYTES = 10 * 1024 * 1024

# Environment keys copied into subprocess env for skills and workspace runs (minimal surface).
_SANDBOX_ENV_EXACT = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOME",
        "PYTHONUTF8",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "WINDIR",
        "LOCALAPPDATA",
        "APPDATA",
        "PYTHONPATH",
        "COMSPEC",
        "PATHEXT",
    }
)


def sandbox_environ() -> dict[str, str]:
    """Build a reduced environment for tool subprocesses (cwd sandbox; no full parent env)."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if v is None:
            continue
        ku = k.upper()
        if k in _SANDBOX_ENV_EXACT or ku.startswith("HIVE_") or "TAVILY" in ku:
            out[k] = str(v)
    return out


def _authorization_header_present(headers: dict[str, str]) -> bool:
    for k, v in headers.items():
        if k.lower() == "authorization" and (str(v) if v is not None else "").strip():
            return True
    return False


def _read_tavily_key_from_cli_sh(skills_root: Path) -> str | None:
    """Match API_KEY="..." in tavily-search/scripts/cli.sh (same as bash script)."""
    cli = skills_root / "tavily-search" / "scripts" / "cli.sh"
    if not cli.is_file():
        return None
    try:
        text = cli.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"API_KEY\s*=\s*\"([^\"]+)\"", text)
    if m:
        return m.group(1).strip()
    return None


def _resolve_tavily_api_key(skills_root: Path | None) -> str | None:
    for env in ("TAVILY_API_KEY", "HIVE_TAVILY_API_KEY"):
        v = os.environ.get(env)
        if v and v.strip():
            return v.strip()
    if skills_root is not None:
        return _read_tavily_key_from_cli_sh(skills_root.resolve())
    return None


def _maybe_inject_tavily_bearer(url: str, headers: dict[str, str], skills_root: Path | None) -> None:
    if "api.tavily.com" not in url.lower():
        return
    if _authorization_header_present(headers):
        return
    key = _resolve_tavily_api_key(skills_root)
    if key:
        headers["Authorization"] = f"Bearer {key}"


def _truthy_query_param(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _normalize_http_fetch(
    method: str, url: str, body: Any
) -> tuple[str, str, Any, str | None]:
    """Fix common mistaken requests (e.g. GET+query where the API expects POST+JSON).

    Returns (method, url, body, note) where note is a short message if anything was changed.
    """
    m = (method or "GET").upper().strip()
    note: str | None = None
    if body not in (None, {}, []):
        return method, url, body, note
    try:
        parts = urlsplit(url.strip())
    except Exception:
        return method, url, body, note
    host = (parts.hostname or "").lower()
    path = (parts.path or "").rstrip("/")

    if m == "GET" and "api.tavily.com" in host and path.endswith("/search") and parts.query:
        qs = parse_qs(parts.query, keep_blank_values=False)

        def first(key: str) -> str | None:
            vs = qs.get(key)
            if not vs or vs[0] is None:
                return None
            return unquote_plus(str(vs[0]))

        q = first("query") or first("q")
        if not q:
            return method, url, body, note
        json_body: dict[str, Any] = {"query": q}
        mr = first("max_results")
        if mr is not None:
            try:
                json_body["max_results"] = int(mr)
            except ValueError:
                json_body["max_results"] = mr
        for key in ("topic", "search_depth"):
            v = first(key)
            if v is not None:
                json_body[key] = v
        for key in ("include_answer", "auto_parameters"):
            v = first(key)
            if v is not None:
                json_body[key] = _truthy_query_param(v)
        base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        note = "Adjusted request: GET with query parameters was converted to POST with a JSON body for this search endpoint."
        return "POST", base, json_body, note

    return method, url, body, note


def _validate_http_fetch_payload(payload: dict[str, Any]) -> str | None:
    url = str(payload.get("url") or "").strip()
    if not url:
        return "http_fetch requires a non-empty string 'url'."
    if not (url.startswith("http://") or url.startswith("https://")):
        return "http_fetch 'url' must start with http:// or https://."
    method = str(payload.get("method") or "GET").strip()
    if not method:
        return "http_fetch 'method' must be a non-empty string."
    return None


def _resolve_script(skill_root: Path, script_relative: str) -> Path | None:
    """script_relative is relative to skill root (e.g. scripts/cli.sh)."""
    if not script_relative or ".." in script_relative.replace("\\", "/"):
        return None
    p = (skill_root / script_relative).resolve()
    try:
        p.relative_to(skill_root.resolve())
    except ValueError:
        return None
    scripts_root = (skill_root / "scripts").resolve()
    if not str(p).startswith(str(scripts_root)):
        return None
    if not p.is_file():
        return None
    return p


def _safe_workspace_filename(name: str) -> str | None:
    base = (name or "").strip().replace("\\", "/").split("/")[-1]
    if not base or ".." in base or "/" in base or "\\" in base:
        return None
    if not re.match(r"^[A-Za-z0-9._-]+$", base):
        return None
    if not base.endswith(".py"):
        base = base + ".py"
    return base


def http_fetch(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout_sec: float = 60.0,
    skills_root: Path | None = None,
) -> dict[str, Any]:
    """GET/POST via requests (Windows-safe alternative to curl).

    For https://api.tavily.com/*, if Authorization is omitted, uses TAVILY_API_KEY,
    HIVE_TAVILY_API_KEY, or the API_KEY line in skills/tavily-search/scripts/cli.sh.
    """
    m = (method or "GET").upper().strip()
    if m not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
        return {"ok": False, "error": f"Unsupported method: {method}"}
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "Invalid or missing URL (must be http/https)."}
    norm_note: str | None = None
    m, url, body, norm_note = _normalize_http_fetch(m, url, body)
    h: dict[str, str] = {}
    if headers:
        for k, v in headers.items():
            if isinstance(k, str) and isinstance(v, str):
                h[k] = v
    _maybe_inject_tavily_bearer(url, h, skills_root)
    try:
        if m == "GET":
            r = requests.get(url, headers=h, timeout=timeout_sec)
        elif m == "HEAD":
            r = requests.head(url, headers=h, timeout=timeout_sec)
        else:
            if body is None:
                r = requests.request(m, url, headers=h, timeout=timeout_sec)
            elif isinstance(body, (dict, list)):
                r = requests.request(
                    m,
                    url,
                    headers=h,
                    json=body,
                    timeout=timeout_sec,
                )
            else:
                r = requests.request(
                    m,
                    url,
                    headers=h,
                    data=str(body).encode("utf-8"),
                    timeout=timeout_sec,
                )
        text = r.text
        if len(text) > 500_000:
            text = text[:500_000] + "\n...[truncated]"
        out: dict[str, Any] = {
            "ok": True,
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "text": text,
        }
        if norm_note:
            out["host_note"] = norm_note
        return out
    except requests.RequestException as e:
        return {"ok": False, "error": str(e)}


def run_script(
    skill: SkillInfo,
    skills_root: Path,
    script_relative: str,
    args: list[str] | None,
    *,
    timeout_sec: float = 120.0,
) -> dict[str, Any]:
    """Run a file under skills/<id>/scripts/ only."""
    root = skills_root.resolve()
    skill_root = (root / skill.id).resolve()
    try:
        skill_root.relative_to(root)
    except ValueError:
        return {"ok": False, "error": "Invalid skill root."}
    exe = _resolve_script(skill_root, script_relative.replace("\\", "/"))
    if exe is None:
        return {"ok": False, "error": f"Script not allowed or not found: {script_relative}"}
    suffix = exe.suffix.lower()
    if suffix in (".sh", ".bash") and os.name == "nt":
        return {
            "ok": False,
            "error": "Bash scripts are not run on Windows here; use http_fetch or a Python script under scripts/.",
        }
    argv: list[str]
    if suffix == ".py":
        py = shutil.which("python") or shutil.which("python3")
        if not py:
            return {"ok": False, "error": "Python interpreter not found on PATH."}
        argv = [py, str(exe), *(args or [])]
    elif suffix in (".sh", ".bash"):
        sh = shutil.which("bash") or shutil.which("sh")
        if not sh:
            return {"ok": False, "error": "No shell available for .sh script."}
        argv = [sh, str(exe), *(args or [])]
    else:
        argv = [str(exe), *(args or [])]
    if exe.name.lower() == "last30days.py" and not (args or []):
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Error: Please provide a topic to research.\n",
            "retry_hint": (
                "last30days.py requires a non-empty \"args\" array. First element must be the research topic "
                "taken from the sub-task (short phrase). Example: "
                '"args": ["quantum computing hype 2025", "--quick"] or '
                '"args": ["best CRM for startups", "--emit=compact"]. Do not use [].'
            ),
        }
    try:
        proc = subprocess.run(
            argv,
            cwd=str(skill_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            shell=False,
            env=sandbox_environ(),
        )
        out: dict[str, Any] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:500_000],
            "stderr": (proc.stderr or "")[:500_000],
        }
        if not out["ok"]:
            se = (proc.stderr or "").lower()
            hints: list[str] = []
            if "topic" in se and ("provide" in se or "usage:" in se or "required" in se):
                hints.append(
                    "This script needs CLI arguments in \"args\" (first item is usually the topic or query string). "
                    "Copy a concise topic from the sub-task title or description; do not call with args: []."
                )
            if "usage:" in se or "usage" in se:
                hints.append(
                    "Match the script's Usage line: pass positional options in \"args\" in order after the script path."
                )
            if hints:
                out["retry_hint"] = " ".join(hints)
        return out
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout after {timeout_sec}s"}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def run_workspace_python(
    workspace_root: Path,
    payload: dict[str, Any],
    *,
    timeout_sec: float,
) -> dict[str, Any]:
    """Write Python source under workspace_root and execute it (cwd = workspace_root)."""
    root = workspace_root.resolve()
    if not root.is_dir():
        return {"ok": False, "error": "workspace_root is not a directory."}
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "error": "run_workspace_python requires string 'code' with non-empty Python source."}
    raw_name = str(payload.get("filename") or "snippet.py").strip()
    fname = _safe_workspace_filename(raw_name) or "snippet.py"
    target = (root / fname).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return {"ok": False, "error": "Invalid filename path."}
    try:
        target.write_text(code, encoding="utf-8", newline="\n")
    except OSError as e:
        return {"ok": False, "error": f"Failed to write file: {e}"}
    py = shutil.which("python") or shutil.which("python3")
    if not py:
        return {"ok": False, "error": "Python interpreter not found on PATH."}
    extra_args = payload.get("args")
    if not isinstance(extra_args, list):
        extra_args = []
    argv = [py, str(target), *[str(a) for a in extra_args]]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            shell=False,
            env=sandbox_environ(),
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:500_000],
            "stderr": (proc.stderr or "")[:500_000],
            "written_file": str(target),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout after {timeout_sec}s"}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def _kb_resolve_subdir(kb_root: Path, subdir: str | None) -> tuple[Path | None, str | None]:
    try:
        kb_r = kb_root.resolve()
    except OSError as exc:
        return None, str(exc)
    if subdir is None or not str(subdir).strip():
        base = kb_r
    else:
        rel = str(subdir).strip().replace("\\", "/").lstrip("/")
        if not rel or ".." in rel.split("/"):
            return None, "kb_list: subdir must be a relative path without .."
        base = (kb_r / rel).resolve()
    try:
        if not base.is_relative_to(kb_r):
            return None, "Path escapes knowledge base root."
    except (ValueError, TypeError):
        return None, "Path escapes knowledge base root."
    if base.is_symlink() or not base.is_dir():
        return None, "kb_list: subdir must be an existing non-symlink directory under the knowledge base root."
    return base, None


def _kb_normalize_ext_list(raw: Any, default: frozenset[str]) -> frozenset[str]:
    if not isinstance(raw, list):
        return default
    out: set[str] = set()
    for x in raw:
        p = str(x).strip().lower()
        if not p:
            continue
        if not p.startswith("."):
            p = "." + p.lstrip(".")
        if p in default:
            out.add(p)
    return frozenset(out) if out else default


def kb_list_tool(kb_root: Path, payload: dict[str, Any], *, default_exts: frozenset[str]) -> dict[str, Any]:
    sub = payload.get("subdir")
    subdir = str(sub).strip() if isinstance(sub, str) and str(sub).strip() else None
    exts = _kb_normalize_ext_list(payload.get("extensions"), default_exts)
    try:
        max_e = int(payload.get("max_entries") or 50)
    except (TypeError, ValueError):
        max_e = 50
    max_e = max(1, min(max_e, 500))
    base, err = _kb_resolve_subdir(kb_root, subdir)
    if err or base is None:
        return {"ok": False, "error": err or "Invalid subdir."}
    files: list[dict[str, Any]] = []
    try:
        kb_r = kb_root.resolve()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        for p in sorted(base.rglob("*"), key=lambda x: str(x).lower()):
            if len(files) >= max_e:
                break
            try:
                if p.is_symlink():
                    continue
                rp = p.resolve()
                if not rp.is_relative_to(kb_r):
                    continue
            except (OSError, ValueError, RuntimeError):
                continue
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                rel = rp.relative_to(kb_r).as_posix()
                sz = int(p.stat().st_size)
            except OSError:
                continue
            files.append({"path": rel, "size": sz})
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    sub_label = subdir if subdir else "."
    return {"ok": True, "kb_root": str(kb_r), "subdir": sub_label, "files": files}


def kb_read_tool(
    kb_root: Path,
    payload: dict[str, Any],
    *,
    kb_read_max_chars: int,
    allowed_exts: frozenset[str],
) -> dict[str, Any]:
    rel_path = payload.get("path") or payload.get("rel_path")
    if not isinstance(rel_path, str) or not rel_path.strip():
        return {"ok": False, "error": "kb_read requires string 'path' relative to knowledge base root."}
    rel_norm = rel_path.strip().replace("\\", "/").lstrip("/")
    if not rel_norm or ".." in rel_norm.split("/"):
        return {"ok": False, "error": "kb_read path must be relative without .."}
    try:
        kb_r = kb_root.resolve()
        target = (kb_r / rel_norm).resolve()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        if not target.is_relative_to(kb_r):
            return {"ok": False, "error": "Path escapes knowledge base root."}
    except (ValueError, TypeError):
        return {"ok": False, "error": "Path escapes knowledge base root."}
    if target.is_symlink():
        return {"ok": False, "error": "Symlinks are not allowed for kb_read."}
    if not target.is_file():
        return {"ok": False, "error": "kb_read path must be a regular file."}
    ext = target.suffix.lower()
    if ext not in allowed_exts:
        return {"ok": False, "error": f"Extension {ext} not allowed for kb_read."}
    try:
        sz = int(target.stat().st_size)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    if sz > KB_READ_FILE_MAX_BYTES:
        return {
            "ok": False,
            "error": f"File too large ({sz} bytes); max {KB_READ_FILE_MAX_BYTES} bytes.",
        }
    try:
        cap = int(payload.get("max_chars") or kb_read_max_chars)
    except (TypeError, ValueError):
        cap = kb_read_max_chars
    cap = max(100, min(cap, kb_read_max_chars))
    try:
        raw = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    truncated = len(raw) > cap
    content = raw[:cap] if truncated else raw
    rel_out = target.relative_to(kb_r).as_posix()
    return {"ok": True, "path": rel_out, "size": sz, "truncated": truncated, "content": content}


def _http_idempotency_key(payload: dict[str, Any], method: str, url: str) -> str | None:
    key = payload.get("idempotency_key")
    if not isinstance(key, str) or not key.strip():
        return None
    m = (method or "GET").upper().strip()
    if m not in ("GET", "HEAD"):
        return None
    h = hashlib.sha256(f"{m}\n{url}\n{key.strip()}".encode("utf-8")).hexdigest()
    return h


def execute_tool(
    tool: str,
    payload: dict[str, Any],
    *,
    skills_by_id: dict[str, SkillInfo],
    skills_root: Path,
    tool_timeout_sec: float,
    workspace_root: Path | None = None,
    idempotency_cache: dict[str, Any] | None = None,
    idempotency_ttl_sec: float = 600.0,
    idempotency_max_size: int = 128,
    kb_root: Path | None = None,
    kb_read_max_chars: int = 40_000,
    kb_file_extensions: frozenset[str] | None = None,
) -> dict[str, Any]:
    t = (tool or "").strip().lower()
    if t == "http_fetch":
        err = _validate_http_fetch_payload(payload)
        if err:
            return {"ok": False, "error": err}
        method = str(payload.get("method") or "GET")
        url = str(payload.get("url") or "").strip()
        idem_key = _http_idempotency_key(payload, method, url)
        now = time.time()
        if idempotency_cache is not None and idem_key:
            ent = idempotency_cache.get(idem_key)
            if isinstance(ent, tuple) and len(ent) == 2:
                cached, ts = ent
                if now - float(ts) <= idempotency_ttl_sec:
                    return json.loads(json.dumps(cached)) if isinstance(cached, dict) else cached
        result = http_fetch(
            method,
            url,
            headers=payload.get("headers") if isinstance(payload.get("headers"), dict) else None,
            body=payload.get("body"),
            timeout_sec=tool_timeout_sec,
            skills_root=skills_root,
        )
        if idempotency_cache is not None and idem_key and isinstance(result, dict):
            idempotency_cache[idem_key] = (result, now)
            over = len(idempotency_cache) - max(8, int(idempotency_max_size))
            if over > 0:
                for k in list(idempotency_cache.keys())[:over]:
                    idempotency_cache.pop(k, None)
        return result
    if t == "run_script":
        sid = str(payload.get("skill_id") or "").strip()
        if not sid:
            return {"ok": False, "error": "run_script requires string 'skill_id'."}
        sk = skills_by_id.get(sid)
        if sk is None:
            return {"ok": False, "error": "run_script requires a valid skill_id in the payload."}
        rel = str(payload.get("script") or payload.get("script_relative") or "").strip()
        if not rel:
            return {"ok": False, "error": "run_script requires 'script' or 'script_relative'."}
        args = payload.get("args")
        if not isinstance(args, list):
            args = []
        args = [str(a) for a in args]
        return run_script(
            sk,
            skills_root,
            rel,
            args,
            timeout_sec=tool_timeout_sec,
        )
    if t == "run_workspace_python":
        if workspace_root is None:
            return {"ok": False, "error": "run_workspace_python requires an active workspace for this run."}
        return run_workspace_python(workspace_root, payload, timeout_sec=tool_timeout_sec)
    if t == "kb_list":
        if kb_root is None:
            return {
                "ok": False,
                "error": "kb_list requires a configured knowledge base directory (HIVE_KB_DIR or per-run kb_dir).",
            }
        exts = kb_file_extensions if kb_file_extensions else frozenset({".md", ".txt"})
        return kb_list_tool(kb_root, payload, default_exts=exts)
    if t == "kb_read":
        if kb_root is None:
            return {
                "ok": False,
                "error": "kb_read requires a configured knowledge base directory (HIVE_KB_DIR or per-run kb_dir).",
            }
        exts = kb_file_extensions if kb_file_extensions else frozenset({".md", ".txt"})
        return kb_read_tool(
            kb_root,
            payload,
            kb_read_max_chars=max(1000, int(kb_read_max_chars)),
            allowed_exts=exts,
        )
    return {"ok": False, "error": f"Unknown tool: {tool}"}


def format_tool_result(tool: str, result: dict[str, Any]) -> str:
    try:
        return json.dumps({"tool": tool, "result": result}, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)
