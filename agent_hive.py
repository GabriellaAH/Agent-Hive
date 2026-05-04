"""
Multi-agent LLM orchestration: plan → parallel worker/QA threads → merge → final user-facing synthesis.
CLI: python agent_hive.py [prompt ...]  (or stdin if prompt omitted)
Web: python agent_hive.py --web [--host 0.0.0.0] [--port 5000]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from openai import APIConnectionError, APIError, OpenAI

from lm_client import complete_with_retries, estimate_tokens_from_text, get_first_model_id, make_openai_client
from skill_tools import execute_tool, format_tool_result
from skills_registry import (
    SkillInfo,
    default_enabled_path,
    default_skills_dir,
    discover_skills,
    load_enabled_map,
    merge_enabled_with_discovery,
    router_prompt_lines,
)

from claims_ledger import (
    ClaimValidationResult,
    build_excluded_claims_digest,
    claims_for_run_report,
    coerce_claims_document,
    deliverables_coverage_gaps,
    format_claims_for_merge,
    parse_allowed_source_types_csv,
    parse_deliverables_from_env,
    resolve_cross_task_conflicts,
    resolve_evidence_policy,
    validate_claims_document,
)
from hive_env import get_hive_env

_env = get_hive_env()
BASE_URL = _env.base_url
API_KEY = _env.api_key

_SUBAGENT_STOPWORDS = frozenset(
    "the and for are but not you all can her was one our out day get has him his how its may new "
    "now old see two way who did she use any due this that with from have been each will your into "
    "more than only also such other about their what when where which while would could should "
    "then them these they those".split()
)


class HiveRole(str, Enum):
    """LLM call roles for per-role temperature and max_tokens."""

    PLANNER_OUTLINE = "planner_outline"
    PLANNER_EXPAND = "planner_expand"
    PLANNER = "planner"
    PLANNER_CRITIQUE = "planner_critique"
    PLANNER_REPAIR = "planner_repair"
    WORKER = "worker"
    SKILL_ROUTER = "skill_router"
    QA = "qa"
    TASK_QA_REFINEMENT = "task_qa_refinement"
    TASK_QA_REFINEMENT_REPAIR = "task_qa_refinement_repair"
    MERGER = "merger"
    COMPRESS_MERGE = "compress_merge"
    SUMMARIZE_DEPS = "summarize_deps"
    CHECKPOINT = "checkpoint"
    FINAL_CHECK = "final_check"
    FINAL_FIX = "final_fix"
    FINAL_SYNTHESIS = "final_synthesis"
    INTEGRATION_QA = "integration_qa"
    ASSERTION_LEDGER = "assertion_ledger"
    QA_FAIL_DECOMPOSE = "qa_fail_decompose"
    QA_FAIL_DECOMPOSE_REPAIR = "qa_fail_decompose_repair"
    GENERIC = "generic"


def default_role_temperatures() -> dict[str, float]:
    return {
        HiveRole.PLANNER_OUTLINE.value: 0.25,
        HiveRole.PLANNER_EXPAND.value: 0.3,
        HiveRole.PLANNER.value: 0.35,
        HiveRole.PLANNER_CRITIQUE.value: 0.2,
        HiveRole.PLANNER_REPAIR.value: 0.2,
        HiveRole.WORKER.value: 0.55,
        HiveRole.SKILL_ROUTER.value: 0.15,
        HiveRole.QA.value: 0.1,
        HiveRole.TASK_QA_REFINEMENT.value: 0.25,
        HiveRole.TASK_QA_REFINEMENT_REPAIR.value: 0.2,
        HiveRole.MERGER.value: 0.4,
        HiveRole.COMPRESS_MERGE.value: 0.2,
        HiveRole.SUMMARIZE_DEPS.value: 0.15,
        HiveRole.CHECKPOINT.value: 0.25,
        HiveRole.FINAL_CHECK.value: 0.1,
        HiveRole.FINAL_FIX.value: 0.35,
        HiveRole.FINAL_SYNTHESIS.value: 0.35,
        HiveRole.INTEGRATION_QA.value: 0.15,
        HiveRole.ASSERTION_LEDGER.value: 0.12,
        HiveRole.QA_FAIL_DECOMPOSE.value: 0.2,
        HiveRole.QA_FAIL_DECOMPOSE_REPAIR.value: 0.15,
        HiveRole.GENERIC.value: 0.5,
    }


def default_role_max_tokens_factor() -> dict[str, float]:
    """Multipliers applied to cfg.max_tokens (clamped); 1.0 means use global max_tokens."""
    return {
        HiveRole.MERGER.value: 1.35,
        HiveRole.FINAL_SYNTHESIS.value: 1.2,
        HiveRole.COMPRESS_MERGE.value: 0.5,
        HiveRole.SUMMARIZE_DEPS.value: 0.35,
        HiveRole.PLANNER_OUTLINE.value: 0.45,
        HiveRole.INTEGRATION_QA.value: 0.55,
        HiveRole.ASSERTION_LEDGER.value: 0.65,
        HiveRole.QA_FAIL_DECOMPOSE.value: 0.45,
        HiveRole.QA_FAIL_DECOMPOSE_REPAIR.value: 0.4,
    }


@dataclass
class PlanTask:
    id: str
    title: str
    description: str
    acceptance_criteria: str = ""
    depends_on: list[str] = field(default_factory=list)
    key_task: bool = False


@dataclass(frozen=True)
class MicroDecomposeStep:
    title: str
    description: str
    done_when: str


@dataclass
class ParsedPlan:
    tasks: list[PlanTask]
    include_original_prompt: bool
    required_deliverables: list[str] = field(default_factory=list)


def extract_json_blob(text: str) -> str | None:
    """Extract JSON object/array from model output (fenced or raw)."""
    if not text or not text.strip():
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        c = t[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return t[start : i + 1]
    return None


def _parse_task_from_planner_item(item: dict[str, Any], index: int) -> PlanTask | None:
    if not isinstance(item, dict):
        return None
    tid = str(item.get("id", f"task-{index + 1}"))
    title = str(item.get("title", tid))
    desc = str(item.get("description", ""))
    acc = str(item.get("acceptance_criteria", item.get("acceptance", "")))
    raw_deps = item.get("depends_on")
    if raw_deps is None:
        raw_deps = item.get("dependencies")
    if not isinstance(raw_deps, list):
        raw_deps = []
    depends_on = [str(x).strip() for x in raw_deps if x is not None and str(x).strip()]
    key_task = bool(item.get("key_task", item.get("checkpoint", item.get("key", False))))
    return PlanTask(
        id=tid,
        title=title,
        description=desc,
        acceptance_criteria=acc,
        depends_on=depends_on,
        key_task=key_task,
    )


def validate_task_dependencies(tasks: list[PlanTask]) -> None:
    """Ensure depends_on references exist and the graph is acyclic."""
    id_list = [t.id for t in tasks]
    if len(id_list) != len(set(id_list)):
        raise ValueError("Duplicate task ids in plan.")
    ids = set(id_list)
    for t in tasks:
        for d in t.depends_on:
            if d not in ids:
                raise ValueError(f"Task {t.id!r} depends_on unknown id {d!r}.")
    visiting: set[str] = set()
    done: set[str] = set()
    by_id = {t.id: t for t in tasks}

    def visit(node: str) -> None:
        if node in done:
            return
        if node in visiting:
            raise ValueError("Task dependency graph contains a cycle.")
        visiting.add(node)
        pr = by_id.get(node)
        if pr:
            for d in pr.depends_on:
                visit(d)
        visiting.remove(node)
        done.add(node)

    for t in tasks:
        visit(t.id)


def parse_plan(raw: str) -> ParsedPlan:
    blob = extract_json_blob(raw)
    if not blob:
        raise ValueError("No JSON object found in planner output.")
    data = json.loads(blob)
    tasks_raw = data.get("tasks")
    if not isinstance(tasks_raw, list):
        raise ValueError("Planner JSON must contain a 'tasks' array.")
    iop = data.get("include_original_prompt")
    include_original_prompt = True if iop is None else bool(iop)
    out: list[PlanTask] = []
    for i, item in enumerate(tasks_raw):
        pt = _parse_task_from_planner_item(item, i) if isinstance(item, dict) else None
        if pt is not None:
            out.append(pt)
    if not out:
        raise ValueError("No tasks parsed from planner JSON.")
    validate_task_dependencies(out)
    rd_raw = data.get("required_deliverables")
    req_del: list[str] = []
    if isinstance(rd_raw, list):
        req_del = [str(x).strip() for x in rd_raw if x is not None and str(x).strip()]
    return ParsedPlan(tasks=out, include_original_prompt=include_original_prompt, required_deliverables=req_del)


def _append_deliverables_block(text: str, items: list[str]) -> str:
    if not items:
        return text
    block = (
        "\n\n[Orchestrator mandatory deliverables — include explicit task(s) or acceptance lines for each:]\n"
        + "\n".join(f"- {x}" for x in items)
    )
    return text + block


def parse_task_qa_refinement(raw: str, current: PlanTask) -> tuple[PlanTask, bool, str]:
    """Parse planner output that may update task fields after repeated QA failures. Returns (task, adjusted, rationale)."""
    blob = extract_json_blob(raw)
    if not blob:
        raise ValueError("No JSON in task QA refinement output.")
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise ValueError("Task QA refinement must be a JSON object.")
    adj = data.get("adjust_task_spec", data.get("refined"))
    adjust = bool(adj) if adj is not None else False
    rationale = str(data.get("rationale", "")).strip()
    if not adjust:
        return current, False, rationale
    title = data.get("title")
    desc = data.get("description")
    acc = data.get("acceptance_criteria")
    updated = PlanTask(
        id=current.id,
        title=str(title) if title is not None else current.title,
        description=str(desc) if desc is not None else current.description,
        acceptance_criteria=str(acc) if acc is not None else current.acceptance_criteria,
        depends_on=current.depends_on,
        key_task=current.key_task,
    )
    return updated, True, rationale


def parse_qa_verdict(raw: str) -> tuple[bool, str]:
    """Parse QA JSON; if checklist is present, every item must have met=true for pass."""
    blob = extract_json_blob(raw)
    if not blob:
        return False, "QA did not return parseable JSON."
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return False, "QA JSON parse error."
    if not isinstance(data, dict):
        return False, "QA output was not a JSON object."
    checklist = data.get("checklist")
    checklist_ok = True
    checklist_notes: list[str] = []
    if isinstance(checklist, list) and checklist:
        for item in checklist:
            if not isinstance(item, dict):
                checklist_ok = False
                checklist_notes.append("Invalid checklist entry.")
                continue
            met = item.get("met", item.get("satisfied", item.get("ok")))
            iid = item.get("id", item.get("criterion_id", "?"))
            note = str(item.get("note", item.get("detail", ""))).strip()
            if met is not True:
                checklist_ok = False
                checklist_notes.append(f"criterion {iid}: not met" + (f" ({note})" if note else ""))
            elif note:
                checklist_notes.append(f"criterion {iid}: met ({note})")
    passed = bool(data.get("pass", data.get("passed", False)))
    if isinstance(checklist, list) and checklist:
        passed = passed and checklist_ok
    reasons = data.get("reasons") or []
    fixes = data.get("required_fixes") or data.get("fixes") or ""
    detail_parts: list[str] = []
    if isinstance(reasons, list):
        detail_parts.extend(str(x) for x in reasons)
    elif reasons:
        detail_parts.append(str(reasons))
    if fixes:
        detail_parts.append(f"Required fixes: {fixes}")
    if checklist_notes:
        detail_parts.append("Checklist: " + "; ".join(checklist_notes))
    detail = "; ".join(detail_parts) if detail_parts else raw[:500]
    return passed, detail


def parse_qa_fail_decompose(raw: str, max_steps: int) -> tuple[bool, str, list[MicroDecomposeStep]]:
    """Parse micro-plan JSON. Returns (single_shot_ok, rationale, steps). single_shot_ok True => skip micro pipeline."""
    blob = extract_json_blob(raw)
    if not blob:
        return True, "", []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return True, "", []
    if not isinstance(data, dict):
        return True, "", []
    rationale = str(data.get("rationale", "")).strip()
    single = bool(data.get("single_shot_ok", data.get("single_shot_recommended", True)))
    steps_raw = data.get("steps")
    out: list[MicroDecomposeStep] = []
    if isinstance(steps_raw, list) and not single:
        for it in steps_raw:
            if len(out) >= max_steps:
                break
            if not isinstance(it, dict):
                continue
            t = str(it.get("title", "")).strip()
            d = str(it.get("description", "")).strip()
            dw = str(it.get("done_when", it.get("acceptance", ""))).strip()
            if t and d:
                out.append(MicroDecomposeStep(title=t, description=d, done_when=dw))
    if single or len(out) < 2:
        return True, rationale, []
    return False, rationale, out[:max_steps]


def parse_integration_qa_verdict(raw: str) -> tuple[bool, str, list[str]]:
    blob = extract_json_blob(raw)
    if not blob:
        return False, "Integration QA did not return parseable JSON.", []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return False, "Integration QA JSON parse error.", []
    if not isinstance(data, dict):
        return False, "Integration QA output was not a JSON object.", []
    passed = bool(data.get("pass", False))
    issues = data.get("issues") or data.get("critical_issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    rationale = str(data.get("rationale", "")).strip()
    detail = rationale or "; ".join(str(x) for x in issues) or raw[:500]
    return passed, detail, [str(x) for x in issues]


def validate_plan_quality(
    tasks: list[PlanTask],
    *,
    max_plan_tasks: int,
    max_plan_depth_hint: int,
    min_description_chars: int,
    min_acceptance_chars: int,
) -> list[str]:
    errors: list[str] = []
    if len(tasks) > max_plan_tasks:
        errors.append(f"Task count {len(tasks)} exceeds max_plan_tasks ({max_plan_tasks}).")
    depths = longest_path_depths(tasks)
    deepest = max(depths.values()) if depths else 0
    if deepest > max_plan_depth_hint:
        errors.append(
            f"Longest dependency chain length is {deepest}, exceeds max_plan_depth_hint ({max_plan_depth_hint})."
        )
    for t in tasks:
        if len((t.description or "").strip()) < min_description_chars:
            errors.append(
                f"Task {t.id!r}: description shorter than {min_description_chars} characters or empty."
            )
        if len((t.acceptance_criteria or "").strip()) < min_acceptance_chars:
            errors.append(
                f"Task {t.id!r}: acceptance_criteria shorter than {min_acceptance_chars} characters or empty."
            )
    return errors


def _tasks_by_id(tasks: list[PlanTask]) -> dict[str, PlanTask]:
    return {t.id: t for t in tasks}


def longest_path_depths(tasks: list[PlanTask]) -> dict[str, int]:
    """For each task id, length of longest dependency chain ending at this task (node count)."""
    by_id = _tasks_by_id(tasks)
    memo: dict[str, int] = {}

    def depth(tid: str) -> int:
        if tid in memo:
            return memo[tid]
        t = by_id.get(tid)
        if not t:
            memo[tid] = 0
            return 0
        if not t.depends_on:
            memo[tid] = 1
            return 1
        memo[tid] = 1 + max(depth(d) for d in t.depends_on)
        return memo[tid]

    for t in tasks:
        depth(t.id)
    return memo


def critical_path_task_ids(tasks: list[PlanTask]) -> list[str]:
    """Ordered task ids along one longest path (for integration QA)."""
    if not tasks:
        return []
    depths = longest_path_depths(tasks)
    by_id = _tasks_by_id(tasks)
    end = max(depths, key=lambda k: depths.get(k, 0))
    order_rev: list[str] = []
    cur: str | None = end
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        order_rev.append(cur)
        t = by_id.get(cur)
        if not t or not t.depends_on:
            break
        cur = max(t.depends_on, key=lambda d: depths.get(d, 0))
    return list(reversed(order_rev))


def _numbered_acceptance_lines(acceptance_criteria: str) -> str:
    lines = [ln.strip() for ln in (acceptance_criteria or "").splitlines() if ln.strip()]
    if not lines:
        return "(none — treat as single implicit criterion: solution must satisfy the task description.)"
    return "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))


def summarize_dependency_blob_heuristic(
    dep_id: str,
    text: str,
    consumer_task: PlanTask,
    max_chars: int,
) -> str:
    """Rule-based compression of a dependency output for the worker prompt."""
    if len(text) <= max_chars:
        return text
    keywords = _task_keyword_tokens(consumer_task)
    head = text[: max(400, max_chars // 3)].strip()
    tail = text[-(max_chars // 4) :].strip() if len(text) > max_chars // 2 else ""
    scored = _score_text_vs_keywords(text, keywords)
    mid_note = ""
    if scored > 0 and len(text) > max_chars:
        paras = _problem_paragraphs(text)
        scored_paras = sorted(
            ((_score_text_vs_keywords(p, keywords), p) for p in paras), key=lambda x: -x[0]
        )
        chunks: list[str] = []
        used = len(head) + len(tail) + 80
        for _, p in scored_paras[:6]:
            frag = (p[:600] + "…") if len(p) > 600 else p
            if used + len(frag) > max_chars - 40:
                break
            chunks.append(frag)
            used += len(frag) + 2
        if chunks:
            mid_note = "\n\n[Excerpts most relevant to this sub-task]\n" + "\n---\n".join(chunks)
    note = f"[Summary of output from {dep_id}: original length {len(text)} chars]\n"
    out = note + "## Beginning\n" + head + mid_note
    if tail and tail not in head:
        out += "\n## Ending\n" + tail
    if len(out) > max_chars:
        out = out[: max_chars - 20] + "\n...[truncated]"
    return out


def parse_checkpoint_decision(raw: str) -> dict[str, Any]:
    """Parse key-task checkpoint output. Default to continue on failure."""
    default: dict[str, Any] = {"action": "continue", "rationale": "", "tasks": []}
    blob = extract_json_blob(raw)
    if not blob:
        return default
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    action = str(data.get("action", "continue")).strip().lower()
    if action not in ("continue", "finish_early", "replan"):
        action = "continue"
    rationale = str(data.get("rationale", data.get("reason", ""))).strip()
    tasks_raw = data.get("tasks")
    if tasks_raw is None:
        tasks_raw = data.get("new_tasks")
    if not isinstance(tasks_raw, list):
        tasks_raw = []
    return {"action": action, "rationale": rationale, "tasks": tasks_raw}


def parse_final_consistency(raw: str) -> dict[str, Any]:
    blob = extract_json_blob(raw)
    if not blob:
        return {
            "pass": False,
            "critical_issues": ["No parseable JSON returned."],
            "logical_inconsistencies": [],
            "unproven_claims": [],
            "missing_components": [],
            "confidence_score": 0.0,
            "requires_replan": False,
        }
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {
            "pass": False,
            "critical_issues": ["Final consistency JSON parse error."],
            "logical_inconsistencies": [],
            "unproven_claims": [],
            "missing_components": [],
            "confidence_score": 0.0,
            "requires_replan": False,
        }
    if not isinstance(data, dict):
        return {
            "pass": False,
            "critical_issues": ["Final consistency verdict was not a JSON object."],
            "logical_inconsistencies": [],
            "unproven_claims": [],
            "missing_components": [],
            "confidence_score": 0.0,
            "requires_replan": False,
        }
    return {
        "pass": bool(data.get("pass", False)),
        "critical_issues": data.get("critical_issues") or [],
        "logical_inconsistencies": data.get("logical_inconsistencies") or [],
        "unproven_claims": data.get("unproven_claims") or [],
        "missing_components": data.get("missing_components") or [],
        "confidence_score": float(data.get("confidence_score") or 0.0),
        "requires_replan": bool(data.get("requires_replan", False)),
    }


class RunState:
    """Thread-safe status for UI and CLI logging."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "idle"
        self.tasks: dict[str, dict[str, Any]] = {}
        self.max_worker_slots: int = _env.max_parallel_workers
        self.thread_task: dict[int, str | None] = {
            i: None for i in range(1, _env.max_parallel_workers + 1)
        }
        self.log_lines: deque[str] = deque(maxlen=_env.log_max_lines)
        self.io_log: deque[dict[str, Any]] = deque(maxlen=_env.io_log_max_entries)
        self.result: str | None = None
        self.output_file: str | None = None
        self.error: str | None = None
        self.final_check: dict[str, Any] | None = None
        self.cancel_requested = False
        self.run_id: str = ""
        self._call_seq = 0
        self.metrics: dict[str, Any] = {
            "llm_calls": 0,
            "qa_failures": 0,
            "tokens_estimated_cumulative": 0,
            "qa_retry_events": 0,
        }
        self.run_started_monotonic: float = 0.0
        self.plan_tasks: list[dict[str, Any]] = []
        self.claims_report: dict[str, Any] | None = None

    def reset(self, *, max_worker_slots: int | None = None, run_id: str | None = None) -> None:
        slots = max(1, int(max_worker_slots or _env.max_parallel_workers))
        rid = run_id or str(uuid.uuid4())[:12]
        with self._lock:
            self.phase = "idle"
            self.tasks.clear()
            self.max_worker_slots = slots
            self.thread_task = {i: None for i in range(1, slots + 1)}
            self.log_lines.clear()
            self.io_log.clear()
            self.result = None
            self.output_file = None
            self.error = None
            self.final_check = None
            self.cancel_requested = False
            self.run_id = rid
            self._call_seq = 0
            self.metrics = {
                "llm_calls": 0,
                "qa_failures": 0,
                "tokens_estimated_cumulative": 0,
                "qa_retry_events": 0,
            }
            self.run_started_monotonic = time.monotonic()
            self.plan_tasks = []
            self.claims_report = None

    def request_cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True
            self._append_log("Cancel requested.")

    def abort_run(self, message: str) -> None:
        """Mark run failed (worker/orchestrator error); signals other workers to stop."""
        with self._lock:
            self.error = message
            self.phase = "error"
            self.cancel_requested = True
            self._append_log(f"ERROR: {message}")

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self._append_log(f"Phase: {phase}")

    def init_tasks(self, task_ids: list[str]) -> None:
        with self._lock:
            for tid in task_ids:
                self.tasks[tid] = {
                    "status": "queued",
                    "attempt": 0,
                    "last_message": "",
                }

    def task_update(self, task_id: str, **kwargs: Any) -> None:
        with self._lock:
            if task_id not in self.tasks:
                self.tasks[task_id] = {}
            self.tasks[task_id].update(kwargs)
            if "last_message" in kwargs:
                self._append_log(f"[{task_id}] {kwargs['last_message']}")

    def set_thread_task(self, worker_slot: int, task_id: str | None) -> None:
        with self._lock:
            if 1 <= worker_slot <= self.max_worker_slots:
                self.thread_task[worker_slot] = task_id

    def _append_log(self, line: str) -> None:
        self.log_lines.append(line)

    def log(self, message: str) -> None:
        with self._lock:
            self._append_log(message)

    def record_llm(
        self,
        label: str,
        input_text: str,
        output_text: str,
        task_id: str | None = None,
        worker_slot: int | None = None,
        *,
        role: str | None = None,
    ) -> None:
        with self._lock:
            self._call_seq += 1
            entry: dict[str, Any] = {
                "label": label,
                "input": input_text,
                "output": output_text,
                "task_id": task_id,
                "worker_slot": worker_slot,
                "run_id": self.run_id,
                "call_id": self._call_seq,
            }
            if role:
                entry["role"] = role
            self.io_log.append(entry)
            self.metrics["llm_calls"] = int(self.metrics.get("llm_calls", 0)) + 1
            self.metrics["tokens_estimated_cumulative"] = int(
                self.metrics.get("tokens_estimated_cumulative", 0)
            ) + estimate_tokens_from_text(input_text, output_text)

    def record_qa_failure(self) -> None:
        with self._lock:
            self.metrics["qa_failures"] = int(self.metrics.get("qa_failures", 0)) + 1

    def record_qa_retry(self) -> None:
        with self._lock:
            self.metrics["qa_retry_events"] = int(self.metrics.get("qa_retry_events", 0)) + 1

    def set_result(self, text: str | None) -> None:
        with self._lock:
            self.result = text

    def set_output_file(self, path: str | None) -> None:
        with self._lock:
            self.output_file = path
            if path:
                self._append_log(f"Saved result to: {path}")

    def set_final_check(self, verdict: dict[str, Any] | None) -> None:
        with self._lock:
            self.final_check = verdict

    def set_claims_report(self, report: dict[str, Any] | None, *, max_claims: int) -> None:
        """Store truncated claims report for /api/status debug."""
        with self._lock:
            if not report:
                self.claims_report = None
                return
            out = dict(report)
            claims = list(out.get("claims") or [])
            out["claims_total"] = len(claims)
            out["claims"] = claims[: max(1, max_claims)]
            self.claims_report = out

    def set_error(self, message: str | None) -> None:
        with self._lock:
            self.error = message
            if message:
                self._append_log(f"ERROR: {message}")

    def set_plan_snapshot(self, tasks: list[PlanTask]) -> None:
        """Publish planner DAG (ids, titles, edges) for UI."""
        snap = [
            {
                "id": t.id,
                "title": t.title,
                "depends_on": list(t.depends_on),
                "key_task": bool(t.key_task),
            }
            for t in tasks
        ]
        with self._lock:
            self.plan_tasks = snap

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "phase": self.phase,
                "tasks": dict(self.tasks),
                "thread_task": dict(self.thread_task),
                "max_worker_slots": self.max_worker_slots,
                "run_id": self.run_id,
                "metrics": dict(self.metrics),
                "log_lines": list(self.log_lines),
                "io_log": list(self.io_log),
                "plan_tasks": list(self.plan_tasks),
                "result": self.result,
                "output_file": self.output_file,
                "error": self.error,
                "final_check": self.final_check,
                "claims_report": dict(self.claims_report) if self.claims_report else None,
                "cancel_requested": self.cancel_requested,
            }


def _planner_system() -> str:
    return (
        "You are a planning assistant. Output a single JSON object only (no markdown fences), "
        'schema: {"include_original_prompt": true or false, '
        '"required_deliverables": ["optional explicit deliverable lines from the user request"], '
        '"tasks":[{"id":"string","title":"string",'
        '"description":"string","acceptance_criteria":"string",'
        '"depends_on":["id_of_prerequisite_task",...],'
        '"key_task": false}]}. '
        "Field required_deliverables: optional list of concrete deliverables the final answer must contain; "
        "use [] if none. When the orchestrator lists mandatory deliverables in the user message, mirror them here. "
        "Field depends_on: list of task ids that must finish before this task may start; use [] when none. "
        "Only reference ids that exist in this same tasks array. Omit depends_on or use [] for independent work. "
        "Field key_task: true only for checkpoints where progress should be reviewed before continuing—"
        "0 to many per plan; use false for simple problems. "
        "Field include_original_prompt: set true only if implementation sub-agents must see the exact "
        "user wording, large pasted code or data, or tight constraints that cannot be restated briefly without loss. "
        "Otherwise set false. "
        "When include_original_prompt is false: do NOT paste or reproduce the entire user message (or huge chunks of it) "
        "into the JSON. Each task must be self-contained—distill facts, copy only minimal necessary excerpts, "
        "spell out concrete inputs/outputs and constraints—so a sub-agent that never sees the raw user prompt can still complete the task. "
        "When true: keep task fields concise anyway; the orchestrator will attach the full user message to sub-agents. "
        "Prefer a moderate number of tasks: batch related steps into one task when they share context, "
        "instead of over-fragmenting (too many tiny tasks increases merge errors). "
        "Respect the orchestrator hard limits given in the user message (max tasks, max dependency depth). "
        "Each task needs testable acceptance_criteria (multiple lines allowed; each line is one criterion). "
        "When the user message includes [Prior run knowledge base], treat it as material from a previous full attempt: "
        "prefer plans that reuse solid partial results and extend or repair where the listed issues indicate gaps. "
        "When the user message includes [User knowledge base], it is an on-disk folder of reference files (excerpts are "
        "injected; workers can load full text via kb_list / kb_read tools). Plan tasks that cite specific relative paths "
        "and instruct workers to read those files when analysis depends on full content."
    )


def _planner_user(problem: str, *, max_plan_tasks: int, max_plan_depth_hint: int) -> str:
    return (
        f"Problem to solve:\n{problem}\n\n"
        f"Hard limits: at most {max_plan_tasks} tasks; dependency depth (longest chain) at most {max_plan_depth_hint} "
        "(count tasks along a prerequisite chain). Merge or combine work to stay within limits.\n"
        "Set include_original_prompt per the rules in your instructions.\n"
        "Respond with JSON only."
    )


def _planner_outline_system() -> str:
    return (
        "You are a planning assistant. Output a single JSON object only (no markdown fences).\n"
        'Schema: {"summary":"one paragraph","tasks":['
        '{"id":"string","title":"string","one_line_goal":"string","depends_on":["id",...]}'
        '],"notes":"optional"}. '
        "depends_on lists prerequisite ids from the same tasks array. Keep the outline small and stable. "
        "At most the max_tasks limit given in the user message."
    )


def _planner_outline_user(problem: str, *, max_plan_tasks: int) -> str:
    return (
        f"Problem to solve:\n{problem}\n\n"
        f"Produce a short outline only (at most {max_plan_tasks} tasks). Respond with JSON only."
    )


def _planner_expand_system() -> str:
    return (
        "You expand a planning outline into a full execution plan. Output a single JSON object only (no markdown fences), "
        'same final schema as the main planner: {"include_original_prompt": true or false, '
        '"required_deliverables": [], "tasks":[{"id","title",'
        '"description","acceptance_criteria","depends_on":[],"key_task":false}]}. '
        "Follow the outline ids and dependencies unless you must fix inconsistencies. "
        "Each task needs a clear description and multi-line acceptance_criteria when helpful. "
        "Respect max_tasks and max_depth from the user message; batch steps if needed."
    )


def _planner_expand_user(problem: str, outline_json: str, *, max_plan_tasks: int, max_plan_depth_hint: int) -> str:
    return (
        f"Problem to solve:\n{problem}\n\n"
        f"Outline JSON (from previous step):\n{outline_json}\n\n"
        f"Expand into the full plan. Hard limits: at most {max_plan_tasks} tasks; "
        f"longest dependency chain at most {max_plan_depth_hint} tasks.\n"
        "Respond with JSON only."
    )


def _planner_critique_system() -> str:
    return (
        "You fix an invalid or low-quality plan JSON. Output ONE valid JSON object only (no markdown fences), "
        "same schema as the main planner: "
        '{"include_original_prompt": true or false, "required_deliverables": [], '
        '"tasks":[{"id","title","description","acceptance_criteria",'
        '"depends_on":[],"key_task":false}]}. '
        "Fix every issue listed in the user message while preserving intent."
    )


def _planner_critique_user(errors: list[str], broken_or_partial_json: str) -> str:
    err_text = "\n".join(f"- {e}" for e in errors)
    return (
        f"Validation errors to fix:\n{err_text}\n\n"
        f"Current planner output (repair or replace):\n{broken_or_partial_json[:20000]}\n\n"
        "Respond with JSON only."
    )


def _repair_json_prompt(broken: str) -> str:
    return (
        "The following text was invalid or incomplete. Output ONE valid JSON object only, "
        'same schema: {"include_original_prompt": true or false, "required_deliverables": [], '
        '"tasks":[{"id","title","description","acceptance_criteria",'
        '"depends_on":[],"key_task":false}]}. '
        "depends_on lists prerequisite task ids (or []); key_task marks optional review checkpoints.\n\n"
        f"Broken output:\n{broken[:12000]}"
    )


def _task_keyword_tokens(task: PlanTask) -> set[str]:
    blob = f"{task.id} {task.title} {task.description} {task.acceptance_criteria or ''}"
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", blob)
    return {w.lower() for w in words if w.lower() not in _SUBAGENT_STOPWORDS}


def _score_text_vs_keywords(text: str, keywords: set[str]) -> int:
    if not keywords:
        return 0
    low = text.lower()
    return sum(low.count(k) for k in keywords if len(k) >= 3)


def _problem_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 2500:
            buf: list[str] = []
            acc = 0
            for line in p.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if acc + len(line) > 2000 and buf:
                    out.append("\n".join(buf))
                    buf = [line]
                    acc = len(line)
                else:
                    buf.append(line)
                    acc += len(line) + 1
            if buf:
                out.append("\n".join(buf))
        else:
            out.append(p)
    return out


def _worker_system() -> str:
    return (
        "You are an implementation sub-agent. Produce a clear, complete partial solution "
        "for the assigned sub-task only. Use structured sections if helpful."
    )


def _worker_user(
    global_problem: str,
    task: PlanTask,
    previous: str | None,
    qa_feedback: str | None,
    *,
    cfg: HiveConfig,
    include_original_prompt: bool,
    dependency_outputs: list[tuple[str, str]] | None = None,
    workspace_abs: str | None = None,
    prior_tool_evidence: str | None = None,
    prior_run_knowledge: str | None = None,
) -> str:
    problem_ctx = _subagent_overall_problem(
        global_problem, task, include_original_prompt=include_original_prompt, cfg=cfg
    )
    parts = [
        f"Overall problem:\n{problem_ctx}\n",
    ]
    if prior_run_knowledge and str(prior_run_knowledge).strip():
        parts.append(
            "[Prior run knowledge base — reuse factual material when relevant; "
            "prefer extending or correcting over repeating verbatim unless needed for clarity.]\n"
            f"{prior_run_knowledge.strip()}\n"
        )
    parts.append(
        f"Your sub-task (id={task.id}):\nTitle: {task.title}\nDescription: {task.description}\n",
    )
    if workspace_abs:
        parts.append(
            f"Workspace (run-scoped; use run_workspace_python to create and execute Python only under this path):\n"
            f"{workspace_abs}\n"
        )
    if dependency_outputs:
        parts.append("Outputs from prerequisite sub-tasks (use as input; do not repeat verbatim unless needed):\n")
        for dep_id, dep_text in dependency_outputs:
            parts.append(f"--- From sub-task {dep_id} ---\n{dep_text}\n")
    if task.acceptance_criteria:
        parts.append(f"Acceptance criteria:\n{task.acceptance_criteria}\n")
    if previous:
        parts.append(f"Previous attempt (revise or replace):\n{previous}\n")
    if qa_feedback:
        parts.append(f"QA did not pass. Reasons and required fixes:\n{qa_feedback}\n")
    if prior_tool_evidence and str(prior_tool_evidence).strip():
        parts.append(
            "Previous attempt — tool transcript (raw tool outputs from the last run). "
            "Read QA feedback first. If the failure is formatting, structure, wording, or presentation only, "
            "fix the deliverable using this evidence and do NOT repeat the same http_fetch or run_script "
            "unless QA explicitly says facts are missing or wrong.\n\n"
            f"--- Tool transcript ---\n{prior_tool_evidence.strip()}\n"
        )
    parts.append("Provide the updated partial solution for this sub-task.")
    return "\n".join(parts)


def _qa_fail_decompose_system(max_steps: int) -> str:
    return (
        "After a failed QA review of a worker output, you decide whether the sub-task can be fixed in ONE more "
        "worker attempt, or should be split into ordered micro-steps.\n"
        "Output ONE JSON object only (no markdown).\n"
        "Schema:\n"
        "{\n"
        '  "single_shot_ok": true or false,\n'
        '  "rationale": "short text",\n'
        '  "steps": [\n'
        '    {"title":"...","description":"...","done_when":"..."}\n'
        "  ]\n"
        "}\n"
        f"If single_shot_ok is true, set steps to []. If false, provide between 2 and {max_steps} steps that "
        "together cover the parent acceptance criteria. Each step must be executable with minimal context."
    )


def _qa_fail_decompose_user(
    task: PlanTask,
    qa_failure_detail: str,
    previous_solution: str,
    tool_evidence: str,
    *,
    max_steps: int,
) -> str:
    prev = previous_solution.strip()
    if len(prev) > 10_000:
        prev = prev[:9997] + "..."
    ev = tool_evidence.strip()
    if len(ev) > 8000:
        ev = ev[:7997] + "..."
    return (
        f"Parent sub-task id={task.id}\nTitle: {task.title}\nDescription:\n{task.description}\n\n"
        f"Acceptance criteria:\n{task.acceptance_criteria or '(none)'}\n\n"
        f"QA failure summary:\n{qa_failure_detail}\n\n"
        f"Previous candidate solution (may be incomplete):\n{prev or '(empty)'}\n\n"
        f"Tool transcript from that attempt (may be empty):\n{ev or '(none)'}\n\n"
        f"Decide single_shot_ok vs micro-plan with at most {max_steps} steps. JSON only."
    )


def _qa_fail_decompose_repair(broken: str, max_steps: int) -> str:
    return (
        "The following text was invalid. Output ONE valid JSON object only, same schema: "
        '{"single_shot_ok": true or false, "rationale": "...", "steps": [...]}. '
        f"If single_shot_ok is true, steps must be []. If false, 2-{max_steps} step objects with "
        "title, description, done_when.\n\n"
        f"Broken output:\n{broken[:12000]}"
    )


def _micro_step_user(
    global_problem: str,
    parent: PlanTask,
    step_index: int,
    step_total: int,
    step: MicroDecomposeStep,
    prior_outputs: str,
    *,
    cfg: HiveConfig,
    include_original_prompt: bool,
    workspace_abs: str | None,
    prior_run_knowledge: str | None = None,
) -> str:
    problem_ctx = _subagent_overall_problem(
        global_problem, parent, include_original_prompt=include_original_prompt, cfg=cfg
    )
    parts = [
        f"Overall problem:\n{problem_ctx}\n",
    ]
    if prior_run_knowledge and str(prior_run_knowledge).strip():
        parts.append(
            "[Prior run knowledge base]\n"
            + prior_run_knowledge.strip()
            + "\n"
        )
    parts.extend(
        [
        f"Micro-plan step {step_index + 1} of {step_total} for parent sub-task id={parent.id}.\n"
        f"Parent title: {parent.title}\n",
        f"Step title: {step.title}\nObjective:\n{step.description}\n",
        f"Step completion signal:\n{step.done_when or '(satisfy relevant parent acceptance lines)'}\n",
        ]
    )
    if workspace_abs:
        parts.append(
            f"Workspace (run-scoped; use run_workspace_python only under this path):\n{workspace_abs}\n"
        )
    parts.append(
        f"Parent acceptance criteria (the merged result must still satisfy this):\n{parent.acceptance_criteria or '(none)'}\n"
    )
    po = prior_outputs.strip()
    if po:
        parts.append(f"Outputs from earlier micro-steps:\n{po}\n")
    parts.append("Produce this step's deliverable only; keep content useful for later merge.")
    return "\n".join(parts)


def _run_qa_fail_micro_worker_chain(
    st: RunState | None,
    *,
    steps: list[MicroDecomposeStep],
    user_prompt: str,
    task: PlanTask,
    include_original_prompt: bool,
    cfg: HiveConfig,
    ws_abs: str,
    selected_skills: list[SkillInfo],
    skills_by_id: dict[str, SkillInfo],
    skills_root: Path,
    workspace_root: Path,
    idempotency_cache: dict[str, Any] | None,
    client: OpenAI,
    model: str,
    tid: str,
    worker_slot: int,
    attempt_num: int,
    prior_run_knowledge: str | None = None,
) -> tuple[str, str]:
    accum_parts: list[str] = []
    te_parts: list[str] = []
    n = len(steps)
    chain_so_far = ""
    for i, step in enumerate(steps):
        if st and st.cancel_requested:
            break
        prior = chain_so_far.strip()
        if len(prior) > cfg.dependency_output_max_chars:
            prior = prior[: cfg.dependency_output_max_chars] + "\n...[truncated]"
        base_u = _micro_step_user(
            user_prompt,
            task,
            i,
            n,
            step,
            prior,
            cfg=cfg,
            include_original_prompt=include_original_prompt,
            workspace_abs=ws_abs,
            prior_run_knowledge=prior_run_knowledge,
        )
        sol, te = _hive_worker_with_tools(
            st,
            f"worker (attempt {attempt_num} micro {i + 1}/{n})",
            base_url=cfg.base_url,
            client=client,
            model=model,
            max_tokens=cfg.max_tokens,
            cfg=cfg,
            task_id=tid,
            worker_slot=worker_slot,
            base_system=_worker_system(),
            base_user=base_u,
            selected=selected_skills,
            skills_by_id=skills_by_id,
            skills_root=skills_root,
            max_tool_rounds=cfg.max_tool_rounds,
            tool_timeout_sec=cfg.tool_timeout_sec,
            workspace_root=workspace_root,
            idempotency_cache=idempotency_cache,
        )
        accum_parts.append(f"=== Micro-step {i + 1}: {step.title} ===\n{sol}\n")
        if te.strip():
            te_parts.append(f"### Micro-step {i + 1} tools\n{te.strip()}")
        chain_so_far = "\n\n".join(accum_parts)
    merged = chain_so_far.strip()
    merged_te = _clamp_transcript("\n\n".join(te_parts), cfg.qa_retry_tool_trace_max_chars)
    return merged, merged_te


def _qa_fail_decompose_llm(
    st: RunState | None,
    client: OpenAI,
    model: str,
    cfg: HiveConfig,
    task: PlanTask,
    qa_failure_detail: str,
    previous_solution: str,
    tool_evidence: str,
    tid: str,
    worker_slot: int,
) -> tuple[bool, str, list[MicroDecomposeStep]]:
    mx = cfg.qa_fail_decompose_max_steps
    user = _qa_fail_decompose_user(
        task,
        qa_failure_detail,
        previous_solution,
        tool_evidence,
        max_steps=mx,
    )
    raw = _hive_complete(
        st,
        f"qa_fail_decompose ({tid})",
        base_url=cfg.base_url,
        client=client,
        model=model,
        user=user,
        system=_qa_fail_decompose_system(mx),
        max_tokens=cfg.max_tokens,
        cfg=cfg,
        role=HiveRole.QA_FAIL_DECOMPOSE.value,
        task_id=tid,
        worker_slot=worker_slot,
    )
    if not extract_json_blob(raw):
        raw2 = _hive_complete(
            st,
            f"qa_fail_decompose_repair ({tid})",
            base_url=cfg.base_url,
            client=client,
            model=model,
            user=_qa_fail_decompose_repair(raw, mx),
            system=_qa_fail_decompose_system(mx),
            max_tokens=cfg.max_tokens,
            cfg=cfg,
            role=HiveRole.QA_FAIL_DECOMPOSE_REPAIR.value,
            task_id=tid,
            worker_slot=worker_slot,
        )
        raw = raw2
    single, rat, steps = parse_qa_fail_decompose(raw, mx)
    return single, rat, steps


def _skill_router_system() -> str:
    return (
        "You choose which skill modules this sub-task needs. Output ONE JSON object only (no markdown fences), "
        'schema: {"skills":["skill_id1",...]}. Use the minimal set; use [] if none apply. '
        "Only use skill ids from the list provided."
    )


def _skill_router_user(
    problem: str,
    task: PlanTask,
    catalog: str,
    *,
    cfg: HiveConfig,
    include_original_prompt: bool,
    prior_run_knowledge: str | None = None,
) -> str:
    problem_ctx = _subagent_overall_problem(
        problem, task, include_original_prompt=include_original_prompt, cfg=cfg
    )
    kb = ""
    if prior_run_knowledge and str(prior_run_knowledge).strip():
        kb = (
            "[Prior run knowledge base]\n"
            + prior_run_knowledge.strip()
            + "\n\n"
        )
    return (
        f"Overall problem:\n{problem_ctx}\n\n"
        + kb
        + f"Sub-task id={task.id}\nTitle: {task.title}\nDescription:\n{task.description}\n"
        f"Acceptance criteria:\n{task.acceptance_criteria or '(none)'}\n\n"
        f"Enabled skills (pick ids only from here):\n{catalog}\n\n"
        "Respond with JSON only."
    )


def parse_router_skill_ids(raw: str, allowed: set[str]) -> list[str]:
    blob = extract_json_blob(raw)
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    arr = data.get("skills")
    if not isinstance(arr, list):
        return []
    out: list[str] = []
    for x in arr:
        if isinstance(x, str) and x in allowed and x not in out:
            out.append(x)
    return out


def _worker_tools_protocol(workspace_dir_note: str = "", *, kb_root: Path | None = None) -> str:
    ws = (
        f"\nRun-scoped workspace directory (create files only here; Python runs here):\n{workspace_dir_note}\n"
        if workspace_dir_note.strip()
        else ""
    )
    kb = ""
    if kb_root is not None:
        kb = (
            "kb_list: {\"tool\":\"kb_list\",\"subdir\":\"optional/relative/subdir\",\"extensions\":[\".md\",\".txt\"],"
            '"max_entries":50} — list files under the configured user knowledge base (paths relative to KB root).\n'
            "kb_read: {\"tool\":\"kb_read\",\"path\":\"relative/path/from/kb_root.md\",\"max_chars\":20000} — read full "
            "text of one allowed file (UTF-8); max_chars is capped by the server.\n"
        )
    return (
        "\n\n## Tool use\n"
        "Tools are invoked ONLY via markdown JSON code blocks (see examples). Required fields must be present.\n"
        "http_fetch: {\"tool\":\"http_fetch\",\"method\":\"GET|POST|...\",\"url\":\"https://...\", "
        '"headers":{} (optional), "body": null or object, "idempotency_key":"optional for GET dedup"}\n'
        "run_script: {\"tool\":\"run_script\",\"skill_id\":\"skill-folder-id\",\"script\":\"scripts/foo.py\",\"args\":[\"...\"]}. "
        "CRITICAL: \"args\" are argv AFTER the script path. Many skills require a non-empty list (e.g. topic string first). "
        "Never use \"args\": [] for scripts that expect a topic or query — derive it from the sub-task.\n"
        "run_workspace_python: {\"tool\":\"run_workspace_python\",\"code\":\"...python source...\", "
        '"filename":"optional_safe_name.py", "args":[]} — writes code under the workspace and executes it.\n'
        + kb
        + ws
        + "Single tool example:\n"
        '```json\n{"tool":"http_fetch","method":"GET","url":"https://example.com"}\n```\n'
        "Multiple tools:\n"
        '```json\n{"tool_calls":[{"tool":"http_fetch","method":"POST","url":"https://example.com","body":{}}]}\n```\n'
        "Bundled script example (topic-first CLI, e.g. last30days):\n"
        '```json\n{"tool":"run_script","skill_id":"last30days-skill","script":"scripts/last30days.py",'
        '"args":["short research topic from this sub-task","--quick"]}\n```\n'
        "Workspace Python example:\n"
        '```json\n{"tool":"run_workspace_python","code":"print(1+1)","filename":"probe.py"}\n```\n'
        "After tool results are returned, continue until your partial solution is plain text with no further tool blocks."
    )


def _windows_tool_note() -> str:
    if os.name == "nt":
        return (
            "\n\nHost note: This machine is Windows. Use http_fetch for HTTP instead of curl. "
            "Shell .sh scripts are not executed; use Python scripts under scripts/ or http_fetch."
        )
    return ""


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE):
        inner = m.group(1).strip()
        if not inner:
            continue
        try:
            data = json.loads(inner)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        tc = data.get("tool_calls")
        if isinstance(tc, list):
            for x in tc:
                if isinstance(x, dict) and str(x.get("tool", "")).strip():
                    calls.append(x)
        elif str(data.get("tool", "")).strip():
            calls.append(data)
    if calls:
        return calls
    blob = extract_json_blob(text)
    if not blob:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    tc = data.get("tool_calls")
    if isinstance(tc, list):
        return [x for x in tc if isinstance(x, dict) and str(x.get("tool", "")).strip()]
    if str(data.get("tool", "")).strip():
        return [data]
    return []


def _strip_tool_fences(text: str) -> str:
    t = re.sub(r"```(?:json)?\s*[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()
    return t if t else text.strip()


def _clamp_transcript(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n…[transcript truncated]"


def _hive_worker_with_tools(
    st: RunState | None,
    attempt_label: str,
    *,
    base_url: str,
    client: OpenAI,
    model: str,
    max_tokens: int,
    cfg: HiveConfig,
    task_id: str,
    worker_slot: int | None,
    base_system: str,
    base_user: str,
    selected: list[SkillInfo],
    skills_by_id: dict[str, SkillInfo],
    skills_root: Path,
    max_tool_rounds: int,
    tool_timeout_sec: float,
    workspace_root: Path | None,
    idempotency_cache: dict[str, Any] | None,
) -> tuple[str, str]:
    ws_note = str(workspace_root.resolve()) if workspace_root is not None else ""
    evidence_parts: list[str] = []
    max_ev = cfg.qa_retry_tool_trace_max_chars

    def _record_tool_round(idx0: int, calls: list[dict[str, Any]], result_lines: list[str]) -> None:
        chunk_lines: list[str] = [f"### Tool round {idx0 + 1}"]
        for i, c in enumerate(calls):
            cs = json.dumps(c, ensure_ascii=False)
            if len(cs) > 4000:
                cs = cs[:3997] + "..."
            chunk_lines.append(f"Call {i + 1}: {cs}")
        chunk_lines.append("Results:")
        for rl in result_lines:
            cap = max(2000, max_ev // max(1, len(result_lines) * 2))
            if len(rl) > cap:
                rl = rl[: cap - 25] + "\n...[truncated]"
            chunk_lines.append(rl)
        evidence_parts.append("\n".join(chunk_lines))

    if not selected:
        plain = _hive_complete(
            st,
            attempt_label,
            base_url=base_url,
            client=client,
            model=model,
            user=base_user,
            system=base_system,
            max_tokens=max_tokens,
            cfg=cfg,
            role=HiveRole.WORKER.value,
            task_id=task_id,
            worker_slot=worker_slot,
        )
        return (plain.strip() if plain else "", "")
    bodies = "\n\n".join(f"### Skill: {s.id}\n{s.body}" for s in selected)
    script_note = ""
    for s in selected:
        if s.script_paths:
            sp = s.script_paths[:40]
            script_note += f"\n{s.id}: " + ", ".join(sp)
            if len(s.script_paths) > 40:
                script_note += ", ..."
    extra = _worker_tools_protocol(ws_note, kb_root=cfg.kb_root_resolved) + _windows_tool_note()
    full_system = (
        base_system
        + extra
        + "\n\n## Skill content\n"
        + bodies
        + (f"\n## Script paths (relative to skill folder)\n{script_note.strip()}" if script_note else "")
    )
    user = base_user
    last_out = ""
    for round_i in range(max_tool_rounds):
        if st and st.cancel_requested:
            ev = _clamp_transcript("\n\n".join(evidence_parts), max_ev)
            return (_strip_tool_fences(last_out) if last_out else "", ev)
        label = f"{attempt_label} tools r{round_i + 1}"
        last_out = _hive_complete(
            st,
            label,
            base_url=base_url,
            client=client,
            model=model,
            user=user,
            system=full_system,
            max_tokens=max_tokens,
            cfg=cfg,
            role=HiveRole.WORKER.value,
            task_id=task_id,
            worker_slot=worker_slot,
        )
        calls = _extract_tool_calls(last_out)
        if not calls:
            ev = _clamp_transcript("\n\n".join(evidence_parts), max_ev)
            return (_strip_tool_fences(last_out), ev)
        result_lines: list[str] = []
        for call in calls:
            tool_name = str(call.get("tool", "")).strip()
            res = execute_tool(
                tool_name,
                call,
                skills_by_id=skills_by_id,
                skills_root=skills_root,
                tool_timeout_sec=tool_timeout_sec,
                workspace_root=workspace_root,
                idempotency_cache=idempotency_cache,
                idempotency_max_size=cfg.http_idempotency_max_entries,
                kb_root=cfg.kb_root_resolved,
                kb_read_max_chars=cfg.kb_read_max_chars,
                kb_file_extensions=cfg.kb_file_extensions,
            )
            result_lines.append(format_tool_result(tool_name, res))
            if st:
                st.record_llm(
                    f"{attempt_label} tool",
                    json.dumps(call, ensure_ascii=False),
                    format_tool_result(tool_name, res),
                    task_id=task_id,
                    worker_slot=worker_slot,
                    role="tool",
                )
        _record_tool_round(round_i, calls, result_lines)
        user = (
            base_user
            + "\n\n---\n[Your previous reply]\n"
            + last_out
            + "\n\n[Tool results]\n"
            + "\n".join(result_lines)
            + "\n\nContinue: complete the partial solution for this sub-task. "
            "If more tools are needed, use JSON code blocks; otherwise output plain text only."
        )
    if st:
        st.log(f"{attempt_label}: max tool rounds ({max_tool_rounds}) reached; using last output.")
    ev = _clamp_transcript("\n\n".join(evidence_parts), max_ev)
    return (_strip_tool_fences(last_out), ev)


def _qa_system() -> str:
    return (
        "You are a QA reviewer. Respond with ONE JSON object only (no markdown).\n"
        "Schema:\n"
        "{\n"
        '  "pass": true or false,\n'
        '  "checklist": [{"id": 1, "met": true, "note": "short optional note"}, ...],\n'
        '  "reasons": ["..."],\n'
        '  "required_fixes": "short text if fail"\n'
        "}\n"
        "Build checklist from the numbered acceptance criteria lines: one object per line, same order, ids 1..N. "
        "Every criterion must have met true/false. pass may be true only if every checklist item has met=true "
        "and the solution satisfies the task."
    )


def _qa_system_json_retry() -> str:
    return (
        _qa_system()
        + "\n\nYour previous reply was not valid JSON. Output ONLY one JSON object, no prose, no markdown fences."
    )


def _qa_user(
    global_problem: str,
    task: PlanTask,
    candidate: str,
    *,
    cfg: HiveConfig,
    include_original_prompt: bool,
) -> str:
    problem_ctx = _subagent_overall_problem(
        global_problem, task, include_original_prompt=include_original_prompt, cfg=cfg
    )
    numbered = _numbered_acceptance_lines(task.acceptance_criteria or "")
    return (
        f"Overall problem:\n{problem_ctx}\n\n"
        f"Sub-task id={task.id} title={task.title}\n"
        f"Description:\n{task.description}\n\n"
        f"Acceptance criteria (numbered; you must mirror each line in checklist):\n{numbered}\n\n"
        f"Candidate solution:\n{candidate}\n\n"
        "Evaluate and output JSON only."
    )


def _integration_qa_system() -> str:
    return (
        "You are an integration reviewer. Given outputs from sub-tasks along one critical dependency chain, "
        "respond with ONE JSON object only (no markdown): "
        '{"pass": true or false, "issues": ["..."], "rationale": "short text", "suggested_actions": ["optional"]}. '
        "pass is true only if the chain is mutually consistent and there are no contradictions or missing hand-offs."
    )


def _integration_qa_user(problem: str, chain_parts: list[tuple[str, str]]) -> str:
    lines = [f"Original problem:\n{problem}\n", "Critical-path sub-task outputs (in order):\n"]
    for tid, text in chain_parts:
        excerpt = text[:12_000] + ("\n...[truncated]" if len(text) > 12_000 else "")
        lines.append(f"=== Task {tid} ===\n{excerpt}\n")
    lines.append("\nRespond with JSON only.")
    return "\n".join(lines)


_MERGE_LEDGER_INPUT_NOTE = (
    "[Input format: Each completed sub-task below may be a strict claims ledger — verified vs unverified claims "
    "with evidence URLs, source_type, numeric_values for pricing, and caveats. Preserve every number, date, and caveat "
    "when merging; do not treat these as a loose summary.]\n\n"
)

_MERGE_RESEARCH_STRICT_NOTE = (
    "[Research-strict mode: In the narrative body of your merged draft, treat ONLY 'Verified claims' sections "
    "as established facts. Unverified or appendix material must not be stated as certain facts.]\n\n"
)


def _ledger_source_types_hint(cfg: HiveConfig) -> str:
    return ", ".join(sorted(cfg.ledger_allowed_source_types))


def _assertion_ledger_system(cfg: HiveConfig) -> str:
    types_hint = _ledger_source_types_hint(cfg)
    type_line = "factual | pricing | opinion | code_behavior | other"
    if cfg.evidence_policy.name == "academic_strict":
        type_line += " | academic_design | academic_citation | paper_finding | academic_sources_only"
    base = (
        "You extract a strict claims ledger from ONE sub-task solution that already passed QA. "
        "This is NOT a narrative summary.\n"
        "Respond with ONE JSON object only (no markdown), exact keys:\n"
        '  "claims": [ claim objects ],\n'
        '  "unverified_claims": [ same shape as claim objects when evidence is weak or unknown ],\n'
        '  "contradictions": [ free-form strings describing tensions, or [] ]\n'
        "Each claim object MUST have:\n"
        '  "id": stable string e.g. "c1",\n'
        '  "claim": concise factual statement,\n'
        f'  "type": one of {type_line},\n'
        '  "provider": vendor or entity name or "",\n'
        '  "evidence": non-empty array of { "source_url", "source_type", "quote_or_snippet", "retrieved_at" (ISO date) },\n'
        '  "numeric_values": array of { "value", "unit", "meaning" } — for type pricing this array MUST be non-empty; '
        "if the claim text includes $, %, GB, tokens, vectors, per-million, or /mo pricing units, numeric_values MUST "
        "encode every such figure,\n"
        '  "formula": optional string — REQUIRED when the claim is an estimate (approx, estimate, ~) and lists quantities,\n'
        '  "confidence": high | medium | low,\n'
        '  "caveats": array of strings (use [] if none).\n'
        f"Allowed source_type values (use exactly these strings): {types_hint}.\n"
        "Use internal_worker_output only for evidence grounded purely in this task's worker text (quote_or_snippet required); "
        "use unknown only when the source category is genuinely unclear.\n"
        "Put any claim you cannot support with the evidence rules into unverified_claims instead of claims."
    )
    if cfg.evidence_policy.name != "academic_strict":
        return base
    return (
        base
        + "\n\n**academic_strict policy (mandatory):**\n"
        "- **academic_design**: proposal/article/paper structure, scope, or design choices you infer only from this "
        "task's worker output — evidence may be internal_worker_output only (with excerpts).\n"
        "- **academic_citation**: identifying or locating a reference — every evidence row must be "
        "source_type academic_paper or academic_metadata; each needs an https source_url; at least one URL must be a "
        "DOI (e.g. doi.org/10.…), CrossRef/OpenAlex/Semantic Scholar/arXiv (or similar metadata host), or a publisher "
        "article page (not blogs/social).\n"
        "- **paper_finding**: what an external paper reports — internal_worker_output cannot verify this; use "
        "academic_paper and/or academic_metadata with scholarly URLs; include a long excerpt (≥80 chars) from the paper "
        "or, for metadata pages, use academic_metadata with URL + excerpt (abstract/snippet) ≥32 chars.\n"
        "- **academic_sources_only**: use only when the claim is that the work uses exclusively academic references — "
        "then every evidence row must be academic_paper or academic_metadata (no internal_worker_output).\n"
        "- Use **academic_metadata** for CrossRef/OpenAlex/Semantic Scholar/arXiv metadata or abstract pages; pair with "
        "quote_or_snippet from that page.\n"
        "- Do not use internal_worker_output to substantiate external literature; keep those claims in unverified_claims "
        "unless you add proper academic evidence."
    )


def _assertion_ledger_json_retry(cfg: HiveConfig) -> str:
    return (
        _assertion_ledger_system(cfg)
        + "\n\nYour previous reply was not valid JSON with the required top-level keys claims, "
        "unverified_claims, and contradictions (arrays). Output ONLY one JSON object, no prose, no markdown fences."
    )


def _assertion_ledger_user(
    global_problem: str,
    task: PlanTask,
    solution: str,
    *,
    cfg: HiveConfig,
    include_original_prompt: bool,
) -> str:
    problem_ctx = _subagent_overall_problem(
        global_problem, task, include_original_prompt=include_original_prompt, cfg=cfg
    )
    return (
        f"Overall problem (context):\n{problem_ctx}\n\n"
        f"Sub-task id={task.id} title={task.title}\n"
        f"Description:\n{task.description}\n\n"
        f"QA-passed solution to convert into a claims ledger:\n{solution}\n\n"
        "Output JSON only."
    )


def _process_assertion_ledger_raw(
    raw: str,
    task_id: str,
    *,
    cfg: HiveConfig,
) -> tuple[str | None, ClaimValidationResult | None]:
    blob = extract_json_blob(raw)
    if not blob:
        return None, None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None, None
    doc = coerce_claims_document(data)
    if not doc:
        return None, None
    validated = validate_claims_document(
        doc,
        allowed_source_types=cfg.ledger_allowed_source_types,
        pricing_requires_url=cfg.pricing_requires_url,
        research_demote_weak=cfg.research_verified_only_final,
        evidence_policy=cfg.evidence_policy,
    )
    text = format_claims_for_merge(
        validated,
        task_id,
        research_verified_only_body=cfg.research_verified_only_final,
    )
    return (text if text.strip() else None), validated


def _build_post_qa_assertion_ledger(
    st: RunState | None,
    *,
    cfg: HiveConfig,
    base_url: str,
    client: OpenAI,
    model: str,
    user_prompt: str,
    task: PlanTask,
    solution: str,
    include_original_prompt: bool,
    task_id: str,
    worker_slot: int,
) -> tuple[str | None, ClaimValidationResult | None]:
    for qpr in range(cfg.qa_json_max_retries + 1):
        sys = _assertion_ledger_system(cfg) if qpr == 0 else _assertion_ledger_json_retry(cfg)
        raw = _hive_complete(
            st,
            f"assertion_ledger ({task_id}) parse_try_{qpr}",
            base_url=base_url,
            client=client,
            model=model,
            user=_assertion_ledger_user(
                user_prompt,
                task,
                solution,
                cfg=cfg,
                include_original_prompt=include_original_prompt,
            ),
            system=sys,
            max_tokens=cfg.max_tokens,
            cfg=cfg,
            role=HiveRole.ASSERTION_LEDGER.value,
            task_id=task_id,
            worker_slot=worker_slot,
        )
        if not extract_json_blob(raw):
            continue
        text, validated = _process_assertion_ledger_raw(raw, task_id, cfg=cfg)
        if text and validated is not None:
            return text, validated
    return None, None


def _compress_merge_system() -> str:
    return (
        "You compress long sub-task solutions for a downstream merger. For each task id, output a short faithful summary. "
        "Respond with ONE JSON object only (no markdown): "
        '{"parts":[{"task_id":"id","summary":"text"},...]}. '
        "Preserve technical facts, numbers, and decisions; drop redundancy."
    )


def _compress_merge_user(problem: str, parts: list[tuple[str, str]]) -> str:
    lines = [f"Original problem (context):\n{problem[:4000]}\n", "Task outputs to compress:\n"]
    for tid, text in parts:
        lines.append(f"--- {tid} ---\n{text}\n")
    lines.append("\nRespond with JSON only.")
    return "\n".join(lines)


def _task_qa_refinement_system() -> str:
    return (
        "You adjust ONE task definition after many failed QA reviews of worker outputs. "
        "Output a single JSON object only (no markdown fences).\n"
        "Schema:\n"
        "{\n"
        '  "adjust_task_spec": true or false,\n'
        '  "rationale": "short explanation of your decision",\n'
        '  "title": "full title text if adjust_task_spec is true (otherwise omit)",\n'
        '  "description": "full description if true (otherwise omit)",\n'
        '  "acceptance_criteria": "full acceptance criteria if true (otherwise omit)"\n'
        "}\n"
        "When adjust_task_spec is true: supply complete title, description, and acceptance_criteria "
        "(not diffs). When false: omit those three fields or leave them unchanged — requirements stay as-is.\n"
        "Set adjust_task_spec true if failures show the spec is too strict, brittle, ambiguous, contradictory, "
        "or legally impossible to satisfy (e.g. exact word limits when minor variance is acceptable — relax or clarify).\n"
        "Set adjust_task_spec false if failures reflect wrong logic, missing substance, incorrect facts, "
        "or real violations of correct requirements — do not weaken those requirements."
    )


def _task_qa_refinement_user(
    task: PlanTask,
    qa_failure_report: str,
    user_problem_summary: str,
) -> str:
    return (
        f"Original user request (for context — task was derived from this):\n{user_problem_summary}\n\n"
        f"Current task id={task.id}\nTitle: {task.title}\n"
        f"Description:\n{task.description}\n\n"
        f"Acceptance criteria:\n{task.acceptance_criteria or '(none)'}\n\n"
        "Repeated QA failures (each line is one failed review; read all of them):\n"
        f"{qa_failure_report}\n\n"
        "Decide whether to relax/clarify the task specification or keep it. Respond with JSON only."
    )


def _task_qa_refinement_repair(broken: str) -> str:
    return (
        "The following text was invalid. Output ONE valid JSON object only, same schema as before: "
        '{"adjust_task_spec": true or false, "rationale": "...", '
        'optional "title", "description", "acceptance_criteria" when adjust_task_spec is true}.\n\n'
        f"Broken output:\n{broken[:12000]}"
    )


def _aggregated_verified_claims_markdown(ex: dict[str, Any]) -> str:
    lines: list[str] = []
    raw_by_tid = ex.get("assertion_ledgers_raw") or {}
    for tid in sorted(raw_by_tid.keys()):
        doc = raw_by_tid.get(tid) or {}
        vc = doc.get("verified_claims") or []
        if not vc:
            continue
        lines.append(f"### From task `{tid}`")
        for c in vc:
            if isinstance(c, dict):
                lines.append(f"- [{c.get('id', '?')}] {c.get('claim', '')}")
    return "\n".join(lines) if lines else "(no verified claims in this run)"


def _planner_summary_user_problem(user_prompt: str) -> str:
    if len(user_prompt) <= 6000:
        return user_prompt
    return (
        user_prompt[:4500].rstrip()
        + "\n\n[... middle omitted ...]\n\n"
        + user_prompt[-1400:].lstrip()
    )


def _merger_system() -> str:
    return (
        "You are a merger agent. Combine validated partial solutions into one coherent draft that "
        "contains everything needed to answer the original request—not a task-by-task status report. "
        "Preserve concrete outcomes: numbers, conclusions, errors, file paths, and any stdout/stderr "
        "or logs from runs. If workers only delivered source code but the problem also asked for "
        "execution, tests, or observed behavior, keep whatever run output exists in the inputs; do not "
        "drop it. Remove redundancy while keeping facts that the final reader will need."
    )


def _merger_user(
    problem: str,
    parts: list[tuple[str, str]],
    *,
    merge_ledger_mode: bool = False,
    merge_research_strict: bool = False,
    ledger_resolution_messages: list[str] | None = None,
) -> str:
    prefix = ""
    if ledger_resolution_messages:
        prefix += "[Orchestrator ledger resolution — reflect faithfully in the merged draft where relevant]\n"
        for m in ledger_resolution_messages:
            prefix += f"- {m}\n"
        prefix += "\n"
    if merge_research_strict:
        prefix += _MERGE_RESEARCH_STRICT_NOTE
    if merge_ledger_mode:
        prefix += _MERGE_LEDGER_INPUT_NOTE
    lines = [prefix + f"Original problem:\n{problem}\n", "Validated partial solutions:\n"]
    for tid, text in parts:
        lines.append(f"--- Task {tid} ---\n{text}\n")
    lines.append(
        "\nProduce one merged draft: integrate all partials so a reader could answer the original problem "
        "from this text alone, including execution or test output when present in the inputs."
    )
    return "\n".join(lines)


def _checkpoint_system() -> str:
    return (
        "You review progress after a checkpoint sub-task completed. Output ONE JSON object only (no markdown).\n"
        "Schema:\n"
        "{\n"
        '  "action": "continue" | "finish_early" | "replan",\n'
        '  "rationale": "short explanation",\n'
        '  "tasks": []\n'
        "}\n"
        "Rules:\n"
        '- "continue": keep executing the remaining planned sub-tasks as-is.\n'
        '- "finish_early": the work done so far is enough; cancel not-yet-started sub-tasks and merge only completed outputs.\n'
        '- "replan": replace all not-yet-started sub-tasks. Set "tasks" to a full new array of sub-task objects with the same '
        "fields as the planner uses: id, title, description, acceptance_criteria, depends_on (ids of tasks that already "
        'finished or appear earlier in your new list), key_task (optional). Completed tasks stay in the run; your new ids must '
        "not collide with completed task ids.\n"
        "If action is continue or finish_early, use an empty array for tasks. "
        "Prefer finish_early when the original goal is satisfied. Prefer replan when the remaining steps are insufficient or wrong."
    )


def _checkpoint_user(
    problem: str,
    *,
    checkpoint_task: PlanTask,
    completed_order: list[str],
    results_by_id: dict[str, str],
    pending_tasks: list[PlanTask],
    cfg: HiveConfig,
) -> str:
    lim = cfg.checkpoint_task_output_chars
    done_lines: list[str] = []
    for tid in completed_order:
        blob = results_by_id.get(tid, "")
        excerpt = blob[:lim] if blob else ""
        if len(blob) > lim:
            excerpt += "\n...[truncated]"
        done_lines.append(f"=== Completed sub-task {tid} ===\n{excerpt}\n")
    pend_lines: list[str] = []
    for pt in pending_tasks:
        pend_lines.append(
            f"- id={pt.id} key_task={pt.key_task} depends_on={pt.depends_on}\n"
            f"  title: {pt.title}\n"
            f"  description: {pt.description}\n"
            f"  acceptance_criteria: {pt.acceptance_criteria or '(none)'}\n"
        )
    return (
        f"Original problem:\n{problem}\n\n"
        f"Checkpoint sub-task that just completed: id={checkpoint_task.id} title={checkpoint_task.title}\n"
        f"Description:\n{checkpoint_task.description}\n\n"
        "Completed sub-tasks (in order):\n"
        + "\n".join(done_lines)
        + "\nRemaining planned sub-tasks (not started yet):\n"
        + ("\n".join(pend_lines) if pend_lines else "(none)\n")
        + "\nDecide action (continue / finish_early / replan) and respond with JSON only."
    )


def _checkpoint_repair_prompt(broken: str) -> str:
    return (
        "The following text was invalid. Output ONE JSON object only, same schema: "
        '{"action":"continue"|"finish_early"|"replan","rationale":"...","tasks":[]}\n\n'
        f"Broken output:\n{broken[:12000]}"
    )


def _final_check_system() -> str:
    return (
        "You are a final consistency validator. Respond with ONE JSON object only (no markdown), "
        "exact schema:\n"
        "{\n"
        '  "pass": true,\n'
        '  "critical_issues": [],\n'
        '  "logical_inconsistencies": [],\n'
        '  "unproven_claims": [],\n'
        '  "missing_components": [],\n'
        '  "confidence_score": 0.0,\n'
        '  "requires_replan": false\n'
        "}\n"
        "Rules: Only pass if the solution is internally consistent, complete relative to the problem, "
        "and does not contain unsupported factual claims. If failure is due to foundational planning gaps, "
        'set "requires_replan": true.\n'
        "Completeness includes deliverables implied by the problem: e.g. if the user asked for code to be "
        "run (or tests executed, benchmarks, CLI output), the candidate must present actual stated output "
        "when such output exists in upstream material, or clearly state that execution output was not "
        "produced—supplying only source code is not enough in that case. List any such gap in "
        "missing_components."
    )


def _final_check_user(problem: str, candidate: str, *, deliverables_note: str = "") -> str:
    tail = ""
    if deliverables_note.strip():
        tail = f"\n{deliverables_note.strip()}\n"
    return (
        f"Original problem:\n{problem}\n\n"
        f"Candidate final result:\n{candidate}\n"
        f"{tail}\n"
        "Validate and output JSON only."
    )


def _final_fix_system() -> str:
    return (
        "You are a merger-and-fixer agent. Revise the candidate final result to resolve the reported issues. "
        "Do not introduce new unproven claims. Keep the answer complete and consistent. "
        "The text must read as a direct answer to the original user request: lead with outcomes they asked for "
        "(results, numbers, errors, run output). If execution or test output appears anywhere in the candidate "
        "or verdict context, surface it explicitly rather than leaving only code."
    )


def _final_synthesis_system(*, research_strict: bool) -> str:
    if research_strict:
        return (
            "You produce the single definitive reply in RESEARCH-STRICT mode. "
            "The merged draft may contain verified and unverified material—treat ONLY the aggregated verified "
            "claims list in the user message as facts you may state with certainty in the main body.\n"
            "Requirements:\n"
            "- Main body: only restate or synthesize verified claims; do not add new factual assertions.\n"
            "- If verified material is insufficient to answer the user, say so honestly.\n"
            "- Include a section titled exactly `## Not verified / excluded` with bullet points summarizing every "
            "item from the orchestrator \"Excluded / unverified claims digest\" in the user message (rephrase lightly "
            "for readability; do not treat them as facts). If that digest is empty or says there are none, omit this "
            "section.\n"
            "- Put anything else from the merged draft that is not verified under short hypotheses or open questions "
            "in that same section—never as established facts in the main body.\n"
            "- Never invent sources, numbers, or run output.\n"
            "Output plain text or markdown only (no JSON)."
        )
    return (
        "You produce the single definitive reply to the human who wrote the original request. "
        "The merged draft is internal material from sub-agents—not your final format.\n"
        "Requirements:\n"
        "- Answer the original ask end-to-end: conclusions, recommendations, numbers, or errors first when "
        "those are what they wanted.\n"
        "- If they asked for code to be written and executed (or tested), include real execution or test "
        "output when it appears in the merged draft (stdout/stderr, exit codes, sample rows). If that output "
        "is missing from the draft, say honestly that it was not produced upstream—never invent run output.\n"
        "- Do not treat delivering source code alone as sufficient when they explicitly asked for a run or "
        "observed behavior.\n"
        "- Prefer one coherent narrative; use short sections only if they help. Drop procedural chatter about "
        "tasks unless the user asked for process.\n"
        "Output plain text or markdown only (no JSON)."
    )


def _final_synthesis_user(
    problem: str,
    merged_draft: str,
    *,
    merge_ledger_mode: bool = False,
    research_strict: bool = False,
    verified_digest: str = "",
    excluded_digest: str = "",
) -> str:
    max_chars = 120_000
    draft = merged_draft
    if len(draft) > max_chars:
        half = max_chars // 2
        draft = (
            draft[:half].rstrip()
            + "\n\n[... middle of merged draft omitted for length ...]\n\n"
            + draft[-half:].lstrip()
        )
    prefix = ""
    if research_strict:
        prefix += _MERGE_RESEARCH_STRICT_NOTE
    if merge_ledger_mode:
        prefix += _MERGE_LEDGER_INPUT_NOTE
    vd = ""
    if research_strict and verified_digest.strip():
        vd = (
            "\n## Aggregated verified claims (only these may be stated as facts in the main answer)\n"
            f"{verified_digest.strip()}\n\n"
        )
    excl = ""
    if research_strict and merge_ledger_mode and excluded_digest.strip():
        excl = (
            "\n## Excluded / unverified claims digest (orchestrator-validated; for `## Not verified / excluded` only)\n"
            f"{excluded_digest.strip()}\n\n"
        )
    return (
        f"Original request (you must fully satisfy this as far as the evidence allows):\n{problem}\n\n"
        + prefix
        + vd
        + excl
        + f"Merged draft from completed sub-tasks:\n{draft}\n\n"
        "Write the final user-facing answer now."
    )


def _final_fix_user(problem: str, candidate: str, verdict: dict[str, Any]) -> str:
    return (
        f"Original problem:\n{problem}\n\n"
        f"Candidate final result:\n{candidate}\n\n"
        f"Final consistency verdict (must fix):\n{json.dumps(verdict, ensure_ascii=False)}\n\n"
        "Return the revised final result text only."
    )


def _slug_from_prompt(prompt: str, max_len: int = 48) -> str:
    first = (prompt.strip().splitlines() or [""])[0][:120]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", first.strip().lower()).strip("-")
    return (slug[:max_len] or "run").rstrip("-")


def write_hive_result_file(directory: str, user_prompt: str, body: str) -> str:
    """Write final merged result to UTF-8 text; return absolute path."""
    os.makedirs(directory, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug_from_prompt(user_prompt)
    name = f"hive_result_{ts}_{slug}.txt"
    path = os.path.abspath(os.path.join(directory, name))
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    header = (
        "# Agent Hive — final output\n"
        f"# Saved (local time): {stamp}\n\n"
        "## Original problem\n\n"
        f"{user_prompt}\n\n"
        "---\n\n"
        "## Final result\n\n"
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")
    return path


def write_hive_claims_report_file(directory: str, user_prompt: str, report: dict[str, Any]) -> str:
    """Write claims run report JSON; return absolute path."""
    os.makedirs(directory, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug_from_prompt(user_prompt)
    name = f"hive_claims_{ts}_{slug}.json"
    path = os.path.abspath(os.path.join(directory, name))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
    return path


def _parse_kb_extensions_csv(csv: str) -> frozenset[str]:
    out: set[str] = set()
    for part in (csv or "").split(","):
        p = part.strip().lower()
        if not p:
            continue
        if not p.startswith("."):
            p = "." + p.lstrip(".")
        out.add(p)
    return frozenset(out)


def _list_kb_files(root: Path, allowed_exts: frozenset[str], max_files: int) -> list[Path]:
    try:
        root = root.resolve()
    except OSError:
        return []
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        for p in sorted(root.rglob("*"), key=lambda x: str(x).lower()):
            if len(out) >= max_files:
                break
            try:
                if p.is_symlink():
                    continue
                rp = p.resolve()
                if not rp.is_relative_to(root):
                    continue
            except (OSError, ValueError, RuntimeError):
                continue
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
            out.append(p)
    except OSError:
        return []
    return out


class HiveConfig:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        max_tokens_cap: int | None = None,
        max_qa_retries: int | None = None,
        max_parallel_workers: int | None = None,
        max_parallel_workers_cap: int | None = None,
        auto_parallel_workers: bool | None = None,
        output_dir: str | None = None,
        save_final_to_file: bool | None = None,
        final_check_attempts: int | None = None,
        replan_attempts: int | None = None,
        skills_dir: str | None = None,
        enabled_state_path: str | None = None,
        enabled_skill_ids: set[str] | None = None,
        max_tool_rounds: int | None = None,
        tool_timeout_sec: float | None = None,
        qa_refinement_after_failures: int | None = None,
        max_checkpoint_replans: int | None = None,
        role_temperatures: dict[str, float] | None = None,
        role_max_tokens_factors: dict[str, float] | None = None,
        max_plan_tasks: int | None = None,
        max_plan_depth_hint: int | None = None,
        min_task_description_chars: int | None = None,
        min_acceptance_criteria_chars: int | None = None,
        max_planner_critique_rounds: int | None = None,
        use_two_phase_planner: bool | None = None,
        qa_json_max_retries: int | None = None,
        integration_qa_enabled: bool | None = None,
        integration_qa_strict: bool | None = None,
        dependency_summary_max_chars: int | None = None,
        summarize_deps_llm_threshold: int | None = None,
        merge_compress_threshold_chars: int | None = None,
        merge_compress_max_chunks: int | None = None,
        merge_strategy: str | None = None,
        post_qa_assertion_ledger: bool | None = None,
        research_verified_only_final: bool | None = None,
        evidence_policy_preset: str | None = None,
        ledger_allowed_source_types_csv: str | None = None,
        ledger_conflict_resolution: bool | None = None,
        required_deliverables_raw: str | None = None,
        save_claims_report: bool | None = None,
        pricing_requires_url: bool | None = None,
        claims_report_snapshot_max_claims: int | None = None,
        final_synthesis_enabled: bool | None = None,
        max_run_wall_seconds: float | None = None,
        max_estimated_tokens_per_run: int | None = None,
        lm_retry_max: int | None = None,
        lm_retry_base_delay_sec: float | None = None,
        lm_retry_max_delay_sec: float | None = None,
        http_timeout_sec: float | None = None,
        http_idempotency_max_entries: int | None = None,
        workspace_parent_dir: str | None = None,
        dependency_output_max_chars: int | None = None,
        checkpoint_task_output_chars: int | None = None,
        subagent_problem_verbatim_max: int | None = None,
        subagent_compressed_body_max: int | None = None,
        subagent_head_chars: int | None = None,
        subagent_tail_chars: int | None = None,
        qa_retry_tool_trace: bool | None = None,
        qa_retry_tool_trace_max_chars: int | None = None,
        qa_fail_decompose_enabled: bool | None = None,
        qa_fail_decompose_max_steps: int | None = None,
        qa_fail_decompose_min_attempt: int | None = None,
        replan_carryover_enabled: bool | None = None,
        replan_knowledge_max_chars: int | None = None,
        replan_share_workspace: bool | None = None,
        replan_carryover_task_max_chars: int | None = None,
        replan_carryover_merged_max_chars: int | None = None,
        kb_dir: str | None = None,
        kb_index_max_chars: int | None = None,
        kb_index_per_file_head_chars: int | None = None,
        kb_read_max_chars: int | None = None,
        kb_file_extensions: str | None = None,
        kb_max_files: int | None = None,
    ) -> None:
        e = _env
        self.base_url = base_url if base_url is not None else e.base_url
        self.api_key = api_key if api_key is not None else e.api_key
        self.model_override = model if model is not None else e.model
        self.max_tokens = int(max_tokens if max_tokens is not None else e.max_tokens)
        mcap = int(max_tokens_cap if max_tokens_cap is not None else e.max_tokens_cap)
        self.max_tokens_cap = max(256, mcap)
        self.max_qa_retries = int(max_qa_retries if max_qa_retries is not None else e.max_qa_retries)
        self.max_parallel_workers = max(1, int(max_parallel_workers if max_parallel_workers is not None else e.max_parallel_workers))
        self.max_parallel_workers_cap = max(
            1, int(max_parallel_workers_cap if max_parallel_workers_cap is not None else e.max_parallel_workers_cap)
        )
        self.auto_parallel_workers = bool(e.auto_parallel_workers if auto_parallel_workers is None else auto_parallel_workers)
        self.output_dir = output_dir if output_dir is not None else e.output_dir
        self.save_final_to_file = bool(e.save_final_to_file if save_final_to_file is None else save_final_to_file)
        self.final_check_attempts = max(1, int(final_check_attempts if final_check_attempts is not None else e.final_check_attempts))
        self.replan_attempts = max(0, int(replan_attempts if replan_attempts is not None else e.replan_attempts))
        self.skills_dir = skills_dir
        self.enabled_state_path = enabled_state_path
        self.enabled_skill_ids = enabled_skill_ids
        self.max_tool_rounds = max(1, int(max_tool_rounds if max_tool_rounds is not None else e.max_tool_rounds))
        self.tool_timeout_sec = max(5.0, float(tool_timeout_sec if tool_timeout_sec is not None else e.tool_timeout_sec))
        self.qa_refinement_after_failures = max(
            0, int(qa_refinement_after_failures if qa_refinement_after_failures is not None else e.qa_refinement_after_failures)
        )
        self.max_checkpoint_replans = max(
            0, int(max_checkpoint_replans if max_checkpoint_replans is not None else e.max_checkpoint_replans)
        )
        rt = default_role_temperatures()
        if role_temperatures:
            rt.update(role_temperatures)
        self.role_temperatures = rt
        rf = default_role_max_tokens_factor()
        if role_max_tokens_factors:
            rf.update(role_max_tokens_factors)
        self.role_max_tokens_factors = rf
        self.max_plan_tasks = max(1, int(max_plan_tasks if max_plan_tasks is not None else e.max_plan_tasks))
        self.max_plan_depth_hint = max(1, int(max_plan_depth_hint if max_plan_depth_hint is not None else e.max_plan_depth_hint))
        self.min_task_description_chars = max(
            1, int(min_task_description_chars if min_task_description_chars is not None else e.min_task_description_chars)
        )
        self.min_acceptance_criteria_chars = max(
            1,
            int(min_acceptance_criteria_chars if min_acceptance_criteria_chars is not None else e.min_acceptance_criteria_chars),
        )
        self.max_planner_critique_rounds = max(
            0, int(max_planner_critique_rounds if max_planner_critique_rounds is not None else e.max_planner_critique_rounds)
        )
        self.use_two_phase_planner = bool(e.use_two_phase_planner if use_two_phase_planner is None else use_two_phase_planner)
        self.qa_json_max_retries = max(0, int(qa_json_max_retries if qa_json_max_retries is not None else e.qa_json_max_retries))
        self.integration_qa_enabled = bool(
            e.integration_qa_enabled if integration_qa_enabled is None else integration_qa_enabled
        )
        self.integration_qa_strict = bool(e.integration_qa_strict if integration_qa_strict is None else integration_qa_strict)
        self.dependency_summary_max_chars = max(
            500, int(dependency_summary_max_chars if dependency_summary_max_chars is not None else e.dependency_summary_max_chars)
        )
        self.summarize_deps_llm_threshold = max(
            1000,
            int(summarize_deps_llm_threshold if summarize_deps_llm_threshold is not None else e.summarize_deps_llm_threshold),
        )
        self.merge_compress_threshold_chars = max(
            5000,
            int(merge_compress_threshold_chars if merge_compress_threshold_chars is not None else e.merger_input_threshold_chars),
        )
        self.merge_compress_max_chunks = max(
            1, int(merge_compress_max_chunks if merge_compress_max_chunks is not None else e.merge_compress_max_chunks)
        )
        ms = merge_strategy if merge_strategy is not None else e.merge_strategy
        self.merge_strategy = ms if ms in ("single_pass", "hierarchical") else "single_pass"
        self.post_qa_assertion_ledger = bool(
            e.post_qa_assertion_ledger if post_qa_assertion_ledger is None else post_qa_assertion_ledger
        )
        ep_name = (evidence_policy_preset if evidence_policy_preset is not None else e.evidence_policy).strip().lower()
        self.evidence_policy = resolve_evidence_policy(ep_name or "normal")
        csv_src = (
            e.ledger_allowed_source_types_csv
            if ledger_allowed_source_types_csv is None
            else ledger_allowed_source_types_csv
        )
        if str(csv_src or "").strip():
            self.ledger_allowed_source_types = parse_allowed_source_types_csv(csv_src)
        else:
            self.ledger_allowed_source_types = frozenset(self.evidence_policy.allowed_source_types)
        self.ledger_conflict_resolution = bool(
            e.ledger_conflict_resolution if ledger_conflict_resolution is None else ledger_conflict_resolution
        )
        self.required_deliverables_raw = (
            e.required_deliverables_raw if required_deliverables_raw is None else required_deliverables_raw
        )
        self.save_claims_report = bool(e.save_claims_report if save_claims_report is None else save_claims_report)
        self.pricing_requires_url = bool(e.pricing_requires_url if pricing_requires_url is None else pricing_requires_url)
        self.claims_report_snapshot_max_claims = max(
            1,
            int(
                claims_report_snapshot_max_claims
                if claims_report_snapshot_max_claims is not None
                else e.claims_report_snapshot_max_claims
            ),
        )
        rv = bool(
            e.research_verified_only_final if research_verified_only_final is None else research_verified_only_final
        )
        self.research_verified_only_final = rv and self.post_qa_assertion_ledger
        self.final_synthesis_enabled = bool(
            e.final_synthesis_enabled if final_synthesis_enabled is None else final_synthesis_enabled
        )
        wall = max_run_wall_seconds if max_run_wall_seconds is not None else e.max_run_wall_seconds
        self.max_run_wall_seconds = float(wall) if wall is not None and float(wall) > 0 else None
        toks = max_estimated_tokens_per_run if max_estimated_tokens_per_run is not None else e.max_estimated_tokens_per_run
        self.max_estimated_tokens_per_run = int(toks) if toks is not None and int(toks) > 0 else None
        self.lm_retry_max = max(1, int(lm_retry_max if lm_retry_max is not None else e.lm_retry_max))
        self.lm_retry_base_delay_sec = max(
            0.01, float(lm_retry_base_delay_sec if lm_retry_base_delay_sec is not None else e.lm_retry_base_delay_sec)
        )
        self.lm_retry_max_delay_sec = max(
            self.lm_retry_base_delay_sec,
            float(lm_retry_max_delay_sec if lm_retry_max_delay_sec is not None else e.lm_retry_max_delay_sec),
        )
        self.http_timeout_sec = max(1.0, float(http_timeout_sec if http_timeout_sec is not None else e.http_timeout_sec))
        self.http_idempotency_max_entries = max(
            8, int(http_idempotency_max_entries if http_idempotency_max_entries is not None else e.http_idempotency_max_entries)
        )
        wp = workspace_parent_dir if workspace_parent_dir is not None else e.workspace_parent_dir
        self.workspace_parent_dir = wp or "workspace"
        self.dependency_output_max_chars = max(
            500, int(dependency_output_max_chars if dependency_output_max_chars is not None else e.dependency_output_max_chars)
        )
        self.checkpoint_task_output_chars = max(
            200, int(checkpoint_task_output_chars if checkpoint_task_output_chars is not None else e.checkpoint_task_output_chars)
        )
        self.subagent_problem_verbatim_max = max(
            500,
            int(subagent_problem_verbatim_max if subagent_problem_verbatim_max is not None else e.subagent_problem_verbatim_max),
        )
        self.subagent_compressed_body_max = max(
            500,
            int(subagent_compressed_body_max if subagent_compressed_body_max is not None else e.subagent_compressed_body_max),
        )
        self.subagent_head_chars = max(100, int(subagent_head_chars if subagent_head_chars is not None else e.subagent_head_chars))
        self.subagent_tail_chars = max(100, int(subagent_tail_chars if subagent_tail_chars is not None else e.subagent_tail_chars))
        self.qa_retry_tool_trace = bool(e.qa_retry_tool_trace if qa_retry_tool_trace is None else qa_retry_tool_trace)
        self.qa_retry_tool_trace_max_chars = max(
            2000,
            int(
                qa_retry_tool_trace_max_chars
                if qa_retry_tool_trace_max_chars is not None
                else e.qa_retry_tool_trace_max_chars
            ),
        )
        self.qa_fail_decompose_enabled = bool(
            e.qa_fail_decompose_enabled if qa_fail_decompose_enabled is None else qa_fail_decompose_enabled
        )
        self.qa_fail_decompose_max_steps = min(
            24,
            max(
                2,
                int(
                    qa_fail_decompose_max_steps
                    if qa_fail_decompose_max_steps is not None
                    else e.qa_fail_decompose_max_steps
                ),
            ),
        )
        self.qa_fail_decompose_min_attempt = max(
            1,
            int(
                qa_fail_decompose_min_attempt
                if qa_fail_decompose_min_attempt is not None
                else e.qa_fail_decompose_min_attempt
            ),
        )
        self.replan_carryover_enabled = bool(
            e.replan_carryover_enabled if replan_carryover_enabled is None else replan_carryover_enabled
        )
        self.replan_knowledge_max_chars = max(
            4000,
            int(replan_knowledge_max_chars if replan_knowledge_max_chars is not None else e.replan_knowledge_max_chars),
        )
        self.replan_share_workspace = bool(
            e.replan_share_workspace if replan_share_workspace is None else replan_share_workspace
        )
        self.replan_carryover_task_max_chars = max(
            2000,
            int(
                replan_carryover_task_max_chars
                if replan_carryover_task_max_chars is not None
                else e.replan_carryover_task_max_chars
            ),
        )
        self.replan_carryover_merged_max_chars = max(
            4000,
            int(
                replan_carryover_merged_max_chars
                if replan_carryover_merged_max_chars is not None
                else e.replan_carryover_merged_max_chars
            ),
        )
        self.kb_dir = (kb_dir if kb_dir is not None else e.kb_dir) or ""
        self.kb_index_max_chars = max(
            500,
            int(kb_index_max_chars if kb_index_max_chars is not None else e.kb_index_max_chars),
        )
        self.kb_index_per_file_head_chars = max(
            50,
            int(
                kb_index_per_file_head_chars
                if kb_index_per_file_head_chars is not None
                else e.kb_index_per_file_head_chars
            ),
        )
        self.kb_read_max_chars = max(
            1000,
            int(kb_read_max_chars if kb_read_max_chars is not None else e.kb_read_max_chars),
        )
        kb_ext_csv = kb_file_extensions if kb_file_extensions is not None else e.kb_file_extensions
        self.kb_file_extensions = _parse_kb_extensions_csv(str(kb_ext_csv or ""))
        if not self.kb_file_extensions:
            self.kb_file_extensions = frozenset(
                {".md", ".txt", ".rst", ".html", ".htm", ".json", ".yaml", ".yml", ".csv", ".tsv"}
            )
        self.kb_max_files = max(1, int(kb_max_files if kb_max_files is not None else e.kb_max_files))
        self.kb_root_resolved: Path | None = None
        kd = (self.kb_dir or "").strip()
        if kd:
            try:
                p = Path(kd).expanduser().resolve(strict=False)
                if p.is_dir():
                    self.kb_root_resolved = p
            except OSError:
                self.kb_root_resolved = None

    def resolved_worker_count(self) -> int:
        cap = max(1, min(self.max_parallel_workers_cap, 64))
        if self.auto_parallel_workers:
            cpu = os.cpu_count() or 4
            return max(1, min(cap, max(1, cpu)))
        return max(1, min(cap, self.max_parallel_workers))


def compress_problem_for_subtask(original: str, task: PlanTask, cfg: HiveConfig) -> str:
    """Shrink a long user prompt for worker/QA/router: head, tail, and task-relevant excerpts."""
    if len(original) <= cfg.subagent_problem_verbatim_max:
        return original
    keywords = _task_keyword_tokens(task)
    head = original[: cfg.subagent_head_chars].strip()
    tail = (
        original[-cfg.subagent_tail_chars :].strip()
        if len(original) > cfg.subagent_head_chars + cfg.subagent_tail_chars
        else ""
    )
    note = (
        f"[Problem text shortened from {len(original)} characters: beginning and ending are included, "
        "plus excerpts that overlap this sub-task. The planner used the full prompt to build the task list.]\n\n"
    )
    paras = _problem_paragraphs(original)
    scored = [(_score_text_vs_keywords(p, keywords), p) for p in paras]
    scored.sort(key=lambda x: -x[0])
    picked: list[str] = []
    overhead = len(note) + len(head) + len(tail) + 200
    used = overhead
    cmax = cfg.subagent_compressed_body_max
    for score, p in scored:
        if score <= 0:
            break
        if p == head or p == tail:
            continue
        frag = p if len(p) <= 1200 else p[:1200] + "…"
        if used + len(frag) + 12 > cmax:
            if len(picked) >= 2:
                break
            room = cmax - used - 12
            if room > 240:
                picked.append(frag[:room] + "…")
            break
        picked.append(frag)
        used += len(frag) + 12
    if not picked and len(paras) > 2:
        step = max(1, len(paras) // 4)
        for i in range(step, len(paras) - 1, step):
            if len(picked) >= 3:
                break
            p = paras[i]
            if p == head or p == tail:
                continue
            frag = p if len(p) <= 1000 else p[:1000] + "…"
            if used + len(frag) + 12 > cmax:
                break
            picked.append(frag)
            used += len(frag) + 12
    parts: list[str] = [note, "## Original problem — beginning\n", head]
    if picked:
        parts.append("\n\n## Original problem — excerpts relevant to this sub-task\n")
        parts.append("\n\n---\n\n".join(picked))
    if tail and tail != head:
        parts.append("\n\n## Original problem — ending\n")
        parts.append(tail)
    return "".join(parts)


def _subagent_overall_problem(
    original: str,
    task: PlanTask,
    *,
    include_original_prompt: bool,
    cfg: HiveConfig,
) -> str:
    """Text under 'Overall problem' for worker / router / QA."""
    if include_original_prompt:
        return compress_problem_for_subtask(original, task, cfg)
    return (
        "The raw user prompt is omitted here (planner set include_original_prompt to false). "
        "Use only the task fields in this message (title, description, acceptance_criteria)—they must contain "
        "everything needed to complete the sub-task.\n"
    )


def _format_llm_io_log(system: str | None, user: str) -> str:
    chunks: list[str] = []
    if system and system.strip():
        chunks.append(f"[system]\n{system.strip()}")
    chunks.append(f"[user]\n{user}")
    return "\n\n".join(chunks)


def _effective_max_tokens_for_role(cfg: HiveConfig, role: str, base_max: int) -> int:
    fac = float(cfg.role_max_tokens_factors.get(role, 1.0))
    mt = int(base_max * fac)
    return max(256, min(mt, cfg.max_tokens_cap))


def _hive_complete(
    st: RunState | None,
    label: str,
    *,
    base_url: str,
    client: OpenAI,
    model: str,
    user: str,
    system: str | None = None,
    max_tokens: int,
    cfg: HiveConfig,
    role: str = HiveRole.GENERIC.value,
    task_id: str | None = None,
    worker_slot: int | None = None,
) -> str:
    prompt_text = _format_llm_io_log(system, user)
    eff_max = _effective_max_tokens_for_role(cfg, role, max_tokens)
    temp = cfg.role_temperatures.get(role)
    if temp is None:
        temp = default_role_temperatures().get(role, 0.5)
    out = complete_with_retries(
        base_url,
        client,
        model,
        user=user,
        system=system,
        max_tokens=eff_max,
        temperature=float(temp),
        max_retries=cfg.lm_retry_max,
        base_delay_sec=cfg.lm_retry_base_delay_sec,
        max_delay_sec=cfg.lm_retry_max_delay_sec,
        http_timeout_sec=cfg.http_timeout_sec,
    )
    if st:
        st.record_llm(
            label,
            prompt_text,
            out,
            task_id=task_id,
            worker_slot=worker_slot,
            role=role,
        )
        if cfg.max_estimated_tokens_per_run is not None:
            if int(st.metrics.get("tokens_estimated_cumulative", 0)) > cfg.max_estimated_tokens_per_run:
                st.abort_run("Exceeded max_estimated_tokens_per_run budget.")
                raise RuntimeError("Exceeded max_estimated_tokens_per_run budget.")
        if cfg.max_run_wall_seconds is not None:
            if (time.monotonic() - st.run_started_monotonic) > cfg.max_run_wall_seconds:
                st.abort_run("Exceeded max_run_wall_seconds budget.")
                raise RuntimeError("Exceeded max_run_wall_seconds budget.")
    return out


def _parse_compress_merge_output(raw: str) -> dict[str, str]:
    blob = extract_json_blob(raw)
    if not blob:
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    parts = data.get("parts")
    if not isinstance(parts, list):
        return {}
    out: dict[str, str] = {}
    for p in parts:
        if not isinstance(p, dict):
            continue
        tid = str(p.get("task_id", p.get("id", ""))).strip()
        summ = str(p.get("summary", "")).strip()
        if tid and summ:
            out[tid] = summ
    return out


def _maybe_compress_ordered_parts(
    st: RunState | None,
    *,
    cfg: HiveConfig,
    base_url: str,
    client: OpenAI,
    model: str,
    user_prompt: str,
    ordered_parts: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    total = sum(len(t) for _, t in ordered_parts)
    if total <= cfg.merge_compress_threshold_chars:
        return ordered_parts
    chunks: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    cur_len = 0
    max_chunk = max(20_000, cfg.merge_compress_threshold_chars // max(1, cfg.merge_compress_max_chunks))
    for pair in ordered_parts:
        ln = len(pair[1])
        if cur and cur_len + ln > max_chunk:
            chunks.append(cur)
            cur = []
            cur_len = 0
        cur.append(pair)
        cur_len += ln
    if cur:
        chunks.append(cur)
    out_map: dict[str, str] = {}
    for chunk in chunks[: cfg.merge_compress_max_chunks]:
        raw = _hive_complete(
            st,
            "compress_for_merge",
            base_url=base_url,
            client=client,
            model=model,
            user=_compress_merge_user(user_prompt, chunk),
            system=_compress_merge_system(),
            max_tokens=cfg.max_tokens,
            cfg=cfg,
            role=HiveRole.COMPRESS_MERGE.value,
        )
        out_map.update(_parse_compress_merge_output(raw))
    merged_list: list[tuple[str, str]] = []
    for tid, text in ordered_parts:
        merged_list.append((tid, out_map.get(tid, text)))
    return merged_list


def _hierarchical_merge(
    st: RunState | None,
    *,
    cfg: HiveConfig,
    base_url: str,
    client: OpenAI,
    model: str,
    user_prompt: str,
    ordered_parts: list[tuple[str, str]],
    merge_ledger_mode: bool = False,
    merge_research_strict: bool = False,
    ledger_resolution_messages: list[str] | None = None,
) -> str:
    parts = list(ordered_parts)
    chunk_size = max(2, min(5, len(parts) // 2 or 2))
    while len(parts) > 1:
        next_level: list[tuple[str, str]] = []
        for i in range(0, len(parts), chunk_size):
            batch = parts[i : i + chunk_size]
            label = "merger_hierarchical_" + "_".join(t[0] for t in batch)
            merged_chunk = _hive_complete(
                st,
                label,
                base_url=base_url,
                client=client,
                model=model,
                user=_merger_user(
                    user_prompt,
                    batch,
                    merge_ledger_mode=merge_ledger_mode,
                    merge_research_strict=merge_research_strict,
                    ledger_resolution_messages=ledger_resolution_messages,
                ),
                system=_merger_system(),
                max_tokens=cfg.max_tokens,
                cfg=cfg,
                role=HiveRole.MERGER.value,
            )
            new_id = "+".join(t[0] for t in batch)
            next_level.append((new_id, merged_chunk))
        parts = next_level
    return parts[0][1]


def _write_task_artifact(
    artifacts_dir: Path,
    task_id: str,
    *,
    status: str,
    attempt: int,
    solution_excerpt: str,
    qa_note: str = "",
) -> None:
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / f"task_{task_id}_artifact.json"
        payload = {
            "task_id": task_id,
            "status": status,
            "attempt": attempt,
            "qa_note": qa_note,
            "solution_excerpt": solution_excerpt[:80_000],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _carryover_session_root(output_dir: str, session_id: str) -> Path:
    return Path(output_dir).resolve() / "run_carryover" / session_id


def _clamp_carryover_prompt_body(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text.strip():
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    sep = "\n\n...[middle omitted for token budget]...\n\n"
    overhead = len(sep) + 8
    body = max(120, max_chars - overhead)
    head = max(60, (body * 2) // 3)
    tail = body - head
    if tail < 40:
        tail = 40
        head = max(60, body - tail)
    if head + len(sep) + tail > max_chars:
        room = max_chars - len(sep) - 8
        head = max(40, room // 2)
        tail = max(40, room - head)
    return t[:head] + sep + t[-tail:]


def _build_kb_index_digest(
    root: Path,
    files: list[Path],
    *,
    per_file_head: int,
    total_cap: int,
) -> str:
    try:
        root = root.resolve()
    except OSError:
        return ""
    lines: list[str] = []
    for p in files:
        try:
            rel = p.resolve().relative_to(root)
        except ValueError:
            rel_s = p.name
        else:
            rel_s = rel.as_posix()
        head = ""
        try:
            if p.is_symlink():
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
            head = raw[:per_file_head] if per_file_head > 0 else ""
        except OSError:
            head = "(unreadable)"
        lines.append(f"### {rel_s}\n{head}\n")
    blob = "\n".join(lines).strip()
    if total_cap <= 0:
        return blob
    return _clamp_carryover_prompt_body(blob, total_cap)


def _read_carryover_digest_for_cycle(knowledge_dir: Path, *, cfg: HiveConfig) -> str:
    if not cfg.replan_carryover_enabled:
        return ""
    latest = knowledge_dir / "LATEST.md"
    if not latest.is_file():
        return ""
    try:
        raw = latest.read_text(encoding="utf-8")
    except OSError:
        return ""
    return _clamp_carryover_prompt_body(raw, cfg.replan_knowledge_max_chars)


def _trunc_for_carryover(s: str, lim: int) -> str:
    s = s or ""
    if len(s) <= lim:
        return s
    return s[:lim] + "\n...[truncated]..."


def _write_replan_carryover(
    knowledge_dir: Path,
    *,
    attempt_num: int,
    completed_order: list[str],
    results: dict[str, str],
    merged_draft: str,
    final_candidate: str,
    verdict: dict[str, Any],
    claims_report: dict[str, Any],
    cfg: HiveConfig,
    slog: Callable[[str], None],
) -> None:
    if not cfg.replan_carryover_enabled:
        return
    try:
        knowledge_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        slog(f"Replan carryover: could not create knowledge dir ({err})")
        return

    tlim = cfg.replan_carryover_task_max_chars
    mlim = cfg.replan_carryover_merged_max_chars

    md_lines: list[str] = [
        f"# Replan carryover snapshot (outer attempt {attempt_num})",
        "",
        "## Final consistency verdict",
        "```json",
        json.dumps(verdict, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Merged draft (excerpt)",
        _trunc_for_carryover(merged_draft, mlim),
        "",
        "## Final candidate (excerpt)",
        _trunc_for_carryover(final_candidate, mlim),
        "",
        "## Per-task outputs (completion order)",
    ]
    task_blob: dict[str, str] = {}
    for tid in completed_order:
        raw = results.get(tid, "") or ""
        ex = _trunc_for_carryover(raw, tlim)
        task_blob[tid] = ex
        md_lines.extend([f"### Sub-task `{tid}`", "", ex, ""])

    claims_rows = list((claims_report or {}).get("claims") or [])
    cap = min(len(claims_rows), max(40, cfg.claims_report_snapshot_max_claims))
    claims_slice = claims_rows[:cap]
    md_lines.extend(
        [
            "## Claims report (excerpt)",
            "```json",
            json.dumps(claims_slice, ensure_ascii=False, indent=2),
            "```",
        ]
    )

    md_text = "\n".join(md_lines)

    json_payload: dict[str, Any] = {
        "attempt": attempt_num,
        "completed_order": list(completed_order),
        "verdict": verdict,
        "merged_draft_excerpt": _trunc_for_carryover(merged_draft, mlim),
        "final_candidate_excerpt": _trunc_for_carryover(final_candidate, mlim),
        "task_outputs": task_blob,
        "claims_excerpt": claims_slice,
    }

    base = knowledge_dir / f"attempt_{attempt_num}"
    try:
        base.with_suffix(".md").write_text(md_text, encoding="utf-8")
        base.with_suffix(".json").write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (knowledge_dir / "LATEST.md").write_text(md_text, encoding="utf-8")
        slog(f"Replan carryover written under {knowledge_dir} (attempt_{attempt_num}.*, LATEST.md)")
    except OSError as err:
        slog(f"Replan carryover: write failed ({err})")


def run_hive(
    user_prompt: str,
    state: RunState | None = None,
    config: HiveConfig | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> str:
    """Run full pipeline; return final user-facing answer (merge + optional synthesis) or raise."""
    cfg = config or HiveConfig()
    log = print_fn or (lambda s: None)
    st = state

    def slog(msg: str) -> None:
        log(msg)
        if st:
            st.log(msg)

    client = make_openai_client(cfg.base_url, cfg.api_key)
    model = cfg.model_override or get_first_model_id(client)

    skills_root = Path(cfg.skills_dir).resolve() if cfg.skills_dir else default_skills_dir().resolve()
    enabled_path = Path(cfg.enabled_state_path).resolve() if cfg.enabled_state_path else default_enabled_path()
    discovered = discover_skills(skills_root)
    stored = load_enabled_map(enabled_path)
    merged_enabled = merge_enabled_with_discovery(discovered, stored)
    if cfg.enabled_skill_ids is not None:
        enabled_ids = set(cfg.enabled_skill_ids)
    else:
        enabled_ids = {k for k, v in merged_enabled.items() if v}
    skills_by_id: dict[str, SkillInfo] = {s.id: s for s in discovered}

    original_prompt = user_prompt
    replan_used = 0
    n_workers = cfg.resolved_worker_count()
    persistent_deliverables: list[str] = parse_deliverables_from_env(cfg.required_deliverables_raw)

    if st and st.cancel_requested:
        raise RuntimeError("Cancelled before start.")

    carryover_session_id = uuid.uuid4().hex[:12]
    carryover_root = _carryover_session_root(cfg.output_dir, carryover_session_id)
    knowledge_dir = carryover_root / "knowledge"
    slog(f"Replan carryover session: {carryover_root}")

    while True:
        run_id = uuid.uuid4().hex[:12]
        folder_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        try:
            carryover_root.mkdir(parents=True, exist_ok=True)
            knowledge_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        carryover_digest = _read_carryover_digest_for_cycle(knowledge_dir, cfg=cfg)

        kb_digest = ""
        if cfg.kb_root_resolved is not None:
            kb_files = _list_kb_files(cfg.kb_root_resolved, cfg.kb_file_extensions, cfg.kb_max_files)
            if kb_files:
                slog(
                    f"User knowledge base: {cfg.kb_root_resolved} "
                    f"({len(kb_files)} file(s) indexed, cap {cfg.kb_max_files})"
                )
            kb_digest = _build_kb_index_digest(
                cfg.kb_root_resolved,
                kb_files,
                per_file_head=cfg.kb_index_per_file_head_chars,
                total_cap=cfg.kb_index_max_chars,
            )
        user_kb_block = ""
        if cfg.kb_root_resolved is not None and kb_digest.strip():
            user_kb_block = (
                "[User knowledge base]\n"
                f"(On-disk directory: {cfg.kb_root_resolved}; use kb_list / kb_read tools for full files.)\n\n"
                + kb_digest.strip()
            )
        co_kb = carryover_digest.strip() or None
        uk_kb = user_kb_block.strip() or None
        if co_kb and uk_kb:
            prior_kb_for_workers = f"{co_kb}\n\n{uk_kb}"
        elif co_kb:
            prior_kb_for_workers = co_kb
        elif uk_kb:
            prior_kb_for_workers = uk_kb
        else:
            prior_kb_for_workers = None

        if cfg.replan_share_workspace:
            workspace_root = carryover_root / "workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)
        else:
            workspace_root = Path(cfg.workspace_parent_dir).resolve() / folder_ts
            workspace_root.mkdir(parents=True, exist_ok=True)
        artifacts_dir = Path(cfg.output_dir).resolve() / "run_artifacts" / f"{folder_ts}_{run_id}"
        idempotency_cache: dict[str, Any] = {}
        if st:
            st.reset(max_worker_slots=n_workers, run_id=run_id)
            st.set_phase("planning")
        planner_problem = _append_deliverables_block(user_prompt, persistent_deliverables)
        if user_kb_block.strip():
            planner_problem += "\n\n" + user_kb_block.strip() + "\n"
        if carryover_digest.strip():
            planner_problem += (
                "\n\n[Prior run knowledge base]\n"
                f"(On-disk directory: {knowledge_dir})\n\n"
                + carryover_digest.strip()
                + "\n"
            )
        try:
            plan: ParsedPlan
            raw_plan_final: str
            if cfg.use_two_phase_planner:
                outline_raw = _hive_complete(
                    st,
                    "planner_outline",
                    base_url=cfg.base_url,
                    client=client,
                    model=model,
                    user=_planner_outline_user(planner_problem, max_plan_tasks=cfg.max_plan_tasks),
                    system=_planner_outline_system(),
                    max_tokens=cfg.max_tokens,
                    cfg=cfg,
                    role=HiveRole.PLANNER_OUTLINE.value,
                )
                blob_o = extract_json_blob(outline_raw) or outline_raw.strip()
                if not blob_o:
                    slog("Outline empty; falling back to single-phase planner.")
                    raw_plan_final = _hive_complete(
                        st,
                        "planner",
                        base_url=cfg.base_url,
                        client=client,
                        model=model,
                        user=_planner_user(
                            planner_problem,
                            max_plan_tasks=cfg.max_plan_tasks,
                            max_plan_depth_hint=cfg.max_plan_depth_hint,
                        ),
                        system=_planner_system(),
                        max_tokens=cfg.max_tokens,
                        cfg=cfg,
                        role=HiveRole.PLANNER.value,
                    )
                else:
                    expand_raw = _hive_complete(
                        st,
                        "planner_expand",
                        base_url=cfg.base_url,
                        client=client,
                        model=model,
                        user=_planner_expand_user(
                            planner_problem,
                            blob_o,
                            max_plan_tasks=cfg.max_plan_tasks,
                            max_plan_depth_hint=cfg.max_plan_depth_hint,
                        ),
                        system=_planner_expand_system(),
                        max_tokens=cfg.max_tokens,
                        cfg=cfg,
                        role=HiveRole.PLANNER_EXPAND.value,
                    )
                    raw_plan_final = expand_raw
            else:
                raw_plan_final = _hive_complete(
                    st,
                    "planner",
                    base_url=cfg.base_url,
                    client=client,
                    model=model,
                    user=_planner_user(
                        planner_problem,
                        max_plan_tasks=cfg.max_plan_tasks,
                        max_plan_depth_hint=cfg.max_plan_depth_hint,
                    ),
                    system=_planner_system(),
                    max_tokens=cfg.max_tokens,
                    cfg=cfg,
                    role=HiveRole.PLANNER.value,
                )
            try:
                plan = parse_plan(raw_plan_final)
            except (json.JSONDecodeError, ValueError):
                slog("Planner JSON parse failed; attempting repair call.")
                raw_plan2 = _hive_complete(
                    st,
                    "planner_repair",
                    base_url=cfg.base_url,
                    client=client,
                    model=model,
                    user=_append_deliverables_block(_repair_json_prompt(raw_plan_final), persistent_deliverables),
                    system=_planner_system(),
                    max_tokens=cfg.max_tokens,
                    cfg=cfg,
                    role=HiveRole.PLANNER_REPAIR.value,
                )
                plan = parse_plan(raw_plan2)
            for critique_round in range(1, cfg.max_planner_critique_rounds + 1):
                qerrs = validate_plan_quality(
                    plan.tasks,
                    max_plan_tasks=cfg.max_plan_tasks,
                    max_plan_depth_hint=cfg.max_plan_depth_hint,
                    min_description_chars=cfg.min_task_description_chars,
                    min_acceptance_chars=cfg.min_acceptance_criteria_chars,
                )
                if not qerrs:
                    break
                slog(f"Planner critique round {critique_round}: " + "; ".join(qerrs[:5]))
                crit_raw = _hive_complete(
                    st,
                    f"planner_critique_{critique_round}",
                    base_url=cfg.base_url,
                    client=client,
                    model=model,
                    user=_append_deliverables_block(
                        _planner_critique_user(qerrs, raw_plan_final), persistent_deliverables
                    ),
                    system=_planner_critique_system(),
                    max_tokens=cfg.max_tokens,
                    cfg=cfg,
                    role=HiveRole.PLANNER_CRITIQUE.value,
                )
                raw_plan_final = crit_raw
                try:
                    plan = parse_plan(raw_plan_final)
                except (json.JSONDecodeError, ValueError) as parse_err:
                    raise ValueError(f"Planner critique produced invalid plan: {parse_err}") from parse_err
            tail_errs = validate_plan_quality(
                plan.tasks,
                max_plan_tasks=cfg.max_plan_tasks,
                max_plan_depth_hint=cfg.max_plan_depth_hint,
                min_description_chars=cfg.min_task_description_chars,
                min_acceptance_chars=cfg.min_acceptance_criteria_chars,
            )
            if tail_errs:
                raise ValueError("Plan validation failed: " + "; ".join(tail_errs[:12]))
            tasks = plan.tasks
            include_original_prompt = plan.include_original_prompt
            if plan.required_deliverables:
                persistent_deliverables[:] = sorted(
                    set(persistent_deliverables) | set(plan.required_deliverables)
                )
        except Exception as e:
            if st:
                st.set_phase("error")
                st.set_error(str(e))
            raise

        if st:
            st.init_tasks([t.id for t in tasks])
            st.set_plan_snapshot(tasks)
            st.set_phase("executing_tasks")
        results: dict[str, str] = {}
        completed_order: list[str] = []
        sched_lock = threading.Lock()
        checkpoint_lock = threading.Lock()
        ready_q: queue.PriorityQueue[tuple[tuple[int, int], tuple[int, PlanTask | None]]] = queue.PriorityQueue()
        ex: dict[str, Any] = {
            "tasks": tasks,
            "plan_version": 0,
            "early_exit": False,
            "stop_sent": False,
            "checkpoint_replans": 0,
            "placed": set(),
            "sched_seq": 0,
            "router_cache": {},
            "assertion_ledgers": {},
            "assertion_ledgers_raw": {},
            "claims_run_report": {"generated_at": "", "claims": []},
            "ledger_resolution_messages": [],
        }

        def drain_ready_queue() -> None:
            while True:
                try:
                    ready_q.get_nowait()
                except queue.Empty:
                    break
                ready_q.task_done()

        def enqueue_runnable() -> None:
            if ex["early_exit"]:
                return
            ts: list[PlanTask] = ex["tasks"]
            pv: int = ex["plan_version"]
            placed: set[str] = ex["placed"]
            depths = longest_path_depths(ts)
            for t in ts:
                if t.id in results:
                    continue
                if t.id in placed:
                    continue
                if not all(d in results for d in t.depends_on):
                    continue
                placed.add(t.id)
                ex["sched_seq"] += 1
                d = depths.get(t.id, 1)
                ready_q.put(((-d, ex["sched_seq"]), (pv, t)))

        def send_shutdown_nones() -> None:
            if ex["stop_sent"]:
                return
            ex["stop_sent"] = True
            pv = ex["plan_version"]
            for i in range(n_workers):
                ready_q.put(((10**12, i), (pv, None)))

        def apply_checkpoint_decision(finished: PlanTask, dec: dict[str, Any]) -> None:
            action = str(dec.get("action", "continue")).strip().lower()
            rationale = str(dec.get("rationale", "")).strip()
            if rationale:
                slog(f"Checkpoint after {finished.id}: {action} — {rationale[:500]}")
            if action == "continue":
                enqueue_runnable()
                if len(results) == len(ex["tasks"]):
                    send_shutdown_nones()
                return
            if action == "finish_early":
                ex["early_exit"] = True
                ex["plan_version"] += 1
                drain_ready_queue()
                send_shutdown_nones()
                return
            if action != "replan":
                enqueue_runnable()
                if len(results) == len(ex["tasks"]):
                    send_shutdown_nones()
                return
            if ex["checkpoint_replans"] >= cfg.max_checkpoint_replans:
                slog(
                    f"Checkpoint replan skipped (budget {cfg.max_checkpoint_replans}); continuing plan."
                )
                enqueue_runnable()
                if len(results) == len(ex["tasks"]):
                    send_shutdown_nones()
                return
            raw_list = dec.get("tasks")
            if not isinstance(raw_list, list) or not raw_list:
                enqueue_runnable()
                if len(results) == len(ex["tasks"]):
                    send_shutdown_nones()
                return
            new_tasks: list[PlanTask] = []
            for i, it in enumerate(raw_list):
                if isinstance(it, dict):
                    pt = _parse_task_from_planner_item(it, i)
                    if pt is not None:
                        new_tasks.append(pt)
            if not new_tasks:
                enqueue_runnable()
                if len(results) == len(ex["tasks"]):
                    send_shutdown_nones()
                return
            done_ids = set(results.keys())
            try:
                merged_list = [x for x in ex["tasks"] if x.id in done_ids] + new_tasks
                validate_task_dependencies(merged_list)
            except ValueError as err:
                slog(f"Checkpoint replan ignored (invalid dependencies): {err}")
                enqueue_runnable()
                if len(results) == len(ex["tasks"]):
                    send_shutdown_nones()
                return
            ex["checkpoint_replans"] += 1
            ex["tasks"] = merged_list
            ex["plan_version"] += 1
            ex["placed"] = set(results.keys())
            drain_ready_queue()
            if st:
                st.set_plan_snapshot(ex["tasks"])
                for nt in new_tasks:
                    if nt.id not in st.tasks:
                        st.task_update(nt.id, status="queued", last_message="Added by checkpoint replan")
            enqueue_runnable()
            if len(results) == len(ex["tasks"]):
                send_shutdown_nones()

        enqueue_runnable()

        def process_one_task(
            worker_slot: int,
            task: PlanTask,
            dependency_outputs: list[tuple[str, str]],
        ) -> str:
            tid = task.id
            carryover_kb = prior_kb_for_workers
            if st:
                st.task_update(tid, status="running_worker", last_message="Worker started")
            previous: str | None = None
            qa_feedback: str | None = None
            qa_failure_lines: list[str] = []
            attempt = 0
            stored_tool_evidence = ""
            decompose_used = False
            run_micro_pipeline = False
            pending_micro: list[MicroDecomposeStep] | None = None
            last_selected_skills: list[SkillInfo] = []
            ws_abs = str(workspace_root.resolve())
            router_key = hashlib.sha256(
                f"{tid}\n{task.title}\n{task.description}\n{task.acceptance_criteria}\n"
                f"{sorted(enabled_ids)}".encode("utf-8")
            ).hexdigest()
            while attempt < cfg.max_qa_retries:
                attempt += 1
                if st and st.cancel_requested:
                    st.task_update(tid, status="failed", last_message="Cancelled")
                    raise RuntimeError("Cancelled")
                if st:
                    st.task_update(
                        tid,
                        attempt=attempt,
                        status="running_worker",
                        last_message=f"Attempt {attempt}/{cfg.max_qa_retries}: generating",
                    )
                if run_micro_pipeline and pending_micro:
                    if st:
                        st.task_update(
                            tid,
                            status="running_worker",
                            last_message=f"Attempt {attempt}/{cfg.max_qa_retries}: micro-plan ({len(pending_micro)} steps)",
                        )
                    solution, te = _run_qa_fail_micro_worker_chain(
                        st,
                        steps=pending_micro,
                        user_prompt=user_prompt,
                        task=task,
                        include_original_prompt=include_original_prompt,
                        cfg=cfg,
                        ws_abs=ws_abs,
                        selected_skills=last_selected_skills,
                        skills_by_id=skills_by_id,
                        skills_root=skills_root,
                        workspace_root=workspace_root,
                        idempotency_cache=idempotency_cache,
                        client=client,
                        model=model,
                        tid=tid,
                        worker_slot=worker_slot,
                        attempt_num=attempt,
                        prior_run_knowledge=carryover_kb,
                    )
                    run_micro_pipeline = False
                    pending_micro = None
                else:
                    selected_ids: list[str] = []
                    if enabled_ids:
                        catalog = "\n".join(router_prompt_lines(discovered, enabled_ids))
                        if catalog.strip():
                            if router_key in ex["router_cache"]:
                                selected_ids = list(ex["router_cache"][router_key])
                            else:
                                raw_router = _hive_complete(
                                    st,
                                    "skill_router",
                                    base_url=cfg.base_url,
                                    client=client,
                                    model=model,
                                    user=_skill_router_user(
                                        user_prompt,
                                        task,
                                        catalog,
                                        cfg=cfg,
                                        include_original_prompt=include_original_prompt,
                                        prior_run_knowledge=carryover_kb,
                                    ),
                                    system=_skill_router_system(),
                                    max_tokens=cfg.max_tokens,
                                    cfg=cfg,
                                    role=HiveRole.SKILL_ROUTER.value,
                                    task_id=tid,
                                    worker_slot=worker_slot,
                                )
                                selected_ids = parse_router_skill_ids(raw_router, enabled_ids)
                                ex["router_cache"][router_key] = list(selected_ids)
                    selected_skills = [skills_by_id[i] for i in selected_ids if i in skills_by_id]
                    last_selected_skills = selected_skills
                    prior_ev: str | None = None
                    if cfg.qa_retry_tool_trace and stored_tool_evidence.strip():
                        prior_ev = stored_tool_evidence
                    solution, te = _hive_worker_with_tools(
                        st,
                        f"worker (attempt {attempt})",
                        base_url=cfg.base_url,
                        client=client,
                        model=model,
                        max_tokens=cfg.max_tokens,
                        cfg=cfg,
                        task_id=tid,
                        worker_slot=worker_slot,
                        base_system=_worker_system(),
                        base_user=_worker_user(
                            user_prompt,
                            task,
                            previous,
                            qa_feedback,
                            cfg=cfg,
                            include_original_prompt=include_original_prompt,
                            dependency_outputs=dependency_outputs or None,
                            workspace_abs=ws_abs,
                            prior_tool_evidence=prior_ev,
                            prior_run_knowledge=carryover_kb,
                        ),
                        selected=selected_skills,
                        skills_by_id=skills_by_id,
                        skills_root=skills_root,
                        max_tool_rounds=cfg.max_tool_rounds,
                        tool_timeout_sec=cfg.tool_timeout_sec,
                        workspace_root=workspace_root,
                        idempotency_cache=idempotency_cache,
                    )
                stored_tool_evidence = te
                previous = solution
                if st:
                    st.task_update(tid, status="running_qa", last_message="QA review")
                qa_raw = ""
                for qpr in range(cfg.qa_json_max_retries + 1):
                    qa_sys = _qa_system() if qpr == 0 else _qa_system_json_retry()
                    qa_raw = _hive_complete(
                        st,
                        f"qa (attempt {attempt}) parse_try_{qpr}",
                        base_url=cfg.base_url,
                        client=client,
                        model=model,
                        user=_qa_user(
                            user_prompt,
                            task,
                            solution,
                            cfg=cfg,
                            include_original_prompt=include_original_prompt,
                        ),
                        system=qa_sys,
                        max_tokens=cfg.max_tokens,
                        cfg=cfg,
                        role=HiveRole.QA.value,
                        task_id=tid,
                        worker_slot=worker_slot,
                    )
                    if extract_json_blob(qa_raw):
                        break
                passed, detail = parse_qa_verdict(qa_raw)
                if passed:
                    if cfg.post_qa_assertion_ledger:
                        ledger_text, validated = _build_post_qa_assertion_ledger(
                            st,
                            cfg=cfg,
                            base_url=cfg.base_url,
                            client=client,
                            model=model,
                            user_prompt=user_prompt,
                            task=task,
                            solution=solution,
                            include_original_prompt=include_original_prompt,
                            task_id=tid,
                            worker_slot=worker_slot,
                        )
                        if ledger_text:
                            ex["assertion_ledgers"][tid] = ledger_text
                        if validated is not None:
                            ex["assertion_ledgers_raw"][tid] = validated.as_dict()
                            if not ex["claims_run_report"].get("generated_at"):
                                ex["claims_run_report"]["generated_at"] = datetime.datetime.now().isoformat(
                                    timespec="seconds"
                                )
                            for row in claims_for_run_report(
                                validated, task_id=tid, worker_slot=worker_slot
                            ):
                                ex["claims_run_report"]["claims"].append(row)
                    if st:
                        st.task_update(tid, status="passed", last_message="QA passed")
                        _write_task_artifact(
                            artifacts_dir,
                            tid,
                            status="passed",
                            attempt=attempt,
                            solution_excerpt=solution,
                            qa_note="passed",
                        )
                    return solution
                qa_feedback = detail
                qa_failure_lines.append(f"- Attempt {attempt}: {detail}")
                if st:
                    st.record_qa_failure()
                    st.record_qa_retry()
                    st.task_update(
                        tid,
                        status="retrying",
                        last_message=f"QA failed: {detail[:300]}",
                    )
                if (
                    cfg.qa_fail_decompose_enabled
                    and (not decompose_used)
                    and attempt >= cfg.qa_fail_decompose_min_attempt
                ):
                    try:
                        single_ok, dec_rat, micro_steps = _qa_fail_decompose_llm(
                            st,
                            client,
                            model,
                            cfg,
                            task,
                            detail,
                            solution,
                            stored_tool_evidence,
                            tid,
                            worker_slot,
                        )
                        if (not single_ok) and micro_steps:
                            if st:
                                msg = (
                                    f"Task {tid}: QA fail decompose → {len(micro_steps)} micro-steps"
                                    + (f" ({dec_rat[:200]})" if dec_rat else "")
                                )
                                st.log(msg)
                            pending_micro = micro_steps
                            run_micro_pipeline = True
                            decompose_used = True
                            qa_feedback = (
                                ((dec_rat + "\n\n") if dec_rat else "")
                                + "Work was split into micro-steps; the next attempt runs that sequence, then QA."
                            )
                            continue
                    except Exception as dec_err:
                        if st:
                            st.log(f"Task {tid}: QA fail decompose skipped ({dec_err})")
                if (
                    cfg.qa_refinement_after_failures > 0
                    and len(qa_failure_lines) >= cfg.qa_refinement_after_failures
                ):
                    report = "\n".join(qa_failure_lines)
                    summary = _planner_summary_user_problem(user_prompt)
                    ref_user = _task_qa_refinement_user(task, report, summary)
                    try:
                        raw_ref = _hive_complete(
                            st,
                            f"task_qa_refinement ({tid})",
                            base_url=cfg.base_url,
                            client=client,
                            model=model,
                            user=ref_user,
                            system=_task_qa_refinement_system(),
                            max_tokens=cfg.max_tokens,
                            cfg=cfg,
                            role=HiveRole.TASK_QA_REFINEMENT.value,
                            task_id=tid,
                            worker_slot=worker_slot,
                        )
                        try:
                            refined, did_adjust, rat = parse_task_qa_refinement(raw_ref, task)
                        except (json.JSONDecodeError, ValueError):
                            raw_ref2 = _hive_complete(
                                st,
                                f"task_qa_refinement_repair ({tid})",
                                base_url=cfg.base_url,
                                client=client,
                                model=model,
                                user=_task_qa_refinement_repair(raw_ref),
                                system=_task_qa_refinement_system(),
                                max_tokens=cfg.max_tokens,
                                cfg=cfg,
                                role=HiveRole.TASK_QA_REFINEMENT_REPAIR.value,
                                task_id=tid,
                                worker_slot=worker_slot,
                            )
                            refined, did_adjust, rat = parse_task_qa_refinement(raw_ref2, task)
                        if did_adjust:
                            task.title = refined.title
                            task.description = refined.description
                            task.acceptance_criteria = refined.acceptance_criteria
                            if st:
                                st.set_plan_snapshot(ex["tasks"])
                            previous = None
                            stored_tool_evidence = ""
                            ex["router_cache"].pop(router_key, None)
                            hint = (rat + "\n\n") if rat else ""
                            qa_feedback = (
                                f"[Planner refined this task's specification after {len(qa_failure_lines)} QA failure(s).]\n"
                                f"{hint}"
                                "Re-attempt from scratch against the updated acceptance criteria below."
                            )
                            if st:
                                st.task_update(
                                    tid,
                                    status="retrying",
                                    last_message=f"Task spec refined: {rat[:280] if rat else 'see feedback'}",
                                )
                            slog(
                                f"Task {tid}: QA refinement adjusted spec after "
                                f"{len(qa_failure_lines)} failure(s). {rat[:200] if rat else ''}"
                            )
                        else:
                            if st and rat:
                                slog(f"Task {tid}: QA refinement kept spec: {rat[:300]}")
                            if st:
                                st.task_update(
                                    tid,
                                    status="retrying",
                                    last_message=f"QA refinement: keeping spec — {rat[:240] if rat else 'no change'}",
                                )
                    except Exception as refine_err:
                        if st:
                            st.log(
                                f"Task {tid}: task QA refinement failed ({refine_err}); continuing with same spec."
                            )
                    qa_failure_lines.clear()
            if st:
                st.task_update(tid, status="failed", last_message="Max QA retries exceeded")
            raise RuntimeError(f"Task {tid} failed QA after {cfg.max_qa_retries} attempt(s).")

        def worker_loop(worker_slot: int) -> None:
            while True:
                _prio, packet = ready_q.get()
                ver, task = packet
                try:
                    if task is None:
                        return
                    if ver != ex["plan_version"]:
                        continue
                    if ex["early_exit"] and task.id not in results:
                        continue
                    if st and st.cancel_requested:
                        return
                    dep_outputs: list[tuple[str, str]] = []
                    with sched_lock:
                        for dep_id in task.depends_on:
                            blob = results.get(dep_id, "")
                            if not blob:
                                continue
                            if len(blob) > cfg.dependency_output_max_chars:
                                blob = blob[: cfg.dependency_output_max_chars] + "\n...[truncated]"
                            if len(blob) > cfg.summarize_deps_llm_threshold:
                                summ_user = (
                                    f"Summarize the following text for downstream sub-task id={task.id}. "
                                    "Keep facts, numbers, and decisions. Plain text only, max about 3000 characters.\n\n"
                                    + blob[:50_000]
                                )
                                summ = _hive_complete(
                                    st,
                                    f"summarize_dep_{dep_id}_for_{task.id}",
                                    base_url=cfg.base_url,
                                    client=client,
                                    model=model,
                                    user=summ_user,
                                    system="You compress dependency output. Reply with plain text only.",
                                    max_tokens=cfg.max_tokens,
                                    cfg=cfg,
                                    role=HiveRole.SUMMARIZE_DEPS.value,
                                    task_id=task.id,
                                    worker_slot=worker_slot,
                                )
                                blob = summ[: cfg.dependency_summary_max_chars]
                            elif len(blob) > cfg.dependency_summary_max_chars:
                                blob = summarize_dependency_blob_heuristic(
                                    dep_id,
                                    blob,
                                    task,
                                    cfg.dependency_summary_max_chars,
                                )
                            dep_outputs.append((dep_id, blob))
                    if st:
                        st.set_thread_task(worker_slot, task.id)
                    sol = process_one_task(worker_slot, task, dep_outputs)
                    if task.key_task:
                        with checkpoint_lock:
                            with sched_lock:
                                results[task.id] = sol
                                completed_order.append(task.id)
                                snap_results = dict(results)
                                snap_order = list(completed_order)
                                pending_list = [x for x in ex["tasks"] if x.id not in results]
                            raw_ck = _hive_complete(
                                st,
                                f"checkpoint ({task.id})",
                                base_url=cfg.base_url,
                                client=client,
                                model=model,
                                user=_checkpoint_user(
                                    user_prompt,
                                    checkpoint_task=task,
                                    completed_order=snap_order,
                                    results_by_id=snap_results,
                                    pending_tasks=pending_list,
                                    cfg=cfg,
                                ),
                                system=_checkpoint_system(),
                                max_tokens=cfg.max_tokens,
                                cfg=cfg,
                                role=HiveRole.CHECKPOINT.value,
                                task_id=task.id,
                                worker_slot=worker_slot,
                            )
                            dec = parse_checkpoint_decision(raw_ck)
                            with sched_lock:
                                apply_checkpoint_decision(task, dec)
                    else:
                        with sched_lock:
                            results[task.id] = sol
                            completed_order.append(task.id)
                            enqueue_runnable()
                            if len(results) == len(ex["tasks"]):
                                send_shutdown_nones()
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    slog(msg)
                    if st:
                        st.abort_run(msg)
                finally:
                    if st:
                        st.set_thread_task(worker_slot, None)
                    ready_q.task_done()

        threads: list[threading.Thread] = []
        for w in range(1, n_workers + 1):
            th = threading.Thread(
                target=worker_loop,
                args=(w,),
                name=f"hive-worker-{w}",
                daemon=True,
            )
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        if st and st.cancel_requested:
            st.set_phase("error")
            if not st.error:
                st.set_error("Cancelled")
            raise RuntimeError(st.error or "Run cancelled.")

        if not ex["early_exit"]:
            missing = [t.id for t in ex["tasks"] if t.id not in results]
            if missing:
                err = f"Missing results for tasks: {missing}"
                if st:
                    st.set_error(err)
                raise RuntimeError(err)

        if cfg.post_qa_assertion_ledger and cfg.ledger_conflict_resolution and ex.get("assertion_ledgers_raw"):
            raw_map = ex["assertion_ledgers_raw"]
            if isinstance(raw_map, dict) and raw_map:
                by_task = {
                    tid: ClaimValidationResult.from_dict(d) for tid, d in raw_map.items() if isinstance(d, dict)
                }
                if by_task:
                    resolved = resolve_cross_task_conflicts(by_task, enabled=True)
                    ex["ledger_resolution_messages"] = list(
                        dict.fromkeys(m for r in resolved.values() for m in (r.resolution_messages or []))
                    )
                    for tid, r in resolved.items():
                        ex["assertion_ledgers_raw"][tid] = r.as_dict()
                        lt = format_claims_for_merge(
                            r, tid, research_verified_only_body=cfg.research_verified_only_final
                        )
                        if lt.strip():
                            ex["assertion_ledgers"][tid] = lt
                    rows_out: list[dict[str, Any]] = []
                    for tid in completed_order:
                        d = ex.get("assertion_ledgers_raw", {}).get(tid)
                        if not isinstance(d, dict):
                            continue
                        vr = ClaimValidationResult.from_dict(d)
                        rows_out.extend(claims_for_run_report(vr, task_id=tid, worker_slot=0))
                    if rows_out:
                        ex["claims_run_report"]["claims"] = rows_out

        ledgers: dict[str, str] = ex["assertion_ledgers"]
        ordered_parts: list[tuple[str, str]] = []
        for tid in completed_order:
            if tid not in results:
                continue
            if cfg.post_qa_assertion_ledger and ledgers.get(tid):
                ordered_parts.append((tid, ledgers[tid]))
            else:
                ordered_parts.append((tid, results[tid]))
        merge_ledger_mode = cfg.post_qa_assertion_ledger and any(
            bool(ledgers.get(tid)) for tid in completed_order if tid in results
        )
        merge_research_strict = bool(cfg.research_verified_only_final and merge_ledger_mode)
        ledger_msgs = list(ex.get("ledger_resolution_messages") or [])

        if st:
            st.set_phase("merging")

        merger_problem = original_prompt
        if cfg.integration_qa_enabled and len(ordered_parts) > 1:
            cp_ids = critical_path_task_ids(ex["tasks"])
            chain_parts = [(tid, results[tid]) for tid in cp_ids if tid in results]
            if len(chain_parts) >= 2:
                integ_raw = _hive_complete(
                    st,
                    "integration_qa",
                    base_url=cfg.base_url,
                    client=client,
                    model=model,
                    user=_integration_qa_user(original_prompt, chain_parts),
                    system=_integration_qa_system(),
                    max_tokens=cfg.max_tokens,
                    cfg=cfg,
                    role=HiveRole.INTEGRATION_QA.value,
                )
                ipass, idetail, _issues = parse_integration_qa_verdict(integ_raw)
                if not ipass:
                    if cfg.integration_qa_strict:
                        raise RuntimeError(f"Integration QA failed: {idetail}")
                    merger_problem = (
                        original_prompt
                        + "\n\n[Cross-task integration review — address in the merged answer]\n"
                        + idetail
                    )

        parts_for_merge = _maybe_compress_ordered_parts(
            st,
            cfg=cfg,
            base_url=cfg.base_url,
            client=client,
            model=model,
            user_prompt=merger_problem,
            ordered_parts=ordered_parts,
        )
        if cfg.merge_strategy == "hierarchical":
            merged = _hierarchical_merge(
                st,
                cfg=cfg,
                base_url=cfg.base_url,
                client=client,
                model=model,
                user_prompt=merger_problem,
                ordered_parts=parts_for_merge,
                merge_ledger_mode=merge_ledger_mode,
                merge_research_strict=merge_research_strict,
                ledger_resolution_messages=ledger_msgs,
            )
        else:
            merged = _hive_complete(
                st,
                "merger",
                base_url=cfg.base_url,
                client=client,
                model=model,
                user=_merger_user(
                    merger_problem,
                    parts_for_merge,
                    merge_ledger_mode=merge_ledger_mode,
                    merge_research_strict=merge_research_strict,
                    ledger_resolution_messages=ledger_msgs,
                ),
                system=_merger_system(),
                max_tokens=cfg.max_tokens,
                cfg=cfg,
                role=HiveRole.MERGER.value,
            )

        final_draft = merged
        verified_digest = _aggregated_verified_claims_markdown(ex)
        research_strict = bool(cfg.research_verified_only_final)
        excluded_digest = ""
        if research_strict and merge_ledger_mode:
            excluded_digest = build_excluded_claims_digest(ex.get("assertion_ledgers_raw") or {})
        if cfg.final_synthesis_enabled:
            if st:
                st.set_phase("final_synthesis")
            final_draft = _hive_complete(
                st,
                "final_synthesis",
                base_url=cfg.base_url,
                client=client,
                model=model,
                user=_final_synthesis_user(
                    original_prompt,
                    merged,
                    merge_ledger_mode=merge_ledger_mode,
                    research_strict=research_strict,
                    verified_digest=verified_digest,
                    excluded_digest=excluded_digest,
                ),
                system=_final_synthesis_system(research_strict=research_strict),
                max_tokens=cfg.max_tokens,
                cfg=cfg,
                role=HiveRole.FINAL_SYNTHESIS.value,
            )

        # Final consistency check + possible fix/replan loop
        final_text = final_draft
        replan_needed = False
        passed_final = False
        if st:
            st.set_phase("final_check")
        for check_i in range(1, cfg.final_check_attempts + 1):
            gaps = deliverables_coverage_gaps(final_text, persistent_deliverables)
            gap_note = ""
            if gaps:
                gap_note = (
                    "[Orchestrator deliverables not clearly found in the candidate text — verify explicitly:]\n"
                    + "\n".join(f"- {g}" for g in gaps)
                )
            verdict_raw = _hive_complete(
                st,
                f"final_check ({check_i})",
                base_url=cfg.base_url,
                client=client,
                model=model,
                user=_final_check_user(original_prompt, final_text, deliverables_note=gap_note),
                system=_final_check_system(),
                max_tokens=cfg.max_tokens,
                cfg=cfg,
                role=HiveRole.FINAL_CHECK.value,
            )
            verdict = parse_final_consistency(verdict_raw)
            if st:
                st.set_final_check(verdict)
            if verdict.get("pass") is True:
                passed_final = True
                break
            if verdict.get("requires_replan") is True:
                if replan_used >= cfg.replan_attempts:
                    raise RuntimeError(
                        "Final check requested replan but replan budget exhausted."
                    )
                replan_used += 1
                _write_replan_carryover(
                    knowledge_dir,
                    attempt_num=replan_used,
                    completed_order=list(completed_order),
                    results=dict(results),
                    merged_draft=merged,
                    final_candidate=final_text,
                    verdict=dict(verdict),
                    claims_report=dict(ex.get("claims_run_report") or {}),
                    cfg=cfg,
                    slog=slog,
                )
                issue_note = (
                    "Previous attempt failed final consistency validation. "
                    "Be careful to avoid repeating these issues: "
                    f"{json.dumps(verdict, ensure_ascii=False)}"
                )
                user_prompt = f"{original_prompt}\n\n[Final-consistency issues to avoid]\n{issue_note}\n"
                if st:
                    st.reset(max_worker_slots=n_workers, run_id=str(uuid.uuid4())[:12])
                    st.log(
                        f"Replan triggered by final check (attempt {replan_used}/{cfg.replan_attempts})."
                    )
                replan_needed = True
                break

            # Otherwise: ask fixer/merger agent to revise the final answer.
            final_text = _hive_complete(
                st,
                f"final_fix ({check_i})",
                base_url=cfg.base_url,
                client=client,
                model=model,
                user=_final_fix_user(original_prompt, final_text, verdict),
                system=_final_fix_system(),
                max_tokens=cfg.max_tokens,
                cfg=cfg,
                role=HiveRole.FINAL_FIX.value,
            )

        if replan_needed:
            continue
        if not passed_final:
            raise RuntimeError("Final consistency check did not pass within attempt budget.")

        if st:
            st.set_result(final_text)
            st.set_phase("done")

        if cfg.save_final_to_file:
            try:
                out_path = write_hive_result_file(cfg.output_dir, original_prompt, final_text)
                if st:
                    st.set_output_file(out_path)
                else:
                    slog(f"Saved result to: {out_path}")
            except OSError as e:
                msg = f"Could not save result file: {e}"
                slog(msg)
                if st:
                    st.log(msg)

        cr = ex.get("claims_run_report") or {}
        claim_rows = list(cr.get("claims") or [])
        if st and claim_rows:
            st.set_claims_report(cr, max_claims=cfg.claims_report_snapshot_max_claims)
        if claim_rows and (cfg.save_claims_report or cfg.post_qa_assertion_ledger):
            try:
                claims_path = write_hive_claims_report_file(cfg.output_dir, original_prompt, cr)
                msg = f"Saved claims report to: {claims_path}"
                if st:
                    st.log(msg)
                else:
                    slog(msg)
            except OSError as e:
                msg = f"Could not save claims report file: {e}"
                slog(msg)
                if st:
                    st.log(msg)

        return final_text


def _cli_print(s: str) -> None:
    print(s, flush=True)


def main_cli(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argv if argv is not None else sys.argv[1:]
    if not p:
        print("Enter problem (end with Ctrl-Z then Enter on Windows, or Ctrl-D on Unix):", flush=True)
        user = sys.stdin.read().strip()
    else:
        user = " ".join(p).strip()
    if not user:
        print("No problem text provided.", file=sys.stderr)
        return 1
    st = RunState()
    try:
        out = run_hive(user, state=st, print_fn=_cli_print)
    except (APIConnectionError, APIError, RuntimeError, ValueError) as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1
    print("\n--- Final ---\n", flush=True)
    print(out, flush=True)
    return 0


def main() -> None:
    we = get_hive_env()
    parser = argparse.ArgumentParser(description="Agent hive: multi-agent LLM orchestration.")
    parser.add_argument("--web", action="store_true", help="Run Flask UI instead of one-shot CLI.")
    parser.add_argument("--host", default=we.web_host, help="Web bind host (default: HIVE_WEB_HOST).")
    parser.add_argument("--port", type=int, default=we.web_port, help="Web port (default: HIVE_WEB_PORT).")
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Problem text for CLI mode; if omitted, read stdin. Ignored with --web.",
    )
    args = parser.parse_args()
    if args.web:
        from hive_web import run_web

        run_web(host=args.host, port=args.port)
        return
    sys.exit(main_cli(list(args.prompt)))


if __name__ == "__main__":
    main()
