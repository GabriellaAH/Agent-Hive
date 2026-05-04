"""Load Agent Hive settings from environment (.env via python-dotenv). Defaults live only here."""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- Default values (single source; mirrored in .env.example) ---

_DEF_BASE_URL = "http://127.0.0.1:1234"
_DEF_API_KEY = "lm-studio"
_DEF_HTTP_TIMEOUT = 360.0
_DEF_LM_RETRY_MAX = 5
_DEF_LM_RETRY_BASE_DELAY = 1.0
_DEF_LM_RETRY_MAX_DELAY = 60.0

_DEF_MAX_PARALLEL = 4
_DEF_MAX_PARALLEL_CAP = 16

_DEF_OUTPUT_DIR = "hive_outputs"
_DEF_WORKSPACE_PARENT = "workspace"

_DEF_MAX_TOKENS = 150000
_DEF_MAX_QA_RETRIES = 30
_DEF_MAX_PLAN_TASKS = 24
_DEF_MAX_PLAN_DEPTH_HINT = 8
_DEF_MIN_TASK_DESC = 20
_DEF_MIN_ACCEPTANCE = 10
_DEF_MAX_PLANNER_CRITIQUE = 2
_DEF_QA_JSON_MAX_RETRIES = 2
_DEF_FINAL_CHECK_ATTEMPTS = 3
_DEF_REPLAN_ATTEMPTS = 2
_DEF_REPLAN_CARRYOVER_ENABLED = True
_DEF_REPLAN_KNOWLEDGE_MAX_CHARS = 36_000
_DEF_REPLAN_SHARE_WORKSPACE = False
_DEF_REPLAN_CARRYOVER_TASK_MAX_CHARS = 24_000
_DEF_REPLAN_CARRYOVER_MERGED_MAX_CHARS = 48_000
_DEF_QA_REFINEMENT_AFTER_FAILURES = 10
_DEF_MAX_CHECKPOINT_REPLANS = 3

_DEF_DEPENDENCY_SUMMARY_MAX = 4000
_DEF_SUMMARIZE_DEPS_THRESHOLD = 8000
_DEF_MERGER_THRESHOLD = 120_000
_DEF_MERGE_COMPRESS_MAX_CHUNKS = 8
_DEF_MERGE_STRATEGY = "single_pass"
_DEF_POST_QA_ASSERTION_LEDGER = False
_DEF_RESEARCH_VERIFIED_ONLY_FINAL = False
_DEF_LEDGER_ALLOWED_SOURCE_TYPES_CSV = ""
_DEF_EVIDENCE_POLICY = "normal"
_DEF_LEDGER_CONFLICT_RESOLUTION = True
_DEF_REQUIRED_DELIVERABLES_RAW = ""
_DEF_SAVE_CLAIMS_REPORT = False
_DEF_PRICING_REQUIRES_URL = False
_DEF_CLAIMS_REPORT_SNAPSHOT_MAX = 80

_DEF_MAX_TOOL_ROUNDS = 8
_DEF_TOOL_TIMEOUT = 120.0
_DEF_HTTP_IDEMPOTENCY_MAX = 128

_DEF_QA_RETRY_TOOL_TRACE = True
_DEF_QA_RETRY_TOOL_TRACE_MAX_CHARS = 18_000

_DEF_QA_FAIL_DECOMPOSE_ENABLED = False
_DEF_QA_FAIL_DECOMPOSE_MAX_STEPS = 6
_DEF_QA_FAIL_DECOMPOSE_MIN_ATTEMPT = 1

_DEF_DEPENDENCY_OUTPUT_MAX = 10_000
_DEF_CHECKPOINT_TASK_OUTPUT = 3500
_DEF_SUBAGENT_VERBATIM = 5500
_DEF_SUBAGENT_COMPRESSED = 4200
_DEF_SUBAGENT_HEAD = 1600
_DEF_SUBAGENT_TAIL = 1000

_DEF_LOG_MAX_LINES = 200
_DEF_IO_LOG_MAX_ENTRIES = 120

_DEF_WEB_HOST = "127.0.0.1"
_DEF_WEB_PORT = 5000

_DEF_TEST_PROMPT = ""

_DEF_OPENAI_BASE_URL = "https://api.openai.com"

_DEF_KB_DIR = ""
_DEF_KB_INDEX_MAX_CHARS = 12_000
_DEF_KB_INDEX_PER_FILE_HEAD_CHARS = 600
_DEF_KB_READ_MAX_CHARS = 40_000
_DEF_KB_FILE_EXTENSIONS = (
    ".md,.txt,.rst,.html,.htm,.json,.yaml,.yml,.csv,.tsv,.py,.js,.ts,.jsx,.tsx,.xml"
)
_DEF_KB_MAX_FILES = 200


def _load_dotenv_files() -> None:
    explicit = (os.environ.get("DOTENV_PATH") or "").strip()
    if explicit:
        load_dotenv(Path(explicit), override=False)
        return
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=False)


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def _env_opt_str(key: str) -> str | None:
    v = os.environ.get(key)
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def _env_optional_positive_int(key: str) -> int | None:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return None
    n = int(str(v).strip())
    return n if n > 0 else None


def _env_optional_positive_float(key: str) -> float | None:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return None
    x = float(str(v).strip())
    return x if x > 0 else None


def _env_raw_preserve(key: str, default: str = "") -> str:
    """Read env value without stripping internal newlines (for multiline deliverables)."""
    v = os.environ.get(key)
    if v is None:
        return default
    return str(v)


def _resolved_openai_api_key() -> str | None:
    """Web UI OpenAI profile: standard OPENAI_API_KEY."""
    return _env_opt_str("OPENAI_API_KEY")


@dataclass(frozen=True)
class HiveEnvSettings:
    base_url: str
    api_key: str
    model: str | None
    http_timeout_sec: float
    lm_retry_max: int
    lm_retry_base_delay_sec: float
    lm_retry_max_delay_sec: float
    max_parallel_workers: int
    max_parallel_workers_cap: int
    auto_parallel_workers: bool
    output_dir: str
    workspace_parent_dir: str
    max_tokens: int
    max_tokens_cap: int
    max_qa_retries: int
    max_plan_tasks: int
    max_plan_depth_hint: int
    min_task_description_chars: int
    min_acceptance_criteria_chars: int
    max_planner_critique_rounds: int
    use_two_phase_planner: bool
    qa_json_max_retries: int
    integration_qa_enabled: bool
    integration_qa_strict: bool
    dependency_summary_max_chars: int
    summarize_deps_llm_threshold: int
    merger_input_threshold_chars: int
    merge_compress_max_chunks: int
    merge_strategy: str
    post_qa_assertion_ledger: bool
    research_verified_only_final: bool
    evidence_policy: str
    ledger_allowed_source_types_csv: str
    ledger_conflict_resolution: bool
    required_deliverables_raw: str
    save_claims_report: bool
    pricing_requires_url: bool
    claims_report_snapshot_max_claims: int
    final_synthesis_enabled: bool
    final_check_attempts: int
    replan_attempts: int
    replan_carryover_enabled: bool
    replan_knowledge_max_chars: int
    replan_share_workspace: bool
    replan_carryover_task_max_chars: int
    replan_carryover_merged_max_chars: int
    qa_refinement_after_failures: int
    max_checkpoint_replans: int
    max_tool_rounds: int
    tool_timeout_sec: float
    max_run_wall_seconds: float | None
    max_estimated_tokens_per_run: int | None
    http_idempotency_max_entries: int
    qa_retry_tool_trace: bool
    qa_retry_tool_trace_max_chars: int
    qa_fail_decompose_enabled: bool
    qa_fail_decompose_max_steps: int
    qa_fail_decompose_min_attempt: int
    save_final_to_file: bool
    dependency_output_max_chars: int
    checkpoint_task_output_chars: int
    subagent_problem_verbatim_max: int
    subagent_compressed_body_max: int
    subagent_head_chars: int
    subagent_tail_chars: int
    log_max_lines: int
    io_log_max_entries: int
    web_host: str
    web_port: int
    test_prompt: str
    openai_api_key: str | None
    openai_base_url: str
    kb_dir: str
    kb_index_max_chars: int
    kb_index_per_file_head_chars: int
    kb_read_max_chars: int
    kb_file_extensions: str
    kb_max_files: int


@functools.lru_cache(maxsize=1)
def get_hive_env() -> HiveEnvSettings:
    _load_dotenv_files()
    merge_strategy = _env_str("HIVE_MERGE_STRATEGY", _DEF_MERGE_STRATEGY)
    if merge_strategy not in ("single_pass", "hierarchical"):
        merge_strategy = _DEF_MERGE_STRATEGY
    return HiveEnvSettings(
        base_url=_env_str("HIVE_BASE_URL", _DEF_BASE_URL),
        api_key=_env_str("HIVE_API_KEY", _DEF_API_KEY),
        model=_env_opt_str("HIVE_MODEL"),
        http_timeout_sec=_env_float("HIVE_HTTP_TIMEOUT_SEC", _DEF_HTTP_TIMEOUT),
        lm_retry_max=_env_int("HIVE_LM_RETRY_MAX", _DEF_LM_RETRY_MAX),
        lm_retry_base_delay_sec=_env_float("HIVE_LM_RETRY_BASE_DELAY_SEC", _DEF_LM_RETRY_BASE_DELAY),
        lm_retry_max_delay_sec=_env_float("HIVE_LM_RETRY_MAX_DELAY_SEC", _DEF_LM_RETRY_MAX_DELAY),
        max_parallel_workers=_env_int("HIVE_MAX_PARALLEL_WORKERS", _DEF_MAX_PARALLEL),
        max_parallel_workers_cap=_env_int("HIVE_MAX_PARALLEL_WORKERS_CAP", _DEF_MAX_PARALLEL_CAP),
        auto_parallel_workers=_env_bool("HIVE_AUTO_PARALLEL_WORKERS", False),
        output_dir=_env_str("HIVE_OUTPUT_DIR", _DEF_OUTPUT_DIR),
        workspace_parent_dir=_env_str("HIVE_WORKSPACE_PARENT_DIR", _DEF_WORKSPACE_PARENT),
        max_tokens=_env_int("HIVE_MAX_TOKENS", _DEF_MAX_TOKENS),
        max_tokens_cap=_env_int("HIVE_MAX_TOKENS_CAP", _DEF_MAX_TOKENS),
        max_qa_retries=_env_int("HIVE_MAX_QA_RETRIES", _DEF_MAX_QA_RETRIES),
        max_plan_tasks=_env_int("HIVE_MAX_PLAN_TASKS", _DEF_MAX_PLAN_TASKS),
        max_plan_depth_hint=_env_int("HIVE_MAX_PLAN_DEPTH_HINT", _DEF_MAX_PLAN_DEPTH_HINT),
        min_task_description_chars=_env_int("HIVE_MIN_TASK_DESCRIPTION_CHARS", _DEF_MIN_TASK_DESC),
        min_acceptance_criteria_chars=_env_int("HIVE_MIN_ACCEPTANCE_CRITERIA_CHARS", _DEF_MIN_ACCEPTANCE),
        max_planner_critique_rounds=_env_int("HIVE_MAX_PLANNER_CRITIQUE_ROUNDS", _DEF_MAX_PLANNER_CRITIQUE),
        use_two_phase_planner=_env_bool("HIVE_USE_TWO_PHASE_PLANNER", True),
        qa_json_max_retries=_env_int("HIVE_QA_JSON_MAX_RETRIES", _DEF_QA_JSON_MAX_RETRIES),
        integration_qa_enabled=_env_bool("HIVE_INTEGRATION_QA_ENABLED", True),
        integration_qa_strict=_env_bool("HIVE_INTEGRATION_QA_STRICT", False),
        dependency_summary_max_chars=_env_int("HIVE_DEPENDENCY_SUMMARY_MAX_CHARS", _DEF_DEPENDENCY_SUMMARY_MAX),
        summarize_deps_llm_threshold=_env_int("HIVE_SUMMARIZE_DEPS_LLM_THRESHOLD", _DEF_SUMMARIZE_DEPS_THRESHOLD),
        merger_input_threshold_chars=_env_int("HIVE_MERGER_INPUT_THRESHOLD_CHARS", _DEF_MERGER_THRESHOLD),
        merge_compress_max_chunks=_env_int("HIVE_MERGE_COMPRESS_MAX_CHUNKS", _DEF_MERGE_COMPRESS_MAX_CHUNKS),
        merge_strategy=merge_strategy,
        post_qa_assertion_ledger=_env_bool("HIVE_POST_QA_ASSERTION_LEDGER", _DEF_POST_QA_ASSERTION_LEDGER),
        research_verified_only_final=_env_bool(
            "HIVE_RESEARCH_VERIFIED_ONLY_FINAL", _DEF_RESEARCH_VERIFIED_ONLY_FINAL
        ),
        evidence_policy=_env_str("HIVE_EVIDENCE_POLICY", _DEF_EVIDENCE_POLICY),
        ledger_allowed_source_types_csv=_env_str(
            "HIVE_LEDGER_ALLOWED_SOURCE_TYPES", _DEF_LEDGER_ALLOWED_SOURCE_TYPES_CSV
        ),
        ledger_conflict_resolution=_env_bool(
            "HIVE_LEDGER_CONFLICT_RESOLUTION", _DEF_LEDGER_CONFLICT_RESOLUTION
        ),
        required_deliverables_raw=_env_raw_preserve(
            "HIVE_REQUIRED_DELIVERABLES", _DEF_REQUIRED_DELIVERABLES_RAW
        ),
        save_claims_report=_env_bool("HIVE_SAVE_CLAIMS_REPORT", _DEF_SAVE_CLAIMS_REPORT),
        pricing_requires_url=_env_bool("HIVE_PRICING_REQUIRES_URL", _DEF_PRICING_REQUIRES_URL),
        claims_report_snapshot_max_claims=max(
            1,
            _env_int("HIVE_CLAIMS_REPORT_SNAPSHOT_MAX", _DEF_CLAIMS_REPORT_SNAPSHOT_MAX),
        ),
        final_synthesis_enabled=_env_bool("HIVE_FINAL_SYNTHESIS_ENABLED", True),
        final_check_attempts=_env_int("HIVE_FINAL_CHECK_ATTEMPTS", _DEF_FINAL_CHECK_ATTEMPTS),
        replan_attempts=_env_int("HIVE_REPLAN_ATTEMPTS", _DEF_REPLAN_ATTEMPTS),
        replan_carryover_enabled=_env_bool("HIVE_REPLAN_CARRYOVER_ENABLED", _DEF_REPLAN_CARRYOVER_ENABLED),
        replan_knowledge_max_chars=max(
            4000,
            _env_int("HIVE_REPLAN_KNOWLEDGE_MAX_CHARS", _DEF_REPLAN_KNOWLEDGE_MAX_CHARS),
        ),
        replan_share_workspace=_env_bool("HIVE_REPLAN_SHARE_WORKSPACE", _DEF_REPLAN_SHARE_WORKSPACE),
        replan_carryover_task_max_chars=max(
            2000,
            _env_int("HIVE_REPLAN_CARRYOVER_TASK_MAX_CHARS", _DEF_REPLAN_CARRYOVER_TASK_MAX_CHARS),
        ),
        replan_carryover_merged_max_chars=max(
            4000,
            _env_int("HIVE_REPLAN_CARRYOVER_MERGED_MAX_CHARS", _DEF_REPLAN_CARRYOVER_MERGED_MAX_CHARS),
        ),
        qa_refinement_after_failures=_env_int(
            "HIVE_QA_REFINEMENT_AFTER_FAILURES", _DEF_QA_REFINEMENT_AFTER_FAILURES
        ),
        max_checkpoint_replans=_env_int("HIVE_MAX_CHECKPOINT_REPLANS", _DEF_MAX_CHECKPOINT_REPLANS),
        max_tool_rounds=_env_int("HIVE_MAX_TOOL_ROUNDS", _DEF_MAX_TOOL_ROUNDS),
        tool_timeout_sec=_env_float("HIVE_TOOL_TIMEOUT_SEC", _DEF_TOOL_TIMEOUT),
        max_run_wall_seconds=_env_optional_positive_float("HIVE_MAX_RUN_WALL_SECONDS"),
        max_estimated_tokens_per_run=_env_optional_positive_int("HIVE_MAX_ESTIMATED_TOKENS_PER_RUN"),
        http_idempotency_max_entries=_env_int("HIVE_HTTP_IDEMPOTENCY_MAX_ENTRIES", _DEF_HTTP_IDEMPOTENCY_MAX),
        qa_retry_tool_trace=_env_bool("HIVE_QA_RETRY_TOOL_TRACE", _DEF_QA_RETRY_TOOL_TRACE),
        qa_retry_tool_trace_max_chars=max(
            2000,
            _env_int("HIVE_QA_RETRY_TOOL_TRACE_MAX_CHARS", _DEF_QA_RETRY_TOOL_TRACE_MAX_CHARS),
        ),
        qa_fail_decompose_enabled=_env_bool(
            "HIVE_QA_FAIL_DECOMPOSE_ENABLED", _DEF_QA_FAIL_DECOMPOSE_ENABLED
        ),
        qa_fail_decompose_max_steps=min(
            24,
            max(2, _env_int("HIVE_QA_FAIL_DECOMPOSE_MAX_STEPS", _DEF_QA_FAIL_DECOMPOSE_MAX_STEPS)),
        ),
        qa_fail_decompose_min_attempt=max(
            1,
            _env_int("HIVE_QA_FAIL_DECOMPOSE_MIN_ATTEMPT", _DEF_QA_FAIL_DECOMPOSE_MIN_ATTEMPT),
        ),
        save_final_to_file=_env_bool("HIVE_SAVE_FINAL_TO_FILE", True),
        dependency_output_max_chars=_env_int("HIVE_DEPENDENCY_OUTPUT_MAX_CHARS", _DEF_DEPENDENCY_OUTPUT_MAX),
        checkpoint_task_output_chars=_env_int("HIVE_CHECKPOINT_TASK_OUTPUT_CHARS", _DEF_CHECKPOINT_TASK_OUTPUT),
        subagent_problem_verbatim_max=_env_int("HIVE_SUBAGENT_PROBLEM_VERBATIM_MAX", _DEF_SUBAGENT_VERBATIM),
        subagent_compressed_body_max=_env_int("HIVE_SUBAGENT_COMPRESSED_BODY_MAX", _DEF_SUBAGENT_COMPRESSED),
        subagent_head_chars=_env_int("HIVE_SUBAGENT_HEAD_CHARS", _DEF_SUBAGENT_HEAD),
        subagent_tail_chars=_env_int("HIVE_SUBAGENT_TAIL_CHARS", _DEF_SUBAGENT_TAIL),
        log_max_lines=_env_int("HIVE_LOG_MAX_LINES", _DEF_LOG_MAX_LINES),
        io_log_max_entries=_env_int("HIVE_IO_LOG_MAX_ENTRIES", _DEF_IO_LOG_MAX_ENTRIES),
        web_host=_env_str("HIVE_WEB_HOST", _DEF_WEB_HOST),
        web_port=_env_int("HIVE_WEB_PORT", _DEF_WEB_PORT),
        test_prompt=_env_str("HIVE_TEST_PROMPT", _DEF_TEST_PROMPT),
        openai_api_key=_resolved_openai_api_key(),
        openai_base_url=_env_str("HIVE_OPENAI_BASE_URL", _DEF_OPENAI_BASE_URL),
        kb_dir=_env_str("HIVE_KB_DIR", _DEF_KB_DIR),
        kb_index_max_chars=max(
            500,
            _env_int("HIVE_KB_INDEX_MAX_CHARS", _DEF_KB_INDEX_MAX_CHARS),
        ),
        kb_index_per_file_head_chars=max(
            50,
            _env_int("HIVE_KB_INDEX_PER_FILE_HEAD_CHARS", _DEF_KB_INDEX_PER_FILE_HEAD_CHARS),
        ),
        kb_read_max_chars=max(
            1000,
            _env_int("HIVE_KB_READ_MAX_CHARS", _DEF_KB_READ_MAX_CHARS),
        ),
        kb_file_extensions=_env_str(
            "HIVE_KB_FILE_EXTENSIONS",
            _DEF_KB_FILE_EXTENSIONS,
        ),
        kb_max_files=max(1, _env_int("HIVE_KB_MAX_FILES", _DEF_KB_MAX_FILES)),
    )


def clear_hive_env_cache() -> None:
    """For tests that need to reload environment."""
    get_hive_env.cache_clear()
