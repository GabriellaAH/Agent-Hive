"""Discover skills under skills/<id>/SKILL.md and persist enable/disable flags."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_skills import Skill
from pydantic import ValidationError

# Default: project-relative skills folder; override with HIVE_SKILLS_DIR
_DEFAULT_SKILLS = "skills"
_DEFAULT_ENABLED_FILE = "hive_skills_enabled.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def default_skills_dir() -> Path:
    return Path(os.environ.get("HIVE_SKILLS_DIR", _project_root() / _DEFAULT_SKILLS))


def default_enabled_path() -> Path:
    return Path(os.environ.get("HIVE_SKILLS_ENABLED_FILE", _project_root() / _DEFAULT_ENABLED_FILE))


@dataclass
class SkillInfo:
    id: str
    name: str
    description: str
    skill_md_path: Path
    body: str
    script_paths: list[str] = field(default_factory=list)

    def has_scripts(self) -> bool:
        return len(self.script_paths) > 0


def _find_skill_markdown(skill_dir: Path) -> Path | None:
    for name in ("SKILL.md", "skill.md", "Skill.md"):
        p = skill_dir / name
        if p.is_file():
            return p
    return None


def _list_scripts_under_skill(skill_dir: Path) -> list[str]:
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return []
    out: list[str] = []
    for root, _dirs, files in os.walk(scripts_dir):
        for f in files:
            p = Path(root) / f
            rel = p.relative_to(skill_dir)
            out.append(str(rel).replace("\\", "/"))
    return sorted(out)


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    raw = raw.lstrip("\ufeff")
    if not raw.startswith("---"):
        return {}, raw
    m = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?", raw)
    if not m:
        return {}, raw
    body = raw[m.end() :]
    fm_text = m.group(1)
    meta: dict[str, Any] = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in ("name", "description"):
            meta[k] = v
    return meta, body


def _skillinfo_from_agent_skills(raw: str, skill_id: str, md: Path, script_paths: list[str]) -> SkillInfo | None:
    """Parse SKILL.md via agent_skills.Skill; return None if format is unsupported."""
    try:
        sk = Skill.from_skill_md(raw)
    except (ValueError, ValidationError):
        return None
    return SkillInfo(
        id=skill_id,
        name=str(sk.metadata.name or skill_id).strip() or skill_id,
        description=str(sk.metadata.description or "").strip(),
        skill_md_path=md.resolve(),
        body=sk.content.strip(),
        script_paths=script_paths,
    )


def _skillinfo_from_legacy_frontmatter(
    raw: str, skill_id: str, md: Path, script_paths: list[str]
) -> SkillInfo:
    meta, body = _parse_frontmatter(raw)
    name = str(meta.get("name") or skill_id).strip() or skill_id
    desc = str(meta.get("description") or "").strip()
    if not desc:
        first_line = (body.strip().splitlines() or [""])[0].strip("# ").strip()
        desc = (first_line[:240] + "...") if len(first_line) > 240 else first_line
    return SkillInfo(
        id=skill_id,
        name=name,
        description=desc,
        skill_md_path=md.resolve(),
        body=body.strip(),
        script_paths=script_paths,
    )


def discover_skills(skills_root: Path | None = None) -> list[SkillInfo]:
    root = skills_root or default_skills_dir()
    root = root.resolve()
    if not root.is_dir():
        return []
    found: list[SkillInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "__pycache__":
            continue
        md = _find_skill_markdown(child)
        if md is None:
            continue
        skill_id = child.name
        raw = md.read_text(encoding="utf-8", errors="replace")
        script_paths = _list_scripts_under_skill(child)
        info = _skillinfo_from_agent_skills(raw, skill_id, md, script_paths)
        if info is None:
            info = _skillinfo_from_legacy_frontmatter(raw, skill_id, md, script_paths)
        found.append(info)
    return found


def load_enabled_map(path: Path | None = None) -> dict[str, bool]:
    p = path or default_enabled_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, bool] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, bool):
            out[k] = v
    return out


def save_enabled_map(enabled: dict[str, bool], path: Path | None = None) -> None:
    p = path or default_enabled_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(enabled, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def merge_enabled_with_discovery(
    discovered: list[SkillInfo],
    stored: dict[str, bool],
    *,
    default_new_enabled: bool = True,
) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for s in discovered:
        if s.id in stored:
            out[s.id] = stored[s.id]
        else:
            out[s.id] = default_new_enabled
    return out


def get_enabled_skill_ids(
    discovered: list[SkillInfo],
    enabled_map: dict[str, bool] | None = None,
    path: Path | None = None,
) -> set[str]:
    stored = enabled_map if enabled_map is not None else load_enabled_map(path)
    merged = merge_enabled_with_discovery(discovered, stored)
    return {sid for sid, on in merged.items() if on}


def skill_by_id(discovered: list[SkillInfo], skill_id: str) -> SkillInfo | None:
    for s in discovered:
        if s.id == skill_id:
            return s
    return None


def router_prompt_lines(discovered: list[SkillInfo], enabled_ids: set[str]) -> list[str]:
    lines: list[str] = []
    for s in discovered:
        if s.id not in enabled_ids:
            continue
        lines.append(f"- id: {s.id}\n  name: {s.name}\n  description: {s.description[:500]}")
    return lines
