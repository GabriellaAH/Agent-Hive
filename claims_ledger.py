"""Parse, validate, and format strict post-QA claims ledgers (no jsonschema dependency)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

_DEFAULT_SOURCE_TYPES = frozenset(
    {
        "official_pricing_page",
        "official_pricing_calculator",
        "official_docs",
        "academic_paper",
        "academic_metadata",
        "primary_source",
        "news",
        "blog",
        "social",
        "unknown",
        "internal_worker_output",
    }
)
_WEAK_SOURCE_TYPES = frozenset({"blog", "social", "news"})
_CLAIM_TYPES = frozenset({"factual", "pricing", "opinion", "code_behavior", "other"})
_ACADEMIC_STRICT_ONLY_CLAIM_TYPES = frozenset(
    {"academic_design", "academic_citation", "paper_finding", "academic_sources_only"}
)
_ACADEMIC_EVIDENCE_TYPES = frozenset({"academic_paper", "academic_metadata"})
# paper_finding: long excerpt from paper, or shorter excerpt tied to academic_metadata + scholarly URL.
_PAPER_FINDING_LONG_QUOTE_MIN = 80
_ACADEMIC_METADATA_SNIPPET_MIN = 32

_SCHOLARLY_CITATION_HOSTS = frozenset(
    {
        "doi.org",
        "www.doi.org",
        "hdl.handle.net",
        "api.crossref.org",
        "crossref.org",
        "www.crossref.org",
        "openalex.org",
        "api.openalex.org",
        "semanticscholar.org",
        "www.semanticscholar.org",
        "arxiv.org",
        "export.arxiv.org",
        "www.arxiv.org",
    }
)
_PUBLISHER_PAGE_DENY_HOSTS = frozenset(
    {
        "medium.com",
        "twitter.com",
        "www.twitter.com",
        "x.com",
        "reddit.com",
        "www.reddit.com",
        "facebook.com",
        "www.facebook.com",
        "tiktok.com",
        "www.tiktok.com",
    }
)
_DOI_IN_URL = re.compile(r"10\.\d{4,9}/\S+")

# Source authority for conflict resolution (higher = preferred).
_SOURCE_RANK: dict[str, int] = {
    "official_pricing_page": 100,
    "official_pricing_calculator": 100,
    "official_docs": 85,
    "academic_paper": 70,
    "academic_metadata": 68,
    "primary_source": 70,
    "news": 45,
    "blog": 25,
    "social": 20,
    "unknown": 10,
    "internal_worker_output": 15,
}

_STRONG_VENDOR_TYPES = frozenset(
    {"official_docs", "official_pricing_page", "official_pricing_calculator"}
)
_PRICING_OFFICIAL_TYPES = frozenset({"official_pricing_page", "official_pricing_calculator"})


@dataclass(frozen=True)
class EvidencePolicy:
    """Preset-driven evidence rules; tier_mode applies on top of base JSON validation."""

    name: str
    tier_mode: str  # none | strict_research | pricing_strict | academic_strict
    allowed_source_types: frozenset[str]


def resolve_evidence_policy(raw: str | None) -> EvidencePolicy:
    """Map HIVE_EVIDENCE_POLICY to tier rules and default allowlist (CSV may override allowlist in HiveConfig)."""
    s = (raw or "").strip().lower() or "normal"
    if s not in ("loose", "normal", "strict_research", "pricing_strict", "academic_strict"):
        s = "normal"
    tier = (
        s
        if s in ("strict_research", "pricing_strict", "academic_strict")
        else "none"
    )
    return EvidencePolicy(name=s, tier_mode=tier, allowed_source_types=_DEFAULT_SOURCE_TYPES)


def claim_types_for_policy(policy: EvidencePolicy | None) -> frozenset[str]:
    """Base claim types; academic_strict-only types are valid only under that preset."""
    if policy is not None and policy.name == "academic_strict":
        return _CLAIM_TYPES | _ACADEMIC_STRICT_ONLY_CLAIM_TYPES
    return _CLAIM_TYPES


def scholarly_citation_url_ok(url: str) -> bool:
    """DOI in URL, known scholarly metadata hosts, or loose https publisher-page heuristic."""
    u = (url or "").strip()
    if not u:
        return False
    if _DOI_IN_URL.search(u):
        return True
    try:
        p = urlparse(u)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    if host in _SCHOLARLY_CITATION_HOSTS:
        return True
    if host in _PUBLISHER_PAGE_DENY_HOSTS:
        return False
    # Strip common www. for host checks against scholarly set
    if host.startswith("www.") and host[4:] in _SCHOLARLY_CITATION_HOSTS:
        return True
    path = (p.path or "").strip("/")
    if host not in _SCHOLARLY_CITATION_HOSTS and path:
        return True
    return False


def parse_deliverables_from_env(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    s = str(raw).strip()
    if "|" in s and "\n" not in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out: list[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out


def parse_allowed_source_types_csv(csv: str | None) -> frozenset[str]:
    if not csv or not str(csv).strip():
        return _DEFAULT_SOURCE_TYPES
    items = {x.strip() for x in str(csv).split(",") if x.strip()}
    return frozenset(items) if items else _DEFAULT_SOURCE_TYPES


def _today_iso() -> str:
    return date.today().isoformat()


def _confidence_bucket(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in ("high", "medium", "low"):
        return s
    if s in ("1", "true", "yes"):
        return "high"
    return "medium"


def _as_str_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def claim_text_implies_quantities(claim: str) -> bool:
    """Heuristic: claim text references numbers/units that should be mirrored in numeric_values."""
    if not claim or not str(claim).strip():
        return False
    c = claim
    patterns = (
        r"\$[\d,]+(?:\.\d+)?",
        r"\$\s*[\d,]+",
        r"%",
        r"\bGB\b",
        r"\bMB\b",
        r"\bTB\b",
        r"\btokens?\b",
        r"\bvectors?\b",
        r"per\s+million",
        r"per\s+m",
        r"/\s*mo\b",
        r"/mo\b",
        r"per\s+month",
        r"\bUSD\b",
        r"\bEUR\b",
    )
    low = c.lower()
    if any(re.search(p, low, re.I) for p in patterns):
        return True
    if re.search(r"~\s*[\d]", c):
        return True
    return False


def claim_text_suggests_estimate(claim: str) -> bool:
    low = claim.lower()
    if re.search(r"\bestimat", low):
        return True
    if re.search(r"\bapprox", low):
        return True
    if re.search(r"\broughly\b", low):
        return True
    if re.search(r"~\s*[\d]", claim):
        return True
    return False


def _parse_retrieved_at(s: str) -> datetime | None:
    t = (s or "").strip()
    if not t:
        return None
    try:
        if len(t) == 10 and t[4] == "-" and t[7] == "-":
            return datetime.fromisoformat(t + "T00:00:00")
        return datetime.fromisoformat(t.replace("Z", "+00:00")[:19])
    except ValueError:
        return None


def _claim_max_source_rank(evidence: list[dict[str, Any]]) -> int:
    best = 0
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        st = str(ev.get("source_type", "")).strip()
        best = max(best, _SOURCE_RANK.get(st, 0))
    return best


def _claim_max_evidence_date(evidence: list[dict[str, Any]]) -> datetime | None:
    best: datetime | None = None
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        d = _parse_retrieved_at(str(ev.get("retrieved_at", "")))
        if d is not None and (best is None or d > best):
            best = d
    return best


def _norm_key_part(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _normalize_scalar_for_compare(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val).strip().lower()
    return re.sub(r"\s+", "", str(val).strip().lower())


def conflict_key_for_claim(claim: dict[str, Any]) -> str | None:
    """Single key from first numeric_values row + provider; None if insufficient for dedup."""
    prov = _norm_key_part(str(claim.get("provider", "")))
    nvs = claim.get("numeric_values") or []
    if not isinstance(nvs, list) or not nvs:
        return None
    nv0 = nvs[0]
    if not isinstance(nv0, dict):
        return None
    meaning = _norm_key_part(str(nv0.get("meaning", "")))
    unit = _norm_key_part(str(nv0.get("unit", "")))
    val = _normalize_scalar_for_compare(nv0.get("value"))
    if not meaning and not unit:
        return None
    return f"{prov}\x1f{meaning}\x1f{unit}\x1f{val}"


def conflict_group_key(claim: dict[str, Any]) -> str | None:
    """Group key: provider + metric + unit (value excluded — used to find conflicting values)."""
    prov = _norm_key_part(str(claim.get("provider", "")))
    nvs = claim.get("numeric_values") or []
    if not isinstance(nvs, list) or not nvs:
        return None
    nv0 = nvs[0]
    if not isinstance(nv0, dict):
        return None
    meaning = _norm_key_part(str(nv0.get("meaning", "")))
    unit = _norm_key_part(str(nv0.get("unit", "")))
    if not meaning and not unit:
        return None
    return f"{prov}\x1f{meaning}\x1f{unit}"


def _claim_only_weak_third_party(evidence: list[dict[str, Any]]) -> bool:
    if not evidence:
        return True
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        st = str(ev.get("source_type", "")).strip()
        if st not in _WEAK_SOURCE_TYPES:
            return False
    return True


def _tier_violation_message(
    *,
    tier_mode: str,
    ctype: str,
    provider: str,
    ev_copy: list[dict[str, Any]],
) -> str | None:
    if tier_mode == "none":
        return None
    if tier_mode == "strict_research":
        if any(str(e.get("source_type", "")).strip() in _WEAK_SOURCE_TYPES for e in ev_copy):
            return "policy strict_research: weak source_type (blog/news/social) cannot verify"
        return None
    if tier_mode == "academic_strict":
        if any(str(e.get("source_type", "")).strip() in _WEAK_SOURCE_TYPES for e in ev_copy):
            return "policy academic_strict: weak source_type (blog/news/social) cannot verify"
        return None
    if tier_mode == "pricing_strict":
        if ctype == "opinion":
            return None
        if ctype == "pricing":
            if any(str(e.get("source_type", "")).strip() not in _PRICING_OFFICIAL_TYPES for e in ev_copy):
                return "policy pricing_strict: pricing claims require official_pricing_page or official_pricing_calculator"
            return None
        if ctype == "factual" and (provider or "").strip():
            if not any(str(e.get("source_type", "")).strip() in _STRONG_VENDOR_TYPES for e in ev_copy):
                return "policy pricing_strict: factual vendor claims require official_docs or official pricing evidence"
        if _claim_only_weak_third_party(ev_copy):
            return "policy pricing_strict: third-party-only evidence cannot verify"
        return None
    return None


def _academic_strict_claim_violation(
    *,
    ctype: str,
    ev_copy: list[dict[str, Any]],
) -> str | None:
    """Extra rules when HIVE_EVIDENCE_POLICY=academic_strict (after weak-source tier check)."""
    if ctype == "academic_design":
        return None
    if ctype not in _ACADEMIC_STRICT_ONLY_CLAIM_TYPES:
        return None

    def _all_internal(ev: list[dict[str, Any]]) -> bool:
        return bool(ev) and all(
            str(e.get("source_type", "")).strip() == "internal_worker_output" for e in ev
        )

    def _has_academic_evidence(ev: list[dict[str, Any]]) -> bool:
        return any(str(e.get("source_type", "")).strip() in _ACADEMIC_EVIDENCE_TYPES for e in ev)

    if ctype in ("paper_finding", "academic_citation", "academic_sources_only"):
        if _all_internal(ev_copy) or not _has_academic_evidence(ev_copy):
            return (
                "policy academic_strict: internal_worker_output cannot verify external literature claims"
            )

    if ctype == "academic_sources_only":
        for e in ev_copy:
            st = str(e.get("source_type", "")).strip()
            if st not in _ACADEMIC_EVIDENCE_TYPES:
                return (
                    "policy academic_strict: academic_sources_only requires every "
                    "evidence source_type to be academic_paper or academic_metadata"
                )
        return None

    if ctype == "academic_citation":
        for e in ev_copy:
            st = str(e.get("source_type", "")).strip()
            if st not in _ACADEMIC_EVIDENCE_TYPES:
                return (
                    "policy academic_strict: academic_citation requires only "
                    "academic_paper or academic_metadata evidence rows"
                )
        if not any(
            scholarly_citation_url_ok(str(e.get("source_url", "")))
            for e in ev_copy
            if str(e.get("source_type", "")).strip() in _ACADEMIC_EVIDENCE_TYPES
        ):
            return (
                "policy academic_strict: academic_citation requires at least one source_url "
                "with DOI, scholarly aggregator (CrossRef/OpenAlex/Semantic Scholar/arXiv), or publisher https page"
            )
        return None

    if ctype == "paper_finding":
        scholarly_rows: list[tuple[str, int]] = []
        for e in ev_copy:
            st = str(e.get("source_type", "")).strip()
            if st not in _ACADEMIC_EVIDENCE_TYPES:
                continue
            url = str(e.get("source_url", "")).strip()
            q = str(e.get("quote_or_snippet", "")).strip()
            if scholarly_citation_url_ok(url):
                scholarly_rows.append((st, len(q)))
        if not scholarly_rows:
            return (
                "policy academic_strict: paper_finding requires at least one academic_paper or "
                "academic_metadata evidence with a scholarly citation URL (DOI, aggregator, or publisher https)"
            )
        long_ok = any(ln >= _PAPER_FINDING_LONG_QUOTE_MIN for _, ln in scholarly_rows)
        meta_ok = any(
            st == "academic_metadata" and ln >= _ACADEMIC_METADATA_SNIPPET_MIN for st, ln in scholarly_rows
        )
        if not long_ok and not meta_ok:
            return (
                "policy academic_strict: paper_finding requires an excerpt of at least "
                f"{_PAPER_FINDING_LONG_QUOTE_MIN} chars from a scholarly-backed row, OR "
                f"academic_metadata with scholarly URL and excerpt of at least {_ACADEMIC_METADATA_SNIPPET_MIN} chars"
            )
        return None

    return None


def normalize_legacy_assertions_blob(data: dict[str, Any]) -> dict[str, Any]:
    """Convert old assertions[] shape into claims-document shape (all unverified for evidence)."""
    assertions_raw = data.get("assertions")
    if not isinstance(assertions_raw, list):
        return {"claims": [], "unverified_claims": [], "contradictions": []}
    unv: list[dict[str, Any]] = []
    for i, item in enumerate(assertions_raw):
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        ev_text = str(item.get("evidence", "")).strip()
        qu = str(item.get("quantifiers", item.get("numbers", ""))).strip()
        cav = _as_str_list(item.get("caveats", item.get("caveat")))
        if isinstance(item.get("caveats"), str) and item.get("caveats"):
            cav = [str(item["caveats"]).strip()] if str(item["caveats"]).strip() else cav
        num_vals: list[dict[str, Any]] = []
        if qu:
            num_vals.append({"value": qu, "unit": "", "meaning": "from legacy quantifiers field"})
        unv.append(
            {
                "id": f"legacy-{i + 1}",
                "claim": claim,
                "type": "other",
                "provider": "",
                "evidence": [
                    {
                        "source_url": "",
                        "source_type": "internal_worker_output",
                        "quote_or_snippet": ev_text or "(no excerpt; legacy ledger)",
                        "retrieved_at": _today_iso(),
                    }
                ],
                "numeric_values": num_vals,
                "confidence": _confidence_bucket(item.get("confidence")),
                "caveats": cav or ["legacy assertion ledger — re-verify sources"],
            }
        )
    return {"claims": [], "unverified_claims": unv, "contradictions": []}


def coerce_claims_document(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, list):
        return {"claims": parsed, "unverified_claims": [], "contradictions": []}
    if not isinstance(parsed, dict):
        return None
    if "assertions" in parsed and "claims" not in parsed:
        return normalize_legacy_assertions_blob(parsed)
    out = {
        "claims": list(parsed.get("claims") or []) if isinstance(parsed.get("claims"), list) else [],
        "unverified_claims": list(parsed.get("unverified_claims") or [])
        if isinstance(parsed.get("unverified_claims"), list)
        else [],
        "contradictions": list(parsed.get("contradictions") or [])
        if isinstance(parsed.get("contradictions"), list)
        else [],
    }
    return out


@dataclass
class ClaimValidationResult:
    verified_claims: list[dict[str, Any]] = field(default_factory=list)
    unverified_claims: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[Any] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    resolution_messages: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ClaimValidationResult:
        return ClaimValidationResult(
            verified_claims=list(d.get("verified_claims") or []),
            unverified_claims=list(d.get("unverified_claims") or []),
            contradictions=list(d.get("contradictions") or []),
            validation_errors=list(d.get("validation_errors") or []),
            resolution_messages=list(d.get("resolution_messages") or []),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified_claims": list(self.verified_claims),
            "unverified_claims": list(self.unverified_claims),
            "contradictions": list(self.contradictions),
            "validation_errors": list(self.validation_errors),
            "resolution_messages": list(self.resolution_messages),
        }


def _numeric_in_claim_text(claim: str, value: Any, unit: str) -> bool:
    if value is None:
        return False
    c = claim.lower()
    vs = str(value).lower()
    if vs and vs in c:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if str(int(value)) in claim or str(value) in claim:
            return True
    u = (unit or "").strip().lower()
    if u and u in c:
        return True
    return False


def _validate_single_claim(
    claim_obj: dict[str, Any],
    *,
    allowed_types: frozenset[str],
    claim_types: frozenset[str],
    pricing_requires_url: bool,
    research_demote_weak: bool,
    evidence_policy: EvidencePolicy,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Returns (is_verified, errors, normalized_copy)."""
    errs: list[str] = []
    cid = str(claim_obj.get("id", "")).strip()
    claim = str(claim_obj.get("claim", "")).strip()
    ctype = str(claim_obj.get("type", "other")).strip().lower()
    if ctype not in claim_types:
        errs.append(f"claim {cid or '?'}: invalid type {ctype!r}")
        ctype = "other"
    if not cid:
        errs.append("claim missing id")
    if not claim:
        errs.append(f"claim {cid or '?'}: empty claim text")

    formula_raw = claim_obj.get("formula")
    formula = str(formula_raw).strip() if formula_raw is not None else ""

    evidence = claim_obj.get("evidence")
    if not isinstance(evidence, list) or len(evidence) == 0:
        errs.append(f"claim {cid or '?'}: evidence must be a non-empty array")

    ev_copy: list[dict[str, Any]] = []
    if isinstance(evidence, list):
        for j, ev in enumerate(evidence):
            if not isinstance(ev, dict):
                errs.append(f"claim {cid or '?'} evidence[{j}]: not an object")
                continue
            st = str(ev.get("source_type", "")).strip()
            url = str(ev.get("source_url", "")).strip()
            quote = str(ev.get("quote_or_snippet", ev.get("snippet", ""))).strip()
            ret = str(ev.get("retrieved_at", "")).strip() or _today_iso()
            if st not in allowed_types:
                errs.append(f"claim {cid or '?'}: disallowed source_type {st!r}")
            if st not in ("unknown", "internal_worker_output") and not url:
                errs.append(f"claim {cid or '?'}: source_url required for source_type {st!r}")
            if pricing_requires_url and ctype == "pricing" and st in ("unknown", "internal_worker_output") and not url:
                errs.append(f"claim {cid or '?'}: pricing claims require a non-empty source_url")
            if not quote:
                errs.append(f"claim {cid or '?'}: empty quote_or_snippet")
            ev_copy.append(
                {
                    "source_url": url,
                    "source_type": st,
                    "quote_or_snippet": quote,
                    "retrieved_at": ret,
                }
            )

    if pricing_requires_url and ctype == "pricing" and ev_copy:
        if not any(
            str(e.get("source_url", "")).strip().lower().startswith(("http://", "https://"))
            for e in ev_copy
        ):
            errs.append(f"claim {cid or '?'}: pricing requires at least one http(s) source_url")

    numeric_values = claim_obj.get("numeric_values")
    nv_copy: list[dict[str, Any]] = []
    if ctype == "pricing":
        if not isinstance(numeric_values, list) or len(numeric_values) == 0:
            errs.append(f"claim {cid or '?'}: pricing type requires non-empty numeric_values")
        else:
            for k, nv in enumerate(numeric_values):
                if not isinstance(nv, dict):
                    errs.append(f"claim {cid or '?'} numeric_values[{k}]: not an object")
                    continue
                val = nv.get("value")
                unit = str(nv.get("unit", "")).strip()
                meaning = str(nv.get("meaning", "")).strip()
                if val is None or (isinstance(val, str) and not str(val).strip()):
                    errs.append(f"claim {cid or '?'} numeric_values[{k}]: missing value")
                if not unit:
                    errs.append(f"claim {cid or '?'} numeric_values[{k}]: missing unit")
                if not meaning:
                    errs.append(f"claim {cid or '?'} numeric_values[{k}]: missing meaning")
                nv_copy.append({"value": val, "unit": unit, "meaning": meaning})
                if val is not None and not _numeric_in_claim_text(claim, val, unit):
                    errs.append(
                        f"claim {cid or '?'}: numeric value {val!r} ({unit}) not clearly reflected in claim text"
                    )
    else:
        if isinstance(numeric_values, list):
            for nv in numeric_values:
                if isinstance(nv, dict):
                    nv_copy.append(
                        {
                            "value": nv.get("value"),
                            "unit": str(nv.get("unit", "")).strip(),
                            "meaning": str(nv.get("meaning", "")).strip(),
                        }
                    )

    qty_implies = claim_text_implies_quantities(claim)
    if qty_implies:
        if not nv_copy or not any(
            isinstance(nv, dict) and nv.get("value") not in (None, "") and str(nv.get("unit", "")).strip()
            for nv in nv_copy
        ):
            errs.append(
                f"claim {cid or '?'}: claim text implies quantities ($, %, GB, tokens, etc.) "
                "but structured numeric_values is missing or incomplete"
            )
        else:
            for k, nv in enumerate(nv_copy):
                if not isinstance(nv, dict):
                    continue
                val, unit, meaning = nv.get("value"), str(nv.get("unit", "")).strip(), str(nv.get("meaning", "")).strip()
                if val is not None and str(val).strip() and unit and not _numeric_in_claim_text(claim, val, unit):
                    errs.append(
                        f"claim {cid or '?'}: numeric_values[{k}] not clearly reflected in claim text "
                        f"({val!r} {unit})"
                    )

    if claim_text_suggests_estimate(claim) and qty_implies and not formula:
        errs.append(f"claim {cid or '?'}: estimate-style claim with quantities requires non-empty formula field")

    conf = _confidence_bucket(claim_obj.get("confidence"))
    caveats = _as_str_list(claim_obj.get("caveats"))
    if isinstance(claim_obj.get("caveats"), str) and claim_obj.get("caveats"):
        caveats = [str(claim_obj["caveats"]).strip()] if str(claim_obj["caveats"]).strip() else caveats

    norm: dict[str, Any] = {
        "id": cid,
        "claim": claim,
        "type": ctype,
        "provider": str(claim_obj.get("provider", "")).strip(),
        "evidence": ev_copy,
        "numeric_values": nv_copy,
        "confidence": conf,
        "caveats": caveats,
    }
    if formula:
        norm["formula"] = formula

    verified = len(errs) == 0
    if verified:
        tier_err = _tier_violation_message(
            tier_mode=evidence_policy.tier_mode, ctype=ctype, provider=norm["provider"], ev_copy=ev_copy
        )
        if tier_err:
            verified = False
            errs.append(tier_err)

    if verified and evidence_policy.tier_mode == "academic_strict":
        ac_err = _academic_strict_claim_violation(ctype=ctype, ev_copy=ev_copy)
        if ac_err:
            verified = False
            errs.append(ac_err)

    if verified and research_demote_weak and evidence_policy.tier_mode == "none":
        if conf == "low":
            verified = False
            errs.append("demoted: low confidence")
        elif conf == "medium":
            if any((e.get("source_type") in _WEAK_SOURCE_TYPES) for e in ev_copy):
                verified = False
                errs.append("demoted: medium confidence with weak source_type")

    return verified, errs, norm


def validate_claims_document(
    doc: dict[str, Any],
    *,
    allowed_source_types: frozenset[str],
    pricing_requires_url: bool,
    research_demote_weak: bool,
    evidence_policy: EvidencePolicy | None = None,
) -> ClaimValidationResult:
    policy = evidence_policy or resolve_evidence_policy("normal")
    result = ClaimValidationResult()
    result.contradictions = list(doc.get("contradictions") or []) if isinstance(doc.get("contradictions"), list) else []

    for u in doc.get("unverified_claims") or []:
        if isinstance(u, dict) and str(u.get("claim", "")).strip():
            result.unverified_claims.append(u)

    claims_in = doc.get("claims")
    if not isinstance(claims_in, list):
        result.validation_errors.append("top-level 'claims' must be an array")
        return result

    claim_types = claim_types_for_policy(policy)
    for claim_obj in claims_in:
        if not isinstance(claim_obj, dict):
            result.validation_errors.append("claims[] entry is not an object")
            continue
        ok, errs, norm = _validate_single_claim(
            claim_obj,
            allowed_types=allowed_source_types,
            claim_types=claim_types,
            pricing_requires_url=pricing_requires_url,
            research_demote_weak=research_demote_weak,
            evidence_policy=policy,
        )
        if ok:
            result.verified_claims.append(norm)
        else:
            norm["_validation_errors"] = errs
            result.unverified_claims.append(norm)
            for e in errs:
                result.validation_errors.append(f"{norm.get('id', '?')}: {e}")

    return result


def _claim_sort_tuple(claim: dict[str, Any]) -> tuple[int, datetime, str]:
    ev = claim.get("evidence") or []
    if not isinstance(ev, list):
        ev = []
    rank = _claim_max_source_rank(ev)
    dt = _claim_max_evidence_date(ev) or datetime.min.replace(tzinfo=None)
    return (rank, dt, claim.get("id", ""))


def resolve_cross_task_conflicts(
    by_task: dict[str, ClaimValidationResult],
    *,
    enabled: bool = True,
) -> dict[str, ClaimValidationResult]:
    """Deterministic winner for same provider+metric+unit with differing values; demotes losers to unverified."""
    if not enabled or len(by_task) < 2:
        return by_task

    out: dict[str, ClaimValidationResult] = {}
    for tid, res in by_task.items():
        out[tid] = ClaimValidationResult(
            verified_claims=list(res.verified_claims),
            unverified_claims=list(res.unverified_claims),
            contradictions=list(res.contradictions),
            validation_errors=list(res.validation_errors),
            resolution_messages=list(res.resolution_messages),
        )

    indexed: list[tuple[str, dict[str, Any]]] = []
    for tid, res in out.items():
        for c in res.verified_claims:
            if isinstance(c, dict):
                indexed.append((tid, c))

    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for tid, c in indexed:
        gk = conflict_group_key(c)
        if not gk:
            continue
        groups.setdefault(gk, []).append((tid, c))

    used_official_vs_blog_msg = False
    for gk, members in groups.items():
        by_val: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for tid, c in members:
            nvs = c.get("numeric_values") or []
            if not nvs or not isinstance(nvs[0], dict):
                continue
            vk = _normalize_scalar_for_compare(nvs[0].get("value"))
            by_val.setdefault(vk, []).append((tid, c))
        if len(by_val) < 2:
            continue

        flat = [x for xs in by_val.values() for x in xs]
        flat.sort(key=lambda tc: _claim_sort_tuple(tc[1]), reverse=True)
        winner_tid, winner = flat[0]

        winner_stypes = {str(e.get("source_type", "")).strip() for e in (winner.get("evidence") or []) if isinstance(e, dict)}
        winner_is_official_pricing = bool(winner_stypes & _PRICING_OFFICIAL_TYPES)

        for loser_tid, loser in flat[1:]:
            loser_ev = loser.get("evidence") or []
            loser_weak_only = _claim_only_weak_third_party(loser_ev if isinstance(loser_ev, list) else [])

            for t_id, res in out.items():
                if t_id != loser_tid:
                    continue
                try:
                    idx = next(i for i, x in enumerate(res.verified_claims) if x is loser)
                except StopIteration:
                    continue
                res.verified_claims.pop(idx)
                reason = (
                    f"cross-task conflict: superseded by claim {winner.get('id')} in task {winner_tid} "
                    f"(same provider/metric/unit, higher-precedence evidence)"
                )
                loser_copy = dict(loser)
                loser_copy["_validation_errors"] = [reason]
                loser_copy["_conflict_reason"] = reason
                res.unverified_claims.append(loser_copy)
                res.validation_errors.append(f"{loser_copy.get('id', '?')}: {reason}")
                note = (
                    f"Cross-task conflict on {gk.replace(chr(31), ' / ')}: kept [{winner_tid}] "
                    f"{winner.get('id')}; demoted [{loser_tid}] {loser_copy.get('id')}."
                )
                if note not in res.contradictions:
                    res.contradictions.append(note)

            if winner_is_official_pricing and loser_weak_only and not used_official_vs_blog_msg:
                msg = "Conflicting third-party pricing was found; official pricing page was used."
                out[winner_tid].resolution_messages.append(msg)
                used_official_vs_blog_msg = True

    return out


def format_claims_for_merge(
    validated: ClaimValidationResult,
    task_id: str,
    *,
    research_verified_only_body: bool,
) -> str:
    """Human-readable block for one task (merger input)."""
    lines: list[str] = [f"## Claims ledger (task `{task_id}`)", ""]

    def _emit_claim(title: str, c: dict[str, Any], indent: str = "") -> None:
        cid = c.get("id", "?")
        lines.append(f"{indent}### {title} [{cid}]")
        lines.append(f"{indent}- **Claim:** {c.get('claim', '')}")
        lines.append(f"{indent}- **Type:** {c.get('type', '')} | **Confidence:** {c.get('confidence', '')}")
        if c.get("provider"):
            lines.append(f"{indent}- **Provider:** {c['provider']}")
        if c.get("formula"):
            lines.append(f"{indent}- **Formula:** {c.get('formula')}")
        for ev in c.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            lines.append(
                f"{indent}- **Evidence:** {ev.get('source_type', '')} | {ev.get('source_url', '')} | "
                f"retrieved {ev.get('retrieved_at', '')}"
            )
            if ev.get("quote_or_snippet"):
                lines.append(f"{indent}  > {ev.get('quote_or_snippet')}")
        for nv in c.get("numeric_values") or []:
            if isinstance(nv, dict):
                lines.append(f"{indent}- **Numeric:** {nv.get('value')} {nv.get('unit')} — {nv.get('meaning', '')}")
        cav = c.get("caveats") or []
        if cav:
            lines.append(f"{indent}- **Caveats:** {cav}")
        if c.get("_validation_errors"):
            lines.append(f"{indent}- **Validation notes:** {c['_validation_errors']}")
        lines.append("")

    if validated.resolution_messages:
        lines.append("### Ledger resolution (orchestrator)")
        for m in validated.resolution_messages:
            lines.append(f"- {m}")
        lines.append("")

    if research_verified_only_body:
        lines.append("### Verified claims (may state as facts in the narrative)")
        if not validated.verified_claims:
            lines.append("(none — do not invent factual answers beyond what is verified below.)")
            lines.append("")
        for c in validated.verified_claims:
            _emit_claim("Verified", c, "")
        lines.append("### Unverified / failed validation (appendix only; do not state as established facts)")
        for c in validated.unverified_claims:
            _emit_claim("Unverified", c, "")
        if validated.contradictions:
            lines.append("### Contradictions (resolve cautiously; do not assert resolution without evidence)")
            for item in validated.contradictions:
                lines.append(f"- {item}")
            lines.append("")
    else:
        lines.append("### Verified claims")
        for c in validated.verified_claims:
            _emit_claim("Verified", c, "")
        lines.append("### Unverified claims (from model)")
        for c in validated.unverified_claims:
            _emit_claim("Unverified", c, "")
        if validated.contradictions:
            lines.append("### Contradictions")
            for item in validated.contradictions:
                lines.append(f"- {item}")
            lines.append("")

    if validated.validation_errors:
        lines.append("### Validation errors (orchestrator)")
        for e in validated.validation_errors[:40]:
            lines.append(f"- {e}")
        if len(validated.validation_errors) > 40:
            lines.append(f"- ... ({len(validated.validation_errors) - 40} more)")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_excluded_claims_digest(
    raw_by_tid: dict[str, Any],
    *,
    max_chars: int = 10_000,
) -> str:
    """Markdown bullets for final synthesis: unverified / excluded with short reasons."""
    lines: list[str] = []
    for tid in sorted(raw_by_tid.keys()):
        doc = raw_by_tid.get(tid) or {}
        unv = doc.get("unverified_claims") or []
        if not isinstance(unv, list):
            continue
        for c in unv:
            if not isinstance(c, dict):
                continue
            claim = str(c.get("claim", "")).strip().replace("\n", " ")
            if not claim:
                continue
            prov = str(c.get("provider", "")).strip()
            bits: list[str] = []
            if c.get("_conflict_reason"):
                bits.append(str(c["_conflict_reason"]))
            elif c.get("_validation_errors"):
                bits.append("; ".join(str(x) for x in (c.get("_validation_errors") or [])[:3]))
            else:
                ev = c.get("evidence") or []
                stypes = [str(e.get("source_type", "")) for e in ev if isinstance(e, dict)]
                bits.append(f"source_type={','.join(stypes) or 'unknown'}")
            head = f"{prov}: " if prov else ""
            line = f"- **{tid}** [{c.get('id', '?')}] {head}{claim[:400]}{'…' if len(claim) > 400 else ''} — _{bits[0][:500]}_"
            lines.append(line)
    body = "\n".join(lines)
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 20].rstrip() + "\n…(truncated)"


def claims_for_run_report(
    validated: ClaimValidationResult,
    *,
    task_id: str,
    worker_slot: int,
) -> list[dict[str, Any]]:
    """Flatten claims with provenance for JSON export."""
    out: list[dict[str, Any]] = []
    for bucket, label in (
        (validated.verified_claims, "verified"),
        (validated.unverified_claims, "unverified"),
    ):
        for c in bucket:
            row = dict(c)
            row["source_task_id"] = task_id
            row["source_worker_slot"] = worker_slot
            row["verification_bucket"] = label
            out.append(row)
    return out


def deliverables_coverage_gaps(final_text: str, deliverables: list[str]) -> list[str]:
    if not deliverables:
        return []
    low = final_text.lower()
    missing: list[str] = []
    for d in deliverables:
        t = d.strip()
        if not t:
            continue
        key = re.sub(r"\s+", " ", t.lower())[:120]
        if len(key) >= 4 and key in low:
            continue
        words = [w for w in re.findall(r"[a-z0-9]{4,}", t.lower()) if len(w) >= 4]
        if words and any(w in low for w in words[:5]):
            continue
        missing.append(t)
    return missing
