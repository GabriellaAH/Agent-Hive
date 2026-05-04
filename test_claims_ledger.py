"""Unit tests for claims ledger, evidence policy, conflicts (no LLM)."""
from __future__ import annotations

import unittest

from claims_ledger import (
    ClaimValidationResult,
    build_excluded_claims_digest,
    claim_text_implies_quantities,
    conflict_group_key,
    resolve_cross_task_conflicts,
    resolve_evidence_policy,
    scholarly_citation_url_ok,
    validate_claims_document,
)


def _base_evidence(source_type: str, url: str = "https://example.com/p") -> dict:
    return {
        "source_url": url,
        "source_type": source_type,
        "quote_or_snippet": "excerpt",
        "retrieved_at": "2026-05-01",
    }


class TestEvidencePolicy(unittest.TestCase):
    def test_academic_strict_policy(self) -> None:
        policy = resolve_evidence_policy("academic_strict")
        self.assertEqual(policy.name, "academic_strict")
        self.assertEqual(policy.tier_mode, "academic_strict")
        self.assertIn("academic_metadata", policy.allowed_source_types)

    def test_scholarly_citation_url_ok(self) -> None:
        self.assertTrue(scholarly_citation_url_ok("https://doi.org/10.1234/abcd.999"))
        self.assertTrue(scholarly_citation_url_ok("https://api.openalex.org/works/W123"))
        self.assertFalse(scholarly_citation_url_ok("https://medium.com/@x/paper"))
        self.assertFalse(scholarly_citation_url_ok("https://example.com"))

    def test_default_allowlist_includes_calculator(self) -> None:
        policy = resolve_evidence_policy("pricing_strict")
        self.assertIn("official_pricing_calculator", policy.allowed_source_types)
        self.assertEqual(policy.tier_mode, "pricing_strict")

    def test_pricing_strict_demotes_blog_pricing(self) -> None:
        pol = resolve_evidence_policy("pricing_strict")
        doc = {
            "claims": [
                {
                    "id": "c1",
                    "claim": "Acme costs $1 per GB.",
                    "type": "pricing",
                    "provider": "Acme",
                    "evidence": [_base_evidence("blog")],
                    "numeric_values": [{"value": 1, "unit": "USD/GB", "meaning": "storage"}],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(any("pricing_strict" in e for e in r.validation_errors))

    def test_academic_strict_design_internal_only_verifies(self) -> None:
        pol = resolve_evidence_policy("academic_strict")
        doc = {
            "claims": [
                {
                    "id": "d1",
                    "claim": "The proposal will use three empirical chapters.",
                    "type": "academic_design",
                    "provider": "",
                    "evidence": [
                        {
                            "source_url": "",
                            "source_type": "internal_worker_output",
                            "quote_or_snippet": "three empirical chapters are planned in the outline.",
                            "retrieved_at": "2026-05-01",
                        }
                    ],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 1)

    def test_academic_strict_paper_finding_internal_only_fails(self) -> None:
        pol = resolve_evidence_policy("academic_strict")
        doc = {
            "claims": [
                {
                    "id": "p1",
                    "claim": "Smith (2020) finds a positive effect.",
                    "type": "paper_finding",
                    "provider": "",
                    "evidence": [
                        {
                            "source_url": "",
                            "source_type": "internal_worker_output",
                            "quote_or_snippet": "Smith (2020) finds a positive effect on returns.",
                            "retrieved_at": "2026-05-01",
                        }
                    ],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(
            any("internal_worker_output cannot verify" in e for e in r.validation_errors)
        )

    def test_academic_strict_citation_doi_verifies(self) -> None:
        pol = resolve_evidence_policy("academic_strict")
        doc = {
            "claims": [
                {
                    "id": "c1",
                    "claim": "DOI 10.9999/zz.1 indexes the target paper.",
                    "type": "academic_citation",
                    "provider": "",
                    "evidence": [
                        {
                            "source_url": "https://doi.org/10.9999/zz.1",
                            "source_type": "academic_paper",
                            "quote_or_snippet": "landing page title matches the registered work.",
                            "retrieved_at": "2026-05-01",
                        }
                    ],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 1)

    def test_academic_strict_citation_bad_host_fails(self) -> None:
        pol = resolve_evidence_policy("academic_strict")
        doc = {
            "claims": [
                {
                    "id": "c2",
                    "claim": "This blog post is the canonical citation.",
                    "type": "academic_citation",
                    "provider": "",
                    "evidence": [
                        {
                            "source_url": "https://medium.com/story/123",
                            "source_type": "academic_paper",
                            "quote_or_snippet": "not acceptable as scholarly citation URL.",
                            "retrieved_at": "2026-05-01",
                        }
                    ],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(any("academic_citation requires at least one" in e for e in r.validation_errors))

    def test_academic_strict_sources_only_mixed_types_fails(self) -> None:
        pol = resolve_evidence_policy("academic_strict")
        doc = {
            "claims": [
                {
                    "id": "s1",
                    "claim": "Only peer-reviewed academic references are cited.",
                    "type": "academic_sources_only",
                    "provider": "",
                    "evidence": [
                        {
                            "source_url": "https://doi.org/10.1/one",
                            "source_type": "academic_paper",
                            "quote_or_snippet": "first ref",
                            "retrieved_at": "2026-05-01",
                        },
                        {
                            "source_url": "https://vendor.example/docs",
                            "source_type": "official_docs",
                            "quote_or_snippet": "vendor doc cited by mistake",
                            "retrieved_at": "2026-05-01",
                        },
                    ],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(any("academic_sources_only requires every" in e for e in r.validation_errors))

    def test_academic_strict_paper_finding_long_quote_verifies(self) -> None:
        pol = resolve_evidence_policy("academic_strict")
        excerpt = "Q" * 80
        doc = {
            "claims": [
                {
                    "id": "pf1",
                    "claim": "The study reports higher accuracy under treatment.",
                    "type": "paper_finding",
                    "provider": "",
                    "evidence": [
                        {
                            "source_url": "https://doi.org/10.7777/ok.1",
                            "source_type": "academic_paper",
                            "quote_or_snippet": excerpt,
                            "retrieved_at": "2026-05-01",
                        }
                    ],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 1)

    def test_paper_finding_invalid_type_without_academic_strict(self) -> None:
        pol = resolve_evidence_policy("normal")
        doc = {
            "claims": [
                {
                    "id": "x1",
                    "claim": "Any claim.",
                    "type": "paper_finding",
                    "provider": "",
                    "evidence": [_base_evidence("official_docs")],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(any("invalid type" in e for e in r.validation_errors))

    def test_strict_research_demotes_any_weak_evidence(self) -> None:
        pol = resolve_evidence_policy("strict_research")
        doc = {
            "claims": [
                {
                    "id": "c1",
                    "claim": "Acme ships feature X in 2026.",
                    "type": "factual",
                    "provider": "Acme",
                    "evidence": [_base_evidence("official_docs"), _base_evidence("blog", "https://b.com/x")],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)


class TestQuantityFormula(unittest.TestCase):
    def test_implies_quantities(self) -> None:
        self.assertTrue(claim_text_implies_quantities("Price is $0.20 per GB."))
        self.assertTrue(claim_text_implies_quantities("10M vectors per month"))
        self.assertFalse(claim_text_implies_quantities("The product is generally available."))

    def test_quantity_requires_numeric_values_non_pricing(self) -> None:
        pol = resolve_evidence_policy("normal")
        doc = {
            "claims": [
                {
                    "id": "c1",
                    "claim": "Widget costs $5 per unit.",
                    "type": "factual",
                    "provider": "",
                    "evidence": [_base_evidence("official_docs")],
                    "numeric_values": [],
                    "confidence": "high",
                    "caveats": [],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(any("numeric_values" in e for e in r.validation_errors))

    def test_estimate_requires_formula(self) -> None:
        pol = resolve_evidence_policy("normal")
        doc = {
            "claims": [
                {
                    "id": "c1",
                    "claim": "We estimate roughly $20–$65 per million vectors.",
                    "type": "factual",
                    "provider": "Zilliz",
                    "evidence": [_base_evidence("official_docs")],
                    "numeric_values": [
                        {"value": "20-65", "unit": "USD/M vectors", "meaning": "published estimate range"}
                    ],
                    "confidence": "medium",
                    "caveats": ["range"],
                }
            ],
            "unverified_claims": [],
            "contradictions": [],
        }
        r = validate_claims_document(
            doc,
            allowed_source_types=pol.allowed_source_types,
            pricing_requires_url=False,
            research_demote_weak=False,
            evidence_policy=pol,
        )
        self.assertEqual(len(r.verified_claims), 0)
        self.assertTrue(any("formula" in e for e in r.validation_errors))


class TestCrossTaskConflicts(unittest.TestCase):
    def test_official_wins_over_blog(self) -> None:
        claim_official = {
            "id": "w1",
            "claim": "Pine charges $0.10 per GB.",
            "type": "pricing",
            "provider": "Pinecone",
            "evidence": [_base_evidence("official_pricing_page", "https://pinecone.io/pricing")],
            "numeric_values": [{"value": 0.1, "unit": "USD/GB", "meaning": "storage"}],
            "confidence": "high",
            "caveats": [],
        }
        claim_blog = {
            "id": "w2",
            "claim": "Pine charges $0.20 per GB per blogs.",
            "type": "pricing",
            "provider": "Pinecone",
            "evidence": [_base_evidence("blog", "https://random.blog/pine")],
            "numeric_values": [{"value": 0.2, "unit": "USD/GB", "meaning": "storage"}],
            "confidence": "high",
            "caveats": [],
        }
        self.assertEqual(conflict_group_key(claim_official), conflict_group_key(claim_blog))
        a = ClaimValidationResult(verified_claims=[claim_official], unverified_claims=[], contradictions=[])
        b = ClaimValidationResult(verified_claims=[claim_blog], unverified_claims=[], contradictions=[])
        out = resolve_cross_task_conflicts({"t1": a, "t2": b}, enabled=True)
        self.assertEqual(len(out["t1"].verified_claims), 1)
        self.assertEqual(len(out["t2"].verified_claims), 0)
        self.assertEqual(len(out["t2"].unverified_claims), 1)
        msgs = [m for r in out.values() for m in r.resolution_messages]
        self.assertTrue(any("third-party" in m.lower() for m in msgs))


class TestExcludedDigest(unittest.TestCase):
    def test_digest_lists_unverified(self) -> None:
        raw = {
            "t1": {
                "verified_claims": [],
                "unverified_claims": [
                    {
                        "id": "u1",
                        "claim": "Blog said price is low.",
                        "provider": "X",
                        "_validation_errors": ["policy pricing_strict"],
                    }
                ],
                "contradictions": [],
                "validation_errors": [],
                "resolution_messages": [],
            }
        }
        d = build_excluded_claims_digest(raw)
        self.assertIn("u1", d)
        self.assertIn("X", d)


if __name__ == "__main__":
    unittest.main()
