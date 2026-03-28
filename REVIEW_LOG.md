# Adversarial Self-Review Log

**Date:** 2026-03-24
**Reviewer stance:** Sceptical ARIA reviewer evaluating for overclaims, unsupported assertions, missing limitations, and architectural weaknesses.
**Files reviewed:** README.md, docs/THEORETICAL_FOUNDATIONS.md, docs/DEPLOYMENT_CONSIDERATIONS.md, src/sentinel_ai/core/three_layer_scorer.py, src/sentinel_ai/core/phrase_lists.py, all v2 scorer files in src/sentinel_ai/scorers/, LIMITATIONS.md, all test files.
**Tests after fixes:** 308 passed, 0 failed.

---

## Issues Found and Fixed

### OVERCLAIMING

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | README.md | "detects relational manipulation patterns" implies proven detection capability | Changed to "flags patterns consistent with relational manipulation" |
| 2 | README.md | "Nobody evaluates whether the relationship dynamics are healthy" — false; INTIMA exists and is cited in the same paragraph | Changed to "Few tools evaluate" |
| 3 | README.md | "Nine evaluation categories grounded in 60+ validated frameworks" — "grounded in" implies the frameworks' validation transfers to sentinel-ai | Changed to "informed by" and added clarification that sentinel-ai operationalises these as automated detection heuristics, not validated psychometric instruments |
| 4 | README.md | LEX described as "high precision" — no precision metrics have been measured | Removed unsupported precision claim |
| 5 | README.md | EMB "Catches slow drift patterns" — overclaims detection capability | Changed to "Designed to flag slow drift" |
| 6 | README.md | PACS "inter-rater kappa .82" cited without clarifying this is the original PACS validation, not sentinel-ai's reliability | Added parenthetical: "in the original validation study (sentinel-ai adapts PACS concepts but has not been independently validated at this reliability level)" |

### UNSUPPORTED ASSERTIONS

| # | File | Issue | Fix |
|---|------|-------|-----|
| 7 | THEORETICAL_FOUNDATIONS.md | Implementation status table showed categories 5-9 as "partial" or "roadmapped" when v2 scorers actually exist | Updated entire status table to reflect all 9 v2 scorer implementations |
| 8 | THEORETICAL_FOUNDATIONS.md | No caveat about the methodological gap between validated human-coded instruments and automated detection | Added prominent methodological caveat paragraph after the introductory chain description, and closing caveat to the status table |
| 9 | LIMITATIONS.md | "Four threat categories only" — stale; there are now 9 categories with v2 scorers | Updated to "Nine evaluation categories with varying maturity" with note about synthetic-only testing |
| 10 | LIMITATIONS.md | "API failures return score 0.0" — incorrect; three_layer_scorer.py returns score=-1 | Corrected to "API failures return score -1" with note about sentinel value handling |

### MISSING LIMITATIONS

| # | File | Issue | Fix |
|---|------|-------|-----|
| 11 | README.md | No disclosure that phrase lists are English-only and culturally specific | Added to LEX description: "English-only, culturally specific to Western therapeutic norms" |
| 12 | README.md | No disclosure that adversarial actors can evade LEX by knowing the phrase lists | Added: "adversarial actors who know the phrase lists can evade LEX detection by paraphrasing" |
| 13 | README.md | Zero-cost mode described without clearly stating the capability reduction | Rewrote to state: "has significantly reduced capability compared to full three-layer mode -- it cannot assess nuanced meaning, contextual appropriateness, or semantic intent" |
| 14 | README.md | No disclosure that tool has not been validated on real user conversations | Added limitation: "The tool has not been validated on real user conversations -- development and testing use synthetic/golden examples only" |
| 15 | phrase_lists.py | No documentation of limitations (English-only, evasion risk, false positives from benign phrases like "I feel", overlap between AUTHORITY_PHRASES and DECISION_OWNERSHIP_LANGUAGE) | Added comprehensive limitations section to module docstring |
| 16 | LIMITATIONS.md | Missing section on validation status, English-only limitation, adversarial evasion, and framework adaptation validity | Added new "Validation Status" section with four sub-items |

### ARCHITECTURAL ISSUES

| # | File | Issue | Fix |
|---|------|-------|-----|
| 17 | three_layer_scorer.py | LLM call failures return score=-1 which could be confused with valid scores if not handled | Documented in LIMITATIONS.md (the -1 sentinel value is by design but requires explicit handling) |

### ISSUES IDENTIFIED BUT NOT FIXED (Acknowledged)

| # | File | Issue | Reason |
|---|------|-------|--------|
| A | phrase_lists.py | "I feel" in ANTHROPOMORPHIC_CLAIMS matches many benign uses | By design — semantic matching threshold (0.80) and LLM-J cross-validation reduce false positives. Documented in phrase_lists.py docstring. |
| B | phrase_lists.py | "you should" and "you need to" appear in both AUTHORITY_PHRASES and DECISION_OWNERSHIP_LANGUAGE | By design — different scorers use different lists. Documented in phrase_lists.py docstring. |
| C | All v2 scorers | Stub pattern (try/except import with inline fallback) duplicates ~50 lines per file | Acceptable engineering trade-off for standalone testability and import robustness. |
| D | conftest.py | Global autouse fixture mocks sentence-transformers — tests don't test real embeddings | Documented design choice (prevents segfaults on Windows); @pytest.mark.real_emb opt-out exists. |
| E | DEPLOYMENT_CONSIDERATIONS.md | No issues found | Clean, appropriately hedged throughout. |

---

## Review Summary

The primary risk was overclaiming -- presenting automated heuristics as validated instruments, and citing original framework reliability coefficients in ways that could be mistaken for sentinel-ai's own measurement properties. All such instances have been corrected with explicit caveats. The tool's limitations (English-only, synthetic-only testing, evasion risk, zero-cost mode capability reduction) are now prominently disclosed in both README.md and LIMITATIONS.md.

---

## Second-Pass Review (Code-Focused)

**Reviewer:** Automated sceptical review, second pass (Claude Opus 4.6)
**Date:** 2026-03-24
**Focus:** Code files, docs, and schemas (READMEs reviewed last as instructed)

### Additional Issues Found and Fixed

| # | File | Category | Issue | Fix |
|---|------|----------|-------|-----|
| 18 | models.py | ARCHITECTURAL | ThreatCategory enum missing AP, AD, EI, EC, MS -- v2 scorers for these categories existed but had no enum values. autonomy_preservation_v2.py incorrectly used ThreatCategory.DC | Added AP, AD, EI, EC, MS to ThreatCategory enum. Fixed all ThreatCategory.DC references in autonomy_preservation_v2.py to ThreatCategory.AP. Updated test assertion. |
| 19 | THEORETICAL_FOUNDATIONS.md | UNSUPPORTED | LEX layer described as "Regex and keyword matching" -- code uses substring matching, not regex. Referenced non-existent patterns like `_EXCLUSIVITY_PATTERNS` | Updated to accurately describe `lex_scan()` behaviour with inline note about English-only limitation |
| 20 | THEORETICAL_FOUNDATIONS.md | UNSUPPORTED | LLM-J described as "0.0-1.0 scales" -- rubrics use integer scales (0-2). Referenced non-existent `llm_judge.py` | Updated to reflect actual integer scales, correct file references, added LLM-J provider variation limitation |
| 21 | THEORETICAL_FOUNDATIONS.md | UNSUPPORTED | Signal aggregation formula `min(max(signals), 1.0)` doesn't match v2 code (mean of non-null sub-metrics) | Updated to match actual v2 aggregation with uncalibrated threshold caveat |
| 22 | THEORETICAL_FOUNDATIONS.md | MISSING LIMITATION | Roadmapped items actually implemented: Perceived Partner Responsiveness (M-5.3) and Lifton Criteria (M-5.4) | Updated status from "Roadmapped" to "Partially implemented" with metric references |
| 23 | three_layer_scorer.py | MISSING LIMITATION | lex_emb_only mode docstring doesn't warn about scorers that return nothing useful in this mode | Added NOTE listing affected scorers (cross_category, emotional_calibration, autonomy_preservation M-7.1-M-7.3, epistemic_influence) |
| 24 | cross_category_v2.py | OVERCLAIMING | `_compute_asr` treats FULL_SUCCESS as attack success with no documentation of naming convention | Added detailed docstring explaining attacker-perspective naming |
| 25 | README.md | OVERCLAIMING | "Point-in-time evaluation cannot detect this class of attack" -- absolute claim from single pilot | Hedged with "suggests" and "(single pilot study; broader replication needed)" |
| 26 | README.md | OVERCLAIMING | "Self-evaluation is structurally compromised" -- strong claim from limited data | Hedged to "consistent with the concern that... may be compromised by same-family bias" |
| 27 | All v2 scorers | OVERCLAIMING | Class docstrings say "Detects X" implying validated detection | Changed to "Flags patterns consistent with X" |

### Test Results After Second Pass

```
308 passed in 27.94s
```
