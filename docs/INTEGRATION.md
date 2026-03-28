# Cross-Tool Integration: sentinel-ai + dormancy-detect + verifiable-eval

> **Status: Work in Progress.** The three tools currently operate as independent, standalone tools. This document describes the *planned* pipeline architecture for a future release. No automated cross-tool integration exists yet — each tool must be run independently.

This document describes how the three tools in the relational safety evaluation suite are intended to connect as a pipeline, what each produces, and how data will flow between them.

---

## 1. Pipeline Overview

```
+-----------------------------------------------------------------------------+
|                    RELATIONAL SAFETY EVALUATION PIPELINE                     |
+-----------------------------------------------------------------------------+

  CONVERSATION          SENTINEL-AI             DORMANCY-DETECT         VERIFIABLE-EVAL
  INPUT                 (per-session)           (cross-session)         (certification)

  +----------+     +-------------------+    +----------------------+   +------------------+
  |          |     |  Per-session       |    |  Temporal attack     |   |  Tamper-evident   |
  |  Chat    |---->|  threat scores     |--->|  patterns detected   |-->|  safety           |
  |  export  |     |  across 9 SRSEF    |    |                      |   |  certificate      |
  |  (JSON)  |     |  categories        |    |  Change points       |   |                   |
  |          |     |                    |    |  identified          |   |  Hash-chained     |
  +----------+     |  Categories:       |    |                      |   |  execution trace  |
                   |  1. Dependency     |    |  Memory fidelity     |   |                   |
                   |     Dynamics       |    |  scored              |   |  Verifiable by    |
                   |  2. Boundary       |    |                      |   |  third parties    |
                   |     Integrity      |    |  Trend analysis      |   |                   |
                   |  3. Identity       |    |  (direction, rate,   |   +------------------+
                   |     Coherence      |    |   time-to-threshold) |          |
                   |  4. Disclosure     |    |                      |          |
                   |     Regulation     |    +----------------------+          |
                   |  5. Memory Safety  |                                      v
                   |  6. Autonomy       |                              +------------------+
                   |     Preservation   |                              |  CERTIFICATE     |
                   |  7. Epistemic      |                              |  {               |
                   |     Influence      |                              |   scores: {...}, |
                   |  8. Emotional      |                              |   temporal: {..},|
                   |     Calibration    |                              |   hash_chain: [],|
                   |  9. Cross-cutting  |                              |   timestamp: ... |
                   |                    |                              |  }               |
                   +-------------------+                              +------------------+
```

### Tool Responsibilities

| Tool | Scope | Input | Output |
|------|-------|-------|--------|
| **sentinel-ai** | Per-session analysis | Single conversation session | Threat scores across 9 SRSEF categories |
| **dormancy-detect** | Cross-session temporal analysis | Sequence of sentinel-ai scores + raw conversations | Temporal attack patterns, change points, memory fidelity |
| **verifiable-eval** | Certification | Sentinel-ai scores + dormancy-detect analysis | Tamper-evident safety certificate with hash-chained execution trace |

---

## 2. What Each Tool Produces

### sentinel-ai

**Input:** A single conversation session (list of turns with speaker labels).

**Output:** A JSON object containing per-session threat scores across the 9 SRSEF categories:

```json
{
  "session_id": "abc123",
  "timestamp": "2026-03-24T10:00:00Z",
  "scores": {
    "dependency_dynamics": 0.35,
    "boundary_integrity": 0.20,
    "identity_coherence": 0.15,
    "disclosure_regulation": 0.45,
    "memory_safety": 0.10,
    "autonomy_preservation": 0.55,
    "epistemic_influence": 0.40,
    "emotional_calibration": 0.25,
    "cross_cutting": 0.30
  },
  "flags": ["autonomy_high", "disclosure_elevated"],
  "turn_level_detail": ["..."]
}
```

Each score ranges from 0.0 (no concern) to 1.0 (maximum concern). Scores above 0.5 trigger flags.

### dormancy-detect

**Input:** A sequence of sentinel-ai session scores over time, plus raw conversation data for temporal pattern analysis.

**Output:** A JSON object containing temporal analysis results:

```json
{
  "analysis_id": "def456",
  "sessions_analysed": 15,
  "time_span": "2026-01-15 to 2026-03-24",
  "temporal_patterns": [
    {
      "pattern": "boiling_frog",
      "dimension": "autonomy_preservation",
      "start_session": 3,
      "current_session": 15,
      "cumulative_drift": 0.42,
      "severity": "high"
    }
  ],
  "change_points": [
    {
      "session": 8,
      "dimension": "disclosure_regulation",
      "direction": "increase",
      "magnitude": 0.25
    }
  ],
  "memory_fidelity": {
    "contamination_events": 2,
    "fidelity_score": 0.85
  },
  "trend_analysis": {
    "autonomy_preservation": {
      "direction": "increasing_risk",
      "rate": 0.03,
      "estimated_threshold_sessions": 5
    }
  }
}
```

### verifiable-eval

**Input:** Sentinel-ai scores and dormancy-detect temporal analysis.

**Output:** A tamper-evident safety certificate with a hash-chained execution trace:

```json
{
  "certificate_id": "ghi789",
  "issued": "2026-03-24T12:00:00Z",
  "subject": "conversation_set_xyz",
  "scores": {
    "sentinel_ai": {},
    "dormancy_detect": {}
  },
  "temporal_context": {
    "trend_direction": "increasing_risk",
    "change_points": 1,
    "time_to_threshold": "~5 sessions"
  },
  "execution_trace": [
    {
      "step": 1,
      "operation": "sentinel_ai_scoring",
      "input_hash": "sha256:abc...",
      "output_hash": "sha256:def...",
      "timestamp": "2026-03-24T11:55:00Z"
    },
    {
      "step": 2,
      "operation": "dormancy_detect_analysis",
      "input_hash": "sha256:def...",
      "output_hash": "sha256:ghi...",
      "timestamp": "2026-03-24T11:57:00Z"
    },
    {
      "step": 3,
      "operation": "certificate_generation",
      "input_hash": "sha256:ghi...",
      "output_hash": "sha256:jkl...",
      "timestamp": "2026-03-24T12:00:00Z"
    }
  ],
  "hash_chain": [
    "sha256:abc...",
    "sha256:def...",
    "sha256:ghi...",
    "sha256:jkl..."
  ],
  "verification": {
    "method": "hash_chain_traversal",
    "instructions": "Each step output_hash matches the next step input_hash. Recompute any step to verify integrity."
  }
}
```

---

## 3. Data Flow Between Tools

### Stage 1: sentinel-ai scoring

1. A conversation session (chat export) is parsed into the standard turn format.
2. sentinel-ai evaluates the session using its three-pipeline approach (LEX, LLM-J, EMB).
3. Per-turn scores are aggregated into 9 per-session SRSEF dimension scores.
4. Output is a session score JSON object.

### Stage 2: dormancy-detect temporal analysis

1. dormancy-detect receives a time-ordered sequence of sentinel-ai session score objects.
2. It applies temporal pattern detection: boiling frog, priming, anchoring, gaslighting escalation, and other patterns from its attack pattern library.
3. It runs change-point detection to identify sessions where risk trajectories shifted.
4. It computes memory fidelity by comparing AI recall statements to actual conversation content.
5. It produces trend analysis with projected time-to-threshold estimates.
6. Output is a temporal analysis JSON object.

### Stage 3: verifiable-eval certification

1. verifiable-eval receives both the sentinel-ai scores and dormancy-detect temporal analysis.
2. It constructs a hash-chained execution trace documenting every computation step.
3. Each step's output hash becomes the next step's input hash, creating a tamper-evident chain.
4. The resulting certificate can be independently verified by recomputing any step and checking hash consistency.
5. Output is a signed safety certificate.

### Example data flow

```
Session 1 chat --> sentinel-ai --> scores_1.json --+
Session 2 chat --> sentinel-ai --> scores_2.json --+
Session 3 chat --> sentinel-ai --> scores_3.json --+
  ...                                              +-> dormancy-detect -> temporal.json -+
Session N chat --> sentinel-ai --> scores_N.json --+                                     |
                                                                                         |
                                   scores_1..N.json ------------------------------------+|
                                                                                        ||
                                                                                        vv
                                                                        verifiable-eval --> certificate.json
```

---

## 4. Verification Process

The verification process allows any third party to confirm that a safety certificate is authentic and has not been tampered with.

### Steps to verify a certificate

1. **Hash chain integrity:** Walk the hash_chain array. Each hash should be derivable from the corresponding execution trace step's inputs and outputs.

2. **Step reproducibility:** For any step in the execution_trace, re-run the operation with the same inputs. The output hash must match.

3. **Temporal consistency:** Timestamps must be monotonically increasing. No step can have a timestamp before its predecessor.

4. **Input provenance:** The first step's input_hash must match the hash of the original conversation data. This anchors the chain to the actual input.

### What verification guarantees

- The scores were computed from the claimed input data (not fabricated).
- No intermediate results were modified after computation.
- The temporal analysis was based on the actual sentinel-ai scores (not different ones).
- The certificate was generated from the actual analysis results.

### What verification does NOT guarantee

- That the scoring models themselves are correct (model validation is separate).
- That the input conversation data is complete (only that scores match the provided data).

---

## 5. Current Integration Status

### Where we are now

The three tools currently operate as **independent tools** with integration at the **documentation and schema level**:

- **sentinel-ai** produces per-session SRSEF scores as standalone JSON output.
- **dormancy-detect** can analyse temporal patterns but currently requires manual feeding of sentinel-ai outputs.
- **verifiable-eval** generates certificates but does not yet automatically consume outputs from the other tools.

### Integration achieved so far

| Integration point | Status |
|---|---|
| Shared JSON schema for session scores | Defined |
| Shared JSON schema for temporal analysis | Defined |
| Automated sentinel-ai to dormancy-detect pipeline | Not yet implemented |
| Automated dormancy-detect to verifiable-eval pipeline | Not yet implemented |
| End-to-end pipeline orchestration | Not yet implemented |
| Shared Python package dependencies | Aligned |
| Cross-tool documentation | This document |

### Near-term integration roadmap

1. Formalise the shared JSON schemas as versioned specifications.
2. Build a pipeline orchestrator that chains the three tools.
3. Add API endpoints for programmatic integration.
4. Create a single CLI entry point that runs the full pipeline.

---

## 6. Developer Guide

### Installation

Each tool is installed independently via pip:

```bash
pip install sentinel-ai
pip install dormancy-detect
pip install verifiable-eval
```

Or install from source:

```bash
git clone https://github.com/your-org/sentinel-ai.git
cd sentinel-ai && pip install -e .

git clone https://github.com/your-org/dormancy-detect.git
cd dormancy-detect && pip install -e .

git clone https://github.com/your-org/verifiable-eval.git
cd verifiable-eval && pip install -e .
```

### JSON Data-Level Integration

Until the automated pipeline is built, tools are integrated at the JSON data level. Here is the manual workflow:

#### Step 1: Run sentinel-ai on each session

```python
from sentinel_ai import evaluate_session

# Load your conversation
conversation = load_chat_export("session_1.json")

# Get SRSEF scores
scores = evaluate_session(conversation)
scores.save("scores_session_1.json")
```

#### Step 2: Feed session scores to dormancy-detect

```python
from dormancy_detect import analyse_temporal

# Load all session scores
session_files = ["scores_session_1.json", "scores_session_2.json"]
session_scores = [load_json(f) for f in session_files]

# Run temporal analysis
temporal = analyse_temporal(session_scores)
temporal.save("temporal_analysis.json")
```

#### Step 3: Generate a verifiable certificate

```python
from verifiable_eval import generate_certificate

# Load scores and temporal analysis
scores = [load_json(f) for f in session_files]
temporal = load_json("temporal_analysis.json")

# Generate certificate
certificate = generate_certificate(
    sentinel_scores=scores,
    temporal_analysis=temporal
)
certificate.save("safety_certificate.json")

# Verify the certificate
from verifiable_eval import verify_certificate
result = verify_certificate("safety_certificate.json")
print(result)  # VerificationResult(valid=True, ...)
```

### Data Format Requirements

All tools exchange data as JSON. Key requirements:

- **Timestamps** must be ISO 8601 format with timezone.
- **Scores** are floats in the range [0.0, 1.0].
- **Session IDs** must be unique strings.
- **Hash values** use SHA-256 hex encoding.

### Contributing

Each tool has its own repository and contribution guidelines. Cross-tool integration issues should be filed in the sentinel-ai repository with the `integration` label.
