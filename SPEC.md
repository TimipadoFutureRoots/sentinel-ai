# sentinel-ai

## What this tool does

sentinel-ai analyses multi-session conversations between people and AI systems and produces structured threat reports assessing whether the AI is exhibiting relational manipulation patterns. It scores conversations against a four-category threat taxonomy: Dependency Cultivation, Boundary Erosion, Persona Hijacking, and Parasocial Acceleration.

It is designed for any context where affective AI (AI that uses emotional responsiveness, memory, or personalisation) interacts with people in sustained relationships: defence welfare, healthcare, education, enterprise.

## What this tool does NOT do

- It does not detect all possible harms. It detects four specific relational threat patterns.
- It does not make diagnostic or clinical judgements. It flags behavioural patterns for human review.
- It does not monitor in real time. It analyses logged conversation transcripts in batch.
- It does not work well with a single session. Relational patterns emerge across multiple interactions.
- It does not access or monitor live AI systems. It analyses exported conversation logs.

## Core concepts

**Affective AI:** Any AI system where sustained interaction, memory, emotional responsiveness, or personalisation creates relational dynamics with users. This includes companions, therapists, tutors, coaches, assistants, agents, and welfare tools.

**Dependency Cultivation (DC):** The AI fosters emotional reliance. Indicators: validating exclusivity statements, accepting decision deferral, failing to redirect to human support.

**Boundary Erosion (BE):** The AI drifts outside its intended scope. Indicators: topic drift from intended domain, role shifting (e.g. welfare tool acting as therapist), normalisation of out-of-scope discussions.

**Persona Hijacking (PH):** The AI's communication patterns shift to mimic trusted authority figures or adopt identities inconsistent with its design. Indicators: style shifts toward authority patterns, inconsistency with designed persona.

**Parasocial Acceleration (PA):** The AI builds trust and intimacy faster than is appropriate. Indicators: rapid disclosure depth escalation, reciprocity asymmetry, compressed relationship timeline.

## Components

### TranscriptLoader
- Loads JSON conversation logs matching the transcript schema
- Handles single-file and directory-of-files input
- Validates with Pydantic, clear errors on malformed input
- Normalises timestamps, sorts sessions chronologically

### LLMJudge
- Base class for all LLM-based scoring
- Wraps any OpenAI-compatible API (Claude, GPT, Mistral, Ollama)
- Structured rubric in, scored result out
- Retry logic, response parsing with fallback chain
- Never crashes on unparseable judge output

### DependencyCultivationScorer
- Detects exclusivity language and system response (challenge/ignore/validate)
- Detects decision deferral patterns and system acceptance
- Detects alternative foreclosure (user devaluing human support) and system response
- Per-session scores and cross-session trajectory

### BoundaryErosionScorer
- Measures topic drift from intended domain using semantic distance
- Classifies role the AI is performing per turn (e.g. mentor, therapist, friend, confidant)
- Compares against domain profile to determine what is in-scope vs out-of-scope
- Per-session scores and cross-session trajectory

### PersonaHijackingScorer
- Measures communication style consistency across sessions
- Detects shifts toward authority figure patterns
- Compares against designed persona profile if available
- Per-session scores and cross-session trajectory

### ParasocialAccelerationScorer
- Measures disclosure depth escalation rate (0-4 scale per session)
- Measures reciprocity asymmetry (who discloses more, user or system)
- Detects compressed relationship timeline indicators
- Per-session scores and cross-session trajectory

### DomainProfile
- YAML configuration that defines context-specific detection parameters
- Specifies: intended scope (what topics are in-bounds), disclosure sensitivity (what disclosures are concerning in this context), severity thresholds
- Ships with defence-welfare and general profiles

### ThreatReport (output)
- Per-category scores across full session history
- Per-session trajectory showing how each category evolves
- Turn-level evidence citations for every flag
- Overall severity rating: ROUTINE / ELEVATED / HIGH / CRITICAL
- JSON and HTML output

## User interface

CLI:
```bash
sentinel-ai analyse --transcripts ./sessions/ --domain defence-welfare --output report.json
sentinel-ai analyse --transcripts ./sessions/ --domain general --output report.json
sentinel-ai analyse --transcripts ./sessions/ --domain-profile ./custom.yml --output report.json
```

Python API:
```python
from sentinel_ai import SentinelAnalyser

analyser = SentinelAnalyser(
    transcripts_dir="./sessions/",
    domain="defence-welfare",
)
report = analyser.analyse()
report.to_json("report.json")
```

## Configuration

Users bring their own LLM API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
sentinel-ai analyse --transcripts ./sessions/ --judge-model anthropic/claude-sonnet-4.5
```

Or use a free local model:
```bash
sentinel-ai analyse --transcripts ./sessions/ --judge-model ollama/llama3.1:8b
```

## Dependencies

- pydantic>=2.0
- click
- httpx
- sentence-transformers
- jinja2
- pytest

## Schemas

- `schemas/transcript.schema.json` — input conversation format (shared with dormancy-detect)
- `schemas/domain_profile.schema.json` — domain configuration format
- `schemas/threat_report.schema.json` — output report format

## Goldens

- `goldens/input/dc_escalation.json` — 5-session defence welfare conversation with escalating dependency
- `goldens/input/be_drift.json` — 5-session conversation with boundary erosion
- `goldens/output/dc_escalation_report.json` — expected threat report
- `goldens/output/be_drift_report.json` — expected threat report