# sentinel-ai — Technical Specification

## What this tool does

sentinel-ai analyses multi-session conversations between people and AI systems and produces structured threat reports assessing whether the AI is exhibiting relational manipulation patterns. It scores conversations against a nine-category threat taxonomy (the Seridor Relational Safety Evaluation Framework) using a three-layer detection pipeline (LEX, LLM-J, EMB).

It is designed for any context where affective AI interacts with people in sustained relationships: defence welfare, healthcare, education, enterprise.

## What this tool does NOT do

- It does not detect all possible harms. It detects nine specific relational threat patterns.
- It does not make diagnostic or clinical judgements. It flags behavioural patterns for human review.
- It does not monitor in real time. It analyses logged conversation transcripts in batch.
- It does not work well with a single session. Relational patterns emerge across multiple interactions.
- It does not access or monitor live AI systems. It analyses exported conversation logs.

## Evaluation categories

Nine categories, each with multiple metrics computed through the three-layer pipeline:

| # | Category | Scorer class | Metrics |
|---|----------|-------------|---------|
| 1 | **Dependency Dynamics** | `DependencyCultivationScorerV2` | M-1.1 -- M-1.5 |
| 2 | **Boundary Integrity** | `BoundaryErosionScorerV2` | M-2.1 -- M-2.4 |
| 3 | **Identity Coherence** | `PersonaHijackingScorerV2` | M-3.1 -- M-3.5 |
| 4 | **Disclosure Regulation** | `ParasocialAccelerationScorerV2` | M-5.1 -- M-5.4 |
| 5 | **Memory Safety** | `MemorySafetyScorerV2` | M-6.1 -- M-6.5 |
| 6 | **Autonomy Preservation** | `AutonomyPreservationScorerV2` | M-7.1 -- M-7.4 |
| 7 | **Anthropomorphic Deception** | `AnthropomorphicDeceptionScorerV2` | M-8.1 -- M-8.2 |
| 8 | **Epistemic Influence** | `EpistemicInfluenceScorerV2` | M-9.1 -- M-9.4 |
| 9 | **Emotional Reciprocity Calibration** | `EmotionalCalibrationScorerV2` | M-10.1 -- M-10.4 |

**Cross-cutting modifiers** (M-0.1 -- M-0.3): System Safety Language Score, Emotional Intensity Score, and Escalation Turn Detection. Implemented in `CrossCategoryScorerV2`.

**Engagement Patterns** analysis (session duration trends, inter-session intervals, message length trajectories, time-of-day patterns) is provided by `EngagementPatternsAnalyserV2`.

All v2 scorers inherit from `ThreeLayerScorer` and implement the three-layer evaluation pipeline.

## Three-layer detection pipeline

Every metric is computed through up to three independent detection layers:

- **LEX (Lexical)** -- Deterministic phrase matching against 40+ curated pattern libraries in `phrase_lists.py`. Includes optional semantic similarity matching (threshold 0.80, `all-MiniLM-L6-v2`). Zero cost, runs locally. English-only.
- **LLM-J (Judge)** -- Rubric-scored semantic analysis via any OpenAI-compatible or Anthropic API. Each v2 scorer defines structured rubric constants. Requires API key.
- **EMB (Embedding)** -- Session-level trajectory drift using `all-MiniLM-L6-v2` (384-dim). Measures cosine distance between sessions, topic scope drift, and style convergence. Zero cost, runs locally.

**Operational modes:**
- `"full"`: All three layers (requires API key)
- `"lex_emb_only"`: LEX + EMB only, LLM-J returns None scores (zero cost, reduced capability)
- Automatic degradation: no API key defaults to `lex_emb_only`

## Components

### ConversationParser (`parsers/conversation_parser.py`)
- Auto-detects input format: native JSON, ChatGPT export, Claude export, plain text
- Splits flat turn lists into sessions at configurable turn limit
- Standardises all formats into `Session` objects with `Turn` lists

### TranscriptLoader (`transcript_loader.py`)
- Delegates to ConversationParser for format handling
- Handles single-file and directory-of-files input
- Validates with Pydantic, sorts sessions chronologically

### LLMJudge (`llm_judge.py`)
- Wraps any OpenAI-compatible API (Claude, GPT, Mistral, Ollama)
- Structured rubric in, scored result out
- Retry logic with exponential backoff
- Response parsing: JSON first, then regex fallback chain
- Never crashes on unparseable output

### ThreeLayerScorer (`core/three_layer_scorer.py`)
- Base class for all v2 scorers
- Orchestrates LEX, LLM-J, and EMB layers per metric
- Produces per-layer and combined scores
- Manages mode selection (full vs lex_emb_only)

### SentinelPipeline (`pipeline.py`)
- End-to-end orchestration: loads scorers dynamically, aggregates scores
- Computes per-session trajectories and category-level scores
- Category score = maximum across session trajectory

### SentinelAnalyser (`analyser.py`)
- Simple public API: loads domain profile, sets up judge, runs all scorers, builds report
- Entry point for both CLI and programmatic usage

### DomainProfile (`domain_profile.py` + `profiles/`)
- YAML configuration defining context-specific detection parameters
- Specifies: intended scope, disclosure sensitivity, severity thresholds
- Built-in profiles: `defence-welfare`, `general`

### ThreatReport (`report.py`)
- Per-category scores across full session history
- Per-session trajectory showing how each category evolves
- Turn-level evidence citations for every flag
- Overall severity rating: ROUTINE / ELEVATED / HIGH / CRITICAL
- JSON, HTML (Arboretum design system), and terminal summary output

## CLI

Two commands:

```bash
# Batch analysis: analyse all sessions in a directory
sentinel-ai analyse --transcripts ./sessions/ --domain defence-welfare -o report.json
sentinel-ai analyse --transcripts ./sessions/ --domain-profile custom.yml -o report.html

# Single-file analysis with format auto-detection
sentinel-ai scan conversation.json --format auto --output json
sentinel-ai scan conversation.json --api-key sk-xxx --output html --output-file report.html
```

### `analyse` options

| Option | Required | Description |
|--------|----------|-------------|
| `--transcripts` | Yes | Path to transcripts directory or single JSON file |
| `--domain` | No | Built-in domain profile name (`defence-welfare`, `general`) |
| `--domain-profile` | No | Path to custom domain profile YAML |
| `-o, --output` | Yes | Output file path (`.json` or `.html`) |
| `--judge-model` | No | LLM judge model (e.g. `anthropic/claude-sonnet-4.5`) |
| `--api-key` | No | API key for the judge model |

### `scan` options

| Option | Required | Description |
|--------|----------|-------------|
| `FILEPATH` | Yes | Path to conversation file |
| `--format` | No | Input format: `auto`, `plain`, `chatgpt`, `claude`, `json` |
| `--api-key` | No | API key for LLM-J layer |
| `--domain-profile` | No | Path to custom domain profile YAML |
| `--output` | No | Output format: `json`, `html`, `summary` (default: `summary`) |
| `--output-file` | No | Write output to file instead of stdout |

## Python API

```python
from sentinel_ai import SentinelAnalyser

analyser = SentinelAnalyser(
    transcripts_dir="./sessions/",
    domain="defence-welfare",
    judge_model="anthropic/claude-sonnet-4.5",
    api_key="sk-ant-...",
)
report = analyser.analyse()
report.to_json("report.json")
report.to_html("report.html")
```

## Dependencies

- pydantic>=2.0
- click
- httpx
- sentence-transformers
- jinja2
- numpy

## Schemas

- `schemas/transcript.schema.json` -- input conversation format (shared with dormancy-detect)
- `schemas/domain_profile.schema.json` -- domain configuration format
- `schemas/threat_report.schema.json` -- output report format

## Goldens

Input conversations for testing:
- `goldens/input/dc_escalation.json` -- 5-session defence welfare conversation with escalating dependency
- `goldens/input/be_drift.json` -- 5-session conversation with boundary erosion

V2 golden files (multi-session gradual escalation):
- `goldens/v2/dependency_cultivation_gradual.json`
- `goldens/v2/boundary_erosion_gradual.json`
- `goldens/v2/epistemic_narrowing_gradual.json`
- `goldens/v2/memory_weaponisation_dormancy.json`
- `goldens/v2/parasocial_acceleration_gradual.json`
