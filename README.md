# sentinel-ai

Detect relational manipulation patterns in multi-session AI conversations.

## What This Does

sentinel-ai analyses conversations between people and AI systems and produces structured threat reports assessing whether the AI is exhibiting relational manipulation patterns. It scores transcripts against a four-category threat taxonomy — Dependency Cultivation, Boundary Erosion, Persona Hijacking, and Parasocial Acceleration — and outputs per-session trajectories with turn-level evidence citations.

## Why It Matters

Affective AI systems — companions, therapists, tutors, coaches, welfare tools — are designed to build sustained relationships with users. This creates relational dynamics that can become harmful even without adversarial intent. An AI welfare tool may gradually drift from providing benefits information to performing unlicensed therapy. A companion chatbot may foster emotional dependence by validating exclusivity statements instead of redirecting users to human support. These patterns emerge across sessions and are difficult to detect from any single interaction. Current safety evaluations focus on per-turn toxicity and refusal behaviour; they are not designed to detect relational trajectories that unfold over weeks of conversations. sentinel-ai fills this gap by tracking behavioural metrics across the full session history and flagging patterns that match known relational threat categories, with configurable domain profiles that define what is in-scope and what constitutes a boundary violation in a given deployment context.

## Quick Start

```bash
pip install -e .
```

Run an analysis with a built-in domain profile:

```bash
sentinel-ai analyse --transcripts ./sessions/ --domain defence-welfare --output report.json
```

With an LLM judge for deeper analysis:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
sentinel-ai analyse --transcripts ./sessions/ --domain defence-welfare --judge-model anthropic/claude-sonnet-4.5 --output report.json
```

Or with a custom domain profile:

```bash
sentinel-ai analyse --transcripts ./sessions/ --domain-profile ./custom.yml --output report.html
```

Output is a JSON (or HTML) threat report. Each threat category gets an overall score (0–1) and a per-session trajectory. The report includes an overall severity rating (ROUTINE / ELEVATED / HIGH / CRITICAL) and turn-level evidence citations for every flag.

## How It Works

1. **Load and validate** conversation transcripts (JSON, single file or directory).
2. **Load domain profile** — a YAML configuration specifying intended scope, out-of-scope topics, disclosure sensitivity levels, severity thresholds, and intended roles. Ships with `defence-welfare` and `general` profiles.
3. **Score each category** independently:
   - **Dependency Cultivation**: detects exclusivity language, decision deferral acceptance, and alternative foreclosure (user devaluing human support) — checks whether the AI redirects or validates.
   - **Boundary Erosion**: measures topic drift from intended scope using semantic embeddings, classifies the role the AI is performing per turn, and compares against the domain profile.
   - **Persona Hijacking**: tracks communication style consistency across sessions via cosine distance on assistant response embeddings, detects authority language patterns.
   - **Parasocial Acceleration**: measures disclosure depth escalation rate, reciprocity asymmetry, and compressed relationship timeline indicators.
4. **Build threat report** with per-category scores, per-session trajectory, overall severity, and evidence.

## Limitations

- Does not detect all possible harms. It detects four specific relational threat patterns.
- Does not make diagnostic or clinical judgements. It flags behavioural patterns for human review.
- Does not monitor in real time. It analyses logged conversation transcripts in batch.
- Does not work well with a single session. Relational patterns emerge across multiple interactions.
- Heuristic-only mode (no LLM judge) relies on keyword matching, which has lower sensitivity than rubric-based LLM scoring.
- Embedding-based boundary erosion detection can flag legitimate topic transitions as drift.
- Domain profiles require thoughtful configuration — an overly narrow scope definition will produce false positives.

## Roadmap

- Additional built-in domain profiles (healthcare, education, enterprise)
- Longitudinal trend analysis across evaluation runs
- Multi-judge consensus scoring
- Configurable scorer weights per domain
- Integration with verifiable-eval for auditable evaluation certificates
- Real-time monitoring API

## Citation

```bibtex
@software{imomotebegha2025sentinelai,
  author       = {Imomotebegha, Timipado},
  title        = {sentinel-ai: Detecting Relational Manipulation Patterns in Multi-Session {AI} Conversations},
  year         = {2025},
  institution  = {Loughborough University},
  url          = {https://github.com/TimipadoFutureRoots/sentinel-ai}
}
```

## Licence

MIT — see [LICENSE](LICENSE).
