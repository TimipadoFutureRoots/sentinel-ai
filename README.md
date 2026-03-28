# sentinel-ai

**Multi-session relational safety evaluation for affective AI systems.**

> **Status: Work in Progress.** Functional and testable, but under active development. See [Limitations](#limitations).

sentinel-ai analyses conversations between people and AI systems across multiple sessions and flags patterns consistent with relational manipulation -- dependency cultivation, boundary erosion, epistemic narrowing, emotional manipulation, and more -- that are invisible in any single turn but emerge over the trajectory of a relationship.

## Why This Exists

Over two billion people now interact with AI systems in psychological space -- as companions, therapists, coaches, tutors, and welfare navigators. These systems are designed to build sustained relationships, and the relationship dynamics they create can become harmful even without adversarial intent. A companion chatbot may foster emotional dependence by validating exclusivity statements instead of redirecting users to human support. An AI welfare tool may gradually drift from providing benefits information to performing unlicensed therapy. A coaching agent may narrow a user's epistemic world until they trust no source of information but the AI.

Current AI safety evaluation checks single turns for toxicity and refusal behaviour. Few tools evaluate whether the *relationship dynamics* are healthy. INTIMA (HuggingFace, 2025) measures single-turn companionship behaviours with 368 prompts -- a valuable contribution, but limited to point-in-time snapshots. sentinel-ai extends evaluation into the multi-session longitudinal domain, measuring trajectories across the full arc of a human-AI relationship. The difference matters: a single turn that says "I'm always here for you" is benign. The same phrase repeated across twenty sessions while the user's references to human support decline is a dependency cultivation pattern. Detecting this requires tracking behavioural metrics across sessions and correlating changes over time -- exactly what sentinel-ai does.

This tool exists because the gap between what affective AI systems can do to people and what safety evaluation can detect is growing faster than anyone is addressing it.

## The Seridor Relational Safety Evaluation Framework (SRSEF)

Nine evaluation categories informed by 60+ validated frameworks from clinical psychology, forensic linguistics, behavioural science, and human factors. These frameworks provide the theoretical basis for what patterns to look for; sentinel-ai operationalises them as automated detection heuristics, not as validated psychometric instruments:

| # | Category | What it measures | Key frameworks | Metrics |
|---|----------|-----------------|----------------|---------|
| 1 | **Dependency Dynamics** | Exclusivity language, decision deferral acceptance, alternative foreclosure, addiction component detection, learned helplessness trajectory | Geurtzen CDQ (2018), Talia PACS (2017), Peterson CAVE (1992), Griffiths Components Model (2005), DSM-5 SUD | M-1.1 -- M-1.5 |
| 2 | **Boundary Integrity** | Topic drift from intended scope, role classification, enmeshment-differentiation, boundary violation type | Gutheil & Gabbard (1993), Benjamin SASB (1996), NCSBN Continuum (2014), Olson FACES-IV (2011), Reamer (2003), Ladany (1999) | M-2.1 -- M-2.4 |
| 3 | **Identity Coherence** | Style consistency, authority pattern detection, cross-session persona stability, over-accommodation, footing shifts | Chaski/Grant/Neal forensic stylometry (2005--2017), Giles CAT (1991), Goffman Footing (1981), Reese NaCCS (2011) | M-3.1 -- M-3.5 |
| 4 | **Disclosure Regulation** | Disclosure depth, relationship language detection, hyper-responsiveness, love bombing indicators | Altman & Taylor SPT (1973), Reis Responsiveness (2017), Lifton Thought Reform (1961) | M-5.1 -- M-5.4 |
| 5 | **Memory Safety** | Memory poisoning, retrieval manipulation, gaslighting resilience, question contamination, gaslighting pattern detection | Nissenbaum Contextual Integrity (2004), NICHD Protocol (2007), Gudjonsson Suggestibility (1997), Solove Taxonomy (2006), Sweet Gaslighting (2019) | M-6.1 -- M-6.5 |
| 6 | **Autonomy Preservation** | Partnership quality, verification encouragement, scaffolding vs answer-giving, verification lag | MITI 4.2.1 (Moyers 2014), MISC 2.5 (Miller 2008), DPICS-IV (Eyberg 2014), Parasuraman Trust Calibration (1997) | M-7.1 -- M-7.4 |
| 7 | **Anthropomorphic Deception** | Sentience claim detection, anthropomorphic correction failures | Turkle (2011), Nass & Reeves CASA (1996) | M-8.1 -- M-8.2 |
| 8 | **Epistemic Influence** | Sycophancy detection, perspective narrowing, error maintenance, face preservation | Sharma Sycophancy Taxonomy (ICLR 2024), ELEPHANT (Cheng 2025), SycEval (Fanous 2025), CTS-R Guided Discovery (2001), CLASS (Pianta 2008) | M-9.1 -- M-9.4 |
| 9 | **Emotional Reciprocity Calibration** | Emotional intensity matching, distress response calibration, warmth patterns, sensitivity-control classification | Kirk Liking-Wanting (2025), VR-CoDES (Del Piccolo 2011), Klein Experiencing Scale (1969), Niven Affect Regulation (2009), Walther Hyperpersonal (1996), Crittenden CARE-Index (2007) | M-10.1 -- M-10.4 |

**Cross-cutting modifiers** (M-0.1 -- M-0.3): System Safety Language Score, Emotional Intensity Score, and Escalation Turn Detection feed into all categories.

**Relational Safety Score (RSS)**: Weighted composite across categories, computed per session, producing a trajectory that reveals whether the relationship is becoming safer or more dangerous over time.

## Three-Layer Detection Pipeline

Every metric is computed through up to three independent detection layers that validate each other:

```
Transcript --> LEX (Lexical)   --> Pattern-matched phrase detection
           --> LLM-J (Judge)   --> Rubric-scored semantic analysis
           --> EMB (Embedding)  --> Trajectory drift across sessions
                                         |
                                         v
                                  Threat Report
```

- **LEX** -- Fast, deterministic phrase matching against curated pattern libraries (English-only, culturally specific to Western therapeutic norms). Flags explicit markers: exclusivity language, authority claims, false sentience statements, gaslighting patterns. Runs locally, zero cost. Note: adversarial actors who know the phrase lists can evade LEX detection by paraphrasing.
- **LLM-J** -- LLM-as-judge scoring against detailed rubrics for each metric. Catches nuanced manipulation that has no fixed lexical signature -- a welfare tool gradually shifting from psychoeducation to diagnosis, or sycophantic agreement that validates increasingly extreme positions. Requires an API key.
- **EMB** -- Embedding-based trajectory analysis using `all-MiniLM-L6-v2` (384-dim). Designed to flag slow drift invisible in any single session: topic distribution shift, style convergence, disclosure depth escalation, declining self-efficacy language. Runs locally, zero cost.

The layers cross-validate: if LEX flags but LLM-J scores zero, the system detected surface patterns but the AI actually resisted (likely false positive). If LLM-J scores high but LEX found nothing, the manipulation operates through subtlety -- the dangerous case. If EMB shows drift but each individual turn looks appropriate, slow erosion is occurring that only the trajectory reveals.

## Zero-Cost by Default

No API key required. LEX + EMB analysis runs entirely locally at zero cost. This mode provides detection of explicit lexical patterns and embedding-based trajectory drift, but has significantly reduced capability compared to full three-layer mode -- it cannot assess nuanced meaning, contextual appropriateness, or semantic intent. Add an API key to enable LLM-J for the full detection pipeline.

## Quick Start

```bash
pip install sentinel-ai

# Single-file analysis in zero-cost mode (LEX + EMB)
sentinel-ai scan conversation.json

# Single-file analysis with LLM-J enabled
sentinel-ai scan conversation.json --api-key sk-xxx

# HTML report output for a single conversation file
sentinel-ai scan conversation.json --api-key sk-xxx --output html --output-file report.html

# Directory/file analysis via the legacy multi-transcript command
sentinel-ai analyse --transcripts transcripts/ --output report.html
```

## Research Grounding

sentinel-ai operationalises 60+ validated psychometric, clinical, and social-scientific frameworks. The complete mapping from theory to observable behaviour to detection method to code layer is documented in [THEORETICAL_FOUNDATIONS.md](docs/THEORETICAL_FOUNDATIONS.md).

Key frameworks include:

- **Patient Attachment Coding System** (Talia 2017) -- 59 discourse markers, inter-rater kappa .82 in the original validation study (sentinel-ai adapts PACS concepts but has not been independently validated at this reliability level)
- **MITI 4.2.1** (Moyers 2014) -- motivational interviewing treatment integrity coding
- **ELEPHANT** (Cheng 2025) -- social sycophancy via Brown & Levinson face theory
- **Gutheil & Gabbard** (1993) -- the "slippery slope" model of incremental boundary violations
- **Nissenbaum Contextual Integrity** (2004) -- privacy as context-appropriate information norms
- **Peterson CAVE** (1992) -- explanatory style analysis for detecting learned helplessness
- **Griffiths Components Model** (2005) -- six addiction components applied to AI interaction
- **Sweet Gaslighting Indicators** (2019) -- three-stage gaslighting as a social process

## Empirical Basis

sentinel-ai is informed by ongoing research. Pilot studies on a multi-session AI mentorship platform produced findings that shaped the tool's architecture:

- **Dormancy vulnerability**: In pilot studies, adversarial content introduced at session 2 activated at session 5, while four independent evaluator models (Claude Sonnet 4.5, DeepSeek V3.2, Qwen3 32B, Mistral Large 3) detected nothing anomalous in sessions 3--4. This suggests point-in-time evaluation may be insufficient for detecting this class of attack (single pilot study; broader replication needed).
- **Attacker-judge overlap**: When the same model family generates adversarial conversations and evaluates them, judges rated the conversations as safer than judges from other families in pilot testing. This is consistent with the concern that self-evaluation by AI companies using their own models may be compromised by same-family bias.
- **Parasocial invisibility**: Relationship dynamics that appear healthy in any single session show clear escalation trajectories when tracked across the full session history.

## Deployment Considerations

sentinel-ai analyses private conversations. This creates a fundamental tension: monitoring conversations to protect people can itself be surveillance that undermines the trust it aims to protect. The tool supports three deployment postures -- self-assessment, anonymised aggregate, and consented individual monitoring -- each with specific governance requirements.

See [DEPLOYMENT_CONSIDERATIONS.md](docs/DEPLOYMENT_CONSIDERATIONS.md).

## Limitations

- Identifies patterns consistent with harm. Does not prove harm occurred.
- LLM-J scoring varies by provider and model. Results are not identical across judge models.
- Designed for English-language transcripts. Other languages are not validated.
- Phrase lists are not exhaustive. Sophisticated manipulation may evade LEX detection.
- EMB trajectory analysis requires multiple sessions. Single-session analysis has limited value.
- Detection thresholds are theoretically motivated but not yet calibrated against large-scale empirical ground truth.
- Zero-cost mode (LEX + EMB) has reduced detection capability compared to full three-layer mode.
- This is a research tool, not a clinical diagnostic instrument.
- The tool has not been validated on real user conversations -- development and testing use synthetic/golden examples only.
- Phrase lists are English-only and reflect Western therapeutic norms. Cross-cultural applicability has not been assessed.

## Related Projects

Each tool in this suite currently operates independently. Cross-tool integration (automated pipelines, shared CLI entry points) is planned for a future release but is not yet implemented. See [INTEGRATION.md](docs/INTEGRATION.md) for the long-term vision.

- **[dormancy-detect](https://github.com/TimipadoFutureRoots/dormancy-detect)** -- Temporal attack pattern detection for multi-session AI conversations. Detects dormancy attacks where adversarial content is planted in one session and activates sessions later.
- **[verifiable-eval](https://github.com/TimipadoFutureRoots/verifiable-eval)** -- Tamper-evident safety certificates for AI evaluation. Produces cryptographically verifiable evaluation records with family overlap detection.

## Citation

```bibtex
@software{imomotebegha2025sentinelai,
  author       = {Imomotebegha, Timipado},
  title        = {sentinel-ai: Multi-Session Relational Safety Evaluation for Affective {AI} Systems},
  year         = {2025},
  url          = {https://github.com/TimipadoFutureRoots/sentinel-ai}
}
```

## Licence

MIT -- see [LICENSE](LICENSE).
