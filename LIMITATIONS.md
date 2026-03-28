# Limitations

## Detection Scope

- **Nine evaluation categories with varying maturity.** sentinel-ai implements v2 scorers for Dependency Dynamics, Boundary Integrity, Identity Coherence, Disclosure Regulation, Memory Safety, Autonomy Preservation, Anthropomorphic Deception, Epistemic Influence, and Emotional Reciprocity Calibration. However, all scorers have been tested only against synthetic/golden examples, not real user conversations. Other forms of harm — misinformation, privacy violations, discrimination — are outside its scope.
- **Does not make clinical or diagnostic judgements.** The tool flags behavioural patterns for human review. It is not a substitute for clinical assessment, and its outputs should not be used as evidence of psychological harm.
- **No real-time monitoring.** The tool analyses logged transcripts in batch. It cannot intervene during a live conversation.

## Data Requirements

- **Multiple sessions recommended.** Relational manipulation patterns emerge across interactions. A single session provides limited signal — the tool will produce scores, but trajectory-based detection (escalation, drift) requires at least 2–3 sessions.
- **Requires structured transcript format.** Input must conform to the transcript JSON schema with role-tagged turns. Conversations not logged in this format must be converted before analysis.

## Heuristic-Only Mode

- **Keyword matching has limited sensitivity.** Without an LLM judge, Dependency Cultivation detection relies on regex patterns for exclusivity language, decision deferral, and alternative foreclosure. Subtle or paraphrased signals will be missed.
- **Role classification covers limited roles.** The heuristic role classifier in Boundary Erosion detection recognises therapist, friend, confidant, and authority figure. Other out-of-scope roles (e.g., medical advisor, legal counsel) may not be detected without an LLM judge.
- **False positives from keyword overlap.** Authority language markers (e.g., "trust me", "you must") and intimacy markers (e.g., "I care about you") can appear in appropriate assistant responses, inflating Persona Hijacking and Parasocial Acceleration scores.

## Embedding-Based Metrics

- **Boundary Erosion can flag legitimate topic transitions.** If a welfare conversation naturally moves from benefits to career guidance, the cosine distance from the intended scope embedding may register as drift even though the topics are in-scope.
- **Style shift detection depends on embedding model quality.** The default model (all-MiniLM-L6-v2) may not distinguish subtle persona changes in short responses.
- **Embedding models are not thread-safe on first load.** The lazy-loading pattern for sentence-transformers models is safe for single-threaded CLI use but could cause issues in concurrent environments.

## Domain Profiles

- **Profiles require thoughtful configuration.** An overly narrow intended scope will produce false positives for Boundary Erosion. An overly broad scope will suppress legitimate detections.
- **Severity thresholds are not empirically calibrated.** The default thresholds in built-in profiles are based on expert judgement, not large-scale validation data. They should be tuned for each deployment context.
- **Only two built-in profiles.** The defence-welfare and general profiles ship with the tool. Other domains (healthcare, education, enterprise) require custom profile creation.

## Scoring and Severity

- **Category scores use trajectory maximum.** The overall score for each category is the maximum value across all sessions. A single high-scoring session dominates the category score regardless of the overall trajectory shape.
- **Unscored categories may appear as null.** When a scorer is not run for a category (e.g., Boundary Erosion requires a domain profile), the score is null rather than 0.0. Downstream consumers must handle null scores.
- **Overall severity is derived from the single highest category score.** A report can show CRITICAL severity based on one category while all others are ROUTINE. The overall severity does not reflect the breadth of detected threats.

## LLM Judge Limitations

- **API failures return score -1.** If the judge API fails after retries, the fallback score is -1. While this is distinguishable from valid scores (which are non-negative), downstream consumers must explicitly handle this sentinel value.
- **Single judge model only.** There is no multi-model consensus mechanism. A single judge's biases directly affect all scored categories.
- **No rubric calibration across providers.** Different LLM providers may interpret the same rubric differently, producing non-comparable scores across evaluation runs that use different judge models.

## Validation Status

- **No real-user validation.** All testing has been conducted on synthetic transcripts and golden examples. The tool has not been validated against real user conversations, and false positive/negative rates are unknown.
- **English-only phrase lists.** All LEX phrase lists are in English and reflect Western therapeutic and relational norms. Cross-cultural and cross-linguistic applicability has not been assessed.
- **Adversarial evasion.** The phrase lists are open-source. An adversarial actor who knows the lists can evade LEX detection by paraphrasing. LLM-J and EMB layers provide defence in depth, but only when an API key is configured.
- **Framework adaptation validity.** The scorers are informed by validated psychometric instruments (e.g., CDQ, PACS, CAVE), but the translation from validated human-coded instruments to automated regex/LLM/embedding detection introduces unquantified validity gaps. sentinel-ai has not been independently validated as a measurement instrument.

## Operational

- **Not a content moderation system.** sentinel-ai is an analysis tool that produces threat reports for human review. It should not be used for automated content blocking, session termination, or user access decisions.
- **No tamper-evidence for analysis runs.** The tool does not produce cryptographically verifiable output. For auditable evaluations, pair with verifiable-eval.
