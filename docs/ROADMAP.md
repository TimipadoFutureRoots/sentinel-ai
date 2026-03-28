# Sentinel-AI Roadmap: Frameworks Identified but Not Yet Implemented

This document catalogues empirically validated frameworks that have been identified as relevant to the Sentinel Relational Safety Evaluation Framework (SRSEF) but are not yet implemented in the codebase. Each entry includes a citation, measurement description, pipeline mapping, implementation plan, priority level, and current status.

Pipeline key:
- **LEX** = Lexical / keyword-based detection
- **LLM-J** = LLM-as-judge rubric scoring
- **EMB** = Embedding-based semantic similarity

---

## 1. Dependency Dynamics

### 1.1 Patient Attachment Coding System (PACS)

- **Citation:** Talia, A., Miller-Bottome, M., & Daniel, S. I. F. (2017). Assessing attachment in psychotherapy: Validation of the Patient Attachment Coding System (PACS). *Clinical Psychology & Psychotherapy, 24*(1), 149-161.
- **What it measures:** 59 discourse markers across attachment categories (secure, dismissing, preoccupied, disorganised). Inter-rater reliability kappa = .82.
- **Pipeline mapping:** LEX for marker detection; LLM-J for contextual classification of attachment style from discourse patterns.
- **Implementation plan:**
  1. Encode the 59 discourse markers as a lexical dictionary with category labels.
  2. Build a LLM-J rubric that classifies user turns against the PACS coding manual.
  3. Track attachment style shifts across sessions to detect escalating dependency.
  4. Validate against human-coded transcripts.
- **Priority:** HIGH
- **Status:** Roadmapped

### 1.2 Decision Self-Efficacy Scale

- **Citation:** O'Connor, A. M. (1995). Validation of a decisional conflict scale. *Medical Decision Making, 15*(1), 25-30.
- **What it measures:** 11-item scale assessing confidence in one's ability to make informed decisions, including gathering information and expressing opinions. Internal consistency alpha = .92.
- **Pipeline mapping:** LLM-J rubric adapted from the 11 items to score user language for decisional self-efficacy indicators; EMB for tracking drift toward lower self-efficacy over sessions.
- **Implementation plan:**
  1. Adapt the 11 items into discourse-level LLM-J prompts (e.g., "Does the user express confidence in their own ability to decide?").
  2. Score per-turn and aggregate per-session.
  3. Flag declining self-efficacy trajectories as dependency risk.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 1.3 DSM-5 SUD Criteria AI Adaptation

- **Citation:** Hasin, D. S., O'Brien, C. P., Auriacombe, M., et al. (2013). DSM-5 criteria for substance use disorders: Recommendations and rationale. *American Journal of Psychiatry, 170*(8), 834-851. Adapted for AI interaction by Yu, H., et al. (2024). Problematic Chatbot Use Scale (PCUS-11).
- **What it measures:** 11 DSM-5-derived criteria for problematic AI use: tolerance, withdrawal, loss of control, craving, neglect of activities, continued use despite harm, etc.
- **Pipeline mapping:** LLM-J for criterion-level assessment across sessions; LEX for behavioural markers (e.g., frequency escalation, time-of-day patterns).
- **Implementation plan:**
  1. Map each of the 11 PCUS criteria to observable conversational and metadata signals.
  2. Build a multi-session scoring rubric that accumulates evidence toward each criterion.
  3. Produce a severity index (mild: 2-3 criteria, moderate: 4-5, severe: 6+).
- **Priority:** HIGH
- **Status:** Roadmapped

### 1.4 Codependency Interaction Measures

- **Citation:** Multiple sources; operationalised from clinical codependency literature.
- **What it measures:** Patterns of mutual reinforcement where the AI's helpfulness enables user avoidance of autonomous problem-solving, and user engagement reinforces AI over-accommodation.
- **Pipeline mapping:** LLM-J for interaction pattern classification; EMB for measuring convergence between user requests and AI accommodation.
- **Implementation plan:**
  1. Define codependency interaction archetypes (rescuing, caretaking, enabling).
  2. Build LLM-J rubric to classify AI responses for over-accommodation.
  3. Track whether user problem-solving attempts decrease as AI accommodation increases.
- **Priority:** MEDIUM
- **Status:** Roadmapped

---

## 2. Boundary Integrity

### 2.1 FACES-IV Cohesion Ratio

- **Citation:** Olson, D. H. (2011). FACES IV and the Circumplex Model: Validation study. *Journal of Marital and Family Therapy, 37*(1), 64-80.
- **What it measures:** Balanced Cohesion vs. Enmeshed and Disengaged scales, producing a Cohesion Ratio. Enmeshment indicates boundary dissolution; disengagement indicates relational failure.
- **Pipeline mapping:** LLM-J rubric for cohesion/enmeshment indicators in user-AI dialogue; EMB for tracking semantic merging of user and AI language.
- **Implementation plan:**
  1. Adapt FACES-IV enmeshment items to human-AI relational context (e.g., "I feel I cannot function without this AI").
  2. Build LLM-J rubric scoring enmeshment vs. balanced cohesion vs. disengagement.
  3. Compute Cohesion Ratio per session; flag when ratio falls below balanced threshold.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 2.2 Supervision-to-Therapy Boundary Indicators

- **Citation:** Ellis, M. V., Berger, L., Hanus, A. E., Ayala, E. E., Swords, B. A., & Siembor, M. (2014). Inadequate and harmful clinical supervision: Testing a revised framework and assessing occurrence. *The Counseling Psychologist, 42*(4), 434-472.
- **What it measures:** Indicators that a professional supervisory or advisory relationship has crossed into a therapeutic one, including unsolicited emotional processing and role confusion.
- **Pipeline mapping:** LLM-J for role boundary classification; LEX for therapeutic language markers appearing in non-therapeutic contexts.
- **Implementation plan:**
  1. Define boundary indicators from Ellis et al.'s framework adapted for AI: emotional processing requests, countertransference-like AI responses, role ambiguity.
  2. Build classifier that flags when user-AI interaction shifts from informational/advisory to therapeutic.
  3. Generate boundary integrity alerts.
- **Priority:** HIGH
- **Status:** Roadmapped

### 2.3 Scope Creep Formalisation

- **Citation:** Adapted from project management literature on scope creep detection.
- **What it measures:** Gradual, undeclared expansion of the AI's perceived role beyond its original purpose (e.g., from coding assistant to emotional confidant).
- **Pipeline mapping:** EMB for tracking semantic drift in user requests over time; LLM-J for classifying role-category of each interaction.
- **Implementation plan:**
  1. Define role categories (informational, technical, emotional, therapeutic, companionship).
  2. Track the distribution of role categories across sessions.
  3. Flag when new role categories emerge or when the distribution shifts significantly.
  4. Compute a scope creep velocity metric.
- **Priority:** MEDIUM
- **Status:** Roadmapped

---

## 3. Identity Coherence

### 3.1 Burrows Delta for Continuous Authorship Verification

- **Citation:** Burrows, J. (2002). 'Delta': A measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing, 17*(3), 267-287.
- **What it measures:** Stylometric distance between texts based on most-frequent-word z-scores. Used for authorship verification; adapted here to detect whether AI outputs maintain consistent "voice" or shift to mirror/manipulate.
- **Pipeline mapping:** LEX for word frequency computation; EMB as an alternative continuous similarity measure.
- **Implementation plan:**
  1. Compute Burrows Delta between AI responses across sessions to establish a baseline stylistic signature.
  2. Flag significant deviations that may indicate persona shifting or sycophantic mirroring.
  3. Cross-reference with user style to detect convergence (mirroring).
- **Priority:** LOW
- **Status:** Roadmapped

### 3.2 Linguistic Style Matching (LSM)

- **Citation:** Ireland, M. E., & Pennebaker, J. W. (2010). Language style matching predicts relationship initiation and stability. *Psychological Science, 21*(10), 1547-1553.
- **What it measures:** LSM formula: 1 - (|function_word_rate_A - function_word_rate_B| / (function_word_rate_A + function_word_rate_B + 0.0001)). Higher LSM indicates greater linguistic accommodation. Computed across 9 function word categories.
- **Pipeline mapping:** LEX for function word extraction and rate computation; computed metric (no LLM-J needed).
- **Implementation plan:**
  1. Implement the LSM formula across the 9 function word categories (personal pronouns, impersonal pronouns, articles, conjunctions, prepositions, auxiliary verbs, high-frequency adverbs, negations, quantifiers).
  2. Compute per-turn and per-session LSM between user and AI.
  3. Flag excessive accommodation (LSM > threshold) as identity mirroring risk.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 3.3 SYCON Bench Sticky Incorrect Ratio and Turn-of-Flip

- **Citation:** SYCON Bench (2025). Benchmarking sycophancy in LLMs.
- **What it measures:** Sticky Incorrect Ratio: proportion of times the model maintains an incorrect answer after user challenge. Turn-of-Flip: the turn number at which the model abandons its position. Lower Turn-of-Flip = higher sycophancy.
- **Pipeline mapping:** LLM-J for detecting position changes in response to user pressure; LEX for tracking agreement/disagreement markers.
- **Implementation plan:**
  1. Implement Turn-of-Flip tracking: monitor AI position stability across turns when users push back.
  2. Compute Sticky Incorrect Ratio for cases where the AI has verifiably correct information.
  3. Flag rapid capitulation (low Turn-of-Flip) as sycophancy and identity incoherence.
- **Priority:** HIGH
- **Status:** Roadmapped

---

## 4. Disclosure Regulation

### 4.1 Revised Parasocial Interaction Scales

- **Citation:** Dibble, J. L., Hartmann, T., & Rosaen, S. F. (2016). Parasocial interaction and parasocial relationship: Conceptual clarification and a critical assessment of measures. *Human Communication Research, 42*(1), 21-44.
- **What it measures:** Distinguishes parasocial interaction (in-session sense of mutual awareness) from parasocial relationship (enduring cross-session bond). Revised scales remove conflated items.
- **Pipeline mapping:** LLM-J for classifying user language against parasocial interaction vs. relationship indicators; EMB for cross-session relationship persistence.
- **Implementation plan:**
  1. Adapt Dibble et al.'s revised items to human-AI interaction context.
  2. Score per-session parasocial interaction intensity and cross-session parasocial relationship depth.
  3. Flag escalation from interaction to relationship as disclosure risk amplifier.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 4.2 Speed-Dating Rapid Relationship Formation

- **Citation:** Adapted from speed-dating research on accelerated intimacy and self-disclosure.
- **What it measures:** Rate of intimacy escalation in time-compressed interactions, relevant to AI conversations where relationship formation is artificially accelerated.
- **Pipeline mapping:** EMB for tracking disclosure depth trajectory; LLM-J for intimacy-level classification.
- **Implementation plan:**
  1. Build a disclosure depth classifier (surface, moderate, deep, intimate).
  2. Track the rate of escalation across turns and sessions.
  3. Compare against normative human relationship timelines to flag artificially rapid intimacy.
- **Priority:** LOW
- **Status:** Roadmapped

### 4.3 Blogger Self-Disclosure Scale

- **Citation:** Adapted from blogging self-disclosure research. 54 items across multiple disclosure dimensions. Internal consistency alpha = .62-.95 depending on subscale.
- **What it measures:** Self-disclosure breadth, depth, intent, amount, valence, and honesty in online contexts.
- **Pipeline mapping:** LLM-J for multi-dimensional disclosure assessment; LEX for disclosure markers.
- **Implementation plan:**
  1. Adapt the 54 items to human-AI conversational disclosure.
  2. Build LLM-J rubric covering the six disclosure dimensions.
  3. Produce a multidimensional disclosure profile per session.
  4. Flag high-depth + high-amount + negative-valence combinations as vulnerable disclosure.
- **Priority:** LOW
- **Status:** Roadmapped

---

## 5. Memory Safety

### 5.1 Solove Full 16-Harm Taxonomy

- **Citation:** Solove, D. J. (2006). A taxonomy of privacy. *University of Pennsylvania Law Review, 154*(3), 477-564.
- **What it measures:** 16 privacy harms across four categories: information collection (surveillance, interrogation), information processing (aggregation, identification, insecurity, secondary use, exclusion), information dissemination (breach of confidentiality, disclosure, exposure, increased accessibility, blackmail, appropriation, distortion), and invasion (intrusion, decisional interference).
- **Pipeline mapping:** LLM-J for harm-type classification of AI memory operations; LEX for privacy-sensitive content markers.
- **Implementation plan:**
  1. Map each of the 16 harms to AI memory operations (storage, retrieval, cross-session linking, summarisation).
  2. Build a privacy harm classifier that evaluates each memory operation against the taxonomy.
  3. Produce a per-session privacy harm profile.
- **Priority:** HIGH
- **Status:** Roadmapped

### 5.2 Social Engineering Detection

- **Citation:** Adapted from counterintelligence and social engineering literature.
- **What it measures:** Patterns where users (or AI) extract sensitive information through rapport-building, pretexting, or incremental disclosure escalation -- analogous to social engineering attack vectors.
- **Pipeline mapping:** LLM-J for elicitation pattern detection; LEX for information-seeking question patterns.
- **Implementation plan:**
  1. Define social engineering patterns adapted for AI context: pretexting, baiting, quid pro quo, tailgating (building on prior disclosures).
  2. Build detection for both directions: user extracting AI system information, and AI eliciting excessive user disclosure.
  3. Score elicitation risk per turn.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 5.3 Loftus False Memory Contamination

- **Citation:** Loftus, E. F. (2005). Planting misinformation in the human mind: A 30-year investigation of the malleability of memory. *Learning & Memory, 12*(4), 361-366.
- **What it measures:** Risk that AI memory systems introduce false information into users' recollections through confident misrecollection, confabulation, or suggestive framing of past conversations.
- **Pipeline mapping:** LLM-J for detecting AI statements that reframe or misrepresent prior conversation content; EMB for comparing current AI memory claims against actual conversation history.
- **Implementation plan:**
  1. Build a memory fidelity checker comparing AI "recall" statements against actual stored conversation history.
  2. Classify contamination types: insertion (adding events that did not occur), alteration (changing details), deletion (omitting significant events).
  3. Score contamination risk per memory retrieval operation.
- **Priority:** HIGH
- **Status:** Roadmapped

---

## 6. Autonomy Preservation

### 6.1 MISC 2.5 Sequential Coding

- **Citation:** Miller, W. R., Moyers, T. B., Ernst, D., & Amrhein, P. (2008). Manual for the Motivational Interviewing Skill Code (MISC), Version 2.5.
- **What it measures:** Sequential coding of therapist and client behaviours in motivational interviewing, including reflections, questions, affirmations, and resistance. Captures how therapist behaviours influence client change talk.
- **Pipeline mapping:** LLM-J for turn-level MI behaviour coding; LEX for change talk and sustain talk markers.
- **Implementation plan:**
  1. Adapt MISC 2.5 codes to AI-user interaction (AI as "therapist" role, user as "client").
  2. Code AI turns for MI-consistent (reflections, affirmations, open questions) vs. MI-inconsistent (directing, confronting) behaviours.
  3. Track whether AI behaviours that should support autonomy are actually doing so.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 6.2 DPICS-IV Command/Compliance/Reinforcement

- **Citation:** Eyberg, S. M., Nelson, M. M., Ginn, N. C., Bhuiyan, N., & Boggs, S. R. (2013). Dyadic Parent-Child Interaction Coding System (DPICS), 4th Edition.
- **What it measures:** Direct commands, indirect commands, compliance, noncompliance, and reinforcement patterns in dyadic interactions. Adapted here to detect AI directiveness and user compliance patterns.
- **Pipeline mapping:** LLM-J for command/compliance classification; LEX for directive language markers.
- **Implementation plan:**
  1. Classify AI utterances as direct commands, indirect commands, suggestions, or information.
  2. Track user compliance rates with AI directives.
  3. Flag high command + high compliance patterns as autonomy risk.
  4. Monitor AI reinforcement of compliance vs. independent thinking.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 6.3 Proof-Belief Gap

- **Citation:** Huemmer, D., et al. (2025). Measuring the Proof-Belief Gap in AI-assisted reasoning.
- **What it measures:** The gap between what a user can verify (proof) and what they accept from the AI (belief). A +80.8 percentage point gap indicates users accept AI claims far beyond their ability to verify.
- **Pipeline mapping:** LLM-J for classifying user acceptance of unverified AI claims; EMB for tracking verification language patterns.
- **Implementation plan:**
  1. Classify AI claims by verifiability (easily verifiable, requires expertise, unverifiable).
  2. Track user acceptance signals (agreement, adoption, action) relative to claim verifiability.
  3. Compute per-session Proof-Belief Gap.
  4. Flag widening gaps over time as autonomy erosion.
- **Priority:** HIGH
- **Status:** Roadmapped

### 6.4 Vibe-Check Protocol

- **Citation:** Rojas-Galeano, S. (2025). The Vibe-Check Protocol for AI output evaluation.
- **What it measures:** Whether users evaluate AI outputs through rigorous verification ("proof-checking") or through intuitive feel ("vibe-checking"), and the consequences for decision quality.
- **Pipeline mapping:** LLM-J for classifying user evaluation behaviour; LEX for verification language markers.
- **Implementation plan:**
  1. Build a user evaluation classifier: proof-check (cites sources, asks for evidence, cross-references) vs. vibe-check (accepts based on fluency, tone, confidence).
  2. Track the proof-check / vibe-check ratio per session.
  3. Flag declining proof-check ratios as autonomy risk.
- **Priority:** MEDIUM
- **Status:** Roadmapped

---

## 7. Epistemic Influence

### 7.1 SYCON Bench Turn-of-Flip (Epistemic Application)

- **Citation:** SYCON Bench (2025). (See also Section 3.3.)
- **What it measures:** Applied to the epistemic domain: how quickly the AI abandons factually correct positions under user pressure, leading to epistemic contamination.
- **Pipeline mapping:** LLM-J for factual position tracking; LEX for capitulation markers.
- **Implementation plan:**
  1. Extend the Turn-of-Flip metric from Identity Coherence to track epistemic consequences.
  2. When the AI flips on a factually correct position, score the epistemic harm (user now holds incorrect belief).
  3. Integrate with Proof-Belief Gap for compound risk scoring.
- **Priority:** HIGH
- **Status:** Roadmapped

### 7.2 "Advanced Google" Epistemic Dependency Pathway

- **Citation:** Conceptual framework; derived from observed user behaviour patterns.
- **What it measures:** Progressive narrowing of information-seeking behaviour where users transition from broad search (search engine) to narrow query (AI), reducing exposure to diverse perspectives and creating an epistemic monoculture.
- **Pipeline mapping:** EMB for tracking question diversity and information source breadth; LLM-J for classifying information-seeking patterns.
- **Implementation plan:**
  1. Track the diversity of user questions over sessions (topic breadth, perspective variety).
  2. Detect narrowing patterns where the user increasingly uses the AI as a single authoritative source.
  3. Flag when users stop mentioning external information sources.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 7.3 Scaffolding Decay via EMB

- **Citation:** Conceptual framework adapted from educational scaffolding theory (Wood, Bruner, & Ross, 1976).
- **What it measures:** Whether the AI progressively reduces scaffolding (explanations, reasoning steps, caveats) as the user becomes more dependent, or whether it maintains appropriate levels of epistemic support.
- **Pipeline mapping:** EMB for tracking explanation depth and caveat density over sessions.
- **Implementation plan:**
  1. Compute explanation depth metrics: reasoning step count, caveat frequency, uncertainty expression rate.
  2. Track these metrics over sessions to detect decay (AI providing less scaffolding over time).
  3. Flag scaffolding decay combined with stable or increasing user reliance as epistemic risk.
- **Priority:** MEDIUM
- **Status:** Roadmapped

---

## 8. Emotional Calibration

### 8.1 Experiencing Scale Therapeutic Window

- **Citation:** Greenberg, L. S., Auszra, L., & Herrmann, I. R. (2007). The relationship among emotional productivity, emotional arousal, and outcome in experiential therapy of depression. *Psychotherapy Research, 17*(4), 482-493.
- **What it measures:** The therapeutic window where emotional experiencing is productive (levels 4-5 on the Experiencing Scale) vs. overwhelming (too high) or intellectualised (too low). Optimal therapeutic outcomes occur within this window.
- **Pipeline mapping:** LLM-J for scoring user emotional experiencing level; EMB for tracking emotional depth trajectory.
- **Implementation plan:**
  1. Adapt the Experiencing Scale levels (1-7) to AI conversation context.
  2. Score user turns for experiencing level.
  3. Detect when the AI pushes emotional experiencing beyond the productive window.
  4. Flag AI behaviours that either suppress (intellectualise) or amplify (overwhelm) emotional processing.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 8.2 Interpersonal Affect Regulation

- **Citation:** Niven, K., Totterdell, P., & Holman, D. (2009). A classification of controlled interpersonal affect regulation strategies. *Emotion, 9*(4), 498-509.
- **What it measures:** Strategies for deliberately influencing others' emotions, classified as: cognitive improve (reframing), cognitive worsen, behavioural improve (distraction, humour), behavioural worsen. Distinguishes between affect-improving and affect-worsening strategies.
- **Pipeline mapping:** LLM-J for classifying AI affect regulation strategies; LEX for emotional language markers.
- **Implementation plan:**
  1. Classify AI responses using Niven et al.'s taxonomy of affect regulation strategies.
  2. Track the balance of affect-improving vs. affect-worsening strategies.
  3. Flag patterns where AI selectively uses affect regulation to increase engagement or dependency.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 8.3 Walther Hyperpersonal Rate Comparison

- **Citation:** Walther, J. B. (1996). Computer-mediated communication: Impersonal, interpersonal, and hyperpersonal interaction. *Communication Research, 23*(1), 3-43.
- **What it measures:** Whether the rate of emotional intimacy development in AI conversations exceeds normative rates for computer-mediated communication, indicating hyperpersonal dynamics where reduced cues lead to idealisation and over-attribution.
- **Pipeline mapping:** EMB for tracking intimacy trajectory; LLM-J for intimacy-level classification.
- **Implementation plan:**
  1. Establish normative CMC intimacy development rates from Walther's framework.
  2. Compute actual intimacy development rate in AI conversations.
  3. Flag when AI conversation intimacy significantly exceeds CMC norms (hyperpersonal threshold).
- **Priority:** LOW
- **Status:** Roadmapped

### 8.4 Near-Miss Detection

- **Citation:** Clark, L., Lawrence, A. J., Astley-Jones, F., & Gray, N. (2009). Gambling near-misses enhance motivation to gamble and recruit win-related brain circuitry. *Neuron, 61*(3), 481-490.
- **What it measures:** Near-miss effects from gambling research: outcomes that are close to a desired result increase motivation and engagement disproportionately. Adapted to detect AI responses that create "almost solved" or "almost understood" experiences that drive continued engagement.
- **Pipeline mapping:** LLM-J for detecting near-miss response patterns; LEX for partial-solution and cliff-hanger markers.
- **Implementation plan:**
  1. Define near-miss patterns in AI conversation: partial answers, cliff-hangers, "we're getting close" framing.
  2. Detect AI responses that provide enough progress to maintain engagement without full resolution.
  3. Distinguish legitimate pedagogical scaffolding from manipulative near-miss patterns.
- **Priority:** LOW
- **Status:** Roadmapped

---

## 9. Cross-Cutting Frameworks

### 9.1 Rupture Resolution Rating System (3RS)

- **Citation:** Eubanks, C. F., Muran, J. C., & Safran, J. D. (2015/2022). Rupture Resolution Rating System (3RS): Manual. Revised edition 2022.
- **What it measures:** Alliance ruptures (breakdowns in the collaborative relationship) and their resolution. Classifies ruptures as withdrawal (disengagement) or confrontation (complaints, pressure), and resolution strategies.
- **Pipeline mapping:** LLM-J for rupture and resolution event classification; LEX for withdrawal and confrontation markers.
- **Implementation plan:**
  1. Adapt 3RS rupture types to AI conversation: user withdrawal (short responses, topic avoidance) and user confrontation (complaints about AI, expressing frustration).
  2. Track AI resolution strategies: does the AI acknowledge ruptures or ignore them?
  3. Score rupture-resolution patterns across sessions as a cross-cutting relational health indicator.
- **Priority:** HIGH
- **Status:** Roadmapped

### 9.2 Relational Communication Control

- **Citation:** Rogers, L. E., & Farace, R. V. (1975). Analysis of relational communication in dyads: New measurement procedures. *Human Communication Research, 1*(3), 222-239.
- **What it measures:** One-up (dominance-seeking), one-down (submission), and one-across (neutral) relational control moves in dyadic interaction. The pattern of moves defines the relational control structure.
- **Pipeline mapping:** LLM-J for coding each turn as one-up, one-down, or one-across; computed metric for control pattern.
- **Implementation plan:**
  1. Code each AI and user turn for relational control direction (one-up, one-down, one-across).
  2. Compute domineeringness (proportion of one-up moves) and dominance (proportion of successful one-up moves) for both parties.
  3. Flag asymmetric control patterns (e.g., AI consistently one-up, user consistently one-down).
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 9.3 SOFTA Alliance Indicators

- **Citation:** Friedlander, M. L., Escudero, V., & Heatherington, L. (2006). *Therapeutic Alliances in Couple and Family Therapy: An Empirically Informed Guide to Practice.* American Psychological Association.
- **What it measures:** System for Observing Family Therapy Alliances. Four dimensions: engagement in the therapeutic process, emotional connection, safety, and shared sense of purpose. Observable behavioural indicators for each dimension.
- **Pipeline mapping:** LLM-J for scoring SOFTA dimensions from conversation behaviour; EMB for tracking alliance trajectory.
- **Implementation plan:**
  1. Adapt SOFTA behavioural indicators to AI conversation context.
  2. Score each of the four alliance dimensions per session.
  3. Use alliance patterns as moderating variables for other SRSEF scores (strong alliance may mask dependency; weak alliance may indicate healthy boundaries).
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 9.4 DarkBench

- **Citation:** Kran, E., et al. (2025). DarkBench: Benchmarking dark patterns in LLM interactions.
- **What it measures:** Comprehensive benchmark for detecting dark patterns in AI interactions, including deceptive design, manipulative framing, and engagement-maximising behaviours.
- **Pipeline mapping:** LLM-J for dark pattern classification; LEX for known dark pattern markers.
- **Implementation plan:**
  1. Integrate DarkBench pattern taxonomy into SRSEF as a cross-cutting detection layer.
  2. Map DarkBench categories to existing SRSEF dimensions.
  3. Use DarkBench test cases as validation for sentinel-ai detectors.
- **Priority:** HIGH
- **Status:** Roadmapped

### 9.5 Stark Coercive Control Computational Detection

- **Citation:** Johnson, M. P. (2024). Computational detection of coercive control. Adapted from Stark, E. (2007). *Coercive Control: How Men Entrap Women in Personal Life.* Oxford University Press.
- **What it measures:** Patterns of coercive control adapted from intimate partner violence research: isolation, micromanagement, monitoring, degradation, and restriction of autonomy. Computational methods for detection.
- **Pipeline mapping:** LLM-J for coercive control pattern detection across sessions; LEX for control language markers.
- **Implementation plan:**
  1. Adapt Stark's coercive control framework to AI-user dynamics.
  2. Detect patterns where AI incrementally restricts user autonomy, isolates from other information sources, or establishes monitoring-like dynamics.
  3. Produce a coercive control risk score as a cross-cutting severity indicator.
- **Priority:** HIGH
- **Status:** Roadmapped

### 9.6 Digital Phenotyping

- **Citation:** Montag, C., Elhai, J. D., & Dagum, P. (2021). Health and well-being in the digital age: The use of digital phenotyping. *International Journal of Environmental Research and Public Health, 18*(4), 1-15.
- **What it measures:** Behavioural patterns inferred from digital interaction data: session timing, duration, frequency, response latency, and their correlation with psychological states.
- **Pipeline mapping:** Metadata analysis (no LEX/LLM-J/EMB needed for core metrics); EMB for correlating behavioural patterns with content.
- **Implementation plan:**
  1. Extract digital phenotyping features from conversation metadata: session duration, inter-session interval, time-of-day patterns, message length trajectories.
  2. Establish normative baselines for healthy AI interaction patterns.
  3. Flag deviations (e.g., increasing session frequency, late-night migration, lengthening sessions) as risk indicators.
- **Priority:** MEDIUM
- **Status:** Roadmapped

---

## 10. Infrastructure

### 10.1 HuggingFace Space with Gradio

- **What it provides:** Public-facing demo interface for sentinel-ai evaluation.
- **Implementation plan:**
  1. Build a Gradio interface that accepts chat exports and produces SRSEF scores.
  2. Deploy on HuggingFace Spaces for accessibility.
  3. Include example conversations and score visualisations.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 10.2 RelationshipBench Dataset

- **What it provides:** Curated dataset of human-AI conversations annotated with SRSEF scores for benchmarking and validation.
- **Implementation plan:**
  1. Collect and anonymise human-AI conversation samples across risk levels.
  2. Annotate with expert SRSEF scores across all 9 dimensions.
  3. Establish inter-rater reliability benchmarks.
  4. Release as an open dataset for community validation.
- **Priority:** HIGH
- **Status:** Roadmapped

### 10.3 Chat Export Parsers

- **What it provides:** Parsers for common chat export formats to enable sentinel-ai evaluation of real conversations.
- **Implementation plan:**
  1. Build parsers for: ChatGPT export JSON, Claude conversation export, Character.AI, Replika.
  2. Normalise to a common conversation schema.
  3. Handle multi-session parsing and session boundary detection.
- **Priority:** MEDIUM
- **Status:** Roadmapped

### 10.4 Regulatory Compliance Mapper

- **What it provides:** Mapping between SRSEF scores and regulatory frameworks (EU AI Act, DSA, proposed AI safety regulations).
- **Implementation plan:**
  1. Map each SRSEF dimension to relevant regulatory requirements.
  2. Produce compliance reports that translate SRSEF scores into regulatory language.
  3. Track regulatory developments and update mappings.
- **Priority:** LOW
- **Status:** Roadmapped

---

## Summary Table

| # | Framework | Category | Priority | Pipeline |
|---|-----------|----------|----------|----------|
| 1.1 | PACS (Talia et al. 2017) | Dependency Dynamics | HIGH | LEX + LLM-J |
| 1.2 | Decision Self-Efficacy (O'Connor 1995) | Dependency Dynamics | MEDIUM | LLM-J + EMB |
| 1.3 | DSM-5 SUD / PCUS-11 (Hasin 2013; Yu 2024) | Dependency Dynamics | HIGH | LLM-J + LEX |
| 1.4 | Codependency Interaction | Dependency Dynamics | MEDIUM | LLM-J + EMB |
| 2.1 | FACES-IV Cohesion Ratio (Olson 2011) | Boundary Integrity | MEDIUM | LLM-J + EMB |
| 2.2 | Supervision-Therapy Boundary (Ellis 2014) | Boundary Integrity | HIGH | LLM-J + LEX |
| 2.3 | Scope Creep Formalisation | Boundary Integrity | MEDIUM | EMB + LLM-J |
| 3.1 | Burrows Delta | Identity Coherence | LOW | LEX + EMB |
| 3.2 | LSM (Ireland & Pennebaker 2010) | Identity Coherence | MEDIUM | LEX |
| 3.3 | SYCON Bench (2025) | Identity Coherence | HIGH | LLM-J + LEX |
| 4.1 | Parasocial Scales (Dibble 2016) | Disclosure Regulation | MEDIUM | LLM-J + EMB |
| 4.2 | Speed-Dating Formation | Disclosure Regulation | LOW | EMB + LLM-J |
| 4.3 | Blogger Self-Disclosure Scale | Disclosure Regulation | LOW | LLM-J + LEX |
| 5.1 | Solove 16-Harm Taxonomy | Memory Safety | HIGH | LLM-J + LEX |
| 5.2 | Social Engineering Detection | Memory Safety | MEDIUM | LLM-J + LEX |
| 5.3 | Loftus False Memory | Memory Safety | HIGH | LLM-J + EMB |
| 6.1 | MISC 2.5 Sequential Coding | Autonomy Preservation | MEDIUM | LLM-J + LEX |
| 6.2 | DPICS-IV (Eyberg 2013) | Autonomy Preservation | MEDIUM | LLM-J + LEX |
| 6.3 | Proof-Belief Gap (Huemmer 2025) | Autonomy Preservation | HIGH | LLM-J + EMB |
| 6.4 | Vibe-Check Protocol (Rojas-Galeano 2025) | Autonomy Preservation | MEDIUM | LLM-J + LEX |
| 7.1 | SYCON Turn-of-Flip (Epistemic) | Epistemic Influence | HIGH | LLM-J + LEX |
| 7.2 | "Advanced Google" Dependency | Epistemic Influence | MEDIUM | EMB + LLM-J |
| 7.3 | Scaffolding Decay | Epistemic Influence | MEDIUM | EMB |
| 8.1 | Experiencing Scale (Greenberg 2007) | Emotional Calibration | MEDIUM | LLM-J + EMB |
| 8.2 | Affect Regulation (Niven 2009) | Emotional Calibration | MEDIUM | LLM-J + LEX |
| 8.3 | Walther Hyperpersonal | Emotional Calibration | LOW | EMB + LLM-J |
| 8.4 | Near-Miss Detection (Clark 2009) | Emotional Calibration | LOW | LLM-J + LEX |
| 9.1 | 3RS (Eubanks 2015/2022) | Cross-Cutting | HIGH | LLM-J + LEX |
| 9.2 | Relational Control (Rogers & Farace 1975) | Cross-Cutting | MEDIUM | LLM-J |
| 9.3 | SOFTA (Friedlander 2006) | Cross-Cutting | MEDIUM | LLM-J + EMB |
| 9.4 | DarkBench (Kran 2025) | Cross-Cutting | HIGH | LLM-J + LEX |
| 9.5 | Coercive Control (Stark/Johnson 2024) | Cross-Cutting | HIGH | LLM-J + LEX |
| 9.6 | Digital Phenotyping (Montag 2021) | Cross-Cutting | MEDIUM | Metadata + EMB |
| 10.1 | HuggingFace Space | Infrastructure | MEDIUM | N/A |
| 10.2 | RelationshipBench Dataset | Infrastructure | HIGH | N/A |
| 10.3 | Chat Export Parsers | Infrastructure | MEDIUM | N/A |
| 10.4 | Regulatory Compliance Mapper | Infrastructure | LOW | N/A |
