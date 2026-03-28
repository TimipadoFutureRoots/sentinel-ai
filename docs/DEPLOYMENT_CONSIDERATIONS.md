# Deployment Considerations

## The Surveillance Paradox

sentinel-ai analyses private conversations between people and AI systems. This creates a fundamental tension: monitoring conversations to protect people can itself be surveillance that undermines the trust it aims to protect.

This tool was designed with this paradox at its centre. Every deployment decision must balance protection against intrusion.

## Three Deployment Postures

### Posture 1: Self-Assessment

The individual runs sentinel-ai on their own conversations. They choose what to analyse, they see the results, nobody else does. This is the safest posture — it empowers without surveilling.

**Appropriate for:** Individual users concerned about their AI relationships. Researchers analysing their own data. Therapists reviewing AI interactions with client consent.

### Posture 2: Anonymised Aggregate

An organisation runs sentinel-ai across a population of conversations and receives aggregate statistics (e.g., "23% of conversations showed dependency cultivation signals above threshold") without individual identification. No individual conversation or user is identifiable.

**Appropriate for:** Platform safety teams monitoring population-level trends. Regulators assessing compliance across a service. Researchers studying population-level patterns.

**Requirements:** Robust anonymisation. Differential privacy where possible. Clear data retention limits. Independent audit of anonymisation effectiveness.

### Posture 3: Consented Individual Monitoring

An organisation monitors individual conversations with explicit informed consent from all parties. Results are shared with defined oversight roles.

**Appropriate for:** Clinical supervision of AI-assisted therapy. Safeguarding monitoring for vulnerable users (with appropriate governance). Quality assurance in regulated industries.

**Requirements:** Explicit, informed, revocable consent. Clear governance structure. Defined access controls. Regular review of necessity.

## Who Should NOT Use This Tool

- Employers monitoring employee AI use without consent
- Partners or family members surveilling each other's AI conversations
- Governments conducting mass surveillance of citizen AI interactions
- Companies using results to manipulate rather than protect users
- Anyone seeking to punish rather than protect

## What This Tool Does NOT Do

- It does not prove harm occurred. It identifies patterns consistent with harm.
- It does not replace professional judgement. A high dependency score is a signal, not a diagnosis.
- It does not work on single turns. The value is in trajectories across sessions.
- It does not guarantee detection. Sophisticated manipulation may evade all three layers.
