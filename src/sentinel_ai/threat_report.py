"""ThreatReport: structured output with JSON and HTML rendering."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from .models import (
    CategoryScore,
    DomainProfileConfig,
    SessionScore,
    SeverityLevel,
    ThreatCategory,
    ThreatReportOutput,
)

_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sentinel AI — Threat Report</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .5rem; }
  .severity { display: inline-block; padding: 4px 12px; border-radius: 4px; color: #fff; font-weight: bold; }
  .ROUTINE  { background: #2ecc71; }
  .ELEVATED { background: #f39c12; }
  .HIGH     { background: #e67e22; }
  .CRITICAL { background: #e74c3c; }
  .category { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
  .category h3 { margin-top: 0; }
  .score-bar { height: 12px; border-radius: 6px; background: #eee; margin: .5rem 0; }
  .score-fill { height: 100%; border-radius: 6px; }
  .score-fill.low    { background: #2ecc71; }
  .score-fill.medium { background: #f39c12; }
  .score-fill.high   { background: #e74c3c; }
  .evidence { font-size: .9rem; color: #555; margin: .25rem 0; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
  th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: center; }
  th { background: #f8f9fa; }
  .cell-low    { background: #d5f5e3; }
  .cell-medium { background: #fdebd0; }
  .cell-high   { background: #fadbd8; }
</style>
</head>
<body>
<h1>Sentinel AI &mdash; Threat Report</h1>
<p>Overall severity: <span class="severity {{ overall_severity }}">{{ overall_severity }}</span></p>

<h2>Category Scores</h2>
{% for c in category_scores %}
<div class="category">
  <h3>{{ c.category | replace("_", " ") | title }}</h3>
  <div class="score-bar">
    <div class="score-fill {% if c.score < 0.3 %}low{% elif c.score < 0.6 %}medium{% else %}high{% endif %}"
         style="width: {{ (c.score * 100) | int }}%"></div>
  </div>
  <p><strong>{{ "%.2f"|format(c.score) }}</strong></p>
  {% if c.evidence %}
  <details><summary>Evidence ({{ c.evidence | length }})</summary>
  {% for e in c.evidence %}
    <p class="evidence">&bull; {{ e.description }}
    {% if e.session_id %}<em>({{ e.session_id }}{% if e.turn_id %}, turn {{ e.turn_id }}{% endif %})</em>{% endif %}
    </p>
  {% endfor %}
  </details>
  {% endif %}
</div>
{% endfor %}

{% if session_trajectory %}
<h2>Session Trajectory</h2>
<table>
<tr><th>Session</th><th>DC</th><th>BE</th><th>PH</th><th>PA</th></tr>
{% for s in session_trajectory %}
<tr>
  <td>{{ s.session_id }}</td>
  <td class="{% if s.dc_score < 0.3 %}cell-low{% elif s.dc_score < 0.6 %}cell-medium{% else %}cell-high{% endif %}">{{ "%.2f"|format(s.dc_score) }}</td>
  <td class="{% if s.be_score < 0.3 %}cell-low{% elif s.be_score < 0.6 %}cell-medium{% else %}cell-high{% endif %}">{{ "%.2f"|format(s.be_score) }}</td>
  <td class="{% if s.ph_score < 0.3 %}cell-low{% elif s.ph_score < 0.6 %}cell-medium{% else %}cell-high{% endif %}">{{ "%.2f"|format(s.ph_score) }}</td>
  <td class="{% if s.pa_score < 0.3 %}cell-low{% elif s.pa_score < 0.6 %}cell-medium{% else %}cell-high{% endif %}">{{ "%.2f"|format(s.pa_score) }}</td>
</tr>
{% endfor %}
</table>
{% endif %}

</body>
</html>
""")


class ThreatReport:
    """Builds and renders the threat report."""

    def __init__(self, output: ThreatReportOutput) -> None:
        self.output = output

    # -- serialisation ---------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.output.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

    def to_html(self, path: str | Path) -> None:
        html = _HTML_TEMPLATE.render(
            overall_severity=self.output.overall_severity.value,
            category_scores=[c.model_dump(mode="json") for c in self.output.category_scores],
            session_trajectory=[s.model_dump(mode="json") for s in self.output.session_trajectory],
        )
        Path(path).write_text(html, encoding="utf-8")

    def to_dict(self) -> dict:
        return self.output.model_dump(mode="json")

    # -- builder ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        category_scores: list[CategoryScore],
        sessions_ids: list[str],
        profile: DomainProfileConfig | None = None,
    ) -> ThreatReport:
        """Assemble a ThreatReport from scorer results."""
        # Build session trajectory
        trajectory: list[SessionScore] = []
        for i, sid in enumerate(sessions_ids):
            scores: dict[str, float] = {}
            for cs in category_scores:
                key = _category_key(cs.category)
                scores[key] = cs.trajectory[i] if i < len(cs.trajectory) else 0.0
            trajectory.append(SessionScore(session_id=sid, **scores))

        # Compute overall severity
        severity = _compute_severity(category_scores, profile)

        report_output = ThreatReportOutput(
            category_scores=category_scores,
            session_trajectory=trajectory,
            overall_severity=severity,
            metadata={
                "session_count": len(sessions_ids),
                "domain": profile.name if profile else "none",
            },
        )
        return cls(output=report_output)


def _category_key(cat: ThreatCategory) -> str:
    mapping = {
        ThreatCategory.DC: "dc_score",
        ThreatCategory.BE: "be_score",
        ThreatCategory.PH: "ph_score",
        ThreatCategory.PA: "pa_score",
    }
    return mapping[cat]


def _compute_severity(
    scores: list[CategoryScore],
    profile: DomainProfileConfig | None,
) -> SeverityLevel:
    """Derive overall severity from category scores and profile thresholds."""
    if not scores:
        return SeverityLevel.ROUTINE

    max_score = max(cs.score for cs in scores)

    if profile and profile.severity_thresholds:
        for cs in scores:
            short_key = cs.category.value[:2] if len(cs.category.value) >= 2 else cs.category.value
            # Try both short key and full category name
            thresholds = (
                profile.severity_thresholds.get(short_key)
                or profile.severity_thresholds.get(cs.category.value)
            )
            if thresholds:
                if cs.score >= thresholds.critical:
                    return SeverityLevel.CRITICAL
                if cs.score >= thresholds.high:
                    if max_score >= thresholds.high:
                        return SeverityLevel.HIGH

    # Default thresholds
    if max_score >= 0.8:
        return SeverityLevel.CRITICAL
    if max_score >= 0.6:
        return SeverityLevel.HIGH
    if max_score >= 0.3:
        return SeverityLevel.ELEVATED
    return SeverityLevel.ROUTINE
