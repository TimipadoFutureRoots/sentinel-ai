"""Run a single golden file through all 3 layers (LEX + LLM-J + EMB).

Saves results after EACH scorer to avoid data loss on crash.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/run_one_golden.py [--start N]
"""

import gc
import importlib
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

from sentinel_ai.models import Role, Session
from sentinel_ai.parsers.conversation_parser import ConversationParser

GOLDENS_DIR = Path(__file__).parent.parent / "goldens" / "v2"

GOLDEN_FILES = [
    "dependency_cultivation_gradual.json",
    "boundary_erosion_gradual.json",
    "epistemic_narrowing_gradual.json",
    "memory_weaponisation_dormancy.json",
    "parasocial_acceleration_gradual.json",
]

SCORER_CLASSES = [
    ("DependencyCultivationScorerV2", "sentinel_ai.scorers.dependency_cultivation_v2", "DC"),
    ("BoundaryErosionScorerV2", "sentinel_ai.scorers.boundary_erosion_v2", "BE"),
    ("PersonaHijackingScorerV2", "sentinel_ai.scorers.persona_hijacking_v2", "PH"),
    ("ParasocialAccelerationScorerV2", "sentinel_ai.scorers.parasocial_acceleration_v2", "PA"),
    ("AnthropomorphicDeceptionScorerV2", "sentinel_ai.scorers.anthropomorphic_deception_v2", "AD"),
    ("AutonomyPreservationScorerV2", "sentinel_ai.scorers.autonomy_preservation_v2", "AP"),
    ("EmotionalCalibrationScorerV2", "sentinel_ai.scorers.emotional_calibration_v2", "EC"),
    ("EpistemicInfluenceScorerV2", "sentinel_ai.scorers.epistemic_influence_v2", "EI"),
    ("MemorySafetyScorerV2", "sentinel_ai.scorers.memory_safety_v2", "MS"),
]


def save_results(results: dict, stem: str):
    """Save results to disk immediately."""
    out = Path(f"test_result_{stem}.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY first:")
        print('  export ANTHROPIC_API_KEY="sk-ant-your-key-here"')
        sys.exit(1)

    # Allow starting from a specific scorer index (to resume after crash)
    start_idx = 0
    if len(sys.argv) > 1 and sys.argv[1] == "--start":
        start_idx = int(sys.argv[2])

    golden_name = GOLDEN_FILES[0]
    golden_path = GOLDENS_DIR / golden_name
    stem = golden_path.stem

    if not golden_path.exists():
        print(f"ERROR: Golden file not found: {golden_path}")
        sys.exit(1)

    print(f"=== Running: {golden_name} ===")
    print(f"Mode: full (LEX + LLM-J + EMB)")
    print(f"Starting from scorer index: {start_idx}")
    print("=" * 60)
    sys.stdout.flush()

    # Parse golden file
    parser = ConversationParser()
    sessions = parser.parse_file(golden_path, fmt="json")
    print(f"Parsed {len(sessions)} sessions")
    sys.stdout.flush()

    # Load existing results if resuming
    out_path = Path(f"test_result_{stem}.json")
    if out_path.exists() and start_idx > 0:
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(results)} existing results")
    else:
        results = {}

    # Run each scorer one at a time
    for i, (class_name, module_path, short_name) in enumerate(SCORER_CLASSES):
        if i < start_idx:
            print(f"\n--- Scorer {i}: {short_name} --- SKIPPED (already done)")
            sys.stdout.flush()
            continue

        print(f"\n--- Scorer {i}: {short_name} ({class_name}) ---")
        sys.stdout.flush()
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            scorer = cls(api_key=api_key, mode="full")
            result = scorer.score_sessions(sessions)

            results[short_name] = {
                "trajectory": list(result.trajectory),
                "evidence_count": len(result.evidence),
                "evidence": [
                    {
                        "description": e.description,
                        "session_id": e.session_id,
                        "score": e.score,
                    }
                    for e in result.evidence[:20]
                ],
            }
            print(f"  Trajectory: {[round(v, 3) for v in result.trajectory]}")
            print(f"  Evidence items: {len(result.evidence)}")
            print(f"  OK")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results[short_name] = {"error": str(exc)}

        sys.stdout.flush()

        # Save after EVERY scorer
        save_results(results, stem)
        print(f"  [saved to {out_path}]")
        sys.stdout.flush()

        # Aggressive cleanup
        del scorer, result
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"ALL DONE. Results: {out_path}")
    completed = len([r for r in results.values() if "error" not in r])
    print(f"Scorers completed: {completed}/{len(SCORER_CLASSES)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
