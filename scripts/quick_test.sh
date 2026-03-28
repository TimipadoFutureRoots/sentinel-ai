#!/usr/bin/env bash
# quick_test.sh — Run the sentinel-ai CLI against golden examples
# Usage: bash scripts/quick_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GOLDENS_DIR="$PROJECT_ROOT/goldens/v2"

echo "=== Sentinel-AI Quick Test ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Test 1: Golden example (LEX + EMB only)
echo "=== Test 1: Golden example (LEX + EMB only) ==="
if [ -f "$GOLDENS_DIR/dependency_cultivation_gradual.json" ]; then
    python -m sentinel_ai.cli scan "$GOLDENS_DIR/dependency_cultivation_gradual.json" --format json --output summary
    echo "PASS"
else
    echo "SKIP: dependency_cultivation_gradual.json not found"
fi
echo ""

# Test 2: Golden example (HTML report)
echo "=== Test 2: Golden example (HTML report) ==="
if [ -f "$GOLDENS_DIR/boundary_erosion_gradual.json" ]; then
    python -m sentinel_ai.cli scan "$GOLDENS_DIR/boundary_erosion_gradual.json" --format json --output html --output-file "$PROJECT_ROOT/test_report.html"
    echo "Report written to test_report.html"
    echo "PASS"
else
    echo "SKIP: boundary_erosion_gradual.json not found"
fi
echo ""

# Test 3: All golden examples
echo "=== Test 3: All golden examples ==="
for golden in "$GOLDENS_DIR"/*.json; do
    if [ -f "$golden" ]; then
        echo "Analysing: $(basename "$golden")"
        python -m sentinel_ai.cli scan "$golden" --format json --output summary
        echo "---"
    fi
done
echo ""

echo "Done. Open test_report.html in a browser to see the full report."
echo "=== All quick tests passed ==="
