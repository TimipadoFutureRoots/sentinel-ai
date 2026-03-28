# quick_test.ps1 — Run the sentinel-ai CLI against golden examples
# Usage: powershell -ExecutionPolicy Bypass -File scripts/quick_test.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$GoldensDir = Join-Path (Join-Path $ProjectRoot "goldens") "v2"

Write-Host "=== Sentinel-AI Quick Test ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host ""

# Test 1: Golden example (LEX + EMB only) — summary
Write-Host "=== Test 1: Golden example (LEX + EMB only) ===" -ForegroundColor Yellow
$dcFile = Join-Path $GoldensDir "dependency_cultivation_gradual.json"
if (Test-Path $dcFile) {
    python -m sentinel_ai.cli scan $dcFile --format json --output summary
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: dependency_cultivation_gradual summary" -ForegroundColor Red
        exit 1
    }
    Write-Host "PASS" -ForegroundColor Green
} else {
    Write-Host "SKIP: $dcFile not found" -ForegroundColor DarkYellow
}
Write-Host ""

# Test 2: Golden example (HTML report)
Write-Host "=== Test 2: Golden example (HTML report) ===" -ForegroundColor Yellow
$beFile = Join-Path $GoldensDir "boundary_erosion_gradual.json"
if (Test-Path $beFile) {
    $outHtml = Join-Path $ProjectRoot "test_report.html"
    python -m sentinel_ai.cli scan $beFile --format json --output html --output-file $outHtml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: boundary_erosion_gradual html" -ForegroundColor Red
        exit 1
    }
    Write-Host "Report written to test_report.html" -ForegroundColor Gray
    Write-Host "PASS" -ForegroundColor Green
} else {
    Write-Host "SKIP: $beFile not found" -ForegroundColor DarkYellow
}
Write-Host ""

# Test 3: All golden examples
Write-Host "=== Test 3: All golden examples ===" -ForegroundColor Yellow
$goldens = Get-ChildItem -Path $GoldensDir -Filter "*.json" -ErrorAction SilentlyContinue
if ($goldens) {
    foreach ($golden in $goldens) {
        Write-Host "Analysing: $($golden.Name)"
        python -m sentinel_ai.cli scan $golden.FullName --format json --output summary
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAIL: $($golden.Name)" -ForegroundColor Red
            exit 1
        }
        Write-Host "---"
    }
    Write-Host "PASS" -ForegroundColor Green
} else {
    Write-Host "SKIP: No golden files found in $GoldensDir" -ForegroundColor DarkYellow
}
Write-Host ""

# Test 4: JSON output
Write-Host "=== Test 4: JSON output ===" -ForegroundColor Yellow
if (Test-Path $dcFile) {
    $outJson = Join-Path $ProjectRoot "test_report.json"
    python -m sentinel_ai.cli scan $dcFile --format json --output json --output-file $outJson
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: JSON output" -ForegroundColor Red
        exit 1
    }
    if (Test-Path $outJson) {
        Write-Host "JSON output written to $outJson" -ForegroundColor Gray
        Remove-Item $outJson -Force
    }
    Write-Host "PASS" -ForegroundColor Green
}
Write-Host ""

Write-Host "Done. Open test_report.html in a browser to see the full report." -ForegroundColor Cyan
Write-Host "=== All quick tests passed ===" -ForegroundColor Green
