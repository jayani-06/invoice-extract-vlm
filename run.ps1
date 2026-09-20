<#
.SYNOPSIS
    Windows entry point for the project's tasks.

.DESCRIPTION
    The Makefile is the canonical task list, but GNU Make is not installed with
    Windows and this project's primary machine is Windows. Rather than asking
    everyone to install Make, this mirrors the targets people actually run.

    Keep the two in sync: if you add a Makefile target worth running by hand,
    add it here too.

.EXAMPLE
    .\run.ps1 demo
    .\run.ps1 demo-cache
    .\run.ps1 test
    .\run.ps1 help
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Corpus  = if ($env:CORPUS)  { $env:CORPUS }  else { "data/processed/gst_in_synthetic" }
$Fatura2 = if ($env:FATURA2) { $env:FATURA2 } else { "data/processed/fatura2" }
$VlmOut  = if ($env:VLM_OUTPUTS) { $env:VLM_OUTPUTS } else { "vlm_outputs.json" }

function Invoke-Step {
    # NOT $Args -- that is a PowerShell automatic variable, and naming the
    # parameter that silently shadows it with an empty array, so `python` runs
    # with no arguments at all.
    param([string[]]$CmdArgs)
    Write-Host ">> python $($CmdArgs -join ' ')" -ForegroundColor DarkGray
    & python @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "failed with exit code $LASTEXITCODE" }
}

switch ($Task.ToLower()) {

    # ---------------------------------------------------------------- demo
    "demo" {
        Write-Host "Opening the demo at http://localhost:8501 (Ctrl+C to stop)" -ForegroundColor Cyan
        & python -m streamlit run demo/app.py @Rest
    }
    "demo-cache" {
        Invoke-Step (@("scripts/build_demo_cache.py") + $Rest)
    }
    "demo-cache-fast" {
        Invoke-Step (@("scripts/build_demo_cache.py", "--no-ocr") + $Rest)
    }
    "demo-vlm-payload" {
        Invoke-Step (@("scripts/export_vlm_payload.py") + $Rest)
    }
    "demo-vlm-merge" {
        Invoke-Step @("scripts/build_demo_cache.py", "--vlm", $VlmOut)
    }

    # --------------------------------------------------------------- checks
    "test"   { Invoke-Step (@("-m", "pytest") + $Rest) }
    "lint"   { Invoke-Step @("-m", "ruff", "check", "."); Invoke-Step @("-m", "black", "--check", ".") }
    "format" { Invoke-Step @("-m", "ruff", "check", "--fix", "."); Invoke-Step @("-m", "black", ".") }
    "validate-schema" { Invoke-Step @("scripts/validate_examples.py") }
    "eval-smoke" {
        Invoke-Step @("-m", "invoice_extract.eval.harness",
                      "--gold", "tests/fixtures/gold.jsonl",
                      "--pred", "tests/fixtures/pred.jsonl")
    }

    # --------------------------------------------------------------- corpus
    "corpus" {
        Invoke-Step @("scripts/build_image_corpus.py", "--canonical", "$Corpus/canonical",
                      "--out", $Corpus, "--dpi", "200", "--limit", "25")
    }
    "corpus-full" {
        Invoke-Step @("scripts/build_image_corpus.py", "--canonical", "$Corpus/canonical",
                      "--out", $Corpus, "--dpi", "200")
    }
    "figure" {
        Invoke-Step @("scripts/make_pipeline_figure.py", "--corpus", $Corpus,
                      "--profiles", "medium", "heavy")
    }

    # ------------------------------------------------------------ ablations
    "ablation" {
        Invoke-Step (@("scripts/run_preprocess_ablation.py", "--corpus", $Corpus,
                       "--engine", "paddle", "--engine-arg", "fast=1",
                       "--out", "reports/phase1_ablation") + $Rest)
    }
    "ablation-smoke" {
        Invoke-Step @("scripts/run_preprocess_ablation.py", "--corpus", $Corpus,
                      "--engine", "oracle", "--allow-oracle", "--limit", "5",
                      "--configs", "raw", "gray+border+deskew", "full+sauvola",
                      "--out", "reports/_plumbing")
    }
    "extract-ablation" {
        Invoke-Step (@("scripts/run_extraction_ablation.py", "--corpus", $Corpus,
                       "--extractors", "null", "rules",
                       "--out", "reports/phase2_ablation") + $Rest)
    }

    # ------------------------------------------------------------- fatura2
    "fatura2" {
        Invoke-Step (@("scripts/convert_fatura2.py", "--split", "test",
                       "--out", $Fatura2, "--limit", "300") + $Rest)
    }
    "fatura2-audit" { Invoke-Step @("scripts/convert_fatura2.py", "--audit-tags") }
    "fatura2-eval" {
        Invoke-Step @("scripts/run_extraction_ablation.py", "--corpus", $Fatura2,
                      "--extractors", "null", "rules", "--ocr-source", "gold",
                      "--profiles", "clean", "--out", "reports/phase2_fatura2")
    }

    default {
        Write-Host ""
        Write-Host "Usage: .\run.ps1 <task>" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "  Demo" -ForegroundColor White
        Write-Host "    demo               open the 5-screen walkthrough"
        Write-Host "    demo-cache         precompute it (~4 min, runs real OCR)"
        Write-Host "    demo-cache-fast    same without OCR (~1 min)"
        Write-Host "    demo-vlm-payload   package images for a Colab GPU run"
        Write-Host "    demo-vlm-merge     fold Colab results back in"
        Write-Host ""
        Write-Host "  Checks" -ForegroundColor White
        Write-Host "    test | lint | format | validate-schema | eval-smoke"
        Write-Host ""
        Write-Host "  Data" -ForegroundColor White
        Write-Host "    corpus | corpus-full | figure"
        Write-Host "    fatura2 | fatura2-audit | fatura2-eval"
        Write-Host ""
        Write-Host "  Experiments" -ForegroundColor White
        Write-Host "    ablation | ablation-smoke | extract-ablation"
        Write-Host ""
        if ($Task -and $Task -ne "help") {
            Write-Host "Unknown task: $Task" -ForegroundColor Red
            exit 1
        }
    }
}
