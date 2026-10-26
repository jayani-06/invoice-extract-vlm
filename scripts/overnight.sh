#!/bin/bash
# Everything outstanding, unattended and resumable.
#
#   bash scripts/overnight.sh
#
# Safe to re-run at any point: each step writes a marker when it completes and
# is skipped on a later run. If the machine or the session dies halfway, run
# the same command again and it picks up from the step that did not finish.
#
# Order matters. The corpus regeneration invalidates the image corpus, the
# ledger, Phase 3, Phase 4 and the demo cache, so everything downstream of it
# runs after it, in dependency order.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8

MARK="reports/_overnight"
LOG="$MARK/log.txt"
mkdir -p "$MARK"

say() {
  printf '\n########## %s  [%s] ##########\n' "$*" "$(date +%H:%M)" | tee -a "$LOG"
}
note() { printf '%s\n' "$*" | tee -a "$LOG"; }

# Run a step once. $1 = marker name, $2 = description, rest = command.
step() {
  local mark="$MARK/$1.done"; shift
  local what="$1"; shift
  if [ -f "$mark" ]; then
    note "SKIP  $what (already done)"
    return 0
  fi
  say "$what"
  if "$@" >>"$LOG" 2>&1; then
    date +%s >"$mark"
    note "OK    $what"
  else
    note "FAIL  $what -- see $LOG. Re-run this script to retry from here."
    return 1
  fi
}

say "overnight run starting"

# ---------------------------------------------------------------- 1. SROIE
# A separate process launched earlier may still be working on this; it reads
# data/processed/sroie and nothing below touches that, so the only conflict is
# CPU. Wait a while for it, then give up and move on rather than blocking the
# whole night on one run.
if [ ! -f reports/phase2/rules_sroie_e2e/run.json ]; then
  say "waiting for the SROIE end-to-end run (up to 90 min)"
  for _ in $(seq 1 90); do
    [ -f reports/phase2/rules_sroie_e2e/run.json ] && break
    sleep 60
  done
fi
if [ -f reports/phase2/rules_sroie_e2e/run.json ]; then
  note "SROIE end-to-end: present"
  date +%s >"$MARK/sroie.done"
else
  note "SROIE end-to-end: STILL MISSING after the wait. Starting it here."
  python scripts/run_extraction.py --corpus data/processed/sroie --extractor rules \
    --ocr-source engine --ocr-engine paddle --preprocess-config adaptive \
    --profile photo --doc-ids reports/phase2/sroie_test.selection.json \
    --out reports/phase2/rules_sroie_e2e >>"$LOG" 2>&1 \
    && date +%s >"$MARK/sroie.done" || note "SROIE run failed; continuing"
fi

# ------------------------------------------------- 2. archive, then regrow
if [ ! -f "$MARK/archive.done" ]; then
  say "archiving the 150-document results for an auditable before/after"
  mkdir -p reports/_corpus150
  for d in phase3_rag phase4_validation; do
    [ -d "reports/$d" ] && cp -r "reports/$d" "reports/_corpus150/$d" 2>/dev/null
  done
  date +%s >"$MARK/archive.done"
fi

step corpus   "1/7  generating 650 invoices" \
  python scripts/generate_synthetic_gst_invoices.py --count 650 \
    --out data/processed/gst_in_synthetic || exit 1

step images   "2/7  rendering 650 x 4 profiles = 2600 pages (~16 min)" \
  python scripts/build_image_corpus.py \
    --canonical data/processed/gst_in_synthetic/canonical \
    --out data/processed/gst_in_synthetic || exit 1

step ledger   "3/7  ledger" \
  python scripts/build_ledger.py || exit 1

# --------------------------------------------------------------- 3. phases
step phase3_vote "4/7  Phase 3, similarity-weighted vote (the control)" \
  python scripts/run_rag_eval.py --retriever tfidf --categoriser vote \
    --out reports/phase3_rag || exit 1

# The one thing in this project that is implemented, verified against real
# weights, and has never produced a number. A local 0.5B model reading the
# retrieved neighbours -- the generation step the plan describes and the
# weighted vote above is the control for.
step phase3_rag "5/7  Phase 3, LLM generation step (never measured before)" \
  python scripts/run_rag_eval.py --retriever tfidf --categoriser rag \
    --out reports/phase3_generated || note "generation step failed; vote numbers stand"

step phase4   "6/7  Phase 4 on the larger corpus" \
  python scripts/run_validation_eval.py --out reports/phase4_validation || exit 1

# The demo cache goes stale the moment the corpus changes, and
# test_the_cache_is_not_stale fails until it is rebuilt -- correctly.
step democache "7/7  demo cache (real OCR on a handful of pages)" \
  python scripts/build_demo_cache.py || note "demo cache failed; the staleness test will fail"

# ----------------------------------------------------------------- 4. check
say "full test suite"
python -m pytest -m "not slow" >>"$LOG" 2>&1
TESTS=$(tail -40 "$LOG" | grep -E "passed|failed" | tail -1)
note "tests: ${TESTS:-see $LOG}"

# --------------------------------------------------------------- 5. summary
say "writing the summary"
python scripts/overnight_summary.py >"$MARK/SUMMARY.md" 2>>"$LOG" \
  && note "summary: $MARK/SUMMARY.md" \
  || note "summary failed; numbers are still in reports/"

say "overnight run finished"
note "Read $MARK/SUMMARY.md first, then $LOG for detail."
