"""Field-level and line-item-level metrics for comparing a predicted
canonical-schema document against its gold annotation.

Designed to be run before any model exists (per the project plan): the
harness only needs *some* gold + prediction JSONL in canonical schema shape,
which is exactly what `tests/fixtures/{gold,pred}.jsonl` provide for a smoke
test, and what a VLM baseline's output should be post-processed into.

Metric choices:
- Numeric fields (amounts, quantities, rates): relative-tolerance match
  (default 1%), not exact match — OCR/VLM rounding shouldn't count as wrong.
- Date fields: exact match on the normalized ISO string (dates don't have
  a meaningful "close enough").
- Everything else text (names, addresses, ids, descriptions): ANLS
  (Average Normalized Levenshtein Similarity), the standard metric from
  DocVQA/SROIE-style extraction tasks — a similarity score in [0,1], with
  score < 0.5 conventionally counted as "incorrect" for accuracy purposes,
  while the *score itself* is also reported (so near-misses are visible,
  not just a binary right/wrong).
- Line items: matched gold<->pred via optimal assignment (Hungarian
  algorithm) on a cost combining description similarity and total-amount
  closeness, then scored field-by-field only on matched pairs; unmatched
  items count against precision/recall directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

ANLS_THRESHOLD = 0.5
DEFAULT_NUMERIC_TOLERANCE = 0.01

NUMERIC_LEAF_NAMES = {
    "unit_price",
    "quantity",
    "discount",
    "tax_rate",
    "tax_amount",
    "line_total",
    "subtotal",
    "discount_total",
    "tax_total",
    "shipping",
    "round_off",
    "grand_total",
    "amount",
    "rate",
}
DATE_LEAF_NAMES = {"invoice_date", "due_date"}

# Fields used for the document-level "critical fields all correct" summary metric.
CRITICAL_FIELDS = [
    "invoice_meta.invoice_number",
    "invoice_meta.invoice_date",
    "totals.grand_total",
    "parties.vendor.name",
]

# Top-level keys walked into scalar-field flattening; deliberately excludes
# pages/source/annotator/grounding/schema_version/doc_id (identity/provenance,
# not extraction targets) and line_items/tax_lines (handled separately).
SCALAR_ROOTS = ["parties", "invoice_meta", "totals", "payment_info"]


def flatten_scalars(doc: dict[str, Any]) -> dict[str, Any]:
    """Flatten the scalar (non-list) portions of a canonical document into
    {dotted.path: value}. Missing branches (e.g. no payment_info) are simply
    absent from the result rather than erroring.
    """
    out: dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            out[prefix] = node

    for root in SCALAR_ROOTS:
        if root in doc and doc[root] is not None:
            walk(root, doc[root])
    return out


def anls(a: str, b: str) -> float:
    a, b = a.strip().lower(), b.strip().lower()
    if not a and not b:
        return 1.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def is_numeric_field(field_path: str) -> bool:
    return field_path.rsplit(".", 1)[-1] in NUMERIC_LEAF_NAMES


def is_date_field(field_path: str) -> bool:
    return field_path.rsplit(".", 1)[-1] in DATE_LEAF_NAMES


@dataclass
class FieldComparison:
    correct: bool
    score: float  # in [0, 1]; for numeric/date this is 1.0 or 0.0, for text it's the ANLS score


def compare_field(
    field_path: str, gold: Any, pred: Any, numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE
) -> FieldComparison:
    if gold is None and pred is None:
        return FieldComparison(correct=True, score=1.0)
    if gold is None or pred is None:
        return FieldComparison(correct=False, score=0.0)  # hallucination or missed value

    if is_numeric_field(field_path):
        try:
            g, p = float(gold), float(pred)
        except (TypeError, ValueError):
            return FieldComparison(correct=False, score=0.0)
        ok = abs(g - p) <= numeric_tolerance * max(abs(g), 1.0)
        return FieldComparison(correct=ok, score=1.0 if ok else 0.0)

    if is_date_field(field_path):
        ok = str(gold).strip() == str(pred).strip()
        return FieldComparison(correct=ok, score=1.0 if ok else 0.0)

    score = anls(str(gold), str(pred))
    return FieldComparison(correct=score >= ANLS_THRESHOLD, score=score)


@dataclass
class FieldReport:
    n: int = 0
    n_correct: int = 0
    score_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.n if self.n else float("nan")


@dataclass
class LineItemMatchResult:
    n_gold: int
    n_pred: int
    n_matched: int
    field_reports: dict[str, FieldReport] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else float("nan")

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gold if self.n_gold else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:  # NaN-safe
            return float("nan")
        return 2 * p * r / (p + r)


LINE_ITEM_FIELDS = [
    "description",
    "hsn_sac_code",
    "quantity",
    "unit_price",
    "discount",
    "tax_rate",
    "tax_amount",
    "line_total",
]
LINE_ITEM_MATCH_COST_THRESHOLD = 0.6  # pairs costlier than this are treated as unmatched


def _line_item_cost(a: dict, b: dict) -> float:
    desc_sim = anls(str(a.get("description") or ""), str(b.get("description") or ""))
    a_total, b_total = a.get("line_total"), b.get("line_total")
    if a_total is not None and b_total is not None:
        amount_sim = 1.0 if abs(a_total - b_total) <= 0.01 * max(abs(a_total), 1.0) else 0.0
    else:
        amount_sim = 0.5  # unknown, don't penalize fully
    similarity = 0.7 * desc_sim + 0.3 * amount_sim
    return 1.0 - similarity


def match_line_items(gold_items: list[dict], pred_items: list[dict]) -> LineItemMatchResult:
    n_gold, n_pred = len(gold_items), len(pred_items)
    field_reports = {f: FieldReport() for f in LINE_ITEM_FIELDS}

    if n_gold == 0 or n_pred == 0:
        return LineItemMatchResult(
            n_gold=n_gold, n_pred=n_pred, n_matched=0, field_reports=field_reports
        )

    cost = [[_line_item_cost(g, p) for p in pred_items] for g in gold_items]
    gold_idx, pred_idx = linear_sum_assignment(cost)

    n_matched = 0
    for gi, pi in zip(gold_idx, pred_idx, strict=True):
        if cost[gi][pi] > LINE_ITEM_MATCH_COST_THRESHOLD:
            continue
        n_matched += 1
        g, p = gold_items[gi], pred_items[pi]
        for f_name in LINE_ITEM_FIELDS:
            path = (
                f"line_items.{f_name}"  # not a real field_path, just routes numeric/date detection
            )
            cmp = compare_field(path, g.get(f_name), p.get(f_name))
            fr = field_reports[f_name]
            fr.n += 1
            fr.n_correct += int(cmp.correct)
            fr.score_sum += cmp.score

    return LineItemMatchResult(
        n_gold=n_gold, n_pred=n_pred, n_matched=n_matched, field_reports=field_reports
    )


def match_tax_lines(
    gold_lines: list[dict], pred_lines: list[dict], tolerance: float = DEFAULT_NUMERIC_TOLERANCE
) -> FieldReport:
    """Tax lines have a small closed vocabulary of `type` (CGST/SGST/IGST/...),
    so match by type directly rather than an assignment problem."""
    pred_by_type = {t["type"]: t for t in pred_lines}
    report = FieldReport()
    for g in gold_lines:
        p = pred_by_type.get(g["type"])
        cmp = compare_field(
            "tax_lines.amount", g.get("amount"), p.get("amount") if p else None, tolerance
        )
        report.n += 1
        report.n_correct += int(cmp.correct)
        report.score_sum += cmp.score
    return report


@dataclass
class DocumentComparisonReport:
    doc_id: str
    field_comparisons: dict[str, FieldComparison]
    line_items: LineItemMatchResult
    tax_lines: FieldReport
    critical_fields_correct: bool


def compare_documents(gold: dict[str, Any], pred: dict[str, Any]) -> DocumentComparisonReport:
    gold_flat = flatten_scalars(gold)
    pred_flat = flatten_scalars(pred)
    all_paths = sorted(set(gold_flat) | set(pred_flat))

    comparisons = {
        path: compare_field(path, gold_flat.get(path), pred_flat.get(path)) for path in all_paths
    }
    critical_ok = all(
        comparisons.get(p, FieldComparison(False, 0.0)).correct for p in CRITICAL_FIELDS
    )

    return DocumentComparisonReport(
        doc_id=gold.get("doc_id", pred.get("doc_id", "<unknown>")),
        field_comparisons=comparisons,
        line_items=match_line_items(gold.get("line_items", []), pred.get("line_items", [])),
        tax_lines=match_tax_lines(gold.get("tax_lines", []), pred.get("tax_lines", [])),
        critical_fields_correct=critical_ok,
    )


@dataclass
class AggregateReport:
    n_docs: int
    per_field: dict[str, FieldReport]
    line_items_micro: LineItemMatchResult
    tax_lines: FieldReport
    doc_critical_accuracy: float
    macro_field_accuracy: float
    micro_field_accuracy: float


def aggregate(doc_reports: list[DocumentComparisonReport]) -> AggregateReport:
    per_field: dict[str, FieldReport] = {}
    for dr in doc_reports:
        for path, cmp in dr.field_comparisons.items():
            fr = per_field.setdefault(path, FieldReport())
            fr.n += 1
            fr.n_correct += int(cmp.correct)
            fr.score_sum += cmp.score

    total_gold = sum(dr.line_items.n_gold for dr in doc_reports)
    total_pred = sum(dr.line_items.n_pred for dr in doc_reports)
    total_matched = sum(dr.line_items.n_matched for dr in doc_reports)
    line_items_micro = LineItemMatchResult(
        n_gold=total_gold, n_pred=total_pred, n_matched=total_matched
    )

    tax_report = FieldReport()
    for dr in doc_reports:
        tax_report.n += dr.tax_lines.n
        tax_report.n_correct += dr.tax_lines.n_correct
        tax_report.score_sum += dr.tax_lines.score_sum

    accuracies = [fr.accuracy for fr in per_field.values() if fr.n]
    macro_acc = sum(accuracies) / len(accuracies) if accuracies else float("nan")
    total_n = sum(fr.n for fr in per_field.values())
    total_correct = sum(fr.n_correct for fr in per_field.values())
    micro_acc = total_correct / total_n if total_n else float("nan")

    n_critical_ok = sum(dr.critical_fields_correct for dr in doc_reports)

    return AggregateReport(
        n_docs=len(doc_reports),
        per_field=per_field,
        line_items_micro=line_items_micro,
        tax_lines=tax_report,
        doc_critical_accuracy=n_critical_ok / len(doc_reports) if doc_reports else float("nan"),
        macro_field_accuracy=macro_acc,
        micro_field_accuracy=micro_acc,
    )
