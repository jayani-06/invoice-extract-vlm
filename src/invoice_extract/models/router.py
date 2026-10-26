"""Take each field from whichever extractor is better at that field.

The two extractors fail in opposite directions, which is the most actionable
result this project has. End-to-end, on identical documents:

| | our layouts | FATURA2 |
|---|---|---|
| `rules` | 0.653 | 0.645 |
| `vlm:qwen2-vl-2b` | **0.843** | 0.432 |

Per field the complementarity is sharper than the totals suggest. On FATURA2
the VLM takes `parties.buyer.name` from 0.000 to 0.560 and
`invoice_meta.payment_terms` from 0.000 to 0.460 -- fields with no stable
label to anchor on -- while the regex holds `totals.grand_total` and
`invoice_meta.invoice_number`, which have reliable labels and which the VLM
misreads.

Neither extractor dominates, so picking one throws away whatever the other is
better at.

**How the choice is made, and what it is not.** There is no usable per-field
confidence to route on: the rules baseline produces none at all, and the VLM's
token log-probabilities over a four-digit field are not calibrated (the same
reason `rag/generate.py` takes its confidence from retrieved evidence rather
than from the model). So the router learns, per field, which extractor is more
often right -- **fitted on a training split and evaluated on a held-out one**,
because a router fitted on the documents it is scored against is just an
oracle wearing a hat.

That makes this a *static* per-field policy, not a per-document decision. It
cannot tell that the VLM happened to misread one particular total. What it can
do is stop using an extractor for the fields it is reliably bad at, which the
measurements say is most of the available gain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from invoice_extract.eval.metrics import compare_documents

#: A field only switches away from the default when the challenger wins by
#: more than this. Two extractors within a point of each other on a 30-document
#: split are not distinguishable, and switching on that noise would make the
#: policy depend on which documents happened to land in the fit.
MIN_MARGIN = 0.05


@dataclass
class FieldChoice:
    field_path: str
    winner: str
    scores: dict[str, float]
    margin: float
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_path,
            "winner": self.winner,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "margin": round(self.margin, 4),
            "n": self.n,
        }


@dataclass
class RoutingPolicy:
    """Which extractor supplies each field."""

    default: str
    choices: dict[str, FieldChoice] = field(default_factory=dict)

    def winner_for(self, field_path: str) -> str:
        choice = self.choices.get(field_path)
        return choice.winner if choice else self.default

    def as_dict(self) -> dict[str, Any]:
        return {
            "default": self.default,
            "fields": {k: v.as_dict() for k, v in sorted(self.choices.items())},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RoutingPolicy:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            default=raw["default"],
            choices={
                k: FieldChoice(v["field"], v["winner"], v["scores"], v["margin"], v["n"])
                for k, v in (raw.get("fields") or {}).items()
            },
        )


def per_field_scores(
    gold: dict[str, dict], predictions: dict[str, dict]
) -> dict[str, tuple[float, int]]:
    """Mean per-field score for one extractor, scored by the frozen harness.

    Uses `compare_documents` rather than a comparison written here, so the
    router is optimising exactly the quantity the project reports.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    empty = {"doc_id": "", "line_items": [], "tax_lines": []}

    for doc_id, gold_doc in gold.items():
        report = compare_documents(gold_doc, predictions.get(doc_id, empty))
        for path, result in report.field_comparisons.items():
            totals[path] += result.score
            counts[path] += 1
    return {p: (totals[p] / counts[p], counts[p]) for p in totals if counts[p]}


def fit(
    gold: dict[str, dict],
    by_extractor: dict[str, dict[str, dict]],
    default: str,
    min_margin: float = MIN_MARGIN,
) -> RoutingPolicy:
    """Learn which extractor to trust for each field.

    `gold` and `by_extractor` must cover the **fitting** documents only. The
    caller is responsible for keeping the evaluation split out; this function
    cannot tell and will happily fit an oracle if handed the test set.
    """
    scored = {name: per_field_scores(gold, preds) for name, preds in by_extractor.items()}
    all_fields = {p for s in scored.values() for p in s}

    policy = RoutingPolicy(default=default)
    for path in sorted(all_fields):
        scores = {name: s.get(path, (0.0, 0))[0] for name, s in scored.items()}
        n = max(s.get(path, (0.0, 0))[1] for s in scored.values())
        best = max(scores, key=lambda k: scores[k])
        margin = scores[best] - scores[default]
        # Only record a field where the challenger clears the margin. Anything
        # else keeps the default, so the policy stays small and auditable.
        if best != default and margin >= min_margin:
            policy.choices[path] = FieldChoice(path, best, scores, margin, n)
    return policy


def _assign(target: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _read(source: dict, path: str) -> Any:
    node: Any = source
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def route(policy: RoutingPolicy, by_extractor: dict[str, dict], doc_id: str) -> dict[str, Any]:
    """Build one document by taking each field from its chosen extractor.

    Starts from the default extractor's whole document, so everything the
    policy has no opinion about -- line items, tax lines, identity metadata --
    comes from one coherent source rather than being stitched together.
    """
    base = by_extractor[policy.default].get(doc_id)
    if base is None:
        return {}
    merged = json.loads(json.dumps(base))

    for path, choice in policy.choices.items():
        donor = by_extractor.get(choice.winner, {}).get(doc_id)
        if donor is None:
            continue
        _assign(merged, path, _read(donor, path))
    return merged
