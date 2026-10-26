"""The **G** in RAG: an LLM assigning a category from retrieved context.

Phase 3 shipped retrieval and a similarity-weighted vote. That is retrieval and
aggregation; it is not generation, and the plan specifically describes an LLM
reading retrieved examples and deciding. This module is that step, and it is
the difference between "we built a nearest-neighbour classifier" and "we built
RAG".

The vote is not thrown away -- it is the control. A generator that cannot beat
a weighted vote over the same retrieved context has added a model and bought
nothing, and that comparison is the only honest way to report this.

Three things the design has to get right, because each is a way to produce a
plausible wrong number:

**Constrained output.** The model may only answer with a code from the chart of
accounts, or `ABSTAIN`. A free-text answer that happens to look like a category
name is not a category, so anything unparseable or out-of-vocabulary becomes an
abstention, never a guess. `_parse_choice` is where that is enforced.

**Abstention has to survive.** The vote path abstains when neighbours are weak
or disagree, and that behaviour is the reason the system is safe to put in
front of a ledger. A generator that always answers would show higher coverage
and be strictly worse. The prompt offers ABSTAIN explicitly and the parser
treats an unusable answer as one.

**No context, no answer.** When retrieval returns nothing, there is nothing to
generate *from*, and asking the model anyway turns RAG into zero-shot
classification by the back door. The retrieval gate runs first and short-
circuits, exactly as the vote path does.

The `Generator` Protocol keeps the LLM swappable: a local transformers model is
what runs here, and an HTTP client would satisfy the same interface. Tests use
`ScriptedGenerator`, but the reported numbers come from a real local model --
`models/vlm.py` was unit-tested through a fake for weeks and the fake told us
nothing about whether generation worked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from invoice_extract.rag.categorize import (
    MIN_NEIGHBOUR_SIMILARITY,
    CategoryDecision,
    _query_for,
)
from invoice_extract.rag.store import Match, Retriever, VendorStats
from invoice_extract.rag.taxonomy import (
    CHART_OF_ACCOUNTS,
    UNCATEGORISED,
    account_name,
)

#: Small enough to run on CPU, instruction-tuned enough to follow a
#: constrained-output instruction. 0.5B is deliberately modest: if a model this
#: size beats the vote the finding is strong, and if it loses that is also
#: informative about whether the generation step earns its cost at all.
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

ABSTAIN = "ABSTAIN"

SYSTEM_PROMPT = (
    "You are an accounts-payable clerk assigning expense categories. "
    "You answer with exactly one account code and nothing else."
)


class GeneratorUnavailableError(RuntimeError):
    """The generator could not be loaded. Never silently downgraded: a run that
    fell back to the vote must not be reported as a generated result."""


class Generator(Protocol):
    """Anything that can answer a constrained categorisation question."""

    def complete(self, system: str, user: str, max_new_tokens: int = 16) -> str: ...


# --------------------------------------------------------------- the prompt


def build_prompt(
    description: str,
    matches: list[Match],
    *,
    vendor: str | None = None,
    hsn: str | None = None,
    usual: dict[str, int] | None = None,
) -> str:
    """Assemble the retrieved context into a decision question.

    Neighbours are listed with their similarity and their human-assigned
    category, closest first, because that ordering is itself evidence and a
    small model reads position as importance.
    """
    allowed = [c for c in CHART_OF_ACCOUNTS if c != UNCATEGORISED]
    lines = [
        "Assign an expense account code to this invoice line item.",
        "",
        f"Line item: {description}",
    ]
    if vendor:
        lines.append(f"Vendor: {vendor}")
    if hsn:
        lines.append(f"HSN/SAC code: {hsn}")

    lines += ["", "Similar line items previously categorised by a human:"]
    for i, m in enumerate(matches, 1):
        lines.append(
            f"{i}. {m.entry.description!r} "
            f"(vendor: {m.entry.vendor or 'unknown'}) "
            f"-> {m.entry.category} {account_name(m.entry.category)} "
            f"[similarity {m.score:.2f}]"
        )

    if usual:
        seen = ", ".join(f"{code} {account_name(code)} ({n}x)" for code, n in sorted(usual.items()))
        lines += ["", f"This vendor has previously been booked to: {seen}"]

    lines += [
        "",
        "Allowed account codes:",
    ]
    for code in allowed:
        lines.append(f"  {code} = {account_name(code)}")

    lines += [
        "",
        "Rules:",
        "- Reply with ONE account code from the list above, or the single word " f"{ABSTAIN}.",
        "- No explanation, no punctuation, no other words.",
        f"- Answer {ABSTAIN} if the examples above do not support a confident "
        "choice, or if they disagree. An abstention is reviewed by a human and "
        "is always better than a wrong account code.",
        "",
        "Account code:",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- the parsing

_CODE = re.compile(r"\b(\d{4})\b")


def _parse_choice(response: str) -> str | None:
    """Return a valid account code, or None for anything else.

    None means abstain. Deliberately strict: a model that replies "probably
    Office Supplies (5100)" has not followed a constrained-output instruction,
    and the safe reading of a disobedient answer is no answer. The one
    concession is a code embedded in prose, because that is a formatting
    failure rather than an uncertain decision -- and only when exactly one
    valid code appears, since two codes is genuine ambiguity.
    """
    if not response or not response.strip():
        return None
    text = response.strip()

    if ABSTAIN.lower() in text.lower():
        return None

    # Ambiguity is checked BEFORE the clean-answer path, not after. "5010 or
    # 5020" starts with a valid code, so reading the first token first would
    # return 5010 and silently discard the model telling us it could not
    # choose. Two codes is an abstention.
    found = {c for c in _CODE.findall(text) if c in CHART_OF_ACCOUNTS and c != UNCATEGORISED}
    if len(found) != 1:
        return None

    bare = text.split()[0].strip(".,:;'\"")
    if bare in CHART_OF_ACCOUNTS and bare != UNCATEGORISED:
        return bare
    # A single valid code embedded in prose is a formatting failure rather than
    # an uncertain decision, so it is honoured.
    return next(iter(found))


# ------------------------------------------------------------- the backends


@dataclass
class LocalGenerator:
    """A local instruction-tuned transformers model, greedy, on CPU.

    Greedy rather than sampled so the run is reproducible: a categorisation
    number that moves between runs cannot be compared against the vote.
    """

    model_name: str = DEFAULT_GENERATOR_MODEL
    _model: Any = None
    _tokenizer: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise GeneratorUnavailableError(
                f"transformers and torch are required: {exc}\n  pip install -e '.[rag]'"
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception as exc:  # pragma: no cover - network / cache dependent
            raise GeneratorUnavailableError(f"could not load {self.model_name!r}: {exc}") from exc
        self._model.eval()

    def complete(self, system: str, user: str, max_new_tokens: int = 16) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )


@dataclass
class ScriptedGenerator:
    """A deterministic stand-in for tests.

    Exists to exercise prompt construction, parsing and the abstention paths
    without a model. It is **never** used for a reported number: `vlm.py` was
    tested through a fake for weeks and the fake could not tell us the
    generation call was wrong.
    """

    replies: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    default: str = ABSTAIN

    def complete(self, system: str, user: str, max_new_tokens: int = 16) -> str:
        self.prompts.append(user)
        if self.replies:
            return self.replies.pop(0)
        return self.default


def get_generator(kind: str = "local", **kwargs) -> Generator:
    if kind == "local":
        return LocalGenerator(**kwargs)
    if kind == "scripted":
        return ScriptedGenerator(**kwargs)
    raise KeyError(f"unknown generator {kind!r}; have 'local' and 'scripted'")


# -------------------------------------------------------- the full RAG path


@dataclass
class GeneratedDecision(CategoryDecision):
    """A `CategoryDecision` that records what the model was asked and said.

    Subclassed rather than replaced so every existing consumer -- the demo, the
    eval harness, the end-to-end test -- keeps working unchanged.
    """

    raw_response: str = ""
    generated: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = super().as_dict()
        d["raw_response"] = self.raw_response
        d["generated"] = self.generated
        return d


def categorize_line_item_rag(
    item: dict[str, Any],
    retriever: Retriever,
    generator: Generator,
    *,
    vendor: str | None = None,
    k: int = 5,
    vendor_stats: VendorStats | None = None,
) -> GeneratedDecision:
    """Retrieve, augment, generate. The complete path.

    Returns a decision whose `category` is `UNCATEGORISED` whenever the model
    abstained, answered unusably, or there was no context to generate from --
    so the abstention guarantee the vote path provides is preserved exactly.
    """
    description = (item.get("description") or "").strip()
    if not description:
        return GeneratedDecision(
            UNCATEGORISED, 0.0, reason="line item has no description", generated=False
        )

    matches = [
        m
        for m in retriever.search(_query_for(item, vendor), k=k)
        if m.score >= MIN_NEIGHBOUR_SIMILARITY
    ]
    if not matches:
        # No retrieved context means there is nothing to generate *from*.
        # Prompting anyway would silently turn this into zero-shot
        # classification and the result would no longer be a RAG measurement.
        return GeneratedDecision(
            UNCATEGORISED,
            0.0,
            reason="no similar item in the ledger (not sent to the model)",
            neighbours=[],
            generated=False,
        )

    usual = None
    if vendor_stats and vendor:
        usual = dict(vendor_stats.usual_categories(vendor)) or None

    prompt = build_prompt(
        description,
        matches,
        vendor=vendor,
        hsn=item.get("hsn_sac_code"),
        usual=usual,
    )
    raw = generator.complete(SYSTEM_PROMPT, prompt)
    choice = _parse_choice(raw)

    if choice is None:
        return GeneratedDecision(
            UNCATEGORISED,
            0.0,
            neighbours=matches,
            reason=f"model abstained or answered unusably ({raw.strip()[:40]!r})",
            raw_response=raw,
            generated=True,
        )

    # Confidence from the retrieved evidence, not from the model: a 0.5B model's
    # self-reported certainty is worth nothing, and token log-probabilities over
    # a 4-digit code are not calibrated either. The honest signal available here
    # is how strongly the retrieved context supports the answer the model gave.
    supporting = sum(m.score for m in matches if m.entry.category == choice)
    total = sum(m.score for m in matches) or 1.0
    share = supporting / total
    confidence = share * min(1.0, matches[0].score / 0.8)

    return GeneratedDecision(
        category=choice,
        confidence=confidence,
        votes={choice: supporting},
        neighbours=matches,
        reason=(
            f"model chose {choice}; {share:.0%} of retrieved context agrees, "
            f"closest match {matches[0].score:.2f}"
        ),
        raw_response=raw,
        generated=True,
    )


def categorize_document_rag(
    doc: dict[str, Any],
    retriever: Retriever,
    generator: Generator,
    *,
    k: int = 5,
    vendor_stats: VendorStats | None = None,
) -> list[GeneratedDecision]:
    vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
    return [
        categorize_line_item_rag(
            item, retriever, generator, vendor=vendor, k=k, vendor_stats=vendor_stats
        )
        for item in doc.get("line_items") or []
    ]
