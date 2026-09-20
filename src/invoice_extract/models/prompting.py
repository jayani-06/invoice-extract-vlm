"""Prompt construction and response parsing for VLM extraction.

Three things live here, all of which are testable without a GPU:

- **The extraction schema** — the subset of the canonical schema a model is
  actually asked to produce. `doc_id`, `source`, `pages` and `grounding` are
  the caller's to fill in; a model that invents them is hallucinating identity
  metadata, which is worse than leaving it out.

- **The prompt** — including the hybrid image+OCR variant. Whether OCR text
  helps is an empirical question on our data (refs [3] and [7] say it should),
  so it is a flag, not a hard-coded choice.

- **Response parsing** — models wrap JSON in prose, markdown fences, or both,
  and sometimes emit trailing commas. `extract_json_object` recovers the
  payload from all of that. This is the single most load-bearing piece of
  non-model code in Phase 2: a parse failure scores as a totally missed
  document, so a fragile parser would show up as a model quality problem.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: JSON Schema for the model's output. Deliberately a *reduced* mirror of
#: schema/invoice_schema.json — same field names and types, minus the
#: caller-owned identity fields. Usable directly as a grammar for constrained
#: decoding (outlines / lm-format-enforcer / vLLM guided decoding).
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["parties", "invoice_meta", "line_items", "tax_lines", "totals"],
    "properties": {
        "document_type": {"type": "string", "enum": ["invoice", "receipt"]},
        "currency": {"type": ["string", "null"], "pattern": "^[A-Z]{3}$"},
        "parties": {
            "type": "object",
            "additionalProperties": False,
            "required": ["vendor"],
            "properties": {
                role: {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "address": {"type": ["string", "null"]},
                        "gstin": {"type": ["string", "null"]},
                        "tax_id": {"type": ["string", "null"]},
                        "phone": {"type": ["string", "null"]},
                        "email": {"type": ["string", "null"]},
                    },
                }
                for role in ("vendor", "buyer", "ship_to")
            },
        },
        "invoice_meta": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "invoice_number": {"type": ["string", "null"]},
                "invoice_date": {"type": ["string", "null"]},
                "due_date": {"type": ["string", "null"]},
                "po_number": {"type": ["string", "null"]},
                "payment_terms": {"type": ["string", "null"]},
            },
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "line_no": {"type": ["integer", "null"]},
                    "description": {"type": ["string", "null"]},
                    "hsn_sac_code": {"type": ["string", "null"]},
                    "quantity": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "discount": {"type": ["number", "null"]},
                    "tax_rate": {"type": ["number", "null"]},
                    "tax_amount": {"type": ["number", "null"]},
                    "line_total": {"type": ["number", "null"]},
                },
            },
        },
        "tax_lines": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": ["string", "null"],
                        "enum": [
                            "CGST",
                            "SGST",
                            "IGST",
                            "CESS",
                            "VAT",
                            "GST",
                            "SALES_TAX",
                            "OTHER",
                            None,
                        ],
                    },
                    "rate": {"type": ["number", "null"]},
                    # Nullable here even though the canonical schema requires a
                    # number, so the prompt template can show `null` like every
                    # other field. A concrete example value (we used 123.45) is
                    # copied verbatim by small models -- Qwen2-VL-2B parroted it
                    # in 8/8 responses -- which corrupts the very field it is
                    # meant to illustrate. `coerce_document` drops tax lines
                    # that arrive without an amount, so permitting null here
                    # costs nothing downstream.
                    "amount": {"type": ["number", "null"]},
                },
            },
        },
        "totals": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                key: {"type": ["number", "null"]}
                for key in (
                    "subtotal",
                    "discount_total",
                    "tax_total",
                    "shipping",
                    "round_off",
                    "grand_total",
                )
            }
            | {"amount_in_words": {"type": ["string", "null"]}},
        },
        "payment_info": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                key: {"type": ["string", "null"]}
                for key in (
                    "bank_name",
                    "account_number",
                    "ifsc_or_swift",
                    "upi_id",
                    "mode",
                    "terms",
                )
            },
        },
    },
}

TARGET_SKELETON = """{
  "document_type": "invoice",
  "currency": "INR",
  "parties": {
    "vendor": {"name": null, "address": null, "gstin": null, "phone": null, "email": null},
    "buyer":  {"name": null, "address": null, "gstin": null}
  },
  "invoice_meta": {
    "invoice_number": null, "invoice_date": null, "due_date": null,
    "po_number": null, "payment_terms": null
  },
  "line_items": [
    {"line_no": null, "description": null, "hsn_sac_code": null, "quantity": null,
     "unit_price": null, "tax_rate": null, "tax_amount": null, "line_total": null}
  ],
  "tax_lines": [{"type": null, "rate": null, "amount": null}],
  "totals": {
    "subtotal": null, "discount_total": null, "tax_total": null,
    "shipping": null, "round_off": null, "grand_total": null
  },
  "payment_info": {"bank_name": null, "account_number": null, "ifsc_or_swift": null, "upi_id": null}
}"""

SYSTEM_PROMPT = (
    "You are an information extraction system for invoices and receipts. "
    "You return only JSON. You never explain, apologise, or add commentary."
)

_INSTRUCTIONS = """Extract the structured data from this document image.

Rules:
- Return ONE JSON object matching the template below. No markdown, no prose.
- Use null for any field that is not present on the document. Never guess.
- Copy values exactly as printed. Do not reformat dates or amounts.
- Amounts and quantities must be JSON numbers, not strings.
- line_items must contain one entry per row of the line-item table.
- tax_lines must contain one entry per tax component shown (CGST, SGST, IGST, ...).
  Use an empty list if the document shows no tax at all.
- The "totals" object is REQUIRED. Always emit it, with grand_total filled in
  from the document. Never omit it.
- The template below shows the SHAPE ONLY. Every value in it is null. Replace
  each null with what the document says, or leave it null if the document does
  not say. Never copy a value out of the template itself.
- Do not invent rows, taxes, or parties that are not on the document.

Template:
"""


def build_prompt(ocr_text: str | None = None, max_ocr_chars: int = 4000) -> str:
    """Build the user prompt, optionally with OCR text as a hint.

    The OCR block is placed *after* the instructions and explicitly framed as a
    fallible aid. Presented as authoritative, a model will copy OCR errors
    verbatim rather than reading the pixels -- which would make the hybrid mode
    strictly worse than vision-only on any page the OCR mangled.
    """
    prompt = _INSTRUCTIONS + TARGET_SKELETON

    if ocr_text:
        text = ocr_text.strip()
        if len(text) > max_ocr_chars:
            # Truncate from the middle: headers and totals sit at the two ends
            # of a page, and those carry the critical fields.
            head = max_ocr_chars // 2
            tail = max_ocr_chars - head
            text = text[:head] + "\n...\n" + text[-tail:]
        prompt += (
            "\n\nAn OCR engine read the following text from this image. It may "
            "contain recognition errors, so trust the image where they disagree:\n"
            "<ocr>\n" + text + "\n</ocr>"
        )

    return prompt


# ------------------------------------------------------------ response parsing

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _find_balanced_object(text: str) -> str | None:
    """Return the first complete `{...}` block, respecting strings and escapes.

    A plain regex cannot do this: invoice text is full of braces inside string
    values, and a non-greedy match stops at the first one.
    """
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(response: str) -> dict[str, Any] | None:
    """Recover a JSON object from a model response, or None if there is none.

    Tried in order of decreasing confidence: the whole response, then a
    markdown code fence, then the first balanced brace block, and finally the
    same with trailing commas stripped. Each fallback is a real failure mode
    observed from instruction-tuned models, not defensive padding.
    """
    if not response or not response.strip():
        return None

    candidates: list[str] = [response.strip()]
    fence = _FENCE.search(response)
    if fence:
        candidates.append(fence.group(1).strip())
    balanced = _find_balanced_object(response)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
            # Some models wrap the object in a single-element list.
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                return parsed[0]
    return None
