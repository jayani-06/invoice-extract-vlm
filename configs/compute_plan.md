# Compute plan — VLM baseline inference

Target environments: **Colab Pro** and **Kaggle** (both give T4 GPUs, 16GB
VRAM). This doc exists so the quantization/model-size decision is made once,
up front, rather than re-litigated every notebook run.

## Environment facts (verify against current quotas before a big run — these
change without notice)

| Env | GPU | VRAM | Session limit | Notes |
|---|---|---|---|---|
| Colab Pro | T4 (usually), sometimes A100/L4 depending on availability | 16GB (T4) | ~24h, disconnects on idle | Priority access, not guaranteed hardware |
| Colab Pro+ | as above, background execution | 16GB (T4) | Longer background runtime | Worth it only if runs exceed a few hours |
| Kaggle | 2x T4 (or 1x P100) | 16GB each | 12h/session, ~30h GPU-quota/week | Free; 2 GPUs enables simple data-parallel sharding |

**Design for T4, 16GB, single-GPU-per-worker.** Treat any A100/L4 you get on
Colab as a bonus, not the baseline you plan around — the eval harness and
inference script should run within a T4's budget so the plan survives
whatever hardware you're actually handed that day.

## Quantization decision: made now, not deferred

**4-bit NF4 (via `bitsandbytes`) for any model ≥3B params; fp16 for anything
smaller.** Rationale:

- A 7B-class VLM in fp16 is ~14GB of weights alone — that's the *entire* T4
  budget before you've loaded an image, run the vision encoder, or allocated
  a KV cache. Not viable.
- The same model in 4-bit NF4 is ~3.5-4GB of weights, leaving ~10-12GB for
  vision-encoder activations (image tokens can be substantial for
  high-resolution invoice scans), KV cache, and framework overhead.
- Below ~3B params, fp16 already fits comfortably (~2-6GB), and quantization
  error starts to matter proportionally more on smaller models doing
  fine-grained field extraction — so skip it there and keep fp16 for quality.
- `bitsandbytes` NF4 is the default choice over GPTQ/AWQ because it needs no
  separate calibration/quantization pass — you load `from_pretrained(...,
  quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...))` directly
  against the base checkpoint, which matters when you're iterating on model
  choice rather than settled on one.
- See `src/invoice_extract/utils/quantization.py` for the loading helper.

## Candidate models (baseline shortlist — pick 1-2 to actually run first)

| Model | Params | Type | T4 fit | Notes |
|---|---|---|---|---|
| Donut (naver-clova-ix/donut-base-finetuned-cord-v2) | ~200M | OCR-free VLM (image->text, no separate OCR step) | fp16, trivial | Purpose-built for exactly this task shape (CORD); good sanity-check baseline before trying general VLMs |
| LayoutLMv3-large | ~355M | OCR-based (needs a separate OCR engine, e.g. Tesseract/PaddleOCR, feeding boxes+text in) | fp16, trivial | Strong on FATURA/SROIE-style layouts; adds an OCR dependency+failure mode |
| Qwen2-VL-2B-Instruct | 2B | General VLM, strong doc/OCR understanding | fp16 fits easily | Good balance of quality vs. simplicity; recommended first general-VLM baseline |
| Qwen2-VL-7B-Instruct | 7B | General VLM | 4-bit NF4 required | Try after 2B if quality isn't enough; ~2-3x slower per doc |
| InternVL2-2B / -8B | 2B / 8B | General VLM | 2B: fp16; 8B: 4-bit | Alternative to Qwen2-VL, competitive on doc benchmarks |
| PaliGemma-3B | 3B | General VLM | fp16 (borderline) or 4-bit | Google's doc-VQA-oriented model; needs a task-specific prompt format |

**Recommended order:** Donut (fastest signal, near-zero setup) -> Qwen2-VL-2B
zero/few-shot with a structured-JSON prompt matching `schema/invoice_schema.json`
-> only reach for a 7B+ model with 4-bit if the 2B ceiling is clearly the
bottleneck and not the prompt/post-processing.

## Memory budget (Qwen2-VL-7B-Instruct, 4-bit NF4, T4 16GB)

Rough, order-of-magnitude — validate with `nvidia-smi` during your first real
run and adjust:

- Quantized weights: ~4.0 GB
- Vision encoder activations (1 image, moderate resolution ~1024px): ~1.5-3 GB
- KV cache (short generation, <512 output tokens, batch size 1): ~1-2 GB
- Framework/CUDA context overhead: ~1-1.5 GB
- **Total: ~8-10 GB**, leaving headroom for batch size 2 or slightly higher
  image resolution before hitting the 16GB ceiling. Batch size 1 is the safe
  default; only increase after confirming headroom empirically.

## Batching & throughput strategy

- Batch size 1 for any 4-bit 7B+ model on a single T4 — VLM generation with
  image inputs batches poorly anyway (padding to the longest image/sequence
  wastes memory) and correctness matters far more than throughput at
  baseline stage.
- For fp16 sub-3B models (Donut, Qwen2-VL-2B), batch 4-8 is reasonable; tune
  empirically starting from 4.
- Kaggle's 2 GPUs: shard the dataset in half (odd/even doc_id or a fixed
  split file) and run two independent inference processes, one per GPU —
  don't bother with true multi-GPU model parallelism at baseline stage, it's
  not needed for models this size.
- Expect roughly 3-8 seconds/document for a 2B model, 8-20s for a 4-bit 7B,
  on a T4 — measure on your first 20 documents and extrapolate before
  committing to a full 10k-document FATURA run; if it's too slow, that's a
  reason to subsample FATURA for the baseline pass rather than force the
  full set through Colab's session limits.

## Checkpointing & session-limit survival

Colab/Kaggle sessions disconnect (idle timeout, 12/24h cap, or just bad
luck) — inference must be resumable:

1. Write predictions incrementally, one JSON line per document, flushing
   after every doc (`predictions.jsonl`, matching the harness's expected
   format — see `src/invoice_extract/eval/harness.py`).
2. Before processing a doc_id, check if it's already in the output file and
   skip it — makes re-running the same cell after a disconnect safe and
   idempotent.
3. Periodically copy `predictions.jsonl` to Google Drive (Colab) or save as
   a Kaggle Dataset version (Kaggle) — don't rely on the ephemeral
   session disk surviving a disconnect.
4. Model weights: let `transformers`/`huggingface_hub` cache to
   `/content/drive/...` (Colab, if Drive is mounted) or Kaggle's
   `/kaggle/working` + dataset-attached weights, so a reconnect doesn't
   re-download multi-GB checkpoints.

## What's NOT decided yet (deliberately)

- Exact prompt template for structured-JSON extraction (depends on which
  model you settle on — Qwen2-VL and InternVL have different preferred
  output-formatting conventions).
- Whether to add a lightweight OCR pre-pass (PaddleOCR/Tesseract) even for
  "OCR-free" VLMs, to ground field extraction with boxes for the
  `grounding[]` part of the schema — worth revisiting once accuracy numbers
  from the harness show whether ungrounded extraction is good enough.
- Fine-tuning vs. zero/few-shot only — this plan is inference-only baseline;
  a fine-tuning compute plan (likely needing more than a T4, or LoRA on a
  T4) is a follow-up once the baseline harness shows where zero-shot falls
  short.

See `notebooks/colab_baseline_inference.ipynb` for a runnable skeleton that
implements points 1-4 above end-to-end against `tests/fixtures/`.
