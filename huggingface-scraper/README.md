# Hugging Face Scraper

This utility crawls public Hugging Face model IDs and detects likely typo variants in:

- owner namespaces (for example `EleutherAI` vs `EluetherAI`)
- full model IDs

## Usage

```bash
python huggingface-scraper/scrape_hf_models.py
```

Quick smoke run:

```bash
python huggingface-scraper/scrape_hf_models.py --max-models 20000 --top-k 10
```

Model IDs only (skip owner typo detection):

```bash
python huggingface-scraper/scrape_hf_models.py --skip-owner-typos
```

Model IDs only, strict typo patterns in slug text (e.g. `lllama` -> `llama`, `gtp` -> `gpt`):

```bash
python huggingface-scraper/scrape_hf_models.py --skip-owner-typos --model-slug-typo-only
```

Save crawled model records to JSON (for offline re-runs):

```bash
python huggingface-scraper/scrape_hf_models.py --max-models 500000 --save-models-json analysis_out/hf_models_500k.json
```

Scrape only (no typo detection) with periodic throttling pause:

```bash
python huggingface-scraper/scrape_hf_models.py --scrape-only --max-models 900000 --save-models-json analysis_out/hf_models_900k.json
```

Run typo detection from a saved JSON file (no API crawl):

```bash
python huggingface-scraper/scrape_hf_models.py --input-models-json analysis_out/hf_models_500k.json --skip-owner-typos
```

You can also pass a previous typo report JSON to `--input-models-json`; the script will reconstruct model records from `model_id_typos`.

Output is written to `analysis_out/huggingface_typos_<timestamp>.json` unless `--output-file` is set.

## Useful flags

- `--owner-max-distance` / `--model-max-distance`: max Damerau-Levenshtein distance
- `--owner-min-similarity` / `--model-min-similarity`: minimum string similarity
- `--max-signature-positions`: number of sampled delete-signatures per token
- `--max-bucket-size`: skip very large candidate buckets to limit noisy matches
- `--owner-min-popularity-ratio` / `--model-min-popularity-ratio`: require canonical item to be sufficiently more popular than the suspected typo
- `--model-ignore-same-owner-variants`: ignore full model ID pairs from the same owner namespace
- `--skip-owner-typos`: disable owner typo detection and process only model IDs
- `--scrape-only`: crawl and save model records only, skip typo detection
- `--save-models-json`: save crawled model records for later offline analysis
- `--input-models-json`: load model records from JSON and skip network crawl
- `--model-slug-typo-only`: keep only model-slug letter typos and ignore case/separator-only differences
- `--model-slug-max-alpha-distance`: max alphabetic edit distance used by `--model-slug-typo-only`
- `--throttle-every-models`: pause crawling after every N models (default 200000)
- `--throttle-sleep-seconds`: pause duration in seconds (default 30)
- `--rate-limit-sleep-seconds`: wait duration before retrying after rate limit errors
- `--rate-limit-max-retries`: maximum consecutive rate-limit retries

## Rate-limit notes

- The crawler now retries rate-limited requests with delay instead of failing immediately.
- Default throttling pauses for 30 seconds every 200000 models to reduce API pressure.
- If retries are exhausted, partial crawl results are still preserved.
- Combine `--scrape-only` and `--save-models-json` for large dataset collection before typo analysis.
