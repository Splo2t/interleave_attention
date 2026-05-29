# Interleave Attention Evaluation

Evaluation code for the CIKM 2026 interleave-attention experiments. The repository keeps only the benchmark runners, analysis helpers, dashboards, and lightweight reproducibility scripts needed for the paper. Large checkpoints, Hugging Face caches, generated benchmark JSONL files, and sweep outputs are intentionally excluded from git.

## Repository Layout

- `krong_eval/`: main benchmark package for standard multiple-choice and reranking tasks.
- `krong_eval_variants/`: KoBEST surface-form stress benchmark package.
- `eval_mmlu_kmmlu_hf_krong.py`: single-model evaluator for main benchmarks.
- `eval_variant_hf_krong.py`: single-model evaluator for KoBEST variant data.
- `run_eval_checkpoint_sweep.py`: checkpoint sweep runner.
- `download_kobest_variant_data.py`: downloads materialized KoBEST variant JSONL data from Hugging Face Hub.
- `prepare_kobest_query_context_stress.py`: materializes the HF query-context stress dataset into the local `kobest_variant` JSONL layout.
- `scripts/`: conversion, sweep, and export wrappers used in the experiments.
- `docs/`: benchmark notes and publication checklist.

The three paper-facing folders are curated views over the same source tree:

- `Korean Surface-Form Stress Tests/`
- `Controlled Relevance Scoring/`
- `Main benchmark results/`

Most files inside these folders are symlinks to the canonical files at the repository root.

## Benchmarks Kept In This Release

Main benchmark results:

```text
mmlu, kmmlu, kobest, click, csatqa, arc_easy, arc_challenge, hellaswag, openbookqa
```

Controlled relevance scoring:

```text
korean_rerank
```

Korean surface-form stress tests:

```text
kobest_variant
```

Non-paper benchmark implementations and local exploratory benchmark data are not included.

## Setup

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA machines, install the PyTorch build that matches your driver/CUDA stack if the default pip build is not appropriate.

Optional public reranker baselines use:

```bash
pip install -r requirements-rerank.txt
```

If you use gated Hugging Face models or private datasets, authenticate first:

```bash
huggingface-cli login
```

## Data Policy

Generated data and large artifacts are not stored in git:

- `variant_benchmarks/`
- `rerank_candidates/`
- `sweep_results*/`
- `variant_eval_outputs*/`
- `logs/`, `outputs/`
- model checkpoint files such as `.safetensors`, `.bin`, `.pt`, `.ckpt`, `.gguf`

Keep these files local, or publish benchmark data through Hugging Face Hub as dataset repositories.

## Prepare KoBEST Surface-Form Data From Hugging Face

`kobest_variant` expects this local layout:

```text
variant_benchmarks/<variant_name>/
  manifest.json
  kobest/
    boolq/{train,validation,test}.jsonl
    copa/{train,validation,test}.jsonl
    hellaswag/{train,validation,test}.jsonl
    sentineg/{train,validation,test}.jsonl
    wic/{train,validation,test}.jsonl
```

### Query-context stress data

The query-context stress dataset can be materialized from Hugging Face rows with:

```bash
python prepare_kobest_query_context_stress.py \
  --dataset-id dilab-cau/kobest-query-context-stress-v3 \
  --output-root variant_benchmarks/kobest_query_context_stress_v3 \
  --original-output-root variant_benchmarks/kobest_query_context_stress_v3_original \
  --cache-dir ./hf_cache/huggingface/datasets \
  --overwrite
```

### P25, P50, and josa-hard data

For the paper stress sets, publish each materialized local folder to Hugging Face with the layout above, then download it with `download_kobest_variant_data.py`.

Suggested dataset repo names:

```text
Splo2t/ko-random-p25
Splo2t/ko-random-p50
Splo2t/ko-josa-preserve-compaction-hard
```

Download examples:

```bash
python download_kobest_variant_data.py \
  --repo-id Splo2t/ko-random-p25 \
  --variant-name ko_random_p25 \
  --output-root variant_benchmarks \
  --overwrite

python download_kobest_variant_data.py \
  --repo-id Splo2t/ko-random-p50 \
  --variant-name ko_random_p50 \
  --output-root variant_benchmarks \
  --overwrite

python download_kobest_variant_data.py \
  --repo-id Splo2t/ko-josa-preserve-compaction-hard \
  --variant-name ko_josa_preserve_compaction_hard \
  --output-root variant_benchmarks \
  --overwrite
```

Maintainer upload example after creating a dataset repo on Hugging Face:

```bash
huggingface-cli upload \
  Splo2t/ko-random-p25 \
  /path/to/variant_benchmarks/ko_random_p25 \
  . \
  --repo-type dataset
```

Repeat for `ko_random_p50` and `ko_josa_preserve_compaction_hard`.

## Run Main Benchmarks

Single benchmark smoke test:

```bash
python eval_mmlu_kmmlu_hf_krong.py \
  --ckpt_path meta-llama/Llama-3.1-8B \
  --task mmlu \
  --subjects abstract_algebra \
  --cache_root ./hf_cache \
  --limit 10 \
  --out_json outputs/smoke_mmlu.json
```

Full paper task list for a local or HF checkpoint:

```bash
python run_eval_checkpoint_sweep.py \
  --single-ckpt-path /path/to/checkpoint-or-hf-model \
  --single-ckpt-name my_model \
  --single-ckpt-step 0 \
  --tasks mmlu,kmmlu,kobest,click,csatqa,arc_easy,arc_challenge,hellaswag,openbookqa \
  --result-root sweep_results/my_model_main \
  --cache_root ./hf_cache
```

Checkpoint directory sweep:

```bash
python run_eval_checkpoint_sweep.py \
  --checkpoints-root /path/to/converted_checkpoints \
  --checkpoint-pattern 'checkpoint-*' \
  --tasks mmlu,kmmlu,kobest,click,csatqa,arc_easy,arc_challenge,hellaswag,openbookqa \
  --result-root sweep_results/my_checkpoint_sweep \
  --cache_root ./hf_cache
```

## Run KoBEST Surface-Form Stress Tests

Example for P25:

```bash
python eval_variant_hf_krong.py \
  --ckpt_path /path/to/checkpoint-or-hf-model \
  --task kobest_variant \
  --variant_data_root variant_benchmarks/ko_random_p25 \
  --variant_name ko_random_p25 \
  --kobest_tasks boolq,copa,hellaswag,sentineg,wic \
  --kobest_split test \
  --k_shot 5 \
  --continuation_scoring oneshot \
  --cache_root ./hf_cache \
  --out_json outputs/ko_random_p25_predictions.json \
  --save_item_predictions
```

Change `--variant_data_root` and `--variant_name` to run P50 or josa-hard.

## Run Controlled Relevance Scoring

`korean_rerank` consumes a JSONL file with one row per query and candidate passages. See `docs/korean_rerank_benchmark.md` for the expected schema.

```bash
python eval_mmlu_kmmlu_hf_krong.py \
  --ckpt_path /path/to/checkpoint-or-hf-model \
  --task korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_hardneg.jsonl \
  --rerank_max_candidates 100 \
  --cache_root ./hf_cache \
  --out_json outputs/korean_rerank.json
```

## Dashboard

```bash
python serve_eval_dashboard.py \
  --result-root sweep_results/my_checkpoint_sweep \
  --host 127.0.0.1 \
  --port 8765
```

Open `http://127.0.0.1:8765/`.

## Included Checkpoint Links

`Main benchmark results/model_checkpoints/` contains symlinks to the selected local CPT checkpoints used for the paper comparison:

- Stage2 Interleave 18k
- Matched decoder-only CPT control 18k
- Vanilla CPT 19k

The symlinks document the selected checkpoints without committing model weights to git.

## Verification

A minimal local smoke test was run with `limit=1` for:

```text
mmlu, kmmlu, kobest, csatqa, click, arc_easy, arc_challenge, hellaswag, openbookqa, korean_rerank, kobest_variant
```

This confirms the package imports, dataset loaders, model scoring loop, and variant runner are wired correctly. It is not a meaningful accuracy measurement.

## Publishing

This working tree has `origin` set to `https://github.com/Splo2t/interleave_attention.git`. After reviewing the staged files, publish with:

```bash
git push -u origin main
```

Recommended first commit message:

```text
Prepare paper evaluation benchmark release
```

See `docs/github_publish_checklist.md` before pushing.
