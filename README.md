# Interleave Attention Evaluation

Evaluation code for the CIKM 2026 interleave-attention experiments. The repository keeps only the benchmark runners, analysis helpers, dashboards, and lightweight reproducibility scripts needed for the paper. Large checkpoints, Hugging Face caches, generated benchmark JSONL files, and sweep outputs are intentionally excluded from git.

## Repository Layout

- `krong_eval/`: main benchmark package for standard multiple-choice and reranking tasks.
- `krong_eval_variants/`: KoBEST surface-form stress benchmark package.
- `eval_paper_benchmarks.py`: single-model evaluator for main benchmarks.
- `eval_variant_hf_krong.py`: single-model evaluator for KoBEST variant data.
- `run_eval_checkpoint_sweep.py`: checkpoint sweep runner.
- `prepare_kobest_surface_form_stress.py`: builds P25, P50, and josa-hard KoBEST stress data from public `skt/kobest_v1`.
- `prepare_kobest_query_context_stress.py`: materializes the HF query-context stress dataset into the local `kobest_variant` JSONL layout.
- `download_kobest_variant_data.py`: optional helper for downloading pre-materialized KoBEST variant JSONL data from a dataset repository.
- `scripts/`: conversion, sweep, and export wrappers used in the experiments.
- `training/stage2_train.py`: sanitized Stage-2 CPT training reference script for interleave and matched decoder-only controls.
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

Optional Stage-2 training reference code uses:

```bash
pip install -r requirements-training.txt
```

The training script also expects the original local training modules `kormo`, `krong_tokenizer`, and `llama_interleave` to be available on `PYTHONPATH`. These are training-time dependencies; running the benchmark evaluators does not require them.

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

For anonymous review, these stress sets are generated locally from the public KoBEST dataset instead of being stored in this git repository or uploaded to a separate data host.

```bash
python prepare_kobest_surface_form_stress.py \
  --variant ko_random_p25 \
  --output-root variant_benchmarks/ko_random_p25 \
  --cache-root ./hf_cache \
  --overwrite

python prepare_kobest_surface_form_stress.py \
  --variant ko_random_p50 \
  --output-root variant_benchmarks/ko_random_p50 \
  --cache-root ./hf_cache \
  --overwrite

python prepare_kobest_surface_form_stress.py \
  --variant ko_josa_preserve_compaction_hard \
  --output-root variant_benchmarks/ko_josa_preserve_compaction_hard \
  --cache-root ./hf_cache \
  --overwrite
```

The script downloads `skt/kobest_v1`, writes the local `kobest_variant` JSONL layout, and records counts plus transformation metadata in `manifest.json`. If a non-anonymous artifact is prepared later, the optional `download_kobest_variant_data.py` helper can fetch the same materialized layout from a dataset repository.

## Stage-2 Training Reference

`training/stage2_train.py` contains the Stage-2 CPT training reference code used for the interleave model family and the matched decoder-only CPT controls. It is included for artifact completeness, but it is not required for running the benchmark scripts.

The public copy intentionally does not include any W&B API key or private credential. Configure logging outside the repository when needed:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=WriteQ-stage2
export WANDB_ENTITY=minjucau
python training/stage2_train.py --report_to wandb ...
```

The script expects local Stage-1 checkpoints, packed Stage-2 training data, and the original training-only modules (`kormo`, `krong_tokenizer`, `llama_interleave`) to be available in the training environment. Evaluation-time custom model code is bundled in the downloaded interleave checkpoint directory, as described below.

## Run Main Benchmarks

Single benchmark smoke test:

```bash
python eval_paper_benchmarks.py \
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

Change `--variant_data_root` and `--variant_name` to run P50 or josa-hard after generating those folders with `prepare_kobest_surface_form_stress.py`.

## Run Controlled Relevance Scoring

`korean_rerank` consumes a JSONL file with one row per query and candidate passages. See `docs/korean_rerank_benchmark.md` for the expected schema.

```bash
python eval_paper_benchmarks.py \
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

## Model Checkpoints

Model weights are not committed to git. For artifact review, the selected checkpoint folders are shared through Google Drive:

https://drive.google.com/drive/folders/1uzIRIoGpfu65aX7BnYDjp_16xUWi4PyX?usp=sharing

The table below records the complete local checkpoint directories that were uploaded:

| Paper name | Google Drive folder name | Local source path to upload |
| --- | --- | --- |
| Stage2 Interleave 18k | `stage2_interleave_18k` | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints_interleave_full_enc4096_mlm00/checkpoint-18000` |
| Matched decoder-only CPT control 18k | `matched_decoder_only_cpt_control_18k` | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-normal-random-new/checkpoint-18000` |
| Vanilla CPT 19k | `vanilla_cpt_19k` | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-1b-cpt/checkpoint-19000` |

Recommended Google Drive layout:

```text
interleave_attention_checkpoints/
  stage2_interleave_18k/
  matched_decoder_only_cpt_control_18k/
  vanilla_cpt_19k/
```

Each folder should be uploaded as a full Hugging Face `from_pretrained` checkpoint directory, including `config.json`, tokenizer files, generation config if present, and all model weight shards such as `model.safetensors` or `model-*.safetensors`. The custom interleave/Krong implementation code is bundled with the downloaded checkpoint directory; inspect files such as `_modeling_krong.py` and `_processing_krong.py` inside the checkpoint. The evaluation scripts load these checkpoints with `trust_remote_code=True` when needed.

After downloading from Google Drive, place the folders under a local ignored directory such as:

```text
checkpoints/
  stage2_interleave_18k/
  matched_decoder_only_cpt_control_18k/
  vanilla_cpt_19k/
```

Then run, for example:

```bash
python run_eval_checkpoint_sweep.py \
  --single-ckpt-path checkpoints/stage2_interleave_18k \
  --single-ckpt-name stage2_interleave_18k \
  --single-ckpt-step 18000 \
  --tasks mmlu,kmmlu,kobest,click,csatqa,arc_easy,arc_challenge,hellaswag,openbookqa \
  --result-root sweep_results/stage2_interleave_18k_main \
  --cache_root ./hf_cache
```

`Main benchmark results/model_checkpoints/` keeps lightweight symlinks that document the original local source paths, but external users should use the Google Drive downloads above.

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
