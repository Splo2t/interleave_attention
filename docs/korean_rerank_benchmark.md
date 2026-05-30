# Korean Reranking Benchmark

This benchmark evaluates whether a language model can act as a Korean relevance scorer for retrieval or QA-style candidate ranking.
It is designed to connect the existing continuation-scoring evaluation framework with CIKM-style ranking metrics.

## Motivation

Most existing evaluations in this repository are multiple-choice tasks. They measure whether the model assigns the highest continuation likelihood to the correct option. The Korean reranking benchmark generalizes the same idea to retrieval candidates:

```text
score(q, p) = log P("예" | q, p) - log P("아니오" | q, p)
```

where `q` is a Korean query and `p` is a candidate passage. A better reranker should place relevant passages above non-relevant passages.

This is useful for testing the hypothesis that interleave-style models improve candidate discrimination, not merely final multiple-choice accuracy.

## Task Definition

Input:

- A Korean query.
- A list of candidate passages, usually produced by BM25 or a dense retriever.
- Relevance labels from qrels or benchmark judgments.

For each query-passage pair, the evaluator builds the prompt:

```text
질문: {query}

문서: {passage}

이 문서는 질문에 답하는 데 관련이 있다:
```

Then it scores two continuations:

```text
 예
 아니오
```

The default reranking score is:

```text
logP(" 예") - logP(" 아니오")
```

The candidates are sorted by this score.

## Metrics

The benchmark reports:

- `nDCG@1`, `nDCG@5`, `nDCG@10`, `nDCG@100`
- `MRR@1`, `MRR@5`, `MRR@10`, `MRR@100`
- `Recall@1`, `Recall@5`, `Recall@10`, `Recall@100`
- `avg_margin`: best positive score minus best negative score
- `positive_margin_rate`: fraction of queries where the best positive beats the best negative
- `num_queries`, `num_positive_queries`, `num_candidates`, `avg_candidates_per_query`

The margin is especially important for the paper because it tests whether the model separates the gold/relevant candidate from the strongest distractor.

## Candidate JSONL Format

The evaluator expects one query per line:

```json
{
  "query_id": "q1",
  "query": "한국어 질문",
  "candidates": [
    {
      "doc_id": "d1",
      "title": "문서 제목",
      "text": "후보 passage 본문",
      "retriever_rank": 1,
      "retriever_score": 12.3,
      "relevance": 1
    },
    {
      "doc_id": "d2",
      "title": "문서 제목",
      "text": "negative passage",
      "retriever_rank": 2,
      "retriever_score": 11.7,
      "relevance": 0
    }
  ]
}
```

Flat JSONL is also accepted. If each line contains `query_id`, `query`, `doc_id`, `text`, and `relevance`, the evaluator groups candidates by `query_id`.

## Building Candidate JSONL

Use `prepare_korean_rerank_candidates.py` when you already have:

- query JSONL
- corpus JSONL
- TREC-style run file from BM25 or dense retrieval
- qrels file

Example:

```bash
python prepare_korean_rerank_candidates.py \
  --queries-jsonl data/miracl_ko/queries.dev.jsonl \
  --corpus-jsonl data/miracl_ko/corpus.jsonl \
  --run-file runs/miracl_ko_bm25_top100.trec \
  --qrels-file data/miracl_ko/qrels.dev.tsv \
  --out-jsonl rerank_candidates/miracl_ko_dev_bm25_top100.jsonl \
  --max-candidates 100
```

### Controlled Hard-Negative Candidate Set

For the diagnostic experiment, use `--candidate-mode controlled-hard-negative`. This mode does not rely on whether BM25 happened to retrieve a positive passage in the top-k list. Instead, it injects qrels positives first and then fills the remaining slots with top-ranked non-relevant documents from the run file. This gives each query at least one known positive and several lexical hard negatives.

```bash
python prepare_korean_rerank_candidates.py \
  --queries-jsonl data/miracl_ko/queries.dev.jsonl \
  --corpus-jsonl data/miracl_ko/corpus.jsonl \
  --run-file runs/miracl_ko_bm25_top100.trec \
  --qrels-file data/miracl_ko/qrels.dev.tsv \
  --out-jsonl rerank_candidates/miracl_ko_dev_controlled_hardneg.jsonl \
  --candidate-mode controlled-hard-negative \
  --max-candidates 100 \
  --max-positives 1 \
  --min-negatives 1 \
  --max-run-depth 1000
```

Use `candidate_source` in the output JSONL to audit candidate construction. Positives are marked as `qrels_positive`, and BM25 hard negatives are marked as `hard_negative_run`.

Supported run formats:

```text
qid Q0 docid rank score tag
qid docid score
```

Supported qrels formats:

```text
qid 0 docid rel
qid docid rel
```

## Running The Benchmark

Single model:

```bash
PY=/mnt/nas_server_yhw/envs/eval_krong/bin/python
CACHE_ROOT=/mnt/nas_server_yhw/huggingface

$PY eval_paper_benchmarks.py \
  --ckpt_path /path/to/checkpoint \
  --task korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_dev_bm25_top100.jsonl \
  --rerank_max_candidates 100 \
  --rerank_score_mode diff \
  --dtype bf16 \
  --device_map auto \
  --cache_root "$CACHE_ROOT" \
  --eval_batch_size 4 \
  --continuation_scoring oneshot \
  --dec_max_len 4096 \
  --out_json eval_outputs/miracl_ko_rerank.json
```

Krong/interleave model:

```bash
$PY eval_paper_benchmarks.py \
  --ckpt_path /path/to/krong/checkpoint \
  --model_arch krong \
  --task korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_dev_bm25_top100.jsonl \
  --rerank_max_candidates 100 \
  --rerank_score_mode diff \
  --dtype bf16 \
  --device_map auto \
  --cache_root "$CACHE_ROOT" \
  --eval_batch_size 4 \
  --continuation_scoring oneshot \
  --dec_max_len 4096 \
  --out_json eval_outputs/miracl_ko_rerank_krong.json
```

Checkpoint sweep:

```bash
$PY run_eval_checkpoint_sweep.py \
  --single-ckpt-path /path/to/checkpoint \
  --single-ckpt-name model_name \
  --single-ckpt-step 19000 \
  --tasks korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_dev_bm25_top100.jsonl \
  --rerank_max_candidates 100 \
  --rerank_score_mode diff \
  --result-root sweep_results/model_name_korean_rerank \
  --python-bin "$PY" \
  --dtype bf16 \
  --device_map auto \
  --cache_root "$CACHE_ROOT" \
  --eval_batch_size 4 \
  --continuation_scoring oneshot \
  --dec_max_len 4096
```


### Dataset-Derived Few-Shot Reranking

Few-shot reranking is supported without hand-written demonstrations. Use `--rerank_num_fewshot K` to prepend `K` query-passage-label examples sampled from a reranking JSONL file. Demonstrations are label-balanced when possible and are sampled deterministically with `--rerank_fewshot_seed`. If `--rerank_fewshot_data` is omitted, examples are sampled from `--rerank_data` while excluding the current query id to avoid same-query leakage. For a stricter paper setup, provide a separate train/dev few-shot file via `--rerank_fewshot_data`.

Example:

```bash
$PY run_eval_checkpoint_sweep.py \
  --single-ckpt-path /path/to/checkpoint \
  --single-ckpt-name model_name \
  --single-ckpt-step 19000 \
  --tasks korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_hardneg_dev.jsonl \
  --rerank_max_candidates 20 \
  --rerank_score_mode diff \
  --rerank_num_fewshot 4 \
  --rerank_fewshot_seed 42 \
  --result-root sweep_results_rerank/model_name_miracl_fs4 \
  --python-bin "$PY" \
  --dtype bf16 \
  --device_map auto \
  --cache_root "$CACHE_ROOT" \
  --eval_batch_size 1 \
  --continuation_scoring oneshot \
  --dec_max_len 4096
```

## Recommended Datasets

Recommended main experiment:

- MIRACL Korean with BM25 or dense top-100 candidates.

Recommended supplementary experiment:

- Mr.TyDi Korean with sparse or dense top-100 candidates.

Derived QA reranking:

- KorQuAD 1.0 or KLUE-MRC can be converted into a reranking task, but should be described as a derived task rather than an official retrieval benchmark.

## Interpretation

If interleave improves `nDCG@10`, `MRR@10`, and `avg_margin`, this supports the claim that the model is a stronger Korean candidate scorer.

If accuracy-style benchmarks improve but reranking metrics do not, the paper should state that the benefit is task-dependent and may not transfer to retrieval-stage candidate discrimination.
