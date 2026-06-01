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

## Candidate Data

This trimmed reviewer repository does not include candidate-building scripts.
Provide a prepared reranking JSONL file under `rerank_candidates/`, or pass its
path with `--rerank_data`.

## Running The Benchmark

Single model:

```bash
python eval_paper_benchmarks.py \
  --ckpt_path checkpoints/stage2_interleave_18k \
  --model_arch krong \
  --task korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_dev_bm25_top100.jsonl \
  --rerank_max_candidates 100 \
  --rerank_score_mode diff \
  --dtype bf16 \
  --device_map auto \
  --cache_root ./hf_cache \
  --eval_batch_size 4 \
  --continuation_scoring oneshot \
  --dec_max_len 4096 \
  --out_json eval_outputs/miracl_ko_rerank.json
```

Multiple tasks through the sweep helper:

```bash
python run_eval_checkpoint_sweep.py \
  --single-ckpt-path checkpoints/stage2_interleave_18k \
  --single-ckpt-name stage2_interleave_18k \
  --single-ckpt-step 18000 \
  --tasks korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_dev_bm25_top100.jsonl \
  --rerank_max_candidates 100 \
  --rerank_score_mode diff \
  --result-root sweep_results/model_name_korean_rerank \
  --dtype bf16 \
  --device_map auto \
  --cache_root ./hf_cache \
  --eval_batch_size 4 \
  --continuation_scoring oneshot \
  --dec_max_len 4096
```


### Dataset-Derived Few-Shot Reranking

Few-shot reranking is supported without hand-written demonstrations. Use `--rerank_num_fewshot K` to prepend `K` query-passage-label examples sampled from a reranking JSONL file. Demonstrations are label-balanced when possible and are sampled deterministically with `--rerank_fewshot_seed`. If `--rerank_fewshot_data` is omitted, examples are sampled from `--rerank_data` while excluding the current query id to avoid same-query leakage. For a stricter paper setup, provide a separate train/dev few-shot file via `--rerank_fewshot_data`.

Example:

```bash
python run_eval_checkpoint_sweep.py \
  --single-ckpt-path checkpoints/stage2_interleave_18k \
  --single-ckpt-name stage2_interleave_18k \
  --single-ckpt-step 18000 \
  --tasks korean_rerank \
  --rerank_data rerank_candidates/miracl_ko_hardneg_dev.jsonl \
  --rerank_max_candidates 20 \
  --rerank_score_mode diff \
  --rerank_num_fewshot 4 \
  --rerank_fewshot_seed 42 \
  --result-root sweep_results_rerank/model_name_miracl_fs4 \
  --dtype bf16 \
  --device_map auto \
  --cache_root ./hf_cache \
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
