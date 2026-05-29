# Korean Surface-Form Stress Tests

Curated view for the paper's KoBEST surface-form stress benchmark.

This folder keeps the `kobest_variant` runner, KoBEST stress-data materializer, and paper analysis helpers. The benchmark JSONL data is not committed to git; generate it locally from public `skt/kobest_v1` into `variant_benchmarks/<variant_name>` and pass that path with `--variant_data_root`.

Paper stress variants:

```text
ko_random_p25
ko_random_p50
ko_josa_preserve_compaction_hard
kobest_query_context_stress_v3
```

See the root `README.md` for data materialization and evaluation commands.
