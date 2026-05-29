# Evaluation Scripts

Each subdirectory is one `EXP_NAME`. The scripts are intentionally thin:

- `env.sh`: source/converted/result defaults for that experiment.
- `convert.sh`: materialize or refresh converted checkpoints.
- `benchmark_main7.sh`: run the paper main set `mmlu,kmmlu,kobest,click,csatqa,arc_easy,arc_challenge,hellaswag,openbookqa`.
- `benchmark_extra.sh`: run the paper extra set `click,csatqa,openbookqa`.

Common usage:

```bash
cd -P /mnt/nas_server_yhw/eval_krong

# Convert one checkpoint.
STEP=11000 scripts/checkpoints_interleave_full_enc4096_mlm025_mbert/convert.sh

# Run one checkpoint on main7.
STEP=11000 scripts/checkpoints_interleave_full_enc4096_mlm025_mbert/benchmark_main7.sh

# Run a range.
START_STEP=11000 END_STEP=19000 scripts/checkpoints_interleave_full_enc4096_mlm025_mbert/benchmark_main7.sh

# Run extra tasks for a range.
START_STEP=1000 END_STEP=19000 scripts/checkpoints_interleave_full_enc4096_mlm025_mbert/benchmark_extra.sh
```

Notes:

- If `STEP` is set, scripts run only `checkpoint-${STEP}`.
- If `STEP` is not set, scripts use `CHECKPOINT_PATTERN`, `START_STEP`, `END_STEP`, and `STEP_INTERVAL`.
- For repeated reruns, omit `SKIP_EXISTING_JSON` so the latest `ok` row overrides older failed rows in the dashboard.
- Use `OVERWRITE_MIRROR=1 OVERWRITE_EXPERIMENTS=1` only when you intentionally want to rebuild converted checkpoints.
