# 8B Evaluation Workspace

This folder keeps 8B model experiments separate from the main KRong checkpoint sweeps.

## Layout

- `backend/run_llama31_8b_main7_sweep.sh`: records Llama 3.1 8B scores into `sweep_results_8b`.
- `backend/serve_8b_dashboard.sh`: serves only the 8B sweep result root.
- `frontend/dashboard_frontend_8b.html`: 8B-labeled dashboard frontend.

## Main Benchmark Run

```bash
cd -P /mnt/nas_server_yhw/eval_krong

MODEL=meta-llama/Llama-3.1-8B \
MODEL_LABEL=llama31_8b \
experiments_8b/backend/run_llama31_8b_main7_sweep.sh
```

Use `MODEL=/path/to/local/model` if the model is already downloaded locally.


## Dashboard

```bash
cd -P /mnt/nas_server_yhw/eval_krong

PORT=7816 experiments_8b/backend/serve_8b_dashboard.sh
```

Open `http://<server-ip>:7816/`.

## Common Overrides

- `TASKS=mmlu,kmmlu,kobest`: run a smaller subset.
- `DEVICE_MAP=cuda:0`: force a specific GPU instead of `auto`.
- `LIMIT=100`: smoke-test with 100 examples per task.
- `RUN_NAME=my_run_name`: force a stable result folder name.
- `RESULT_BASE=/path/to/result_root`: store results somewhere else.
