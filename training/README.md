# Stage-2 Training Reference

This folder contains the sanitized Stage-2 CPT training reference code for the paper.

- `stage2_train.py`: interleave and matched decoder-only CPT training script.

Credentials are intentionally not included. Set `WANDB_API_KEY`, `WANDB_PROJECT`, and `WANDB_ENTITY` in the shell if W&B logging is needed, then pass `--report_to wandb`.

The script requires the original training environment modules on `PYTHONPATH`: `kormo`, `krong_tokenizer`, and `llama_interleave`. The benchmark evaluation code in this repository does not require those modules.
