# Selected CPT Checkpoints

This folder records the best-AVG checkpoint setting for each CPT model family. The entries in this directory are lightweight local symlinks only; the actual model weights are not committed to git. The selected checkpoint folders are shared through Google Drive:

https://drive.google.com/drive/folders/1uzIRIoGpfu65aX7BnYDjp_16xUWi4PyX?usp=sharing

| Name | Setting | Local source path to upload | Actual weight size |
| --- | --- | --- | --- |
| `stage2_interleave_18k` | Stage2 Interleave, 18k | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints_interleave_full_enc4096_mlm00/checkpoint-18000` | `model.safetensors` ~3.5 GB |
| `matched_decoder_only_cpt_control_18k` | matched decoder-only CPT control, 18k | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-normal-random-new/checkpoint-18000` | `model.safetensors` ~3.4 GB |
| `vanilla_cpt_19k` | vanilla CPT, 19k | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-1b-cpt/checkpoint-19000` | `model.safetensors` ~2.8 GB |

For artifact review, upload each complete checkpoint directory to Google Drive using the folder names above. After downloading, place them under an ignored local directory such as `checkpoints/<name>/` and pass that path to `--ckpt_path` or `--single-ckpt-path`. The custom interleave/Krong implementation code is included inside the downloaded checkpoint directory, especially files such as `_modeling_krong.py` and `_processing_krong.py`; the decoder-only control and vanilla CPT checkpoints use the corresponding standard Transformers loading path.
