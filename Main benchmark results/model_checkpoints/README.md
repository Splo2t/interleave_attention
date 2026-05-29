# Selected CPT Checkpoints

Local links to the best-AVG checkpoint setting for each CPT model family.

| Name | Setting | Local checkpoint path | Actual weight size |
| --- | --- | --- | --- |
| `stage2_interleave_18k` | Stage2 Interleave, 18k | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints_interleave_full_enc4096_mlm00/checkpoint-18000` | `model.safetensors` ~3.5 GB |
| `matched_decoder_only_cpt_control_18k` | matched decoder-only CPT control, 18k | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-normal-random-new/checkpoint-18000` | `model.safetensors` ~3.4 GB |
| `vanilla_cpt_19k` | vanilla CPT, 19k | `/mnt/nas_server_yhw/converted_checkpoints_for_experiments/checkpoints-1b-cpt/checkpoint-19000` | `model.safetensors` ~2.8 GB |

These are symlinks for local management. Upload the actual checkpoint directories to Hugging Face Hub, object storage, or GitHub Releases if the weights need to be shared outside this machine.
