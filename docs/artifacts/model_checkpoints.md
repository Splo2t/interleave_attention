# Model Checkpoints

Model weights are intentionally excluded from git. For artifact review, the
selected checkpoint folders are shared separately.

Paper checkpoint mapping:

| Paper name | Artifact folder name |
| --- | --- |
| Stage2 Interleave 18k | `stage2_interleave_18k` |
| Matched decoder-only CPT control 18k | `matched_decoder_only_cpt_control_18k` |
| Vanilla CPT 19k | `vanilla_cpt_19k` |

Recommended local layout after download:

```text
checkpoints/
  stage2_interleave_18k/
  matched_decoder_only_cpt_control_18k/
  vanilla_cpt_19k/
```

Each folder should be a complete Hugging Face `from_pretrained` checkpoint
directory, including `config.json`, tokenizer files, generation config if
present, custom remote-code files such as `_modeling_krong.py` and
`_processing_krong.py`, and all model weight shards.
