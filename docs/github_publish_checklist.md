# GitHub Publish Checklist

Use this checklist before making the repository public.

## 1. Confirm ignored local artifacts

```bash
git check-ignore -v checkpoint-19000
git check-ignore -v checkpoints-interleave_full_enc4096_decodkd_copylow
git check-ignore -v sweep_results
git check-ignore -v variant_benchmarks
git check-ignore -v variant_lexicons
git check-ignore -v rerank_candidates
```

These paths are large or generated and should not be committed.

## 2. Inspect files that will be committed

```bash
git status --short
git add --dry-run .
```

Look for accidental data, private paths, model weights, cached datasets, logs, or unpublished notes.

## 3. Audit site-local paths

```bash
rg -n "/mnt/nas_server|/mnt/nas_server_yhw|/home/"
```

Local defaults are acceptable for private lab repos, but public repos should either document them clearly or replace them with portable placeholders.

## 4. Initialize and commit

```bash
git init
git add .
git commit -m "Prepare KRong evaluation workspace"
```

## 5. Add remote and push

```bash
git branch -M main
git remote add origin git@github.com:<owner>/<repo>.git
git push -u origin main
```

## 6. Publish large artifacts separately

Use Hugging Face Hub, object storage, or GitHub Releases for:

- checkpoints
- converted checkpoints
- generated benchmark JSONL files
- full sweep outputs
- spreadsheets and paper tables

Keep the README links updated if these artifacts are made public.
