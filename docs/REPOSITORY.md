# What belongs in this repository

This repository shares the code and instructions needed to prepare leaf datasets,
train MobileNetV4, evaluate a model, and predict from a photograph. Each user
obtains the datasets separately and creates their own training outputs.

## Public files

- Original Python code, including download/import, curation, splitting,
  training, evaluation, prediction, and report-generation scripts.
- Tests that generate synthetic fixtures at runtime.
- Dependency specifications and lockfiles, `.python-version`, `.gitignore`,
  the MIT license, and the sanitized `.env.example` template.
- Setup and execution guides, class definitions and mapping logic, file schemas,
  data-preparation methods, source attribution, and known limitations.

Dataset preparation updates belong here as code and methodology. The resulting
photos, source annotations, and per-image manifests stay local. See
[SOURCES.md](SOURCES.md) for source terms and [DATASET_SCHEMA.md](DATASET_SCHEMA.md)
for the generated file format.

## Local files

| Category | Examples | Reason |
|---|---|---|
| Source and processed datasets | Photos, crops, supplied augmentations, archives, source label tables, inventories, leaf maps | These are separately licensed source materials or generated dataset copies. |
| Prepared per-image records | Manifests, split/group assignments, duplicate audits, prediction records | They can contain source annotations, image paths, and metadata. |
| Model artifacts | Pretrained weights, fine-tuned checkpoints, exported models | The code's MIT license does not grant rights to these weights. |
| Run history | Configurations, logs, metrics, plots, source snapshots, experiment journals | These describe a particular local run and may expose machine paths. |
| Private settings | `.env`, API tokens, Kaggle credentials, key files | These are personal configuration or secrets. |
| Installed and temporary files | Virtual environments, caches, build output, editor settings | Users create these locally from the shared source and lockfiles. |

Keep datasets in `data/`, models and reports in `runs/` or `outputs/`, and
private notes in ignored `docs/local/`. `.gitignore` also covers common artifact
names and extensions if a file is accidentally copied outside those folders.
Only the sanitized `.env.example` is shared among `.env` files.

## Reusable documentation

Public guides describe how to run the commands, what files they produce, and
how to interpret them. Use plain example paths such as `runs/weekend/best.pt`
for files the reader must generate. Markdown links should point to public
source files, documentation, or official websites; links to ignored local
outputs will be broken on GitHub.

Detailed run diaries and machine-specific completion updates belong in the
local experiment folder. Dataset sources, label-merging rules, approximate
Mango grouping, and the lack of healthy-orange examples belong in the public
guides because they affect everyone's use of the code.

A sanitized aggregate-results table or chart is optional documentation.
Excluding all generated images, CSVs, and reports is this repository's chosen
scope, not a blanket rule that every such file is legally unpublishable. A
separately prepared benchmark can describe its data, settings, and limitations
without bundling model weights. Review its source terms and contents before
making a deliberate exception to the ignore rules.

## Check the files before publication

Inspect both tracked files and untracked candidates, then inspect the exact
staged changes before committing:

```bash
git status --short
git ls-files --cached --others --exclude-standard
git diff --cached --stat
git diff --cached
git ls-files -ci --exclude-standard
```

The last command should print nothing. It finds tracked files that now match
ignore rules: `.gitignore` does not remove files already tracked or erase prior
commits. Preserve local data when correcting the index, and inspect affected
history if a sensitive file was committed previously. Review file contents as
well as names; an allowed text file can still contain credentials or local paths.

The [verification guide](VALIDATION.md) covers checks for the code and generated
data. Publishing the source does not publish ignored local datasets or models.
