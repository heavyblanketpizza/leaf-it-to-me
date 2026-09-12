# Dataset sources and import guide

Source records were checked on September 12, 2026. The release counts below explain which images are eligible for this project; your preparation `report.json` records the files actually imported and retained on your computer. Store datasets under `data/raw/` inside the `LEAFIT_DATA` folder configured in your private `.env`. See [the verification guide](VALIDATION.md) for checks and [the README](../README.md) for setup and the full workflow.

Commands below run from the project folder after setup. Load your private settings and create the storage folders (connect your drive first if using external storage):

```bash
set -a
source .env
set +a
mkdir -p "$LEAFIT_DATA/data/raw" "$LEAFIT_DATA/downloads"
```

## License scope and attribution

Publishing this repository shares original code and documentation under
[MIT](../LICENSE); datasets, model weights, and dependencies retain their own
terms. Training uses the datasets and pretrained weights under those terms,
including MangoLeafBD's noncommercial condition. Demonstrating predictions is
a separate use from distributing the data or weights: consider the purpose of
the demonstration and the terms for any source images shown. Dataset licenses
do not automatically become the model's license; [Creative Commons' AI guidance](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)
explains when copyright permission and license conditions apply.

The README banner uses photos and icons licensed through a paid stock
subscription and is excluded from this repository's MIT license.

Please credit the original dataset sources:

- **PlantVillage:** Sharada P. Mohanty, David P. Hughes, and Marcel Salathé (2016),
  [Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
  Source: the [author's dataset card](https://huggingface.co/datasets/mohanty/PlantVillage)
  and [original image repository](https://github.com/spMohanty/PlantVillage-Dataset).
- **Cornell / Plant Pathology 2020:** Thapa et al. (2020),
  [The Plant Pathology 2020 challenge dataset to classify foliar disease of apples](https://arxiv.org/abs/2004.11958).
  Download: [official FGVC7 competition](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data).
- **MangoLeafBD:** the contributors listed on the
  [Mendeley Data v1 record](https://data.mendeley.com/datasets/hxsnvwty3r/1),
  published August 30, 2022, DOI [10.17632/hxsnvwty3r.1](https://doi.org/10.17632/hxsnvwty3r.1).

The source license pages were rechecked on September 12, 2026: PlantVillage's
author card lists CC BY-SA 3.0; MangoLeafBD v1 lists CC BY-NC 3.0; the
[Cornell rules, section B.7](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules)
list Apache 2.0 while retaining competition-specific data restrictions and
precedence. Obtain the data from these sources and review their current terms.

## PlantVillage

The author's [Hugging Face card](https://huggingface.co/datasets/mohanty/PlantVillage) lists **CC BY-SA 3.0**, 54,306 photographs, and color/grayscale/segmented variants. Use color only. The [original GitHub repository](https://github.com/spMohanty/PlantVillage-Dataset) identifies `raw/color` as original RGB photographs.

The official [color training list](https://huggingface.co/datasets/mohanty/PlantVillage/resolve/9e97599868962bd0079b8db4b7f1efa9185fa1e7/splits/color_train.txt) and [color test list](https://huggingface.co/datasets/mohanty/PlantVillage/resolve/9e97599868962bd0079b8db4b7f1efa9185fa1e7/splits/color_test.txt) at the pinned revision contain 54,305 distinct paths; the requested trees account for **13,241** paths:

| Actual class directory | Count |
|---|---:|
| `Apple___Apple_scab` | 630 |
| `Apple___Black_rot` | 621 |
| `Apple___Cedar_apple_rust` | 275 |
| `Apple___healthy` | 1,645 |
| `Cherry_(including_sour)___Powdery_mildew` | 1,052 |
| `Cherry_(including_sour)___healthy` | 854 |
| `Peach___Bacterial_spot` | 2,297 |
| `Peach___healthy` | 360 |
| `Orange___Haunglongbing_(Citrus_greening)` | 5,507 |

The inventory differs by one image from the reported whole-dataset total. Preparation counts actual files instead of assuming the headline total. Orange has only citrus-greening examples, so this selection cannot teach a healthy-versus-diseased orange decision.

The [leaf map](https://huggingface.co/datasets/mohanty/PlantVillage/resolve/9e97599868962bd0079b8db4b7f1efa9185fa1e7/leaf_grouping/leaf-map.json) maps normalized photograph identifiers to lists such as `["Peach___healthy:::54.0"]`. All 13,241 selected inventory entries resolve using the [official loader's normalization](https://huggingface.co/datasets/mohanty/PlantVillage/blob/9e97599868962bd0079b8db4b7f1efa9185fa1e7/plant_village.py): take the filename after `___`, remove `copy` suffixes and image extensions, lowercase, then choose the class-matching suggestion. The loader's license comment says its dataset license was assumed; the current dataset card explicitly declares CC BY-SA 3.0.

Pinned source revisions: GitHub `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`; Hugging Face `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.

The downloader selects only those nine original RGB directories and official grouping/inventory metadata. It excludes other crops and all grayscale and segmented copies. Download or resume with:

```bash
uv run python -m leafit download-plantvillage --output "$LEAFIT_DATA/data/raw/plantvillage"
```

Downloaded original bytes are preserved. The saved `download_receipt.json` records the selected source revisions and file hashes; the preparation report separately checks readability and grouping.

## Cornell Plant Pathology 2020 / FGVC7

The [authors' paper](https://arxiv.org/abs/2004.11958) reports 3,651 captured photographs. Its [methods, section 2.3](https://arxiv.org/pdf/2004.11958), name apple scab, **cedar apple rust**, complex disease patterns, and healthy leaves as the challenge categories. This supports merging the release's `rust` category with PlantVillage's cedar apple rust.

The [official Kaggle data page](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data) explicitly shows **3,642 JPGs and three CSVs**, 823.79 MB. Files are `images/`, `train.csv`, `test.csv`, and `sample_submission.csv`. It describes labels for training images and only image IDs for test images. Its prose calls the combined category `combinations`; the release CSV column is `multiple_diseases`.

The downloadable release contains **3,642 image entries**: **1,821 publicly labeled training photographs** and **1,821 unlabeled test photographs**. Each of the three CSVs has 1,821 data rows. The paper's larger total is not a supervised-training count. The release schemas are:

| CSV | Release columns | Use here |
|---|---|---|
| `train.csv` | `image_id`, `healthy`, `multiple_diseases`, `rust`, `scab` | Public labels; retained |
| `test.csv` | `image_id` | Unlabeled image IDs; not extracted |
| `sample_submission.csv` | `image_id`, `healthy`, `multiple_diseases`, `rust`, `scab` | Submission template, not labels; not extracted |

Every training row has exactly one category marked 1 and the other three marked 0. The actual public label counts and combined mappings are:

| Cornell column | Count | Combined label |
|---|---:|---|
| `healthy` | 516 | `apple_healthy` |
| `multiple_diseases` | 91 | `apple_multiple_diseases` |
| `rust` | 622 | `apple_cedar_apple_rust` |
| `scab` | 592 | `apple_scab` |

Cornell does not supply public physical-leaf identifiers, so exact and visual
duplicate checks provide approximate grouping. The merge script excludes
exact-pixel copies with conflicting labels and records them in
`excluded_conflicts.csv`, preserving source files and labels. Use
`--strict-conflicts` to stop and inspect instead. The lower-level
`python -m leafit prepare` command stops by default unless given
`--exclude-conflicting-duplicates`. Neither command invents a corrected label.
Cornell's `multiple_diseases` remains a separate class; the release does not
identify individual diseases for those images.

The [competition rules](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules) govern access and use. They list Apache 2.0 for the data, with competition rules taking precedence, and restrict redistribution to people who accepted the rules. The project should keep downloaded data private. This is not an unrestricted permission to redistribute the archive.

### Download and import

1. Sign in to Kaggle and open the competition's [data page](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data). Follow the prompt to review and accept the rules.
2. Use **Download all files** and save the archive as `$LEAFIT_DATA/downloads/plant-pathology-2020-fgvc7.zip`. Alternatively, authenticate and download with the CLI using the [official API instructions](https://www.kaggle.com/docs/api):

   ```bash
   uvx --from kaggle kaggle auth login
   uvx --from kaggle kaggle competitions download -c plant-pathology-2020-fgvc7 -p "$LEAFIT_DATA/downloads"
   ```

3. Import either download with the selective helper:

   ```bash
   uv run python scripts/import_cornell_zip.py --archive "$LEAFIT_DATA/downloads/plant-pathology-2020-fgvc7.zip" --output "$LEAFIT_DATA/data/raw/cornell" --delete-archive
   ```

If browser authentication is unavailable, the [official CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) supports a token from [Settings → API](https://www.kaggle.com/settings/api), saved in `~/.kaggle/access_token`. Legacy credentials remain supported: **Legacy API Credentials → Create Legacy API Key**, save `kaggle.json` in `~/.kaggle/`, then `chmod 600 ~/.kaggle/kaggle.json`. Do not put credentials in this repository.

The helper retains `train.csv` and only the training photographs named by its validated public labels. It checks ZIP checksums, image counts and Pillow readability before deleting the supplied ZIP when `--delete-archive` is set; omit that flag to keep the archive. Unlabeled test photos, `test.csv` and submission examples are never extracted. The resulting local import root is `"$LEAFIT_DATA/data/raw/cornell"`. A ZIP saved elsewhere also works by passing its actual path to `--archive`. The README's main preparation command uses all three sources and produces the complete 18-label mapping.

## MangoLeafBD v1

The [official Mendeley record](https://data.mendeley.com/datasets/hxsnvwty3r/1), DOI `10.17632/hxsnvwty3r.1`, lists **CC BY-NC 3.0**: credit the authors and use it noncommercially. It reports 4,000 JPGs, 500 per category, made from approximately 1,800 distinct leaves plus zoomed/rotated versions.

The v1 release contains these eight category directories: `Anthracnose`, `Bacterial Canker`, `Cutting Weevil`, `Die Back`, `Gall Midge`, `Healthy`, `Powdery Mildew`, `Sooty Mould`. Download and import the official
[v1 archive](https://data.mendeley.com/public-api/zip/hxsnvwty3r/download/1) with:

```bash
uv run python scripts/download_mangoleafbd.py --output "$LEAFIT_DATA/data/raw/mangoleafbd"
```

If an automated request receives HTTP 403, open the dataset page and click **Download All**. Save it as `$LEAFIT_DATA/downloads/mangoleafbd-v1.zip`, then import it:

```bash
uv run python scripts/download_mangoleafbd.py --archive "$LEAFIT_DATA/downloads/mangoleafbd-v1.zip" --output "$LEAFIT_DATA/data/raw/mangoleafbd" --remove-archive
```

The importer checks ZIP integrity and image readability, preserves source image
bytes, and writes a local receipt. `--remove-archive` removes a supplied archive
after successful verification; omit it to keep that archive.

The release has no physical-leaf or original-to-augmentation grouping table.
The default merge policy uses capture metadata and visual similarity to keep
one representative per detected Mango family and class, then splits whole
families. It preserves downloaded files and records exclusions in
`mango_curation.csv`. These groups do not certify original-only or independent
physical leaves. See [Mango curation and grouping](MANGO_AUDIT.md) for the
algorithm, limitations, and the `--mango-policy train-only` fallback.

## Pretrained model and dependencies

The starting checkpoint is
[`timm/mobilenetv4_conv_small.e2400_r224_in1k`](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k),
trained on ImageNet-1k by Ross Wightman. Its model card lists Apache 2.0. The
project uses these pretrained weights as a starting point and fine-tunes them
on the three leaf datasets above. It does not download ImageNet training images.

[timm's licensing guidance](https://github.com/huggingface/pytorch-image-models#licenses)
notes that the implications of [ImageNet's noncommercial research and education
terms](https://image-net.org/download.php) for pretrained weights are unclear,
and recommends assuming those restrictions apply. The model card's Apache 2.0
designation alone does not resolve this upstream rights question; the guidance
does not establish that dataset restrictions automatically transfer to weights.

Downloaded weights stay in `.cache/`; fine-tuned checkpoints stay in `runs/`.
Neither is included in Git or relicensed by this project's MIT license.
Dependency names and versions are shared in `pyproject.toml` and `uv.lock`;
the installed third-party packages stay in the ignored `.venv/` and retain
their own licenses.
