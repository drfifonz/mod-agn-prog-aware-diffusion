# Modality-agnostic, progression-aware diffusion augmentation

Source code for reference-to-pathology image generation and downstream classification in OCT and NIH chest radiography. The framework first trains a reference-class generator and then independently initializes each pathology-specific generator from that reference checkpoint. It includes a scratch DDPM comparison, an OCT GAN baseline, within-class interpolation, and a real-only classifier baseline.

**The private OCT dataset is not distributed.** No patient images, patient manifests, checkpoints, historical experiment logs, or cluster submission scripts are included. Obtain the NIH data through its original distribution and use OCT only when you have appropriate access to your own data.

## What this release reproduces

This release implements the documented publication protocol. In particular, mixed stratified training fills class deficits, all classifier strategies share a real validation and test manifest, and macro-F1 is calculated globally on each test run. These differ from parts of the historical working implementation (fixed 70/15/10/5 synthetic subsampling and batch-level F1 aggregation). **Running this release does not automatically reproduce the historical numerical tables.** New results must be generated and reported with their configuration and selection manifests.

The generator architecture and training components are retained from the research implementation. The configured FID monitor uses the retained torchvision ImageNet Inception feature extractor and legacy covariance calculation. Treat it as the training monitor used by this implementation, not as a newly standardized or externally comparable FID benchmark. Real validation intensities are explicitly mapped according to the input normalization so they use the same [0,1] scale as generated images before feature extraction; this fixes a historical scaling mismatch. FID does not establish diagnostic fidelity.

## Installation

Python 3.12 is recommended. Training is intended for a Linux workstation with an NVIDIA GPU; CPU execution is available for small checks. The complete OCT interpolation baseline can require substantial host memory. Intel macOS receives compatible older PyTorch wheels; it is not the intended full-training platform.

```bash
uv sync --extra training --group dev --locked
```

Sampling, manifest preparation, metrics, and the unit tests can run without PyTorch:

```bash
uv sync --group dev --locked
uv run pytest
```

All paths below are relative to the repository working directory. Run commands from the repository root. Generated files belong in `data/`, `manifests/`, or `outputs/`, which are ignored by Git. W&B is disabled in supplied configurations. Generator runs can opt in using the `wandb` configuration; classifier runs use local CSV logs.

## Real-data format and fixed patient splits

Provide one CSV row per real image:

```csv
path,label,patient_id,split
images/example_001.png,reference,example_patient_001,train
images/example_002.png,reference,example_patient_002,val
images/example_003.png,reference,example_patient_003,test
```

The names above are fictional schema examples, not real patient records. Valid OCT labels are `reference`, `benign`, `fluid`, `precancerous`. The selected ChestX labels are `No_Finding`, `Effusion`, `Pneumonia`, `Pneumothorax`. Paths are resolved relative to `--root`; absolute paths are also supported. Every file must exist. Every real row must have a patient ID. An image may have only one target label, and a patient must occur in only one split. The program rejects duplicate paths, conflicting assignments, missing files, and patient overlap.

If splits are already assigned, preparation validates and preserves them. Otherwise leave **all** `split` cells empty: preparation assigns patients to train/validation/test using 80/10/10 patient fractions and a fixed seed. These are patient fractions, not exact image fractions; the split is not class-balanced. If a particular split lacks a class needed for an experiment, provide a suitable preassigned patient split rather than repeatedly redrawing until a favorable result appears.

```bash
uv run medical-gen-prepare \
  --manifest /path/to/your/oct_images.csv --root /path/to/your/oct_images \
  --classes reference benign fluid precancerous \
  --output manifests/oct_ns.csv --seed 42
```

For NIH data, the optional metadata adapter uses `Image Index`, `Finding Labels`, and `Patient ID`. It retains an image only when exactly one selected target label is present; findings outside the selected set are allowed, and No Finding must be the sole source label. Images with multiple selected findings are excluded, not duplicated into multiple classes. This explicit publication policy differs from the original overlapping class-directory export. Missing source files are skipped by the adapter; the resulting manifest records the actual available pool.

```bash
uv run medical-gen-prepare \
  --nih-metadata /path/to/Data_Entry_2017.csv --root /path/to/nih_images \
  --output manifests/chestx_ns.csv --seed 42
```

The NIH image root may contain individual images directly or selected class subdirectories. Download/extract the original data separately; this repository does not redistribute it or require Kaggle credentials.

### Test sampling regimes

The `_ns` manifest retains the available test distribution. To freeze a second test regime with specified class proportions, start from the **already prepared** manifest. Training and validation membership are preserved; only the held-out test subset is downsampled, without replacement:

```bash
uv run medical-gen-prepare \
  --manifest manifests/oct_ns.csv --root . \
  --classes reference benign fluid precancerous \
  --test-distribution examples/oct_test_distribution.json \
  --output manifests/oct_s.csv --seed 42
```

Repeat for ChestX with `examples/chestx_test_distribution.json`. These example mappings specify 70/15/10/5 in reference/benign/fluid/precancerous order for OCT and No Finding/Effusion/Pneumonia/Pneumothorax order for ChestX. They are **explicit experimental target weights**, not a claim about the original clinical prevalence. Rounding and the limiting class determine actual retained counts. This operation is distinct from mixed-training class balancing.

Freeze each manifest once and use the same file for all strategies and repeats within that test regime. Existing outputs are never overwritten. Do not derive patient IDs from arbitrary image names; supply the correct grouping for both eyes and all visits where applicable.

## Generative training

Both generator training and classifier evaluation consume the same patient-disjoint real manifest. Generator training uses only its real training subset; the validation subset monitors checkpoints and the test subset is never used for fitting or selection. OCT images undergo a central 256×512 crop and resize; ChestX uses a full-image resize. Inputs are single-channel. Horizontal flipping is used only for training; vertical flipping is not used. Diffusion inputs are scaled to [0,1] and internally normalized by the diffusion model; GAN inputs are normalized to [-1,1].

Reference training:

```bash
uv run medical-gen-train --config configs/train/oct/reference.yaml
uv run medical-gen-train --config configs/train/chestx/No_Finding.yaml
```

Independent pathology adaptation:

```bash
uv run medical-gen-train --config configs/train/oct/benign.yaml
uv run medical-gen-train --config configs/train/oct/fluid.yaml
uv run medical-gen-train --config configs/train/oct/precancerous.yaml
```

Each pathology configuration points directly to the reference checkpoint. It does not continue from another pathology model. Weights are initialized from the reference model, while the optimizer is initialized anew. Configurations default to 100,000 class-specific steps; the reference stage adds its own training cost. FID validation selects `checkpoints/best.ckpt`; `final.ckpt` is also saved. The initial Inception/ImageNet weights may need to be downloaded on first use.

An explicit resume preserves the Lightning state of the same run:

```bash
uv run medical-gen-train --config configs/train/oct/benign.yaml \
  --resume outputs/generators/oct/benign/checkpoints/last.ckpt
```

Remove `paths.init_checkpoint` when resuming; initialization and resume are mutually exclusive. Use `scratch_<class>.yaml` for a generator trained without reference initialization. Its architecture and class-specific step budget match the corresponding diffusion configuration. The supplied OCT `gan_<class>.yaml` and `smote_<class>.yaml` provide the other baselines. The interpolation baseline computes nearest neighbors within each selected class in flattened pixel space and draws a fresh interpolation coefficient for each generated image; it does not skip interpolation for single-class input. This corrects a single-class fallback in the historical working implementation. GAN uses epoch-based training; the interpolation baseline has no learned neural optimizer.

## Generate synthetic pools

```bash
uv run medical-gen-generate --config configs/generate/oct_progression.yaml
uv run medical-gen-generate --config configs/generate/chestx_progression.yaml
```

Equivalent configs exist for scratch, OCT GAN and OCT SMOTE. By default each class produces 4000 individual PNGs named `sample_000000.png`, etc. Sampling grids can be written for inspection but are **never loaded as classifier observations**. Generation uses DDIM with 500 sampling steps from a 1000-step diffusion process. Model settings in the generation YAML must match the checkpoint. Existing generated directories are not overwritten. A missing checkpoint, failed batch or incomplete class pool writes diagnostics and exits with an error; partial output is not reported as successful.

Public classifier synthetic roots follow this layout:

```text
data/generated/oct/progression/
  reference/sample_000000.png
  benign/sample_000000.png
  fluid/sample_000000.png
  precancerous/sample_000000.png
```

Generate all classes, including the reference class. A larger pool can be requested with `--num-images N --output-root data/generated/oct/larger_pool`; update the classifier's `synthetic_root` accordingly.

## Classifier sampling: exact definitions

All strategies use the **same real validation and test records** from the selected prepared manifest. The selected real training pool is also common to real-only and mixed comparisons; mixed does not apply a second 80% draw of real training data. Validation remains real even for synthetic-only training. Classifier inputs use single-channel images normalized to [-1,1].

### Mixed, stratified: fill class deficits

Let `R_c` be the number of real training images of class `c`, and `M = max(R_c)`. Select `M - R_c` different synthetic images for that class. The combined training set then contains `M` images per class. The largest real class receives no synthetic additions. Sampling is without replacement. If a class lacks enough synthetic samples, preparation fails and reports the quota and available count. It never silently duplicates records, lowers the target, or discards real training images.

Example: real counts `[5,3,2,1]` require synthetic additions `[0,2,3,4]`, giving `[5,5,5,5]` in mixed training.

### Mixed, non-stratified: equal additions

Add `synthetic_per_class` different synthetic images to **every** class, preserving all selected real training images. The default is 4000 additions per class. This retains the absolute differences between real class sizes; it does not guarantee an equal final class distribution. If the pool has fewer images than requested, fail rather than repeat samples.

### Real-only and synthetic-only

Real-only non-stratified uses the real training subset. Synthetic-only non-stratified uses `synthetic_per_class` individual generated samples per class. In their stratified variants, the corresponding training source is downsampled to the explicit `target_distribution` mapping, without replacement. In these non-mixed settings, the word stratified denotes target-distribution sampling, **not** deficit filling. The distinction is intentional and must be retained when describing experiments.

Mixed balancing uses the available **training** counts, not whole-dataset counts or validation/test labels. A pool of 4000 per class may be insufficient to balance ChestX; increase the generated pool before running such an experiment.

## Downstream experiments

```bash
uv run medical-gen-classify --config configs/classify/oct_mixed_s_test_ns.yaml --prepare-only
```

This checks paths and quotas and writes the selected train/validation/test CSVs and `selection_summary.json` before importing the training stack. Use a fresh `experiment_name` or output directory for the actual run because existing experiments are not overwritten:

```bash
uv run medical-gen-classify --config configs/classify/oct_mixed_s_test_ns.yaml
```

The configs cover both modalities, real/synthetic/mixed training, training S/NS, and test S/NS. They default to ResNet34 with ImageNet initialization, 15 maximum epochs, early stopping, one device, and 10 repeats with classifier seeds `42..51`. Sampling seed 42 and manifest membership stay fixed. Repeats measure classifier-training variation on a fixed generator pool; they are not independent generator-training repeats. Choose a new experiment name when comparing scratch, SMOTE or another pool and change `synthetic_root` accordingly. Real-only training needs no synthetic files.

The best classifier checkpoint is selected by real validation loss and restored before test evaluation. Each repeat writes its checkpoint, predictions and metrics; the top-level summary reports means and sample standard deviations across completed repeats. No results are inferred for incomplete repeats.

### Evaluation metric

Macro-F1 is computed from all predictions of a single real test run. Compute F1 separately for each of the four predefined classes, then average with equal class weights. A zero denominator contributes zero. Report per-class precision, recall, F1, support and the confusion matrix, together with ordinary accuracy. Do not average per-batch F1 values or pool repetitions into a single run.

```bash
uv run medical-gen-evaluate \
  --predictions outputs/classification/oct_mixed_s_test_ns/repeat-00/test_predictions.csv \
  --classes reference benign fluid precancerous

uv run medical-gen-evaluate \
  --runs outputs/classification/oct_mixed_s_test_ns/repeat-*/metrics.json
```

Aggregation checks that run class ordering, real-manifest hash, training-selection hash, training protocol and test-selection hash match and that seeds are distinct. It reports the mean and sample SD (`ddof=1`) across runs; SD is undefined for one run. Macro-F1 alone does not establish high sensitivity in every pathology class or clinical readiness.

## Checks and outputs

```bash
uv run pytest
uv run --extra training pytest tests/test_runtime.py
uv run python -m compileall -q src
```

The core tests cover patient isolation, deterministic sampling, deficit filling, equal additions, insufficient pools, exclusion of montage grids, shared held-out data and global metrics. Runtime tests require the optional training dependencies and use artificial images; they do not establish scientific performance. Full GPU training and regeneration of paper tables are separate experiments.

See [examples](examples/) for the CSV schema and target mappings and [configs](configs/) for all supplied workflows. The input manifest is local and private; do not publish output selection CSVs if they contain private paths or patient IDs.

## References and third-party code

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Song, Meng and Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Chawla et al., *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 2002.
- Wang et al., *ChestX-ray8*, CVPR 2017, for the original NIH data source.

The bundled diffusion/U-Net implementation includes adaptations of `lucidrains/denoising-diffusion-pytorch`. Preserve the accompanying [third-party notices](THIRD_PARTY_NOTICES.md) and license text. No additional project-wide license grant is asserted by this release; the repository owner should select one before distributing it under an open-source license.
