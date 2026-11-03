# Publication protocol and historical implementation

This source release is an executable implementation of the documented protocol.
It is not a claim that the historical numerical results were produced by every
implementation detail below.

| Component | Historical working behavior | Publication release |
|---|---|---|
| Mixed stratified training | Subsampled synthetic pool to fixed class weights before adding real images | Add class-specific deficits to match the largest selected real training class |
| Mixed non-stratified training | Concatenated source subsets with additional sampling of real images | Reuse the fixed real training subset and add an equal configured number of synthetic samples per class |
| Synthetic source files | General image-directory enumeration could include montage grids | Only individual `sample_*` images are eligible |
| Real validation/test | Different loaders and sources were available across variants | Fixed patient-disjoint real held-out manifests shared across strategies |
| Real-only/synthetic-only stratification and test S | Target-distribution subsampling | Retained as a separate explicit policy; not confused with mixed class balancing |
| Classifier metric | Logged batch-level F1 values | Global per-run macro-F1, then mean and sample SD across repeats |
| Classifier checkpoint | Current model could be tested without restoring best weights | Restore the best real-validation-loss checkpoint before testing |
| FID real-image scaling | Real [0,1] inputs could be treated as [-1,1] | Explicit real-input normalization matching generated inputs |
| FID feature extractor/covariance | Torchvision ImageNet features and legacy covariance calculation | Retained and identified as the implementation's training monitor, not a standardized external FID comparison |
| Class-specific interpolation | A single-class input could fall back to original-image resampling | Within-class nearest-neighbor interpolation with a fresh coefficient per synthetic image |
| Generation errors | Partial generation could return success | Save diagnostics and return an error if any class quota is incomplete |
| NIH selected-label conflicts | Images could occur in multiple selected class directories | Metadata preparation excludes images with more than one selected target label |

Experiments using this release must report their own prepared manifests,
source-pool sizes, quotas, seeds, checkpoints, and metrics. In particular,
4000 generated images per class may not meet the balancing requirement for
an imbalanced ChestX training set. A shortage is an error, not permission to
reuse images or silently change the target.

Only classifier initialization and training randomness change across the ten
classification repeats by default. The generator pool and patient/test
membership remain fixed. This is not ten independent generator-training runs.
