# Third-party notices

The vendored U-Net, diffusion utilities and attention implementation under
`src/medical_gen/models/ddpm_lib/` contain adaptations of
[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
by Phil Wang (lucidrains), distributed under the MIT license.

Copyright (c) 2020 Phil Wang. The applicable permission and disclaimer are
included in `LICENSES/denoising-diffusion-pytorch-MIT.txt` and must accompany
redistribution of the relevant code. Local adaptations include rectangular
medical-image support and integration with the training framework. This notice
does not imply that an unmodified upstream version produced the paper results.

PyTorch, torchvision, Lightning, diffusers, transformers and
other installed dependencies retain their own licenses; they are dependencies,
not vendored copies in this repository. NIH data and private OCT data are not
covered by a code license and are not distributed here.
