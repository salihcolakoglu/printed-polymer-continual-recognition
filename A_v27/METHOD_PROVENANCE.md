# Method and implementation provenance

Negotiated target construction follows the idea described by Nuri Korhan and Ceren Oner in *Negotiated Representations to Prevent Forgetting in Machine Learning Applications* (arXiv:2312.00237, version 1, 2023). This source is a preprint: https://arxiv.org/abs/2312.00237 .

The authors' implementation is https://github.com/nurikorhan/Negotiated-Representations-for-Continual-Learning . Its MIT license was read through the connected GitHub file interface during this preparation; LICENSE blob SHA: `bd1be1cad44071c79674f4c9b22f8c3634ca933f`. Its notice is reproduced in `THIRD_PARTY_CODE_NOTICE.txt`.

The supplied MATLAB code is a source-informed adaptation, not an exact rerun of the source notebook. For each stage, the learner predicts the current training rows before updates start. A fixed target mixes that prediction with the prescribed class code: `q = (1-alpha)*classCode + alpha*preStagePrediction`. The mixture weights are 0.5, 2/3, and 0.75. Comparison arms use alpha zero.

The implementation uses a fixed global dictionary rather than the source's adaptive assignment of unused codes. Ordinary codes are four one-hot classes padded to eight outputs. Walsh codes use rows 2 through 5 of a Sylvester order-eight Hadamard matrix, mapped to zero and one. The decoder selects the eligible code with the smallest mean squared distance. The eight-output head, fixed dictionary, decoder, and separate class-incremental eligibility define this study's explicit adaptation. Ordinary and Walsh codes have different active-bit counts, which remain part of the compared codebook choices.

No unpublished work by the manuscript's authors is used as an external scientific authority. Preserved computational component names identify the files that produced the saved results. They do not identify a preceding journal publication.
