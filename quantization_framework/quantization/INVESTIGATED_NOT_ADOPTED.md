# Investigated, Not Adopted

`smoothquant.py`, `log_quantizer.py`, `twin_gelu_quantizer.py`, and `erq_correction.py`
are calibration-only PTQ accuracy techniques from the literature, each reimplemented
directly against its public reference source (not paraphrased from the paper text —
see each file's docstring for the specific source files/functions checked):

- **SmoothQuant** (Xiao et al., ICML 2023) — migrates activation outlier difficulty
  into the weight matrix via a per-channel scale.
- **Log2 quantizer** (RepQ-ViT, ICCV 2023 / AdaLog, ECCV 2024) — log-domain
  quantization for post-softmax attention weights.
- **Twin uniform quantizer** (PTQ4ViT, ECCV 2022) — splits post-GELU activation
  quantization at zero with independently-calibrated positive/negative scales.
- **ERQ-style correction** (arXiv:2407.06794) — closed-form ridge-regression
  correction of FP32 weights to compensate for activation-quantization error.

Each was implemented, verified for correctness, and tested in isolation
(`test_*_standalone.py` here; `test_smoothquant_isolated.py`, `test_erq_full_model_accuracy.py`,
`test_erq_single_layer_wiring.py`, `test_attention_hook_wiring.py` in
`revision_v2/experiments_imagenet/`). None improved accuracy enough over the project's
existing per-channel W=A allocation to justify adoption into the shipped method — the
project's final accuracy story uses plain 8-bit quantization without these techniques.

Kept in the repo as a record of the design space explored, not as live method code.
