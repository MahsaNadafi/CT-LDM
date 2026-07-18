# Arbitrary-Scale CT Image Super-Resolution  
### using Latent Diffusion Model and Implicit Neural Decoder

This repository focuses on **arbitrary-scale super-resolution (SR) of CT images** using a **Latent Diffusion Model (LDM)** combined with an **Implicit Neural Decoder (LIIF-style)**.

Unlike conventional SR methods limited to fixed upscaling factors, this framework enables **continuous-resolution CT reconstruction**, allowing high-quality image generation at **any target scale** while preserving anatomical structures and intensity consistency.

The method is designed specifically for **medical imaging**, with:
- Patient-wise data splitting (no leakage)
- Realistic low-resolution degradation
- Clinically relevant evaluation metrics

---

## Project Goals

- Perform **CT image super-resolution at arbitrary scales**
- Avoid hallucination of non-existent anatomical structures
- Ensure **patient-safe generalization**
- Provide a **reproducible research baseline** for medical diffusion SR

---

## Data Preparation

### Dataset Structure

```
data/CT/
├── single-slice-Normal/
│   ├── normal_83_ns002i00001_slice_001.png
│   ├── ...
├── single-slice-COVID19/
│   ├── covid_p16_ns002i00001_slice_001.png
│   ├── ...
```

Each filename **must include a patient ID**, which is used for patient-wise splitting.

---

### Dataset Statistics

**Total patients**
- Normal patients: **149**
- COVID-19 patients: **221**

---

### Patient-Wise Splitting (No Leakage)

```bash
python split.py
```

Generated files:
```
data/train.txt
data/val.txt
data/test.txt
```

## Model Training

### First-Stage Autoencoder

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
python main.py \
  --base configs/first-stage/<config_spec>.yaml \
  -t --gpus 0, --scale_lr False
```

---

### Latent Diffusion Model

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
python main.py \
  --base configs/latent-diffusion/<config_spec>.yaml \
  -t --gpus 0, --scale_lr False
```

---

## Evaluation

### Paper-comparable protocol

The attached ICEE 2025 paper reports the following held-out test results:

| Scale | LR -> HR | PSNR | SSIM |
|---:|---:|---:|---:|
| x8 | 32 -> 256 | 29.811 | 0.841 |
| x4 | 64 -> 256 | 33.073 | 0.907 |
| x2 | 128 -> 256 | 40.385 | 0.959 |

A direct reproduction requires the paper's exact patient manifests: 12,584
training slices, 154 validation slices (four patients), and 2,230 test slices
(58 patients). Numbers from a different split are useful diagnostics but are
not directly comparable to the table above.

Evaluate every test slice with deterministic DDIM sampling and EMA weights:

```bash
python eval_sr.py \
  --exp logs/<exp_path> \
  --ckpt logs/<exp_path>/checkpoints/<checkpoint>.ckpt \
  --split test \
  --lr_size 32 \
  --scale_ratio 8 \
  --num_samples -1 \
  --batch_size 1 \
  --steps 100 \
  --eta 0 \
  --seed 23 \
  --use_ema true
```

Repeat with `--lr_size 64 --scale_ratio 4` and `--lr_size 128
--scale_ratio 2`. The saved YAML includes the paper target and the metric gap.

Use `--num_samples 100`, `--num_samples 500`, or `--num_samples -1` to
evaluate a subset or the complete existing test manifest. This option only
limits evaluation and never regenerates or changes the data split.

To sweep multiple checkpoints, DDIM step counts, and test-set sizes, edit
`configs/evaluation/ldm_sr_32_256.yaml` and run:

```bash
python eval_sr.py --eval_config configs/evaluation/ldm_sr_32_256.yaml
```

### Metrics
- PSNR
- SSIM
- MAE
- LPIPS (optional with `--lpips true`; not used in the paper comparison)

---

## Medical Disclaimer

Research use only. Not clinically validated.

---

## Citation

```bibtex
@inproceedings{nadafi2025ct,
  title={CT Super-Resolution Using Arbitrary Scale Diffusion Model},
  author={Nadafi Ghahnavieh, Mahsa and Masoudnia, Saeed and Soltanian-Zadeh, Hamid},
  booktitle={2025 33rd International Conference on Electrical Engineering (ICEE)},
  year={2025},
  doi={10.1109/ICEE67339.2025.11213738}
}
```
