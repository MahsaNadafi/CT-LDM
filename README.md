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

### Arbitrary-Scale Super-Resolution

```bash
python eval_sr.py --exp logs/<exp_path> --lr_size <input_lr_size> --scale_ratio <scale>
```

### Metrics
- PSNR
- SSIM
- MAE

---

## Medical Disclaimer

Research use only. Not clinically validated.

---

## Citation

```bibtex
@inproceedings{kim2024arbitraryscale,
  title={Arbitrary-Scale Image Generation and Upsampling using Latent Diffusion Model and Implicit Neural Decoder},
  author={Kim, Jinseok and Kim, Tae-Kyun},
  booktitle={CVPR},
  year={2024}
}
```
