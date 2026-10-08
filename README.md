# Zero-Shot Benchmark: Histopathology Foundation Models in Canine Oncology

Evaluation code for the study: **"Heterogeneous Cross-Species Generalization of Histopathology Foundation Models in Canine Oncology: A Zero-Shot Benchmark Revealing Complementary Failure Modes of CONCH and Pathos-Gemma4"** (Villanueva-Castillo et al., 2026, submitted).

## Contents

| Script | Description |
|---|---|
| `benchmark_conch_midog.py` | Zero-shot classification of the MIDOG++ canine subset (148 slides) with CONCH: 4×4 patch grid (900×900 px), background filtering, prompt-based cosine-similarity voting |
| `benchmark_pathos_midog.py` | Zero-shot classification with Pathos-Gemma4 (GGUF, served locally via llama.cpp): 3 patches/slide (30/50/70% of field), temperature 0, majority vote |
| `comparativa_conch_pathos.py` | Head-to-head comparison on the balanced 30-slide subset (10 per tumour type), confusion matrices and figures |

## Result files

- `resultados_conch.json` / `resultados_detallados.csv` — CONCH full-cohort predictions (n = 148)
- `resultados_pathos.json` — Pathos-Gemma4 predictions (balanced subset)
- `comparativa_30slides.csv` — per-slide ground truth vs. model predictions

## Data

Public dataset: **MIDOG++** canine subset — https://doi.org/10.6084/m9.figshare.c.6615571
(148 specimens: cutaneous mast cell tumour n=50, lymphoma n=54, pulmonary carcinoma n=44)

## Models

- CONCH: `MahmoodLab/CONCH` (Hugging Face)
- Pathos-Gemma4: `ByteKnight28/pathos-gemma4-histopathology-GGUF` (Hugging Face, served via llama.cpp)

## Usage

Both benchmarks are strictly zero-shot (no fine-tuning, no parameter updates).

## License

CC-BY 4.0
