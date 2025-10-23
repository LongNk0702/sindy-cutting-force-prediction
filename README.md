# PySINDy Cutting Force Prediction

Repository for **cutting force prediction in milling processes** using a **Taguchi-based dataset (27 experiments)** and **SINDy (Sparse Identification of Nonlinear Dynamics)**.  
Goal: Identify a sparse, interpretable, and extrapolatable dynamic model of cutting force.

---

## Key Highlights
- End-to-end pipeline:  
  `load data` → `preprocess` → `feature library` → `fit SINDy` → `evaluate`
<!-- - Benchmark comparison with baseline models: **Random Forest** and **Gradient Boosting** -->
- Supports both:
  - **Python** (via [PySINDy](https://github.com/dynamicslab/pysindy))
  - **MATLAB** (dedicated scripts for modeling and validation)

---

## Project Structure
See detailed structure and explanations in [`docs/index.md`](docs/index.md).

```
SINDy-Cutting-Force/
│
├── data/ # Taguchi dataset (27 experiments)
├── src/ # Main Python scripts (data prep, modeling, evaluation)
├── matlab/ # MATLAB SINDy implementation
├── notebooks/ # Jupyter notebooks for visualization & testing
├── docs/ # Documentation and notes
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment setup
└── README.md # This file
```
---

## Quick Setup

### Option 1: Conda
```bash
conda env create -f environment.yml
conda activate sindy-cutting-force
pre-commit install

---
## References

Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS, 113(15), 3932–3937.
