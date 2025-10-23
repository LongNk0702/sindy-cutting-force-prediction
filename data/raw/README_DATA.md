# Cutting Force Experiment Data (Taguchi L27)

This folder contains the **experimental datasets** used in the study *"Prediction of Cutting Force in Milling Process Using SINDy"*.

---

## 1. Overview

The data were collected from **27 Taguchi-designed experiments (L27 – 3⁴)**  
for the milling process using an **indexable end mill with rhombic inserts**.

Each experiment varies four cutting parameters across three levels,  
and the resulting **cutting forces (Fx, Fy, Fz)** are measured.

---

## 2. Input Parameters

| Symbol | Parameter | Unit | Level 1 | Level 2 | Level 3 |
|:-------|:-----------|:------|:---------|:---------|:---------|
| X₁ | Cutting speed *(Vc)* | m/min | 80 | 140 | 200 |
| X₂ | Feed per revolution *(fₜ)* | mm/rev | 0.05 | 0.10 | 0.15 |
| X₃ | Axial depth of cut *(a)* | mm | 0.10 | 0.30 | 0.50 |
| X₄ | Radial depth of cut *(b)* | mm | 4 | 8 | 12 |

These parameters were chosen based on:
- The machining capacity of the tool (D=16 mm, 2 flutes).
- Material: **SKD11 tool steel**.
- Recommended ranges from the tool manufacturer.

---

## 3. Experimental Design

- **Design method:** Taguchi Orthogonal Array (L27)
- **Number of factors:** 4
- **Levels per factor:** 3
- **Total experiments:** 27 runs

Each row in the dataset corresponds to one experimental trial with: Run | Vc | ft | a | b | Fx | Fy | Fz


---

## 4. Output Data

| Symbol | Description | Unit |
|:--------|:-------------|:------|
| Fx | Cutting force in X direction | N |
| Fy | Cutting force in Y direction | N |
| Fz | Cutting force in Z direction | N |

*Additional measurements such as vibration or surface roughness can be appended if available.*

---

## 5. File Structure

```
data/
├─ raw/
│ ├─ exp_run_01.csv ... exp_run_27.csv # (optional split)
│ └─ README_DATA.md # This file
├─ processed/
│ ├─ taguchi_L27_processed.csv # Cleaned & normalized dataset
│ └─ scaler.pkl # StandardScaler object for reproducibility
```
---

## 6. Preprocessing Notes

Before using the data for modeling:
1. **Convert units** consistently (e.g., all in mm, N, m/min).  
2. **Normalize** continuous variables with `StandardScaler` or `MinMaxScaler`.  
3. **Split** data for training and testing (e.g., 80 / 20).  
4. **Save** processed results in `data/processed/`.

---