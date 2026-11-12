# Uncertainty-aware MGMT prediction from brain MRI (pilot)

<!-- Write every section below in your own words. The prompts in brackets are only guides — delete them. -->

## Why this matters
[2–3 sentences for a non-expert: what glioblastoma is, why MGMT status decides whether temozolomide helps, and why waiting weeks for a lab result is a problem.]

## What this project does
[One sentence: a model that predicts MGMT status from MRI **and** says "uncertain – refer to lab" when it should not be trusted.]

## Method
1. Data: RSNA-MICCAI Brain Tumor Radiogenomic Classification (Kaggle, 2021), [N] patients, [SEQUENCE] sequence.
2. Preprocessing: [how many slices, how chosen, image size].
3. Model: 5-model deep ensemble of ImageNet-pretrained ResNet18 (5-fold CV, patient-level split).
4. Uncertainty: ensemble disagreement + split conformal prediction (α = 0.10).
5. Stress test: noise, blur, darkening.
6. Explanation: Grad-CAM.

## Results
| Metric | Value |
|---|---|
| Test AUC | |
| Calibration error (ECE) | |
| Conformal coverage (target 0.90) | |
| Share of patients answered confidently | |
| Accuracy on confident patients | |
| Accuracy if answering everyone | |

![Risk–coverage and calibration](figures/risk_coverage_and_calibration.png)
![Stress test](figures/stress_test.png)
![Grad-CAM](figures/gradcam.png)

[2–4 sentences: what do the plots show? Did accuracy rise when the model was allowed to abstain? Did uncertainty rise under stress?]

## Limitations
[Be honest: single sequence, 2D slices, one dataset, small test set, AUC close to chance as in the original challenge.]

## Next steps
[Link to your PhD proposal idea: multi-sequence 3D models, clinical data fusion, external validation (UCSF-PDGM, UPenn-GBM), runtime monitoring.]

## How to run
Open `mgmt_uncertainty_pilot.ipynb` on Kaggle (add the competition data, GPU T4, Internet on) or on Colab (set `PLATFORM = "colab"`, upload `kaggle.json`). Run all cells; results go to `results.json` and `figures/`.

## Related work by me
- LLM confidence pilot: https://github.com/nishikant1744/llm-confidence-pilot
