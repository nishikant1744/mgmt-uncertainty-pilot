# Uncertainty-aware MGMT prediction from brain MRI (pilot)

## Why this matters
Glioblastoma is the most aggressive type of brain cancer. Doctors check a genetic marker called MGMT to decide whether the chemotherapy drug temozolomide is likely to help. At present this needs a tissue sample and a lab test that can take weeks, which delays treatment decisions. Predicting MGMT status from routine MRI scans could give a much faster answer, but only if we can trust the prediction.

## What this project does
This pilot tests whether a deep learning model can predict MGMT status from MRI scans and also say "uncertain – refer to lab" when its prediction should not be trusted.

## Method
1. Data: RSNA-MICCAI Brain Tumor Radiogenomic Classification (Kaggle, 2021), 582 patients, FLAIR sequence.
2. Preprocessing: 12 slices per patient with the most brain tissue, intensity-normalised and resized to 224×224.
3. Model: 5-model deep ensemble of ImageNet-pretrained ResNet18 (5-fold CV, patient-level split).
4. Uncertainty: ensemble disagreement + split conformal prediction (α = 0.10).
5. Stress test: noise, blur, darkening.
6. Explanation: Grad-CAM.

## Results
| Metric | Value |
|---|---|
| Test AUC | 0.63 |
| Calibration error (ECE) | 0.13 |
| Conformal coverage (target 0.90) | 0.97 |
| Share of patients answered confidently | 11% |
| Accuracy on confident patients | 0.70 |
| Accuracy if answering everyone | 0.56 |

![Risk–coverage and calibration](risk_coverage_and_calibration.png)
![Stress test](stress_test.png)
![Grad-CAM](gradcam.png)

When the model answers only the patients it is most confident about, accuracy peaks at about 80% (at around 20% coverage), compared with 56% when it answers everyone. With only 88 test patients this curve is noisy, but it shows that letting the model abstain can make its answers more reliable. However, the stress test revealed a problem: when images were heavily corrupted, AUC dropped to near chance (0.49 with noise σ = 0.1, 0.53 with strong darkening), but the models became more confident, not less, and fewer patients were referred to the lab. Grad-CAM shows that the model sometimes focuses on the tumour, but in other cases it looks at the background or image edges. In one case the model was confident but wrong (p = 0.10 for a methylated tumour), which shows that confidence alone is not a safe signal.

## Limitations
This is a small pilot. Only one MRI sequence (FLAIR) was used, and the model works on 2D slices rather than full 3D scans. The test set has only 88 patients, so all results have wide uncertainty. The overall AUC (0.63) is only slightly better than chance, which is similar to results from the original RSNA-MICCAI challenge. The models also overfit after a few training epochs. AUC varied widely between data splits (0.54 on the calibration set, 0.63 on the test set), so the true performance is uncertain.

## Next steps
The stress-test result shows that ensemble uncertainty alone is not enough to make the model safe. A next step is a runtime monitor that detects unusual or poor-quality scans before the model makes a prediction. Other improvements are using all four MRI sequences, 3D models, adding clinical data such as age, and testing on other datasets such as UCSF-PDGM and UPenn-GBM.

## How to run
Open `mgmt_uncertainty_pilot.ipynb` on Kaggle, add the RSNA-MICCAI competition data, set GPU T4 and Internet on, then Run All. Results are saved to `results.json` and the figure PNGs.

## Related work by me
- LLM confidence pilot: https://github.com/nishikant1744/llm-confidence-pilot
- Grounded medical RAG with answer verification: https://github.com/nishikant1744/grounded-medical-rag
