# EU AI Act Model Card — LabAegis Layer 3 (`GlassBoxMLGuard`)

* **Model Name:** LabAegis Silent Pre-Analytical Artifact Guard (`v0.1.0`)
* **Architecture:** Classical Tabular Tree Ensemble (`XGBClassifier` primary + `RandomForestClassifier` TreeSHAP-verified fallback)
* **Author & Clinical Domain Architect:** Mateusz Jakubowski (*From Pipette to Python*)
* **Intended Purpose:** Detection of **sub-threshold, multivariate pre-analytical specimen integrity errors** (subtle K-EDTA carryover, subtle IV fluid dilution, and suspected Wrong Blood in Tube / WBIT) that remain invisible to single-analyte reference ranges and univariate RCV99 delta checks.
* **Out-of-Scope Uses:** Clinical diagnosis, therapeutic dosing, triage, or autonomous laboratory result authorization.

## 1. Domain Feature Engineering (Explainable by Design)
Instead of raw opaque embeddings, the model consumes 26 auditable laboratory-medicine features derived from 10 core biomarkers (`K, NA, CA, MG, ALP, GLU, CREA, HGB, MCV, RDW`):
* Relative percentage shifts (`delta_*_pct`) and **EFLM biologically normalized Z-deltas** (`bio_z_*` scaled by $\sqrt{CV_I^2 + CV_A^2}$).
* Relational physics indices: `edta_chelation_index`, `edta_k_ca_ratio_shift`, `iv_dilution_mean_drop`, `iv_osmotic_push`, `wbit_fingerprint_l2`, and `wbit_sign_discordance`.

## 2. Benchmark Performance (Hold-Out Stratified Test Set, n = 800)
* **Logistic Regression (Linear Baseline):** Macro F1 = `0.9759` | Macro ROC-AUC = `0.9980`
* **Random Forest (Bagging Glass-Box):** Macro F1 = `0.9784` | Macro ROC-AUC = `0.9978`
* **XGBoost (Champion):** Macro F1 = `0.9810` | Macro ROC-AUC = `0.9989`

## 3. Containment & Human Oversight Safeguards (EU AI Act Art. 13 & 14)
1. **Upstream Deterministic Gate:** Layer 3 is physically barred from executing if Layer 2 (`L1-L5`) raises a hard stop or HITL hold.
2. **Mandatory Local XAI:** Every Layer 3 inference computes exact `TreeSHAP` attributions logged to the SHA-256 `AuditLedger`.
3. **Epistemic Uncertainty Floor:** Any prediction with top-class confidence `< 75%` is automatically escalated to `HOLD_FOR_DIAGNOSTICIAN_REVIEW`.
4. **Emergency Kill-Switch:** Operators can instantly isolate Layer 3 (`set_ml_kill_switch(True)`) while retaining full deterministic `L1-L5` protection.

## 4. Known Limitations & Clinical Validation Roadmap
The reference weights in `v0.1.0` are trained on a physiologically grounded synthetic cohort calibrated to EFLM biological variation. Before any clinical pilot, site-specific calibration and retrospective validation on local, IRB-approved, anonymized LIS data are mandatory.
