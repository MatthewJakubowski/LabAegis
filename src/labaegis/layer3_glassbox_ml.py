"""
Layer 3: Glass-Box ML Guard (EFLM Domain Feature Engineering + Classical ML + Mandatory TreeSHAP).
Strictly enclosed inside Layer 2 Containment & Kill-Switch governance.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .layer0_privacy import AuditLedger, DecisionStatus
from .layer1_ontology import NormalizedObservation
from .layer2_containment import (
    AnalyteMetrologySpec,
    ContainmentVerdict,
    DeterministicContainmentCage,
)

CORE_ML_ANALYTES = ["K", "NA", "CA", "MG", "ALP", "GLU", "CREA", "HGB", "MCV", "RDW"]

ARTIFACT_CLASSES = {
    0: ("CLEAN_BIOLOGICAL", "Czysta zmienność biologiczna", "Clean Biological Variation"),
    1: ("SUBTLE_EDTA", "Subtelna kontaminacja K-EDTA", "Subtle K-EDTA Contamination"),
    2: ("SUBTLE_IV_DILUTION", "Ciche rozcieńczenie kroplówką (IV)", "Subtle IV Fluid Dilution"),
    3: ("SUBTLE_WBIT", "Podejrzenie zamiany probówki (WBIT)", "Suspected Wrong Blood in Tube (WBIT)"),
}


def compute_domain_features(
    current: Dict[str, float],
    baseline: Dict[str, float],
    metrology_table: Dict[str, AnalyteMetrologySpec],
) -> Dict[str, float]:
    """Transforms current/baseline analyte pairs into EFLM Z-deltas and physiological cross-indices."""
    feats: Dict[str, float] = {}
    bio_z: Dict[str, float] = {}

    for code in CORE_ML_ANALYTES:
        curr_v = float(current[code])
        base_v = max(float(baseline[code]), 1e-6)
        delta_pct = ((curr_v - base_v) / base_v) * 100.0
        spec = metrology_table[code]
        total_cv = math.sqrt(spec.cv_i_pct**2 + spec.cv_a_pct**2)
        z_delta = delta_pct / max(total_cv, 0.1)
        feats[f"delta_{code}_pct"] = delta_pct
        feats[f"bio_z_{code}"] = z_delta
        bio_z[code] = z_delta

    k_ca_now = current["K"] / max(current["CA"], 0.1)
    k_ca_base = baseline["K"] / max(baseline["CA"], 0.1)
    feats["edta_k_ca_ratio_shift"] = ((k_ca_now - k_ca_base) / max(k_ca_base, 0.1)) * 100.0
    feats["edta_chelation_index"] = bio_z["K"] - bio_z["CA"] - 0.5 * bio_z["ALP"] - 0.5 * bio_z["MG"]
    feats["iv_dilution_mean_drop"] = float(np.mean([
        feats["delta_HGB_pct"], feats["delta_CA_pct"], feats["delta_CREA_pct"],
        feats["delta_ALP_pct"], feats["delta_MG_pct"],
    ]))
    feats["iv_osmotic_push"] = max(feats["delta_NA_pct"], 0.0) + max(feats["delta_GLU_pct"] * 0.25, 0.0)

    wbit_markers = ["MCV", "RDW", "CREA", "HGB", "ALP"]
    feats["wbit_fingerprint_l2"] = float(np.sqrt(np.mean([bio_z[m] ** 2 for m in wbit_markers])))
    feats["wbit_sign_discordance"] = float(np.std([bio_z[m] for m in wbit_markers]))
    return feats


def _sample_physiological_baseline(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "K":    float(np.clip(rng.normal(4.25, 0.35), 3.4, 5.1)),
        "NA":   float(np.clip(rng.normal(140.0, 2.2), 134.0, 146.0)),
        "CA":   float(np.clip(rng.normal(2.36, 0.09), 2.12, 2.58)),
        "MG":   float(np.clip(rng.normal(0.85, 0.08), 0.65, 1.08)),
        "ALP":  float(np.clip(rng.normal(82.0, 18.0), 42.0, 132.0)),
        "GLU":  float(np.clip(rng.normal(5.05, 0.55), 3.9, 6.8)),
        "CREA": float(np.clip(rng.normal(82.0, 13.0), 55.0, 118.0)),
        "HGB":  float(np.clip(rng.normal(14.3, 1.35), 11.5, 17.5)),
        "MCV":  float(np.clip(rng.normal(88.5, 3.8), 78.0, 98.0)),
        "RDW":  float(np.clip(rng.normal(12.9, 0.75), 11.4, 15.2)),
    }


def _apply_biological_noise(
    base: Dict[str, float],
    rng: np.random.Generator,
    metrology_table: Dict[str, AnalyteMetrologySpec],
) -> Dict[str, float]:
    curr: Dict[str, float] = {}
    hydration_factor = rng.normal(0.0, 0.015)
    for code in CORE_ML_ANALYTES:
        spec = metrology_table[code]
        sigma_rel = math.sqrt(spec.cv_i_pct**2 + spec.cv_a_pct**2) / 100.0
        noise = rng.normal(0.0, sigma_rel)
        shift = noise + (hydration_factor if code in {"HGB", "CREA", "CA", "NA"} else 0.0)
        curr[code] = round(base[code] * (1.0 + shift), 3)
    return curr


def generate_labaegis_cohort(n_samples: int = 4000, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    metro = DeterministicContainmentCage.METROLOGY_TABLE
    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    class_choices = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.55, 0.15, 0.15, 0.15])

    for cls in class_choices:
        base = _sample_physiological_baseline(rng)
        curr = _apply_biological_noise(base, rng, metro)
        if cls == 1:
            severity = rng.uniform(0.08, 0.24)
            curr["K"] = round(min(curr["K"] * (1.0 + severity * 1.05), 5.90), 3)
            curr["CA"] = round(max(curr["CA"] * (1.0 - severity * 0.70), 1.62), 3)
            curr["MG"] = round(max(curr["MG"] * (1.0 - severity * 0.55), 0.48), 3)
            curr["ALP"] = round(max(curr["ALP"] * (1.0 - severity * 0.90), 36.0), 3)
        elif cls == 2:
            dil = rng.uniform(0.08, 0.20)
            is_dextrose = rng.random() < 0.45
            for c in ["HGB", "CA", "MG", "ALP", "CREA", "K"]:
                curr[c] = round(curr[c] * (1.0 - dil), 3)
            if is_dextrose:
                curr["GLU"] = round(curr["GLU"] * rng.uniform(1.18, 1.65), 3)
                curr["NA"] = round(curr["NA"] * (1.0 - dil * 0.35), 3)
            else:
                curr["NA"] = round(curr["NA"] * (1.0 - dil) + 154.0 * dil, 3)
        elif cls == 3:
            other_patient = _sample_physiological_baseline(rng)
            curr = _apply_biological_noise(other_patient, rng, metro)

        rows.append(compute_domain_features(curr, base, metro))
        labels.append(int(cls))

    return pd.DataFrame(rows), np.array(labels)


def train_reference_models(n_samples: int = 3000, seed: int = 42) -> Tuple[XGBClassifier, RandomForestClassifier, List[str]]:
    X_df, y_arr = generate_labaegis_cohort(n_samples=n_samples, seed=seed)
    xgb = XGBClassifier(
        n_estimators=180, max_depth=5, learning_rate=0.07, subsample=0.85,
        colsample_bytree=0.85, eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=160, max_depth=12, min_samples_leaf=3, random_state=seed, n_jobs=-1,
    )
    xgb.fit(X_df, y_arr)
    rf.fit(X_df, y_arr)
    return xgb, rf, list(X_df.columns)


class XAIExplanationItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    feature_value: float
    shap_contribution: float


class MLGuardAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    specimen_id: str
    pseudonym_id: str
    model_name: str
    executed: bool
    final_status: DecisionStatus
    predicted_class_code: str
    predicted_class_pl: str
    predicted_class_en: str
    class_probabilities: Dict[str, float]
    top_shap_drivers: List[XAIExplanationItem]
    message_pl: str
    message_en: str


class GlassBoxMLGuard:
    """Explainable ML layer strictly governed by Layer 2 Containment and HITL ceiling."""

    def __init__(
        self,
        primary_model: Any,
        fallback_tree_model: Any,
        feature_columns: List[str],
        ledger: AuditLedger,
        artifact_threshold: float = 0.45,
        uncertainty_floor: float = 0.75,
    ) -> None:
        self.feature_columns = feature_columns
        self.ledger = ledger
        self.artifact_threshold = artifact_threshold
        self.uncertainty_floor = uncertainty_floor

        try:
            self.explainer = shap.TreeExplainer(primary_model)
            self.model = primary_model
            self.model_name = type(primary_model).__name__
        except Exception:
            self.explainer = shap.TreeExplainer(fallback_tree_model)
            self.model = fallback_tree_model
            self.model_name = f"{type(fallback_tree_model).__name__} (TreeSHAP-Verified)"

    @staticmethod
    def _extract_class_shap(raw_shap: Any, class_idx: int) -> np.ndarray:
        if isinstance(raw_shap, list):
            return np.asarray(raw_shap[class_idx])[0]
        arr = np.asarray(raw_shap)
        if arr.ndim == 3:
            return arr[0, :, class_idx]
        return arr[0]

    def evaluate(
        self,
        observations: List[NormalizedObservation],
        cage_verdict: ContainmentVerdict,
        baseline_values: Dict[str, float],
    ) -> MLGuardAssessment:
        if not cage_verdict.allow_layer3_ml:
            msg_pl = f"BLOKADA POWSTRZYMANIA: Warstwa 3 ML zablokowana (status Warstwy 2: {cage_verdict.final_status.value})."
            msg_en = f"CONTAINMENT BLOCK: Layer 3 ML denied (Layer 2 status: {cage_verdict.final_status.value})."
            self.ledger.record(
                stage="LAYER_3_GLASSBOX_ML_GUARD",
                decision=cage_verdict.final_status,
                legal_basis="EU AI Act Art. 14 / Containment Architecture Enforcement",
                reason=msg_pl,
                payload_for_hash={"specimen_id": cage_verdict.specimen_id, "executed": False},
            )
            return MLGuardAssessment(
                specimen_id=cage_verdict.specimen_id,
                pseudonym_id=cage_verdict.pseudonym_id,
                model_name=self.model_name,
                executed=False,
                final_status=cage_verdict.final_status,
                predicted_class_code="BLOCKED_BY_LAYER_2",
                predicted_class_pl="Zablokowano przez Klatkę Deterministyczną",
                predicted_class_en="Blocked by Deterministic Cage",
                class_probabilities={},
                top_shap_drivers=[],
                message_pl=msg_pl,
                message_en=msg_en,
            )

        curr_map = {o.canonical_code: o.canonical_value for o in observations}
        missing = [c for c in CORE_ML_ANALYTES if c not in curr_map or c not in baseline_values]
        if missing:
            raise ValueError(f"Missing analytes for Layer 3 ML: {missing}")

        feats = compute_domain_features(curr_map, baseline_values, DeterministicContainmentCage.METROLOGY_TABLE)
        X_one = pd.DataFrame([feats])[self.feature_columns]

        probs = self.model.predict_proba(X_one)[0]
        prob_dict = {ARTIFACT_CLASSES[i][0]: round(float(probs[i]), 4) for i in range(4)}

        artifact_probs = probs[1:]
        max_artifact_idx = int(np.argmax(artifact_probs)) + 1
        max_artifact_prob = float(probs[max_artifact_idx])
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])

        explain_idx = max_artifact_idx if max_artifact_prob >= self.artifact_threshold else top_idx
        raw_shap = self.explainer.shap_values(X_one)
        class_shap = self._extract_class_shap(raw_shap, explain_idx)

        order = np.argsort(np.abs(class_shap))[::-1][:6]
        top_drivers = [
            XAIExplanationItem(
                feature=self.feature_columns[idx],
                feature_value=round(float(X_one.iloc[0, idx]), 3),
                shap_contribution=round(float(class_shap[idx]), 4),
            )
            for idx in order
        ]

        cls_code, cls_pl, cls_en = ARTIFACT_CLASSES[explain_idx]

        if max_artifact_prob >= self.artifact_threshold:
            final_status = DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
            msg_pl = f"ALERT XAI (HITL): Wykryto wzorzec '{cls_pl}' ({max_artifact_prob*100:.1f}%). Wymagana weryfikacja diagnosty."
            msg_en = f"XAI ALERT (HITL): Pattern detected '{cls_en}' ({max_artifact_prob*100:.1f}%). Diagnostician review required."
            legal = "EU AI Act Art. 13 (XAI) & Art. 14 (HITL) / Ustawa o medycynie laboratoryjnej"
        elif top_prob < self.uncertainty_floor:
            final_status = DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
            msg_pl = f"NIEPEWNOŚĆ EPISTEMICZNA (HITL): Pewność modelu ({top_prob*100:.1f}%) poniżej progu bezpieczeństwa."
            msg_en = f"EPISTEMIC UNCERTAINTY (HITL): Model confidence ({top_prob*100:.1f}%) below safety floor."
            legal = "EU AI Act Art. 9 & Art. 14 (Uncertainty Escalation)"
        else:
            final_status = DecisionStatus.INGESTED_CLEAN
            msg_pl = f"AUDYT PRZEDANALITYCZNY CZYSTY: Brak wielowymiarowych artefaktów ({probs[0]*100:.1f}%)."
            msg_en = f"PRE-ANALYTICAL AUDIT CLEAR: No silent multivariate artifact detected ({probs[0]*100:.1f}%)."
            legal = "ISO 15189:2022 / Pre-Analytical Quality Assurance"

        assessment = MLGuardAssessment(
            specimen_id=cage_verdict.specimen_id,
            pseudonym_id=cage_verdict.pseudonym_id,
            model_name=self.model_name,
            executed=True,
            final_status=final_status,
            predicted_class_code=cls_code,
            predicted_class_pl=cls_pl,
            predicted_class_en=cls_en,
            class_probabilities=prob_dict,
            top_shap_drivers=top_drivers,
            message_pl=msg_pl,
            message_en=msg_en,
        )

        self.ledger.record(
            stage="LAYER_3_GLASSBOX_ML_GUARD",
            decision=final_status,
            legal_basis=legal,
            reason=msg_pl,
            payload_for_hash={
                "specimen_id": assessment.specimen_id,
                "model": self.model_name,
                "predicted_class": cls_code,
                "probabilities": prob_dict,
                "top_shap": [d.model_dump() for d in top_drivers[:3]],
            },
        )
        return assessment


def plot_xai_audit_card(
    assessment: MLGuardAssessment,
    title_suffix: str = "",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> bool:
    """Publication-grade XAI Audit Card with non-overlapping legend & defensive Containment Guard."""
    if not assessment.executed or not assessment.class_probabilities:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.9), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor("#0F172A")

    labels_short = ["Clean", "Subtle EDTA", "IV Dilution", "Subtle WBIT"]
    keys = ["CLEAN_BIOLOGICAL", "SUBTLE_EDTA", "SUBTLE_IV_DILUTION", "SUBTLE_WBIT"]
    vals = [assessment.class_probabilities[k] * 100.0 for k in keys]
    colors = [
        "#10B981" if k == "CLEAN_BIOLOGICAL"
        else ("#F43F5E" if assessment.predicted_class_code == k else "#475569")
        for k in keys
    ]

    ax1.set_facecolor("#1E293B")
    bars = ax1.barh(labels_short[::-1], vals[::-1], color=colors[::-1], height=0.50)
    ax1.axvline(45.0, color="#F59E0B", linestyle="--", linewidth=1.4, label="Próg HITL / HITL Threshold (45%)")
    ax1.set_xlim(0, 115)
    ax1.set_ylim(-0.55, 4.05)
    ax1.set_xlabel("Prawdopodobieństwo modelu / Probability (%)", color="#E2E8F0", fontsize=9.5)
    ax1.set_title("Rozkład prawdopodobieństw (Layer 3)", color="#F8FAFC", fontsize=11, fontweight="bold", pad=10)
    ax1.tick_params(colors="#CBD5E1")
    ax1.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F59E0B", loc="upper right", fontsize=8.5)
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    for bar, v in zip(bars, vals[::-1]):
        ax1.text(v + 1.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", color="#F8FAFC", fontsize=9, fontweight="bold")

    drivers = assessment.top_shap_drivers[::-1]
    feat_labels = [f"{d.feature}\n(val={d.feature_value:+.2f})" for d in drivers]
    shap_vals = [d.shap_contribution for d in drivers]
    shap_colors = ["#F43F5E" if s > 0 else "#38BDF8" for s in shap_vals]

    ax2.set_facecolor("#1E293B")
    ax2.barh(feat_labels, shap_vals, color=shap_colors, height=0.56)
    ax2.axvline(0.0, color="#94A3B8", linewidth=1.0)
    ax2.set_xlabel(f"Wpływ TreeSHAP na '{assessment.predicted_class_code}'", color="#E2E8F0", fontsize=9.5)
    ax2.set_title("Wyjaśnialność TreeSHAP (EU AI Act Art. 13)", color="#F8FAFC", fontsize=11, fontweight="bold", pad=10)
    ax2.tick_params(colors="#CBD5E1", labelsize=8.5)
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    fig.suptitle(
        f"LabAegis XAI Audit Card — {assessment.specimen_id} {title_suffix} | {assessment.final_status.value}",
        color="#38BDF8", fontsize=12.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return True
