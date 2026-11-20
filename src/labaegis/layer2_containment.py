"""
Layer 2: Deterministic Containment Cage (Cascade L1-L5 + EFLM RCV + EU AI Act Kill-Switch).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .layer0_privacy import AuditLedger, DecisionStatus
from .layer1_ontology import LabAegisOntology, NormalizedObservation, PreAnalyticalRisk


class SpecimenPreAnalyticalMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    hemolysis_index: float = Field(default=0.0, ge=0.0)
    icterus_index: float = Field(default=0.0, ge=0.0)
    lipemia_index: float = Field(default=0.0, ge=0.0)
    centrifugation_delay_min: int = Field(default=30, ge=0)
    clot_detected: bool = False
    qc_westgard_passed: bool = True


class AnalyteMetrologySpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref_low: float
    ref_high: float
    crit_low: Optional[float] = None
    crit_high: Optional[float] = None
    cv_i_pct: float
    cv_a_pct: float

    def rcv_pct(self, z_score: float = 1.96) -> float:
        return round(math.sqrt(2.0) * z_score * math.sqrt(self.cv_a_pct**2 + self.cv_i_pct**2), 2)


class CascadeTrigger(BaseModel):
    model_config = ConfigDict(frozen=True)

    level: str
    severity: str
    analyte_codes: Tuple[str, ...]
    rule_id: str
    message_pl: str
    message_en: str


class ContainmentVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    specimen_id: str
    pseudonym_id: str
    final_status: DecisionStatus
    allow_layer3_ml: bool
    kill_switch_active: bool
    triggers: List[CascadeTrigger]
    rcv_metrics: Dict[str, Dict[str, float]]
    summary_pl: str
    summary_en: str


class DeterministicContainmentCage:
    """L1-L5 deterministic autoverification and containment cascade."""

    METROLOGY_TABLE: Dict[str, AnalyteMetrologySpec] = {
        "K":     AnalyteMetrologySpec(ref_low=3.5,   ref_high=5.1,   crit_low=2.8,  crit_high=6.2,   cv_i_pct=4.1,  cv_a_pct=1.5),
        "NA":    AnalyteMetrologySpec(ref_low=136.0, ref_high=145.0, crit_low=120.0, crit_high=160.0, cv_i_pct=0.5,  cv_a_pct=0.8),
        "CL":    AnalyteMetrologySpec(ref_low=98.0,  ref_high=107.0, crit_low=80.0, crit_high=120.0, cv_i_pct=1.1,  cv_a_pct=1.2),
        "CA":    AnalyteMetrologySpec(ref_low=2.15,  ref_high=2.55,  crit_low=1.50, crit_high=3.25,  cv_i_pct=1.8,  cv_a_pct=1.5),
        "MG":    AnalyteMetrologySpec(ref_low=0.66,  ref_high=1.07,  crit_low=0.40, crit_high=2.00,  cv_i_pct=2.9,  cv_a_pct=2.0),
        "GLU":   AnalyteMetrologySpec(ref_low=3.9,   ref_high=5.5,   crit_low=2.2,  crit_high=25.0,  cv_i_pct=4.9,  cv_a_pct=2.0),
        "CREA":  AnalyteMetrologySpec(ref_low=62.0,  ref_high=106.0, crit_low=None, crit_high=500.0, cv_i_pct=4.5,  cv_a_pct=2.5),
        "UREA":  AnalyteMetrologySpec(ref_low=2.8,   ref_high=8.1,   crit_low=None, crit_high=35.0,  cv_i_pct=13.9, cv_a_pct=3.0),
        "URIC":  AnalyteMetrologySpec(ref_low=202.0, ref_high=416.0, crit_low=None, crit_high=750.0, cv_i_pct=8.3,  cv_a_pct=2.5),
        "ALT":   AnalyteMetrologySpec(ref_low=5.0,   ref_high=41.0,  crit_low=None, crit_high=1000.0, cv_i_pct=10.1, cv_a_pct=3.5),
        "AST":   AnalyteMetrologySpec(ref_low=5.0,   ref_high=40.0,  crit_low=None, crit_high=1000.0, cv_i_pct=9.5,  cv_a_pct=3.5),
        "ALP":   AnalyteMetrologySpec(ref_low=40.0,  ref_high=129.0, crit_low=None, crit_high=1000.0, cv_i_pct=5.4,  cv_a_pct=3.0),
        "GGTP":  AnalyteMetrologySpec(ref_low=8.0,   ref_high=61.0,  crit_low=None, crit_high=1000.0, cv_i_pct=8.9,  cv_a_pct=3.0),
        "BIL_T": AnalyteMetrologySpec(ref_low=3.4,   ref_high=20.5,  crit_low=None, crit_high=250.0,  cv_i_pct=20.9, cv_a_pct=3.5),
        "CRP":   AnalyteMetrologySpec(ref_low=0.0,   ref_high=5.0,   crit_low=None, crit_high=200.0,  cv_i_pct=34.1, cv_a_pct=4.0),
        "TSH":   AnalyteMetrologySpec(ref_low=0.27,  ref_high=4.20,  crit_low=None, crit_high=50.0,   cv_i_pct=17.7, cv_a_pct=4.0),
        "CHOL":  AnalyteMetrologySpec(ref_low=3.0,   ref_high=5.0,   crit_low=None, crit_high=15.0,   cv_i_pct=5.3,  cv_a_pct=2.0),
        "TG":    AnalyteMetrologySpec(ref_low=0.4,   ref_high=1.7,   crit_low=None, crit_high=15.0,   cv_i_pct=19.8, cv_a_pct=2.5),
        "HGB":   AnalyteMetrologySpec(ref_low=12.0,  ref_high=17.5,  crit_low=7.0,  crit_high=21.0,   cv_i_pct=2.8,  cv_a_pct=1.2),
        "WBC":   AnalyteMetrologySpec(ref_low=4.0,   ref_high=10.0,  crit_low=1.5,  crit_high=50.0,   cv_i_pct=10.8, cv_a_pct=2.5),
        "RBC":   AnalyteMetrologySpec(ref_low=4.0,   ref_high=5.8,   crit_low=2.0,  crit_high=7.5,    cv_i_pct=2.8,  cv_a_pct=1.5),
        "PLT":   AnalyteMetrologySpec(ref_low=150.0, ref_high=400.0, crit_low=30.0, crit_high=1000.0, cv_i_pct=7.2,  cv_a_pct=3.5),
        "MCV":   AnalyteMetrologySpec(ref_low=80.0,  ref_high=96.0,  crit_low=55.0, crit_high=125.0,  cv_i_pct=0.8,  cv_a_pct=1.0),
        "RDW":   AnalyteMetrologySpec(ref_low=11.5,  ref_high=14.5,  crit_low=None, crit_high=30.0,   cv_i_pct=1.7,  cv_a_pct=2.0),
    }

    def __init__(self, ontology: LabAegisOntology, ledger: AuditLedger) -> None:
        self.ontology = ontology
        self.ledger = ledger
        self._ml_kill_switch_active: bool = False

    def set_ml_kill_switch(self, active: bool, operator_id: str, reason: str) -> None:
        self._ml_kill_switch_active = active
        state_pl = "AKTYWOWANY" if active else "DEZAKTYWOWANY"
        self.ledger.record(
            stage="LAYER_2_GOVERNANCE_KILL_SWITCH",
            decision=DecisionStatus.KILL_SWITCH_ACTIVE if active else DecisionStatus.INGESTED_CLEAN,
            legal_basis="EU AI Act Art. 14(4)(e) (Human Stop / Kill-Switch Mechanism)",
            reason=f"Kill-Switch ML {state_pl} przez operatora '{operator_id}': {reason}",
            payload_for_hash={"active": active, "operator_id": operator_id, "reason": reason},
        )

    def evaluate_panel(
        self,
        observations: List[NormalizedObservation],
        meta: SpecimenPreAnalyticalMeta,
        baseline_values: Optional[Dict[str, float]] = None,
    ) -> ContainmentVerdict:
        if not observations:
            raise ValueError("Panel must contain at least one NormalizedObservation.")

        specimen_id = observations[0].specimen_id
        pseudonym_id = observations[0].pseudonym_id
        baseline = baseline_values or {}

        triggers: List[CascadeTrigger] = []
        rcv_metrics: Dict[str, Dict[str, float]] = {}
        panel_map: Dict[str, NormalizedObservation] = {o.canonical_code: o for o in observations}

        if not meta.qc_westgard_passed:
            triggers.append(CascadeTrigger(
                level="L1_TECHNICAL", severity="HARD_STOP", analyte_codes=tuple(panel_map.keys()),
                rule_id="L1_WESTGARD_QC_FAIL",
                message_pl="L1 STOP: Niespełniona kontrola jakości analizatora (reguły Westgarda).",
                message_en="L1 STOP: Analyzer Westgard QC failure.",
            ))

        if meta.clot_detected:
            triggers.append(CascadeTrigger(
                level="L1_TECHNICAL", severity="HARD_STOP", analyte_codes=tuple(panel_map.keys()),
                rule_id="L1_SPECIMEN_CLOT",
                message_pl="L1 STOP: Wykryto skrzep / mikroskrzep w próbce. Konieczne ponowne pobranie.",
                message_en="L1 STOP: Clot/microclot detected in specimen. Redraw required.",
            ))

        for code, obs in panel_map.items():
            concept = self.ontology.concepts[code]

            if obs.status == DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW:
                triggers.append(CascadeTrigger(
                    level="L1_TECHNICAL", severity="HITL_HOLD", analyte_codes=(code,),
                    rule_id="L1_ONTOLOGY_UNCONFIRMED",
                    message_pl=f"L1 HITL: Parametr {code} ma niepotwierdzone mapowanie z Warstwy 1.",
                    message_en=f"L1 HITL: Analyte {code} carries unconfirmed Layer 1 mapping.",
                ))

            low_p, high_p = concept.plausibility_bounds
            if not (low_p <= obs.canonical_value <= high_p):
                triggers.append(CascadeTrigger(
                    level="L1_TECHNICAL", severity="HARD_STOP", analyte_codes=(code,),
                    rule_id="L1_IMPLAUSIBLE_VALUE",
                    message_pl=f"L1 STOP: Wartość fizjologicznie niemożliwa dla {concept.name_pl}: {obs.canonical_value} {obs.canonical_unit}.",
                    message_en=f"L1 STOP: Physiologically implausible value for {concept.name_en}: {obs.canonical_value} {obs.canonical_unit}.",
                ))

            if meta.hemolysis_index >= 50.0 and PreAnalyticalRisk.HEMOLYSIS in concept.preanalytical_links:
                triggers.append(CascadeTrigger(
                    level="L1_TECHNICAL", severity="HARD_STOP", analyte_codes=(code,),
                    rule_id="L1_HEMOLYSIS_ONTOLOGY_LINK",
                    message_pl=f"L1 STOP: Hemoliza (H={meta.hemolysis_index}) interferuje z podatnym parametrem {concept.name_pl} [link ontologii: HEMOLYSIS].",
                    message_en=f"L1 STOP: Hemolysis (H={meta.hemolysis_index}) interferes with {concept.name_en} [ontology link: HEMOLYSIS].",
                ))

            if meta.lipemia_index >= 150.0 and PreAnalyticalRisk.LIPEMIA_INTERFERENCE in concept.preanalytical_links:
                triggers.append(CascadeTrigger(
                    level="L1_TECHNICAL", severity="HITL_HOLD", analyte_codes=(code,),
                    rule_id="L1_LIPEMIA_ONTOLOGY_LINK",
                    message_pl=f"L1 HITL: Lipemia (L={meta.lipemia_index}) może zaburzać pomiar {concept.name_pl}.",
                    message_en=f"L1 HITL: Lipemia (L={meta.lipemia_index}) may interfere with {concept.name_en}.",
                ))

            if meta.centrifugation_delay_min > 120 and PreAnalyticalRisk.GLYCOLYSIS_DELAY in concept.preanalytical_links:
                triggers.append(CascadeTrigger(
                    level="L1_TECHNICAL", severity="HITL_HOLD", analyte_codes=(code,),
                    rule_id="L1_GLYCOLYSIS_DELAY",
                    message_pl=f"L1 HITL: Opóźnienie odwirowania ({meta.centrifugation_delay_min} min) -> ryzyko glikolizy in vitro ({concept.name_pl}).",
                    message_en=f"L1 HITL: Centrifugation delay ({meta.centrifugation_delay_min} min) -> in vitro glycolysis risk ({concept.name_en}).",
                ))

            spec = self.METROLOGY_TABLE[code]
            if (spec.crit_low is not None and obs.canonical_value < spec.crit_low) or (
                spec.crit_high is not None and obs.canonical_value > spec.crit_high
            ):
                triggers.append(CascadeTrigger(
                    level="L2_CRITICAL", severity="HARD_STOP", analyte_codes=(code,),
                    rule_id="L2_CRITICAL_VALUE",
                    message_pl=f"L2 STOP [KRYTYCZNY]: {concept.name_pl} = {obs.canonical_value} {obs.canonical_unit} przekracza próg krytyczny ({spec.crit_low} / {spec.crit_high}).",
                    message_en=f"L2 STOP [CRITICAL]: {concept.name_en} = {obs.canonical_value} {obs.canonical_unit} breaches panic threshold ({spec.crit_low} / {spec.crit_high}).",
                ))

            in_ref_now = spec.ref_low <= obs.canonical_value <= spec.ref_high
            if not in_ref_now:
                triggers.append(CascadeTrigger(
                    level="L3_REFERENCE", severity="INFO", analyte_codes=(code,),
                    rule_id="L3_OUTSIDE_REFERENCE",
                    message_pl=f"L3 INFO: {concept.name_pl} ({obs.canonical_value} {obs.canonical_unit}) poza zakresem ref. [{spec.ref_low}-{spec.ref_high}].",
                    message_en=f"L3 INFO: {concept.name_en} ({obs.canonical_value} {obs.canonical_unit}) outside ref. range [{spec.ref_low}-{spec.ref_high}].",
                ))

            if code in baseline and baseline[code] > 0:
                prev_val = baseline[code]
                pct_change = round(((obs.canonical_value - prev_val) / prev_val) * 100.0, 2)
                rcv_95 = spec.rcv_pct(z_score=1.96)
                rcv_99 = spec.rcv_pct(z_score=2.58)
                rcv_metrics[code] = {
                    "prev": prev_val, "curr": obs.canonical_value,
                    "delta_pct": pct_change, "rcv_95_pct": rcv_95, "rcv_99_pct": rcv_99,
                }
                in_ref_prev = spec.ref_low <= prev_val <= spec.ref_high
                if abs(pct_change) > rcv_99:
                    if in_ref_now and in_ref_prev:
                        triggers.append(CascadeTrigger(
                            level="L4_EFLM_RCV", severity="HITL_HOLD", analyte_codes=(code,),
                            rule_id="L4_SILENT_DRIFT_IN_NORMAL_RANGE",
                            message_pl=(
                                f"L4 HITL [CICHY DRYF W NORMIE]: Zmiana wartości parametru {concept.name_pl} "
                                f"z {prev_val} na {obs.canonical_value} ({pct_change:+.1f}% vs EFLM RCV99={rcv_99:.1f}%), "
                                f"mimo że oba wyniki mieszczą się w normie [{spec.ref_low}-{spec.ref_high}]."
                            ),
                            message_en=(
                                f"L4 HITL [SILENT IN-RANGE DRIFT]: {concept.name_en} shifted {prev_val} -> {obs.canonical_value} "
                                f"({pct_change:+.1f}% vs EFLM RCV99={rcv_99:.1f}%) while both remain inside [{spec.ref_low}-{spec.ref_high}]."
                            ),
                        ))
                    else:
                        triggers.append(CascadeTrigger(
                            level="L4_EFLM_RCV", severity="HITL_HOLD", analyte_codes=(code,),
                            rule_id="L4_RCV99_BREACH",
                            message_pl=f"L4 HITL [DELTA RCV99]: Istotna zmiana metrologiczna {concept.name_pl}: {prev_val} -> {obs.canonical_value} ({pct_change:+.1f}% > RCV99={rcv_99:.1f}%).",
                            message_en=f"L4 HITL [DELTA RCV99]: Significant shift in {concept.name_en}: {prev_val} -> {obs.canonical_value} ({pct_change:+.1f}% > RCV99={rcv_99:.1f}%).",
                        ))

        if "K" in panel_map and "CA" in panel_map:
            k_val = panel_map["K"].canonical_value
            ca_val = panel_map["CA"].canonical_value
            alp_note = ""
            affected = ["K", "CA"]
            if "ALP" in panel_map and panel_map["ALP"].canonical_value < 35.0:
                alp_note = f" + spadek ALP ({panel_map['ALP'].canonical_value} U/L)"
                affected.append("ALP")
            if k_val >= 6.0 and ca_val <= 1.5:
                triggers.append(CascadeTrigger(
                    level="L5_CROSS_CHECK", severity="HARD_STOP", analyte_codes=tuple(affected),
                    rule_id="L5_OVERT_EDTA_CONTAMINATION",
                    message_pl=f"L5 STOP [KONTAMINACJA EDTA]: Klasyczna sygnatura chelatacji! K+={k_val} mmol/L przy Ca2+={ca_val} mmol/L{alp_note}.",
                    message_en=f"L5 STOP [EDTA CONTAMINATION]: Classic chelation signature! K+={k_val} mmol/L with Ca2+={ca_val} mmol/L{alp_note}.",
                ))

        if "HGB" in panel_map and "RBC" in panel_map and panel_map["RBC"].canonical_value > 0:
            mch = (panel_map["HGB"].canonical_value * 10.0) / panel_map["RBC"].canonical_value
            if not (15.0 <= mch <= 45.0):
                triggers.append(CascadeTrigger(
                    level="L5_CROSS_CHECK", severity="HARD_STOP", analyte_codes=("HGB", "RBC"),
                    rule_id="L5_HEMATOLOGY_MCH_IMPOSSIBLE",
                    message_pl=f"L5 STOP [NIESPÓJNOŚĆ MORFOLOGII]: Wyliczone MCH={mch:.1f} pg poza granicą fizjologii (15-45 pg).",
                    message_en=f"L5 STOP [CBC INCONSISTENCY]: Derived MCH={mch:.1f} pg outside physiological bounds (15-45 pg).",
                ))

        has_hard_stop = any(t.severity == "HARD_STOP" for t in triggers)
        has_hitl_hold = any(t.severity == "HITL_HOLD" for t in triggers)

        if has_hard_stop:
            final_status = DecisionStatus.STOP_DETERMINISTIC_CAGE
            allow_ml = False
            summary_pl = "TWARDY STOP KLATKI (L1-L5): Próbka zatrzymana deterministycznie. Warstwa ML zablokowana."
            summary_en = "DETERMINISTIC CAGE HARD STOP (L1-L5): Specimen halted. ML layer blocked."
            legal = "ISO 15189:2022 & Ustawa o medycynie laboratoryjnej"
        elif has_hitl_hold:
            final_status = DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
            allow_ml = False
            summary_pl = "WSTRZYMANO DO DECYZJI DIAGNOSTY (HITL): Wykryto odchylenie metrologiczne lub przedanalityczne w L1-L5."
            summary_en = "HELD FOR DIAGNOSTICIAN REVIEW (HITL): Pre-analytical or metrological alert raised in L1-L5."
            legal = "EU AI Act Art. 14 & ISO 15189:2022"
        elif self._ml_kill_switch_active:
            final_status = DecisionStatus.KILL_SWITCH_ACTIVE
            allow_ml = False
            summary_pl = "KLATKA L1-L5 CZYSTA, ALE AKTYWNY KILL-SWITCH: Warstwa ML odcięta na żądanie operatora."
            summary_en = "L1-L5 CLEAN, KILL-SWITCH ENGAGED: ML layer isolated by operator."
            legal = "EU AI Act Art. 14(4)(e) (Active Human Kill-Switch)"
        else:
            final_status = DecisionStatus.INGESTED_CLEAN
            allow_ml = True
            summary_pl = "KLATKA L1-L5 PRZEJŚCIE CZYSTE: Dopuszczono panel do Warstwy 3 (Szklana Skrzynka ML + SHAP)."
            summary_en = "L1-L5 CLEARED: Panel permitted to enter Layer 3 (Glass-Box ML + SHAP)."
            legal = "ISO 15189:2022 / Containment Layer Cleared"

        verdict = ContainmentVerdict(
            specimen_id=specimen_id,
            pseudonym_id=pseudonym_id,
            final_status=final_status,
            allow_layer3_ml=allow_ml,
            kill_switch_active=self._ml_kill_switch_active,
            triggers=triggers,
            rcv_metrics=rcv_metrics,
            summary_pl=summary_pl,
            summary_en=summary_en,
        )

        self.ledger.record(
            stage="LAYER_2_DETERMINISTIC_CAGE",
            decision=final_status,
            legal_basis=legal,
            reason=summary_pl,
            payload_for_hash={
                "specimen_id": specimen_id,
                "allow_layer3_ml": allow_ml,
                "rules_fired": [t.rule_id for t in triggers],
            },
        )
        return verdict
