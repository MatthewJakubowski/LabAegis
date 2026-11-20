"""Comprehensive regulatory, ontological, metrological, and XAI test suite for LabAegis."""

import pytest

from labaegis import (
    AuditLedger,
    DecisionStatus,
    DeterministicContainmentCage,
    GlassBoxMLGuard,
    LabAegisOntology,
    OntologyContainmentError,
    PIIViolationError,
    SpecimenPreAnalyticalMeta,
    ZeroPIIGate,
    train_reference_models,
)


def _base_payload(test_name: str, value: float, unit: str, specimen_id: str = "SPEC_T01", comment: str = "OK"):
    return {
        "pseudonym_id": "PSEUDO_TEST_0001",
        "specimen_id": specimen_id,
        "raw_test_name": test_name,
        "raw_value": value,
        "raw_unit": unit,
        "collected_at_utc": "2026-11-20T20:00:00Z",
        "ward_code": "LAB_QA",
        "comment": comment,
    }


def test_01_zero_pii_blocks_smuggled_pesel():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    with pytest.raises(PIIViolationError):
        gate.ingest(_base_payload("Potas [S]", 4.2, "mmol/L", comment="Pilne PESEL 86053112345"))
    assert ledger.events[-1].decision == DecisionStatus.REJECTED_PII_VIOLATION
    assert ledger.verify_integrity() is True


def test_02_audit_ledger_detects_retroactive_tampering():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    gate.ingest(_base_payload("Potas [S]", 4.2, "mmol/L"))
    assert ledger.verify_integrity() is True
    object.__setattr__(ledger._events[0], "reason", "TAMPERED")
    assert ledger.verify_integrity() is False


def test_03_ontology_24_biomarkers_ucum_and_fhir_r4():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    assert len(onto.concepts) == 24

    obs_crea = onto.harmonize(gate.ingest(_base_payload("Kreatynina [S]", 1.0, "mg/dL")))
    assert obs_crea.canonical_code == "CREA"
    assert obs_crea.canonical_value == 88.4
    assert obs_crea.canonical_unit == "umol/L"

    obs_hgb = onto.harmonize(gate.ingest(_base_payload("Hemoglobina (HGB)", 142.0, "g/L")))
    assert obs_hgb.mapping_method == "DETERMINISTIC_COMPOSITE_CONFIRMED"
    assert obs_hgb.canonical_value == 14.2

    fhir = obs_crea.to_fhir_r4()
    assert fhir["resourceType"] == "Observation"
    assert fhir["code"]["coding"][0]["code"] == "14682-9"


def test_04_ontology_hard_stops_conflicting_lis_label():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    rec = gate.ingest(_base_payload("Potas (NA) [S]", 4.3, "mmol/L"))
    with pytest.raises(OntologyContainmentError):
        onto.harmonize(rec)


def test_05_ontology_holds_wrong_matrix_for_hitl():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    obs = onto.harmonize(gate.ingest(_base_payload("Kreatynina (mocz)", 110.0, "mg/dL")))
    assert obs.status == DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW


def test_06_cage_overt_edta_triggers_hard_stop_blocking_ml():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    cage = DeterministicContainmentCage(onto, ledger)
    obs = [
        onto.harmonize(gate.ingest(_base_payload("Potas [S]", 7.6, "mmol/L"))),
        onto.harmonize(gate.ingest(_base_payload("Wapń całkowity", 0.90, "mmol/L"))),
        onto.harmonize(gate.ingest(_base_payload("ALP", 20.0, "U/L"))),
    ]
    verdict = cage.evaluate_panel(obs, meta=SpecimenPreAnalyticalMeta())
    assert verdict.final_status == DecisionStatus.STOP_DETERMINISTIC_CAGE
    assert verdict.allow_layer3_ml is False
    assert any(t.rule_id == "L5_OVERT_EDTA_CONTAMINATION" for t in verdict.triggers)


def test_07_cage_ontology_driven_selective_hemolysis():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    cage = DeterministicContainmentCage(onto, ledger)
    obs = [
        onto.harmonize(gate.ingest(_base_payload("Potas [S]", 5.3, "mmol/L"))),
        onto.harmonize(gate.ingest(_base_payload("Kreatynina [S]", 82.0, "umol/L"))),
    ]
    verdict = cage.evaluate_panel(obs, meta=SpecimenPreAnalyticalMeta(hemolysis_index=110.0))
    hemo_triggers = [t for t in verdict.triggers if t.rule_id == "L1_HEMOLYSIS_ONTOLOGY_LINK"]
    assert len(hemo_triggers) == 1
    assert hemo_triggers[0].analyte_codes == ("K",)


def test_08_cage_catches_silent_drift_inside_normal_range_eflm_rcv():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    cage = DeterministicContainmentCage(onto, ledger)
    obs = [onto.harmonize(gate.ingest(_base_payload("Kreatynina [S]", 96.0, "umol/L")))]
    verdict = cage.evaluate_panel(obs, meta=SpecimenPreAnalyticalMeta(), baseline_values={"CREA": 65.0})
    assert verdict.final_status == DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
    assert any(t.rule_id == "L4_SILENT_DRIFT_IN_NORMAL_RANGE" for t in verdict.triggers)


def test_09_eu_ai_act_kill_switch_isolates_layer3_ml():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    cage = DeterministicContainmentCage(onto, ledger)
    xgb, rf, cols = train_reference_models(n_samples=600, seed=42)
    ml_guard = GlassBoxMLGuard(xgb, rf, cols, ledger)

    cage.set_ml_kill_switch(True, operator_id="DIAGNOSTA_QA", reason="Scheduled audit")
    obs = [onto.harmonize(gate.ingest(_base_payload("Potas [S]", 4.2, "mmol/L")))]
    verdict = cage.evaluate_panel(obs, meta=SpecimenPreAnalyticalMeta())
    assert verdict.final_status == DecisionStatus.KILL_SWITCH_ACTIVE
    assert verdict.allow_layer3_ml is False

    res = ml_guard.evaluate(obs, verdict, baseline_values={"K": 4.2})
    assert res.executed is False
    assert res.final_status == DecisionStatus.KILL_SWITCH_ACTIVE


def test_10_end_to_end_layer3_catches_sub_rcv_edta_and_wbit_with_shap():
    ledger = AuditLedger()
    gate = ZeroPIIGate(ledger)
    onto = LabAegisOntology(ledger)
    cage = DeterministicContainmentCage(onto, ledger)
    xgb, rf, cols = train_reference_models(n_samples=2500, seed=42)
    ml_guard = GlassBoxMLGuard(xgb, rf, cols, ledger)

    baseline = {
        "K": 4.20, "NA": 140.0, "CA": 2.40, "MG": 0.88, "ALP": 88.0,
        "GLU": 5.00, "CREA": 80.0, "HGB": 14.5, "MCV": 89.0, "RDW": 12.6,
    }
    sub_edta_raw = [
        ("Potas [S]", 4.73, "mmol/L"), ("Sód [S]", 139.8, "mmol/L"),
        ("Wapń całkowity", 2.22, "mmol/L"), ("Magnez [S]", 0.80, "mmol/L"),
        ("ALP", 74.8, "U/L"), ("Glukoza", 5.02, "mmol/L"),
        ("Kreatynina", 80.5, "umol/L"), ("HGB", 14.4, "g/dL"),
        ("MCV", 89.1, "fL"), ("RDW-CV", 12.7, "%"),
    ]
    obs_e = [onto.harmonize(gate.ingest(_base_payload(n, v, u, "SPEC_E2E_EDTA"))) for n, v, u in sub_edta_raw]
    v_e = cage.evaluate_panel(obs_e, meta=SpecimenPreAnalyticalMeta(), baseline_values=baseline)
    assert v_e.allow_layer3_ml is True
    r_e = ml_guard.evaluate(obs_e, v_e, baseline_values=baseline)
    assert r_e.executed is True
    assert r_e.predicted_class_code == "SUBTLE_EDTA"
    assert r_e.final_status == DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
    assert len(r_e.top_shap_drivers) == 6

    sub_wbit_raw = [
        ("Potas [S]", 4.55, "mmol/L"), ("Sód [S]", 141.5, "mmol/L"),
        ("Wapń całkowity", 2.28, "mmol/L"), ("Magnez [S]", 0.79, "mmol/L"),
        ("ALP", 71.0, "U/L"), ("Glukoza", 5.45, "mmol/L"),
        ("Kreatynina", 91.0, "umol/L"), ("HGB", 13.4, "g/dL"),
        ("MCV", 85.5, "fL"), ("RDW-CV", 13.6, "%"),
    ]
    obs_w = [onto.harmonize(gate.ingest(_base_payload(n, v, u, "SPEC_E2E_WBIT"))) for n, v, u in sub_wbit_raw]
    v_w = cage.evaluate_panel(obs_w, meta=SpecimenPreAnalyticalMeta(), baseline_values=baseline)
    assert v_w.allow_layer3_ml is True
    r_w = ml_guard.evaluate(obs_w, v_w, baseline_values=baseline)
    assert r_w.executed is True
    assert r_w.predicted_class_code == "SUBTLE_WBIT"
    assert r_w.final_status == DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
    assert ledger.verify_integrity() is True
