# LabAegis — Regulatory & Clinical Compliance Matrix (PL / EU)

> **Scope Boundary (IVDR Regulation (EU) 2017/746):** LabAegis is an open-source **Pre-Analytical Quality Assurance & Semantic Interoperability Research Framework**. It does **not** perform clinical diagnosis, treatment recommendation, or autonomous result release.

| Regulation / Standard | Core Legal Requirement | Architectural Enforcement in LabAegis |
| :--- | :--- | :--- |
| **Polish Laboratory Medicine Act** *(Ustawa o medycynie laboratoryjnej, Dz.U. 2022 poz. 2280)* | Result authorization is the exclusive statutory competence of a licensed Laboratory Diagnostician. | `DecisionStatus` hard ceiling: anomalies can never exceed `HOLD_FOR_DIAGNOSTICIAN_REVIEW`. No auto-release path exists for flagged panels. |
| **EU AI Act (2024/1689)** *(Art. 9, 12, 13, 14 — High-Risk AI Governance)* | Risk management, immutable event logging, mandatory explainability, and real human oversight with stop capability. | Mandatory **TreeSHAP** attribution (`Layer 3`), SHA-256 `AuditLedger` (`Layer 0`), epistemic uncertainty floor, and one-call `set_ml_kill_switch()` (`Layer 2`). |
| **GDPR / RODO** *(Art. 9 & Art. 25 — Privacy by Design)* | Protection of special-category health data and structural data minimization. | Active `ZeroPIIGate` blocks forbidden PII keys and regex-detected PESEL/email/phone strings before ontology ingestion; raw PII is never written to logs. |
| **ISO 15189:2022** *(Medical Laboratories — Quality & Competence)* | Pre-analytical phase control, verification of processes, measurement uncertainty, and full traceability. | Ontology-driven HIL rules, **EFLM RCV** metrological delta checks, cryptographic SHA-256 chain, and 100% deterministic `pytest` regression suite. |
| **EHDS / Interoperability** *(HL7 FHIR R4, LOINC, UCUM)* | Cross-border semantic standardization of laboratory data. | Native 24-biomarker LOINC/UCUM harmonization with built-in `.to_fhir_r4()` `Observation` export. |
