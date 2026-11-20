"""
Layer 1: LabAegis Clinical Ontology & Silo Breaker (Polish LIS -> LOINC / UCUM / HL7 FHIR R4).
Encodes 24 core general-panel biomarkers with pre-analytical vulnerability links and
deterministic parenthetical conflict cross-checking.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from .layer0_privacy import AuditLedger, DecisionStatus, PseudonymizedLabRecord


class SpecimenMatrix(str, Enum):
    SERUM_OR_PLASMA = "SERUM_OR_PLASMA"
    WHOLE_BLOOD_EDTA = "WHOLE_BLOOD_EDTA"


class PreAnalyticalRisk(str, Enum):
    HEMOLYSIS = "HEMOLYSIS"
    EDTA_CONTAMINATION = "EDTA_CONTAMINATION"
    IV_FLUID_DILUTION = "IV_FLUID_DILUTION"
    GLYCOLYSIS_DELAY = "GLYCOLYSIS_DELAY"
    LIPEMIA_INTERFERENCE = "LIPEMIA_INTERFERENCE"
    WBIT_FINGERPRINT = "WBIT_FINGERPRINT"
    CLOT_OR_MICROCLOT = "CLOT_OR_MICROCLOT"


class BiomarkerConcept(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_code: str
    loinc_code: str
    name_en: str
    name_pl: str
    clinical_group: str
    required_matrix: SpecimenMatrix
    canonical_unit: str
    unit_to_canonical_factor: Dict[str, float]
    plausibility_bounds: Tuple[float, float]
    lis_aliases: Tuple[str, ...]
    preanalytical_links: Tuple[PreAnalyticalRisk, ...]


class NormalizedObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    pseudonym_id: str
    specimen_id: str
    collected_at_utc: str
    ward_code: Optional[str]

    canonical_code: str
    loinc_code: str
    name_en: str
    name_pl: str
    clinical_group: str
    required_matrix: SpecimenMatrix

    canonical_value: float
    canonical_unit: str

    raw_test_name: str
    raw_value: float
    raw_unit: str
    conversion_factor_applied: float
    mapping_method: str
    mapping_confidence: float
    status: DecisionStatus
    message_pl: str
    message_en: str

    def to_fhir_r4(self) -> Dict[str, Any]:
        """Exports harmonized observation as a standard HL7 FHIR R4 Observation resource."""
        fhir_status = (
            "preliminary"
            if self.status == DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
            else "registered"
        )
        return {
            "resourceType": "Observation",
            "status": fhir_status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": self.loinc_code,
                        "display": self.name_en,
                    }
                ],
                "text": f"{self.name_pl} ({self.raw_test_name})",
            },
            "subject": {"identifier": {"system": "urn:labaegis:pseudonym", "value": self.pseudonym_id}},
            "specimen": {"identifier": {"system": "urn:labaegis:specimen", "value": self.specimen_id}},
            "effectiveDateTime": self.collected_at_utc,
            "valueQuantity": {
                "value": self.canonical_value,
                "unit": self.canonical_unit,
                "system": "http://unitsofmeasure.org",
                "code": self.canonical_unit,
            },
            "note": [
                {
                    "text": (
                        f"LabAegis Ontology | mapping={self.mapping_method} "
                        f"(conf={self.mapping_confidence:.2f}) | raw={self.raw_value} {self.raw_unit}"
                    )
                }
            ],
        }


class OntologyContainmentError(ValueError):
    """Raised on conflicting concepts, unknown units, or unmappable LIS labels."""


class LabAegisOntology:
    """24-biomarker semantic ontology breaking Polish LIS naming and unit silos."""

    LIS_NOISE_TOKENS = {
        "cito", "pilne", "stat", "surowica", "sur", "osocze", "krew", "pelna",
        "badanie", "oznaczenie", "poziom", "stezenie", "aktywnosc", "jonogram",
        "morfologia", "panel", "profil", "s", "p", "b", "edta", "met", "ise",
    }

    def __init__(self, ledger: AuditLedger) -> None:
        self.ledger = ledger
        self.concepts: Dict[str, BiomarkerConcept] = self._build_24_biomarker_ontology()
        self._alias_index: Dict[str, str] = self._build_alias_index()

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        text = text.replace("ł", "l").replace("Ł", "L")
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    @classmethod
    def normalize_test_name(cls, raw_name: str, strip_noise: bool = False) -> str:
        text = cls._strip_diacritics(raw_name).lower()
        text = re.sub(r"[^a-z0-9+]+", " ", text).strip()
        if not strip_noise:
            return re.sub(r"\s+", " ", text)
        tokens = [t for t in text.split() if t not in cls.LIS_NOISE_TOKENS]
        return " ".join(tokens) if tokens else text

    @classmethod
    def normalize_unit(cls, raw_unit: str) -> str:
        u = raw_unit.strip().replace("\u00b5", "u").replace("\u03bc", "u")
        u = cls._strip_diacritics(u).lower()
        return re.sub(r"\s+", "", u)

    def _build_24_biomarker_ontology(self) -> Dict[str, BiomarkerConcept]:
        raw_concepts = [
            BiomarkerConcept(
                canonical_code="K", loinc_code="2823-3", name_en="Potassium", name_pl="Potas",
                clinical_group="ELECTROLYTES", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "meq/l": 1.0, "mg/dl": 0.2558},
                plausibility_bounds=(1.0, 15.0),
                lis_aliases=("potas", "k", "k+", "potas k", "potas k+", "potassium"),
                preanalytical_links=(PreAnalyticalRisk.HEMOLYSIS, PreAnalyticalRisk.EDTA_CONTAMINATION, PreAnalyticalRisk.IV_FLUID_DILUTION),
            ),
            BiomarkerConcept(
                canonical_code="NA", loinc_code="2951-2", name_en="Sodium", name_pl="Sód",
                clinical_group="ELECTROLYTES", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "meq/l": 1.0},
                plausibility_bounds=(90.0, 200.0),
                lis_aliases=("sod", "na", "na+", "sod na", "sod na+", "sodium"),
                preanalytical_links=(PreAnalyticalRisk.IV_FLUID_DILUTION, PreAnalyticalRisk.LIPEMIA_INTERFERENCE),
            ),
            BiomarkerConcept(
                canonical_code="CL", loinc_code="2075-0", name_en="Chloride", name_pl="Chlorki",
                clinical_group="ELECTROLYTES", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "meq/l": 1.0},
                plausibility_bounds=(50.0, 160.0),
                lis_aliases=("chlorki", "chlor", "cl", "cl-", "chloride"),
                preanalytical_links=(PreAnalyticalRisk.IV_FLUID_DILUTION,),
            ),
            BiomarkerConcept(
                canonical_code="CA", loinc_code="2000-8", name_en="Calcium, total", name_pl="Wapń całkowity",
                clinical_group="ELECTROLYTES", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "mg/dl": 0.2495, "meq/l": 0.5},
                plausibility_bounds=(0.2, 6.0),
                lis_aliases=("wapn", "wapn calkowity", "ca", "ca2+", "wapn ca", "calcium", "total calcium"),
                preanalytical_links=(PreAnalyticalRisk.EDTA_CONTAMINATION, PreAnalyticalRisk.IV_FLUID_DILUTION),
            ),
            BiomarkerConcept(
                canonical_code="MG", loinc_code="2601-3", name_en="Magnesium", name_pl="Magnez",
                clinical_group="ELECTROLYTES", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "mg/dl": 0.4114, "meq/l": 0.5},
                plausibility_bounds=(0.1, 5.0),
                lis_aliases=("magnez", "mg", "mg2+", "magnez mg", "magnesium"),
                preanalytical_links=(PreAnalyticalRisk.EDTA_CONTAMINATION, PreAnalyticalRisk.HEMOLYSIS),
            ),
            BiomarkerConcept(
                canonical_code="GLU", loinc_code="14749-6", name_en="Glucose", name_pl="Glukoza",
                clinical_group="METABOLIC_RENAL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "mg/dl": 0.0555},
                plausibility_bounds=(0.5, 80.0),
                lis_aliases=("glukoza", "glu", "glukoza na czczo", "glucose", "glikemia"),
                preanalytical_links=(PreAnalyticalRisk.GLYCOLYSIS_DELAY, PreAnalyticalRisk.IV_FLUID_DILUTION),
            ),
            BiomarkerConcept(
                canonical_code="CREA", loinc_code="14682-9", name_en="Creatinine", name_pl="Kreatynina",
                clinical_group="METABOLIC_RENAL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="umol/L", unit_to_canonical_factor={"umol/l": 1.0, "mg/dl": 88.4, "mmol/l": 1000.0},
                plausibility_bounds=(10.0, 2500.0),
                lis_aliases=("kreatynina", "crea", "krea", "creatinine"),
                preanalytical_links=(PreAnalyticalRisk.WBIT_FINGERPRINT, PreAnalyticalRisk.IV_FLUID_DILUTION),
            ),
            BiomarkerConcept(
                canonical_code="UREA", loinc_code="22664-7", name_en="Urea", name_pl="Mocznik",
                clinical_group="METABOLIC_RENAL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "mg/dl": 0.1665},
                plausibility_bounds=(0.5, 100.0),
                lis_aliases=("mocznik", "urea"),
                preanalytical_links=(PreAnalyticalRisk.IV_FLUID_DILUTION,),
            ),
            BiomarkerConcept(
                canonical_code="URIC", loinc_code="14933-6", name_en="Uric acid", name_pl="Kwas moczowy",
                clinical_group="METABOLIC_RENAL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="umol/L", unit_to_canonical_factor={"umol/l": 1.0, "mg/dl": 59.48, "mmol/l": 1000.0},
                plausibility_bounds=(30.0, 1500.0),
                lis_aliases=("kwas moczowy", "kw moczowy", "uric", "ua", "uric acid"),
                preanalytical_links=(PreAnalyticalRisk.WBIT_FINGERPRINT,),
            ),
            BiomarkerConcept(
                canonical_code="ALT", loinc_code="1742-6", name_en="Alanine aminotransferase (ALT)", name_pl="ALT (AlAT)",
                clinical_group="LIVER_PANEL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="U/L", unit_to_canonical_factor={"u/l": 1.0, "iu/l": 1.0, "ukat/l": 60.0},
                plausibility_bounds=(1.0, 15000.0),
                lis_aliases=("alt", "alat", "gpt", "aminotransferaza alaninowa", "alt alat"),
                preanalytical_links=(PreAnalyticalRisk.HEMOLYSIS,),
            ),
            BiomarkerConcept(
                canonical_code="AST", loinc_code="1920-8", name_en="Aspartate aminotransferase (AST)", name_pl="AST (AspAT)",
                clinical_group="LIVER_PANEL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="U/L", unit_to_canonical_factor={"u/l": 1.0, "iu/l": 1.0, "ukat/l": 60.0},
                plausibility_bounds=(1.0, 15000.0),
                lis_aliases=("ast", "aspat", "got", "aminotransferaza asparaginianowa", "ast aspat"),
                preanalytical_links=(PreAnalyticalRisk.HEMOLYSIS,),
            ),
            BiomarkerConcept(
                canonical_code="ALP", loinc_code="6768-6", name_en="Alkaline phosphatase (ALP)", name_pl="Fosfataza alkaliczna (ALP)",
                clinical_group="LIVER_PANEL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="U/L", unit_to_canonical_factor={"u/l": 1.0, "iu/l": 1.0, "ukat/l": 60.0},
                plausibility_bounds=(1.0, 5000.0),
                lis_aliases=("alp", "fosfataza alkaliczna", "fosfataza zasadowa", "fosfataza alkaliczna alp", "alkaline phosphatase"),
                preanalytical_links=(PreAnalyticalRisk.EDTA_CONTAMINATION,),
            ),
            BiomarkerConcept(
                canonical_code="GGTP", loinc_code="2324-2", name_en="Gamma-glutamyl transferase (GGT)", name_pl="GGTP (GGT)",
                clinical_group="LIVER_PANEL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="U/L", unit_to_canonical_factor={"u/l": 1.0, "iu/l": 1.0, "ukat/l": 60.0},
                plausibility_bounds=(1.0, 5000.0),
                lis_aliases=("ggtp", "ggt", "gamma glutamylotransferaza"),
                preanalytical_links=(PreAnalyticalRisk.WBIT_FINGERPRINT,),
            ),
            BiomarkerConcept(
                canonical_code="BIL_T", loinc_code="14631-6", name_en="Total Bilirubin", name_pl="Bilirubina całkowita",
                clinical_group="LIVER_PANEL", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="umol/L", unit_to_canonical_factor={"umol/l": 1.0, "mg/dl": 17.1},
                plausibility_bounds=(0.5, 1000.0),
                lis_aliases=("bilirubina", "bilirubina calkowita", "bil t", "tbil", "total bilirubin"),
                preanalytical_links=(PreAnalyticalRisk.HEMOLYSIS,),
            ),
            BiomarkerConcept(
                canonical_code="CRP", loinc_code="1988-5", name_en="C-reactive protein (CRP)", name_pl="Białko C-reaktywne (CRP)",
                clinical_group="INFLAMMATION_ENDOCRINE_LIPIDS", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mg/L", unit_to_canonical_factor={"mg/l": 1.0, "mg/dl": 10.0},
                plausibility_bounds=(0.0, 600.0),
                lis_aliases=("crp", "bialko c reaktywne", "bialko crp", "crp ilosciowo"),
                preanalytical_links=(PreAnalyticalRisk.LIPEMIA_INTERFERENCE,),
            ),
            BiomarkerConcept(
                canonical_code="TSH", loinc_code="3016-3", name_en="Thyroid stimulating hormone (TSH)", name_pl="TSH (Tyreotropina)",
                clinical_group="INFLAMMATION_ENDOCRINE_LIPIDS", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="m[IU]/L", unit_to_canonical_factor={"m[iu]/l": 1.0, "miu/l": 1.0, "uiu/ml": 1.0, "u[iu]/ml": 1.0},
                plausibility_bounds=(0.001, 200.0),
                lis_aliases=("tsh", "tyreotropina", "tsh 3 generacji", "tsh iii"),
                preanalytical_links=(PreAnalyticalRisk.WBIT_FINGERPRINT,),
            ),
            BiomarkerConcept(
                canonical_code="CHOL", loinc_code="14647-2", name_en="Total Cholesterol", name_pl="Cholesterol całkowity",
                clinical_group="INFLAMMATION_ENDOCRINE_LIPIDS", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "mg/dl": 0.02586},
                plausibility_bounds=(0.5, 30.0),
                lis_aliases=("cholesterol", "cholesterol calkowity", "chol", "tc", "total cholesterol"),
                preanalytical_links=(PreAnalyticalRisk.LIPEMIA_INTERFERENCE, PreAnalyticalRisk.WBIT_FINGERPRINT),
            ),
            BiomarkerConcept(
                canonical_code="TG", loinc_code="14927-8", name_en="Triglycerides", name_pl="Trójglicerydy",
                clinical_group="INFLAMMATION_ENDOCRINE_LIPIDS", required_matrix=SpecimenMatrix.SERUM_OR_PLASMA,
                canonical_unit="mmol/L", unit_to_canonical_factor={"mmol/l": 1.0, "mg/dl": 0.01129},
                plausibility_bounds=(0.1, 60.0),
                lis_aliases=("trojglicerydy", "triglicerydy", "tg", "triglycerides"),
                preanalytical_links=(PreAnalyticalRisk.LIPEMIA_INTERFERENCE,),
            ),
            BiomarkerConcept(
                canonical_code="HGB", loinc_code="718-7", name_en="Hemoglobin", name_pl="Hemoglobina (HGB)",
                clinical_group="HEMATOLOGY", required_matrix=SpecimenMatrix.WHOLE_BLOOD_EDTA,
                canonical_unit="g/dL", unit_to_canonical_factor={"g/dl": 1.0, "g/l": 0.1, "mmol/l": 1.6114},
                plausibility_bounds=(2.0, 26.0),
                lis_aliases=("hemoglobina", "hgb", "hb", "hemoglobin"),
                preanalytical_links=(PreAnalyticalRisk.IV_FLUID_DILUTION, PreAnalyticalRisk.LIPEMIA_INTERFERENCE, PreAnalyticalRisk.CLOT_OR_MICROCLOT),
            ),
            BiomarkerConcept(
                canonical_code="WBC", loinc_code="6690-2", name_en="Leukocytes (WBC)", name_pl="Leukocyty (WBC)",
                clinical_group="HEMATOLOGY", required_matrix=SpecimenMatrix.WHOLE_BLOOD_EDTA,
                canonical_unit="10*9/L", unit_to_canonical_factor={"10*9/l": 1.0, "10^9/l": 1.0, "10^3/ul": 1.0, "tys/ul": 1.0, "k/ul": 1.0, "g/l": 1.0, "/ul": 0.001},
                plausibility_bounds=(0.05, 500.0),
                lis_aliases=("leukocyty", "wbc", "krwinki biale", "leukocytes"),
                preanalytical_links=(PreAnalyticalRisk.CLOT_OR_MICROCLOT, PreAnalyticalRisk.IV_FLUID_DILUTION),
            ),
            BiomarkerConcept(
                canonical_code="RBC", loinc_code="789-8", name_en="Erythrocytes (RBC)", name_pl="Erytrocyty (RBC)",
                clinical_group="HEMATOLOGY", required_matrix=SpecimenMatrix.WHOLE_BLOOD_EDTA,
                canonical_unit="10*12/L", unit_to_canonical_factor={"10*12/l": 1.0, "10^12/l": 1.0, "10^6/ul": 1.0, "mln/ul": 1.0, "m/ul": 1.0, "t/l": 1.0},
                plausibility_bounds=(0.5, 10.0),
                lis_aliases=("erytrocyty", "rbc", "krwinki czerwone", "erythrocytes"),
                preanalytical_links=(PreAnalyticalRisk.IV_FLUID_DILUTION, PreAnalyticalRisk.CLOT_OR_MICROCLOT),
            ),
            BiomarkerConcept(
                canonical_code="PLT", loinc_code="777-3", name_en="Platelets (PLT)", name_pl="Płytki krwi (PLT)",
                clinical_group="HEMATOLOGY", required_matrix=SpecimenMatrix.WHOLE_BLOOD_EDTA,
                canonical_unit="10*9/L", unit_to_canonical_factor={"10*9/l": 1.0, "10^9/l": 1.0, "10^3/ul": 1.0, "tys/ul": 1.0, "k/ul": 1.0, "g/l": 1.0, "/ul": 0.001},
                plausibility_bounds=(1.0, 3000.0),
                lis_aliases=("plytki krwi", "plytki", "plt", "trombocyty", "platelets"),
                preanalytical_links=(PreAnalyticalRisk.CLOT_OR_MICROCLOT, PreAnalyticalRisk.EDTA_CONTAMINATION),
            ),
            BiomarkerConcept(
                canonical_code="MCV", loinc_code="787-2", name_en="Mean Corpuscular Volume (MCV)", name_pl="MCV (Średnia objętość krwinki)",
                clinical_group="HEMATOLOGY", required_matrix=SpecimenMatrix.WHOLE_BLOOD_EDTA,
                canonical_unit="fL", unit_to_canonical_factor={"fl": 1.0, "um3": 1.0},
                plausibility_bounds=(45.0, 145.0),
                lis_aliases=("mcv", "srednia objetosc krwinki", "srednia objetosc erytrocytu"),
                preanalytical_links=(PreAnalyticalRisk.WBIT_FINGERPRINT,),
            ),
            BiomarkerConcept(
                canonical_code="RDW", loinc_code="788-0", name_en="Red Cell Distribution Width (RDW-CV)", name_pl="RDW-CV (Rozpiętość rozkładu erytrocytów)",
                clinical_group="HEMATOLOGY", required_matrix=SpecimenMatrix.WHOLE_BLOOD_EDTA,
                canonical_unit="%", unit_to_canonical_factor={"%": 1.0},
                plausibility_bounds=(8.0, 40.0),
                lis_aliases=("rdw", "rdw cv", "rdw-cv", "anizocytoza"),
                preanalytical_links=(PreAnalyticalRisk.WBIT_FINGERPRINT,),
            ),
        ]
        return {c.canonical_code: c for c in raw_concepts}

    def _build_alias_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for code, concept in self.concepts.items():
            for alias in concept.lis_aliases:
                index[self.normalize_test_name(alias, strip_noise=False)] = code
        return index

    def _resolve_concept(self, raw_test_name: str) -> Tuple[BiomarkerConcept, str, float]:
        norm_full = self.normalize_test_name(raw_test_name, strip_noise=False)
        if norm_full in self._alias_index:
            return self.concepts[self._alias_index[norm_full]], "DETERMINISTIC_EXACT", 1.0

        norm_clean = self.normalize_test_name(raw_test_name, strip_noise=True)
        if norm_clean in self._alias_index:
            return self.concepts[self._alias_index[norm_clean]], "DETERMINISTIC_NOISE_STRIPPED", 1.0

        paren_groups = re.findall(r"\(([^)]+)\)", raw_test_name)
        if paren_groups:
            base_raw = re.sub(r"\([^)]*\)", " ", raw_test_name)
            norm_base = self.normalize_test_name(base_raw, strip_noise=True)
            base_code = self._alias_index.get(norm_base)
            paren_codes = set()
            for grp in paren_groups:
                norm_grp = self.normalize_test_name(grp, strip_noise=True)
                if norm_grp in self._alias_index:
                    paren_codes.add(self._alias_index[norm_grp])
            if base_code is not None and paren_codes:
                if paren_codes == {base_code}:
                    return self.concepts[base_code], "DETERMINISTIC_COMPOSITE_CONFIRMED", 1.0
                raise OntologyContainmentError(
                    f"CONFLICTING_CONCEPTS [STOP]: Sprzeczna etykieta LIS '{raw_test_name}' "
                    f"wskazuje jednocześnie na różne parametry ({base_code} vs {sorted(paren_codes)})."
                )

        best_alias = None
        best_score = 0.0
        for alias in self._alias_index.keys():
            score = difflib.SequenceMatcher(None, norm_clean, alias).ratio()
            if score > best_score:
                best_score = score
                best_alias = alias

        if best_alias is not None and best_score >= 0.72:
            concept = self.concepts[self._alias_index[best_alias]]
            return concept, "FUZZY_SUGGESTION_REQUIRES_HITL", round(best_score, 3)

        raise OntologyContainmentError(
            f"UNMAPPED_TEST [STOP]: Nieznany parametr LIS '{raw_test_name}' (brak bezpiecznego dopasowania w ontologii)."
        )

    def harmonize(self, record: PseudonymizedLabRecord) -> NormalizedObservation:
        try:
            concept, method, confidence = self._resolve_concept(record.raw_test_name)
        except OntologyContainmentError as exc:
            self.ledger.record(
                stage="LAYER_1_CLINICAL_ONTOLOGY",
                decision=DecisionStatus.STOP_DETERMINISTIC_CAGE,
                legal_basis="ISO 15189:2022 (Pre-analytical & Data Integrity)",
                reason=str(exc),
                payload_for_hash={"specimen_id": record.specimen_id, "raw_test_name": record.raw_test_name},
            )
            raise

        norm_unit = self.normalize_unit(record.raw_unit)
        if norm_unit not in concept.unit_to_canonical_factor:
            err_msg = (
                f"UNKNOWN_UNIT [STOP]: Nieobsługiwana jednostka '{record.raw_unit}' "
                f"dla parametru {concept.canonical_code} ({concept.name_pl})."
            )
            self.ledger.record(
                stage="LAYER_1_CLINICAL_ONTOLOGY",
                decision=DecisionStatus.STOP_DETERMINISTIC_CAGE,
                legal_basis="ISO 15189:2022 / UCUM Metrology Enforcement",
                reason=err_msg,
                payload_for_hash={"specimen_id": record.specimen_id, "raw_unit": record.raw_unit},
            )
            raise OntologyContainmentError(err_msg)

        factor = concept.unit_to_canonical_factor[norm_unit]
        canonical_val = round(record.raw_value * factor, 4)

        if confidence < 1.0:
            status = DecisionStatus.HOLD_FOR_DIAGNOSTICIAN_REVIEW
            msg_pl = (
                f"WSTRZYMANO (HITL): Niejednoznaczna nazwa LIS '{record.raw_test_name}'. "
                f"Sugestia ontologii: {concept.name_pl} [LOINC {concept.loinc_code}] (pewność {confidence*100:.1f}%)."
            )
            msg_en = (
                f"HELD (HITL): Ambiguous LIS label '{record.raw_test_name}'. "
                f"Ontology suggestion: {concept.name_en} [LOINC {concept.loinc_code}] (confidence {confidence*100:.1f}%)."
            )
            legal_basis = "EU AI Act Art. 14 (Human Oversight) & Ustawa o medycynie laboratoryjnej"
        else:
            status = DecisionStatus.INGESTED_CLEAN
            msg_pl = (
                f"ZHARMONIZOWANO: '{record.raw_test_name}' -> {concept.name_pl} [LOINC {concept.loinc_code}] | "
                f"{record.raw_value} {record.raw_unit} -> {canonical_val} {concept.canonical_unit}."
            )
            msg_en = (
                f"HARMONIZED: '{record.raw_test_name}' -> {concept.name_en} [LOINC {concept.loinc_code}] | "
                f"{record.raw_value} {record.raw_unit} -> {canonical_val} {concept.canonical_unit}."
            )
            legal_basis = "ISO 15189:2022 / LOINC & UCUM Standardization"

        obs = NormalizedObservation(
            pseudonym_id=record.pseudonym_id,
            specimen_id=record.specimen_id,
            collected_at_utc=record.collected_at_utc,
            ward_code=record.ward_code,
            canonical_code=concept.canonical_code,
            loinc_code=concept.loinc_code,
            name_en=concept.name_en,
            name_pl=concept.name_pl,
            clinical_group=concept.clinical_group,
            required_matrix=concept.required_matrix,
            canonical_value=canonical_val,
            canonical_unit=concept.canonical_unit,
            raw_test_name=record.raw_test_name,
            raw_value=record.raw_value,
            raw_unit=record.raw_unit,
            conversion_factor_applied=factor,
            mapping_method=method,
            mapping_confidence=confidence,
            status=status,
            message_pl=msg_pl,
            message_en=msg_en,
        )

        self.ledger.record(
            stage="LAYER_1_CLINICAL_ONTOLOGY",
            decision=status,
            legal_basis=legal_basis,
            reason=msg_pl,
            payload_for_hash=obs.model_dump(),
        )
        return obs
