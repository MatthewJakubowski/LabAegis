"""LabAegis: Open-Source Clinical Ontology & Containment Engine for Laboratory Medicine."""

from .layer0_privacy import AuditEvent, AuditLedger, DecisionStatus, PIIViolationError, PseudonymizedLabRecord, ZeroPIIGate
from .layer1_ontology import BiomarkerConcept, LabAegisOntology, NormalizedObservation, OntologyContainmentError, PreAnalyticalRisk, SpecimenMatrix
from .layer2_containment import AnalyteMetrologySpec, CascadeTrigger, ContainmentVerdict, DeterministicContainmentCage, SpecimenPreAnalyticalMeta
from .layer3_glassbox_ml import ARTIFACT_CLASSES, CORE_ML_ANALYTES, GlassBoxMLGuard, MLGuardAssessment, XAIExplanationItem, compute_domain_features, generate_labaegis_cohort, plot_xai_audit_card, train_reference_models

__version__ = "0.1.0"
__all__ = [
    "AuditEvent", "AuditLedger", "DecisionStatus", "PIIViolationError", "PseudonymizedLabRecord", "ZeroPIIGate",
    "BiomarkerConcept", "LabAegisOntology", "NormalizedObservation", "OntologyContainmentError", "PreAnalyticalRisk", "SpecimenMatrix",
    "AnalyteMetrologySpec", "CascadeTrigger", "ContainmentVerdict", "DeterministicContainmentCage", "SpecimenPreAnalyticalMeta",
    "ARTIFACT_CLASSES", "CORE_ML_ANALYTES", "GlassBoxMLGuard", "MLGuardAssessment", "XAIExplanationItem",
    "compute_domain_features", "generate_labaegis_cohort", "plot_xai_audit_card", "train_reference_models",
]
