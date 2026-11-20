"""
Layer 0: Sovereignty, Zero-PII Gate & Cryptographic Audit Ledger (SHA-256 Hash Chain).
Compliance: GDPR/RODO Art. 9 & Art. 25, ISO 15189:2022, EU AI Act Art. 12 & Art. 14,
Polish Laboratory Medicine Act (Dz.U. 2022 poz. 2280).
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DecisionStatus(str, Enum):
    """Hard legal decision ceiling. AI can NEVER autonomously authorize a clinical anomaly."""

    INGESTED_CLEAN = "INGESTED_CLEAN"
    REJECTED_PII_VIOLATION = "REJECTED_PII_VIOLATION"
    STOP_DETERMINISTIC_CAGE = "STOP_DETERMINISTIC_CAGE"
    HOLD_FOR_DIAGNOSTICIAN_REVIEW = "HOLD_FOR_DIAGNOSTICIAN_REVIEW"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


class AuditEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    timestamp_utc: str
    stage: str
    decision: DecisionStatus
    legal_basis: str
    reason: str
    payload_hash: str
    prev_hash: str
    event_hash: str


class AuditLedger:
    """Tamper-evident SHA-256 Hash Chain for ISO 15189:2022 and EU AI Act Art. 12 traceability."""

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    @staticmethod
    def _sha256_canonical(data: Any) -> str:
        serialized = json.dumps(
            data, sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def record(
        self,
        stage: str,
        decision: DecisionStatus,
        legal_basis: str,
        reason: str,
        payload_for_hash: Any,
    ) -> AuditEvent:
        index = len(self._events)
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        prev_hash = self._events[-1].event_hash if self._events else self.GENESIS_HASH
        payload_hash = self._sha256_canonical(payload_for_hash)

        event_body = {
            "index": index,
            "timestamp_utc": timestamp_utc,
            "stage": stage,
            "decision": decision.value,
            "legal_basis": legal_basis,
            "reason": reason,
            "payload_hash": payload_hash,
            "prev_hash": prev_hash,
        }
        event_hash = self._sha256_canonical(event_body)
        event = AuditEvent(**event_body, event_hash=event_hash)
        self._events.append(event)
        return event

    def verify_integrity(self) -> bool:
        expected_prev = self.GENESIS_HASH
        for event in self._events:
            if event.prev_hash != expected_prev:
                return False
            recomputed_body = {
                "index": event.index,
                "timestamp_utc": event.timestamp_utc,
                "stage": event.stage,
                "decision": event.decision.value,
                "legal_basis": event.legal_basis,
                "reason": event.reason,
                "payload_hash": event.payload_hash,
                "prev_hash": event.prev_hash,
            }
            if self._sha256_canonical(recomputed_body) != event.event_hash:
                return False
            expected_prev = event.event_hash
        return True

    @property
    def events(self) -> List[AuditEvent]:
        return list(self._events)


class PseudonymizedLabRecord(BaseModel):
    """Strict Zero-PII ingress contract (extra fields forbidden)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pseudonym_id: str = Field(..., min_length=8, max_length=64)
    specimen_id: str = Field(..., min_length=4, max_length=64)
    raw_test_name: str = Field(..., min_length=1, max_length=120)
    raw_value: float
    raw_unit: str = Field(..., min_length=1, max_length=32)
    collected_at_utc: str
    ward_code: Optional[str] = Field(default=None, max_length=32)
    comment: Optional[str] = Field(default=None, max_length=256)

    @field_validator("pseudonym_id")
    @classmethod
    def must_have_pseudonym_prefix(cls, v: str) -> str:
        if not v.startswith("PSEUDO_"):
            raise ValueError("Identifier must be pseudonymized with prefix 'PSEUDO_'.")
        return v


class PIIViolationError(ValueError):
    """Raised when direct or smuggled PII is detected at ingress."""


class ZeroPIIGate:
    """Active GDPR/RODO privacy firewall blocking PESEL, names, emails, and phones."""

    FORBIDDEN_KEYS = {
        "pesel", "imie", "imię", "nazwisko", "first_name", "last_name",
        "full_name", "patient_name", "pacjent", "email", "e_mail",
        "telefon", "phone", "adres", "address", "dob", "data_urodzenia",
        "nr_dowodu", "passport",
    }

    PII_REGEX_PATTERNS = {
        "PESEL_11_DIGITS": re.compile(r"(?<!\d)\d{11}(?!\d)"),
        "EMAIL_ADDRESS": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
        "PHONE_PL": re.compile(r"(?<!\d)(?:\+48[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)"),
    }

    def __init__(self, ledger: AuditLedger) -> None:
        self.ledger = ledger

    def _scan_for_pii(self, raw_payload: Dict[str, Any]) -> List[str]:
        violations: List[str] = []
        for key in raw_payload.keys():
            if key.strip().lower() in self.FORBIDDEN_KEYS:
                violations.append(f"FORBIDDEN_KEY:{key}")
        for key, value in raw_payload.items():
            if isinstance(value, str):
                for pattern_name, regex in self.PII_REGEX_PATTERNS.items():
                    if regex.search(value):
                        violations.append(f"{pattern_name}_IN_FIELD:{key}")
        return violations

    def ingest(self, raw_payload: Dict[str, Any]) -> PseudonymizedLabRecord:
        violations = self._scan_for_pii(raw_payload)
        if violations:
            self.ledger.record(
                stage="LAYER_0_ZERO_PII_GATE",
                decision=DecisionStatus.REJECTED_PII_VIOLATION,
                legal_basis="RODO Art. 9 & Art. 25 (Privacy by Design)",
                reason=f"Zablokowano wyciek PII: {', '.join(violations)}",
                payload_for_hash={"violations": violations},
            )
            raise PIIViolationError(
                f"HARD STOP [RODO]: Wykryto niedozwolone dane PII ({', '.join(violations)})."
            )

        record = PseudonymizedLabRecord(**raw_payload)
        self.ledger.record(
            stage="LAYER_0_ZERO_PII_GATE",
            decision=DecisionStatus.INGESTED_CLEAN,
            legal_basis="RODO Art. 25 / ISO 15189:2022",
            reason="Rekord spseudonimizowany przeszedł walidację Zero-PII.",
            payload_for_hash=record.model_dump(),
        )
        return record
