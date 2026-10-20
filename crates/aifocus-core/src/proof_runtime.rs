//! Bounded proof-status lifecycle ledger.
//!
//! The ledger records proof status transitions and evidence digests, but does
//! not prove claims or verify the meaning of an evidence artifact. Callers
//! must supply independently generated evidence and a trusted verifier.

use std::collections::BTreeMap;
use crate::ai_mode::{
    AiModePolicy, PolicyViolation, ProofRequirement, ProofStatus,
};

pub const MAX_PROOF_RECEIPTS: usize = 16_384;
pub const MAX_PROOF_EVIDENCE_DIGEST_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofReceipt {
    pub sequence: u64,
    pub requirement: ProofRequirement,
    pub status: ProofStatus,
    /// SHA-256 digest of evidence; raw source, prompts and secrets are not stored.
    pub evidence_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofLedgerError {
    ReceiptLimitExceeded,
    InvalidEvidenceDigest,
    EvidenceDigestRequired(ProofRequirement),
    InvalidStatusTransition {
        requirement: ProofRequirement,
        from: ProofStatus,
        to: ProofStatus,
    },
    AlreadySealed,
    PolicyRejected(Vec<PolicyViolation>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofLedger {
    latest: BTreeMap<ProofRequirement, ProofStatus>,
    receipts: Vec<ProofReceipt>,
    sealed: bool,
}

impl Default for ProofLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofLedger {
    pub fn new() -> Self {
        Self {
            latest: BTreeMap::new(),
            receipts: Vec::new(),
            sealed: false,
        }
    }

    pub fn receipts(&self) -> &[ProofReceipt] {
        &self.receipts
    }

    pub fn status(&self, requirement: ProofRequirement) -> Option<ProofStatus> {
        self.latest.get(&requirement).copied()
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    /// Record a status transition only after all validation succeeds.
    pub fn record(
        &mut self,
        requirement: ProofRequirement,
        status: ProofStatus,
        evidence_digest: Option<&str>,
    ) -> Result<&ProofReceipt, ProofLedgerError> {
        if self.sealed {
            return Err(ProofLedgerError::AlreadySealed);
        }
        if self.receipts.len() >= MAX_PROOF_RECEIPTS {
            return Err(ProofLedgerError::ReceiptLimitExceeded);
        }

        if let Some(digest) = evidence_digest {
            if digest.len() != MAX_PROOF_EVIDENCE_DIGEST_BYTES
                || !digest.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(ProofLedgerError::InvalidEvidenceDigest);
            }
        }
        if matches!(status, ProofStatus::Proved | ProofStatus::RuntimeChecked)
            && evidence_digest.is_none()
        {
            return Err(ProofLedgerError::EvidenceDigestRequired(requirement));
        }

        if let Some(previous) = self.latest.get(&requirement).copied() {
            let allowed = matches!(
                (previous, status),
                (ProofStatus::Unknown, ProofStatus::RuntimeChecked | ProofStatus::Proved | ProofStatus::Failed)
                    | (ProofStatus::RuntimeChecked, ProofStatus::Proved | ProofStatus::Failed)
            );
            if !allowed {
                return Err(ProofLedgerError::InvalidStatusTransition {
                    requirement,
                    from: previous,
                    to: status,
                });
            }
        }

        let receipt = ProofReceipt {
            sequence: self.receipts.len() as u64,
            requirement,
            status,
            evidence_digest: evidence_digest.map(str::to_owned),
        };
        self.latest.insert(requirement, status);
        self.receipts.push(receipt);
        Ok(self.receipts.last().expect("receipt was just appended"))
    }

    /// Seal only if every required proof satisfies the configured policy.
    /// A rejected seal leaves the ledger open so missing evidence can be added.
    pub fn seal(&mut self, policy: &AiModePolicy) -> Result<(), ProofLedgerError> {
        if self.sealed {
            return Err(ProofLedgerError::AlreadySealed);
        }
        policy
            .validate_proofs(&self.latest)
            .map_err(ProofLedgerError::PolicyRejected)?;
        self.sealed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_mode::{
        ProofPolicy, TracePolicy, UnknownProofPolicy, VaultPolicy,
    };
    use std::collections::BTreeSet;

    fn policy() -> AiModePolicy {
        AiModePolicy {
            trace: TracePolicy { required: false, include_ai_metadata: false, require_signature: false },
            vault: VaultPolicy::deny_all(),
            proof: ProofPolicy {
                required: [ProofRequirement::TypeSafety, ProofRequirement::EffectPolicy]
                    .into_iter().collect::<BTreeSet<_>>(),
                on_unknown: UnknownProofPolicy::Reject,
            },
        }
    }

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn records_monotonic_receipts_and_allowed_status_progression() {
        let mut ledger = ProofLedger::new();
        assert_eq!(ledger.record(ProofRequirement::TypeSafety, ProofStatus::Unknown, None).unwrap().sequence, 0);
        assert_eq!(ledger.record(ProofRequirement::TypeSafety, ProofStatus::RuntimeChecked, Some(DIGEST_A)).unwrap().sequence, 1);
        assert_eq!(ledger.record(ProofRequirement::TypeSafety, ProofStatus::Proved, Some(DIGEST_B)).unwrap().sequence, 2);
        assert_eq!(ledger.status(ProofRequirement::TypeSafety), Some(ProofStatus::Proved));
        assert_eq!(ledger.receipts().len(), 3);
    }

    #[test]
    fn invalid_updates_are_atomic_and_terminal_states_cannot_regress() {
        let mut ledger = ProofLedger::new();
        ledger.record(ProofRequirement::TypeSafety, ProofStatus::Proved, Some(DIGEST_A)).unwrap();
        let before = ledger.clone();
        assert!(matches!(
            ledger.record(ProofRequirement::TypeSafety, ProofStatus::Unknown, None),
            Err(ProofLedgerError::InvalidStatusTransition { .. })
        ));
        assert_eq!(ledger, before);
        assert_eq!(
            ledger.record(ProofRequirement::EffectPolicy, ProofStatus::Proved, Some("bad")),
            Err(ProofLedgerError::InvalidEvidenceDigest)
        );
        assert_eq!(ledger, before);
    }

    #[test]
    fn proved_and_runtime_checked_require_digest() {
        let mut ledger = ProofLedger::new();
        assert_eq!(
            ledger.record(ProofRequirement::TypeSafety, ProofStatus::Proved, None),
            Err(ProofLedgerError::EvidenceDigestRequired(ProofRequirement::TypeSafety))
        );
        assert_eq!(ledger.record(ProofRequirement::TypeSafety, ProofStatus::RuntimeChecked, None),
            Err(ProofLedgerError::EvidenceDigestRequired(ProofRequirement::TypeSafety)));
        assert!(ledger.receipts().is_empty());
    }

    #[test]
    fn cannot_seal_with_missing_or_unknown_required_proofs() {
        let mut ledger = ProofLedger::new();
        assert!(matches!(ledger.seal(&policy()), Err(ProofLedgerError::PolicyRejected(_))));
        assert!(!ledger.is_sealed());
        ledger.record(ProofRequirement::TypeSafety, ProofStatus::Proved, Some(DIGEST_A)).unwrap();
        ledger.record(ProofRequirement::EffectPolicy, ProofStatus::Unknown, None).unwrap();
        assert!(matches!(ledger.seal(&policy()), Err(ProofLedgerError::PolicyRejected(_))));
        assert!(!ledger.is_sealed());
    }

    #[test]
    fn seals_only_after_required_proofs_pass_and_then_is_immutable() {
        let mut ledger = ProofLedger::new();
        ledger.record(ProofRequirement::TypeSafety, ProofStatus::Proved, Some(DIGEST_A)).unwrap();
        ledger.record(ProofRequirement::EffectPolicy, ProofStatus::RuntimeChecked, Some(DIGEST_B)).unwrap();
        ledger.seal(&policy()).unwrap();
        assert!(ledger.is_sealed());
        assert_eq!(ledger.record(ProofRequirement::Bounds, ProofStatus::Unknown, None), Err(ProofLedgerError::AlreadySealed));
        assert_eq!(ledger.seal(&policy()), Err(ProofLedgerError::AlreadySealed));
    }

    #[test]
    fn digest_validation_rejects_uppercase_and_wrong_length() {
        let mut ledger = ProofLedger::new();
        assert_eq!(ledger.record(ProofRequirement::Bounds, ProofStatus::Unknown, Some("ABC")), Err(ProofLedgerError::InvalidEvidenceDigest));
        assert!(ledger.receipts().is_empty());
    }
}
