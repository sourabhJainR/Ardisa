//! Gate-scoped authenticated Proof evidence admission.
//! Production replay stores must be durable and rollback-resistant.
use std::{collections::{BTreeMap, BTreeSet}, sync::{Arc, Mutex}};
use sha2::{Digest, Sha256};

pub const MAX_ITEMS: usize = 4096;
pub const MAX_EVIDENCE_BYTES: usize = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofEvidence {
    pub gate_id: String,
    pub requirement: String,
    pub status: String,
    pub producer_id: String,
    pub key_id: String,
    pub sequence: u64,
    pub artifact_sha256: String,
    pub evidence: Vec<u8>,
    pub signature: Vec<u8>,
}

pub trait SignatureVerifier: Send + Sync {
    fn verify(&self, key_id: &str, payload: &[u8], signature: &[u8]) -> Result<bool, String>;
}

#[derive(Default, Clone)]
pub struct TrustRegistry {
    keys: BTreeMap<(String, String), Arc<dyn SignatureVerifier>>,
}
impl TrustRegistry {
    pub fn register(&mut self, producer: &str, key: &str, verifier: Arc<dyn SignatureVerifier>) -> Result<(), AdmissionError> {
        valid_id(producer)?; valid_id(key)?;
        let id = (producer.to_owned(), key.to_owned());
        if self.keys.contains_key(&id) { return Err(AdmissionError::DuplicateKey); }
        self.keys.insert(id, verifier); Ok(())
    }
}

pub trait ReplayStore: Send + Sync {
    /// Atomically compare-and-advance all counters. Persistent implementations must
    /// protect this state against backup/restore rollback, not only process restarts.
    fn commit_batch(&self, advances: &[(String, String, u64)]) -> Result<(), AdmissionError>;
}
#[derive(Default)]
pub struct MemoryReplayStore(Mutex<BTreeMap<(String, String), u64>>);
impl MemoryReplayStore {
    pub fn latest(&self, producer: &str, key: &str) -> Option<u64> {
        self.0.lock().ok()?.get(&(producer.into(), key.into())).copied()
    }
}
impl ReplayStore for MemoryReplayStore {
    fn commit_batch(&self, advances: &[(String, String, u64)]) -> Result<(), AdmissionError> {
        let mut state = self.0.lock().map_err(|_| AdmissionError::ReplayStoreUnavailable)?;
        let mut pending = BTreeMap::new();
        for (producer, key, sequence) in advances {
            let id = (producer.clone(), key.clone());
            if *sequence == 0 || state.get(&id).is_some_and(|old| sequence <= old)
                || pending.get(&id).is_some_and(|old| sequence <= old) {
                return Err(AdmissionError::ReplayOrRollback);
            }
            pending.insert(id, *sequence);
        }
        state.extend(pending); Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidId, InvalidDigest, TooManyItems, EvidenceTooLarge, EmptyEvidence, EmptySignature,
    UnknownGate(String), MissingGate(String), UnknownRequirement(String, String),
    MissingRequirement(String, String), DuplicateOutcome(String, String), InvalidStatus,
    DuplicateKey, UnknownProducerKey, SignatureRejected, VerifierUnavailable(String),
    ReplayOrRollback, ReplayStoreUnavailable,
}

fn valid_id(s: &str) -> Result<(), AdmissionError> {
    if s.is_empty() || s.len() > 128 || s.trim() != s
        || !s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) {
        return Err(AdmissionError::InvalidId);
    }
    Ok(())
}
fn valid_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}
fn field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes()); out.extend_from_slice(bytes);
}
impl ProofEvidence {
    pub fn canonical_payload(&self) -> Vec<u8> {
        let mut out = b"ARDISA-PROOF-EVIDENCE-V1".to_vec();
        for part in [self.producer_id.as_bytes(), self.key_id.as_bytes(), self.gate_id.as_bytes(),
            self.requirement.as_bytes(), self.status.as_bytes(), &self.sequence.to_be_bytes(),
            self.artifact_sha256.as_bytes(), digest(&self.evidence).as_bytes()] { field(&mut out, part); }
        out
    }
}

/// required_by_gate must contain every Proof declaration in the source module.
/// Evidence is authenticated and replay-checked before scoped native outcomes are returned.
pub fn admit(
    required_by_gate: &BTreeMap<String, BTreeSet<String>>,
    evidence: &[ProofEvidence],
    artifact_sha256: &str,
    trust: &TrustRegistry,
    replay: &dyn ReplayStore,
) -> Result<Vec<String>, AdmissionError> {
    if !valid_digest(artifact_sha256) { return Err(AdmissionError::InvalidDigest); }
    if required_by_gate.is_empty() { return Err(AdmissionError::MissingGate("<none>".into())); }
    if evidence.len() > MAX_ITEMS { return Err(AdmissionError::TooManyItems); }
    for (gate, requirements) in required_by_gate {
        valid_id(gate)?;
        if requirements.is_empty() { return Err(AdmissionError::MissingGate(gate.clone())); }
        for req in requirements { valid_id(req)?; }
    }
    let mut seen = BTreeMap::<(String, String), String>::new();
    let mut gates = BTreeSet::new();
    let mut advances = Vec::new();
    let mut outcomes = Vec::new();
    for item in evidence {
        for id in [&item.gate_id, &item.requirement, &item.producer_id, &item.key_id] { valid_id(id)?; }
        let requirements = required_by_gate.get(&item.gate_id)
            .ok_or_else(|| AdmissionError::UnknownGate(item.gate_id.clone()))?;
        if !requirements.contains(&item.requirement) {
            return Err(AdmissionError::UnknownRequirement(item.gate_id.clone(), item.requirement.clone()));
        }
        if !matches!(item.status.as_str(), "proved" | "runtime_checked") { return Err(AdmissionError::InvalidStatus); }
        if item.evidence.is_empty() { return Err(AdmissionError::EmptyEvidence); }
        if item.evidence.len() > MAX_EVIDENCE_BYTES { return Err(AdmissionError::EvidenceTooLarge); }
        if item.signature.is_empty() { return Err(AdmissionError::EmptySignature); }
        if item.artifact_sha256 != artifact_sha256 { return Err(AdmissionError::InvalidDigest); }
        let key = (item.gate_id.clone(), item.requirement.clone());
        if seen.insert(key, item.status.clone()).is_some() {
            return Err(AdmissionError::DuplicateOutcome(item.gate_id.clone(), item.requirement.clone()));
        }
        let verifier = trust.keys.get(&(item.producer_id.clone(), item.key_id.clone()))
            .ok_or(AdmissionError::UnknownProducerKey)?;
        match verifier.verify(&item.key_id, &item.canonical_payload(), &item.signature) {
            Ok(true) => (), Ok(false) => return Err(AdmissionError::SignatureRejected),
            Err(e) => return Err(AdmissionError::VerifierUnavailable(e)),
        }
        gates.insert(item.gate_id.clone());
        advances.push((item.producer_id.clone(), item.key_id.clone(), item.sequence));
        outcomes.push(format!("{}:{}={}", item.gate_id, item.requirement, item.status));
    }
    for (gate, requirements) in required_by_gate {
        if !gates.contains(gate) { return Err(AdmissionError::MissingGate(gate.clone())); }
        for req in requirements {
            if !seen.contains_key(&(gate.clone(), req.clone())) {
                return Err(AdmissionError::MissingRequirement(gate.clone(), req.clone()));
            }
        }
    }
    replay.commit_batch(&advances)?;
    outcomes.sort(); Ok(outcomes)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Echo;
    impl SignatureVerifier for Echo {
        fn verify(&self, _: &str, payload: &[u8], signature: &[u8]) -> Result<bool, String> { Ok(payload == signature) }
    }
    fn declarations() -> BTreeMap<String, BTreeSet<String>> {
        BTreeMap::from([
            ("BuildGate".into(), BTreeSet::from(["type_safety".into()])),
            ("ReleaseGate".into(), BTreeSet::from(["bounds".into()])),
        ])
    }
    fn evidence(gate: &str, req: &str, seq: u64) -> ProofEvidence {
        let mut e = ProofEvidence { gate_id: gate.into(), requirement: req.into(), status: "proved".into(),
            producer_id: "ci".into(), key_id: "ci-key".into(), sequence: seq, artifact_sha256: "a".repeat(64),
            evidence: format!("{gate}:{req}").into_bytes(), signature: vec![] };
        e.signature = e.canonical_payload(); e
    }
    fn registry() -> TrustRegistry {
        let mut r = TrustRegistry::default(); r.register("ci", "ci-key", Arc::new(Echo)).unwrap(); r
    }
    #[test] fn validates_all_gates_and_scopes_outcomes() {
        let replay = MemoryReplayStore::default();
        assert_eq!(admit(&declarations(), &[evidence("BuildGate","type_safety",1), evidence("ReleaseGate","bounds",2)],
            &"a".repeat(64), &registry(), &replay).unwrap(),
            vec!["BuildGate:type_safety=proved", "ReleaseGate:bounds=proved"]);
    }
    #[test] fn rejects_unknown_and_cross_gate_evidence() {
        let replay = MemoryReplayStore::default();
        assert_eq!(admit(&declarations(), &[evidence("Unknown","bounds",1)], &"a".repeat(64), &registry(), &replay),
            Err(AdmissionError::UnknownGate("Unknown".into())));
        assert!(matches!(admit(&declarations(), &[evidence("BuildGate","bounds",1)], &"a".repeat(64), &registry(), &replay),
            Err(AdmissionError::UnknownRequirement(_, _))));
    }
    #[test] fn rejects_duplicate_malformed_and_missing_evidence() {
        let replay = MemoryReplayStore::default(); let one = evidence("BuildGate","type_safety",1);
        assert!(matches!(admit(&declarations(), &[one.clone(),one.clone()], &"a".repeat(64), &registry(), &replay),
            Err(AdmissionError::DuplicateOutcome(_, _))));
        let mut bad = evidence("BuildGate","type_safety",1); bad.signature.clear();
        assert_eq!(admit(&declarations(), &[bad], &"a".repeat(64), &registry(), &replay), Err(AdmissionError::EmptySignature));
        assert_eq!(admit(&declarations(), &[one], &"a".repeat(64), &registry(), &replay),
            Err(AdmissionError::MissingGate("ReleaseGate".into())));
    }
    #[test] fn rejects_tampering_replay_and_rollback_atomically() {
        let replay = MemoryReplayStore::default();
        let mut bad = evidence("BuildGate","type_safety",1); bad.status = "runtime_checked".into();
        assert_eq!(admit(&declarations(), &[bad], &"a".repeat(64), &registry(), &replay), Err(AdmissionError::SignatureRejected));
        let bundle = || vec![evidence("BuildGate","type_safety",1), evidence("ReleaseGate","bounds",2)];
        admit(&declarations(), &bundle(), &"a".repeat(64), &registry(), &replay).unwrap();
        assert_eq!(admit(&declarations(), &bundle(), &"a".repeat(64), &registry(), &replay), Err(AdmissionError::ReplayOrRollback));
        assert_eq!(replay.latest("ci","ci-key"), Some(2));
    }
}
