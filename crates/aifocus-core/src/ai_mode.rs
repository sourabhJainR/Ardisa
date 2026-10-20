//! AI Mode policy primitives.
//!
//! These types model security and verification policy independently of source
//! syntax. They do not create OS/hardware isolation and do not claim a program
//! is secure merely because a policy can be represented. Source parsing and
//! compiler enforcement are introduced in later phases.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownProofPolicy {
    /// Reject compilation when a required proof remains unknown.
    Reject,
    /// Permit the build only when the caller records the unknown result.
    Record,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    Clock,
    Random,
    FilesystemRead,
    FilesystemWrite,
    Network,
    Process,
    Secrets,
    UnsafeInterop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProofRequirement {
    TypeSafety,
    Bounds,
    EffectPolicy,
    ContractPostconditions,
    ConstantTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofStatus {
    Proved,
    RuntimeChecked,
    Unknown,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracePolicy {
    pub required: bool,
    /// Include a provider/model identifier and prompt/session digest only.
    /// Raw prompts, secrets, source, and embeddings are never implied.
    pub include_ai_metadata: bool,
    pub require_signature: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultPolicy {
    pub allowed_capabilities: BTreeSet<Capability>,
}

impl VaultPolicy {
    pub fn deny_all() -> Self {
        Self {
            allowed_capabilities: BTreeSet::new(),
        }
    }

    pub fn permits(&self, capability: Capability) -> bool {
        self.allowed_capabilities.contains(&capability)
    }

    /// Validate observed effects against explicit grants. This is a policy
    /// primitive; the compiler/runtime must call it at a real enforcement
    /// boundary for it to provide protection.
    pub fn validate_effects(
        &self,
        observed: impl IntoIterator<Item = Capability>,
    ) -> Result<(), PolicyViolation> {
        let denied = observed
            .into_iter()
            .filter(|capability| !self.permits(*capability))
            .collect::<BTreeSet<_>>();
        if denied.is_empty() {
            Ok(())
        } else {
            Err(PolicyViolation::UndeclaredCapabilities(denied))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofPolicy {
    pub required: BTreeSet<ProofRequirement>,
    pub on_unknown: UnknownProofPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AiModePolicy {
    pub trace: TracePolicy,
    pub vault: VaultPolicy,
    pub proof: ProofPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyViolation {
    UndeclaredCapabilities(BTreeSet<Capability>),
    MissingProofResult(ProofRequirement),
    FailedProof(ProofRequirement),
    UnknownProof(ProofRequirement),
    MissingTrace,
    MissingSignature,
}

/// Runtime state machine for an explicitly declared Phase transition graph.
///
/// This is a runtime API only: source-level `phase` declarations are still
/// rejected by the executable compiler until they are encoded in ARDISA-EXEC-V1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseMachine {
    current: String,
    transitions: BTreeSet<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseMachineError {
    EmptyState,
    EmptyTransitions,
    SelfTransition(String),
    DuplicateTransition { from: String, to: String },
    InitialStateUnknown(String),
    InvalidTransition { from: String, to: String },
}

impl PhaseMachine {
    /// Build a machine from an initial state and a finite set of allowed edges.
    /// Validation is completed before a machine is returned.
    pub fn new(
        initial: impl Into<String>,
        transitions: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self, PhaseMachineError> {
        let initial = initial.into();
        if initial.trim().is_empty() {
            return Err(PhaseMachineError::EmptyState);
        }

        let mut edges = BTreeSet::new();
        let mut states = BTreeSet::new();
        for (from, to) in transitions {
            if from.trim().is_empty() || to.trim().is_empty() {
                return Err(PhaseMachineError::EmptyState);
            }
            if from == to {
                return Err(PhaseMachineError::SelfTransition(from));
            }
            if !edges.insert((from.clone(), to.clone())) {
                return Err(PhaseMachineError::DuplicateTransition { from, to });
            }
            states.insert(from);
            states.insert(to);
        }

        if edges.is_empty() {
            return Err(PhaseMachineError::EmptyTransitions);
        }
        if !states.contains(&initial) {
            return Err(PhaseMachineError::InitialStateUnknown(initial));
        }

        Ok(Self { current: initial, transitions: edges })
    }

    pub fn current(&self) -> &str {
        &self.current
    }

    pub fn can_transition_to(&self, next: &str) -> bool {
        self.transitions.contains(&(self.current.clone(), next.to_string()))
    }

    /// Apply only a declared outgoing edge. On rejection, current state is
    /// unchanged, so failed transitions cannot partially mutate the machine.
    pub fn transition(&mut self, next: &str) -> Result<(), PhaseMachineError> {
        if next.trim().is_empty() {
            return Err(PhaseMachineError::EmptyState);
        }
        if !self.can_transition_to(next) {
            return Err(PhaseMachineError::InvalidTransition {
                from: self.current.clone(),
                to: next.to_string(),
            });
        }
        self.current = next.to_string();
        Ok(())
    }
}

/// A runtime value whose invariant is checked before every committed update.
///
/// This generic API is not yet wired to source-level `cell` declarations or
/// native bytecode. It provides transactional invariant enforcement for callers
/// that explicitly supply a predicate; it is not a sandbox or formal proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellInvariantError {
    InitialValueRejected,
    InvariantViolation,
}

pub struct InvariantCell<T, F>
where
    F: Fn(&T) -> bool,
{
    value: T,
    invariant: F,
}

impl<T, F> InvariantCell<T, F>
where
    F: Fn(&T) -> bool,
{
    /// Construct only when the initial value satisfies the invariant.
    pub fn new(value: T, invariant: F) -> Result<Self, CellInvariantError> {
        if !invariant(&value) {
            return Err(CellInvariantError::InitialValueRejected);
        }
        Ok(Self { value, invariant })
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    /// Validate before commit; rejected values leave the previous value intact.
    pub fn set(&mut self, next: T) -> Result<(), CellInvariantError> {
        if !(self.invariant)(&next) {
            return Err(CellInvariantError::InvariantViolation);
        }
        self.value = next;
        Ok(())
    }

    /// Apply a candidate update to a clone and commit only if it remains valid.
    pub fn try_update(
        &mut self,
        update: impl FnOnce(&mut T),
    ) -> Result<(), CellInvariantError>
    where
        T: Clone,
    {
        let mut candidate = self.value.clone();
        update(&mut candidate);
        self.set(candidate)
    }
}

impl AiModePolicy {
    pub fn validate_proofs(
        &self,
        results: &BTreeMap<ProofRequirement, ProofStatus>,
    ) -> Result<(), Vec<PolicyViolation>> {
        let mut violations = Vec::new();
        for requirement in &self.proof.required {
            match results.get(requirement) {
                None => violations.push(PolicyViolation::MissingProofResult(*requirement)),
                Some(ProofStatus::Failed) => {
                    violations.push(PolicyViolation::FailedProof(*requirement))
                }
                Some(ProofStatus::Unknown)
                    if self.proof.on_unknown == UnknownProofPolicy::Reject =>
                {
                    violations.push(PolicyViolation::UnknownProof(*requirement))
                }
                // Runtime checks are evidence of runtime enforcement, not a
                // formal proof. Keep the status distinct in the report.
                Some(ProofStatus::Proved | ProofStatus::RuntimeChecked | ProofStatus::Unknown) => {}
            }
        }
        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Validate build-attestation prerequisites. Cryptographic signing itself
    /// must be performed by a trusted signing service, not this policy model.
    pub fn validate_trace(
        &self,
        trace_present: bool,
        signature_verified: bool,
    ) -> Result<(), PolicyViolation> {
        if self.trace.required && !trace_present {
            return Err(PolicyViolation::MissingTrace);
        }
        if self.trace.required && self.trace.require_signature && !signature_verified {
            return Err(PolicyViolation::MissingSignature);
        }
        Ok(())
    }
}


/// Maximum number of receipts retained by one in-memory trace ledger.
pub const MAX_TRACE_RECEIPTS: usize = 4096;
pub const MAX_TRACE_TOKEN_BYTES: usize = 128;
pub const MAX_TRACE_SIGNATURE_BYTES: usize = 512;

/// Minimal signer boundary. Production implementations must use a configured,
/// protected signing key and a reviewed signature algorithm; this trait itself
/// does not provide cryptography or key management.
pub trait TraceSigner {
    fn sign(&self, canonical_payload: &[u8]) -> Result<Vec<u8>, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    InvalidToken(&'static str),
    InvalidDigest(&'static str),
    MetadataNotAllowed,
    LedgerLimitExceeded,
    InvalidSignatureLength,
    SigningFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEventInput {
    pub event_kind: String,
    pub actor: String,
    pub subject_sha256: String,
    pub outcome: String,
    /// Optional provider/model identifier; never store a raw prompt here.
    pub provider_id: Option<String>,
    /// Optional digest of the session/prompt, not the session/prompt itself.
    pub session_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceReceipt {
    pub sequence: u64,
    pub event_kind: String,
    pub actor: String,
    pub subject_sha256: String,
    pub outcome: String,
    pub provider_id: Option<String>,
    pub session_sha256: Option<String>,
    /// Exact previous receipt signature, providing a tamper-evident chain when
    /// callers verify each signature and predecessor link with the trusted key.
    pub previous_signature: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TraceLedger {
    receipts: Vec<TraceReceipt>,
}

impl TraceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn receipts(&self) -> &[TraceReceipt] {
        &self.receipts
    }

    /// Sign before committing. Validation or signer failure leaves the ledger
    /// unchanged. Raw prompt/source/secret fields are intentionally not accepted.
    pub fn record<S: TraceSigner>(
        &mut self,
        policy: &TracePolicy,
        event: TraceEventInput,
        signer: Option<&S>,
    ) -> Result<&TraceReceipt, TraceError> {
        if self.receipts.len() >= MAX_TRACE_RECEIPTS {
            return Err(TraceError::LedgerLimitExceeded);
        }
        validate_trace_token(&event.event_kind, "event_kind")?;
        validate_trace_token(&event.actor, "actor")?;
        validate_trace_token(&event.outcome, "outcome")?;
        validate_trace_digest(&event.subject_sha256, "subject_sha256")?;

        if !policy.include_ai_metadata && (event.provider_id.is_some() || event.session_sha256.is_some()) {
            return Err(TraceError::MetadataNotAllowed);
        }
        if let Some(provider) = &event.provider_id {
            validate_trace_token(provider, "provider_id")?;
        }
        if let Some(digest) = &event.session_sha256 {
            validate_trace_digest(digest, "session_sha256")?;
        }
        if policy.required && policy.require_signature && signer.is_none() {
            return Err(TraceError::SigningFailed("required trace signer is not configured".into()));
        }

        let previous_signature = self.receipts.last().map(|r| r.signature.clone()).unwrap_or_default();
        let sequence = self.receipts.len() as u64;
        let payload = canonical_trace_payload(sequence, &event, &previous_signature);
        let signature = match signer {
            Some(signer) => signer.sign(payload.as_bytes()).map_err(TraceError::SigningFailed)?,
            None => Vec::new(),
        };
        if policy.require_signature && signature.is_empty() {
            return Err(TraceError::InvalidSignatureLength);
        }
        if signature.len() > MAX_TRACE_SIGNATURE_BYTES {
            return Err(TraceError::InvalidSignatureLength);
        }

        self.receipts.push(TraceReceipt {
            sequence,
            event_kind: event.event_kind,
            actor: event.actor,
            subject_sha256: event.subject_sha256,
            outcome: event.outcome,
            provider_id: event.provider_id,
            session_sha256: event.session_sha256,
            previous_signature,
            signature,
        });
        Ok(self.receipts.last().expect("receipt was just appended"))
    }
}

fn validate_trace_token(value: &str, field: &'static str) -> Result<(), TraceError> {
    if value.is_empty()
        || value.len() > MAX_TRACE_TOKEN_BYTES
        || !value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':'))
    {
        return Err(TraceError::InvalidToken(field));
    }
    Ok(())
}

fn validate_trace_digest(value: &str, field: &'static str) -> Result<(), TraceError> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(TraceError::InvalidDigest(field));
    }
    Ok(())
}

fn canonical_trace_payload(sequence: u64, event: &TraceEventInput, previous_signature: &[u8]) -> String {
    let previous_hex = previous_signature.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!(
        "ARDISA-TRACE-V1|{sequence}|{}|{}|{}|{}|{}|{}|{previous_hex}",
        event.event_kind,
        event.actor,
        event.subject_sha256.to_ascii_lowercase(),
        event.outcome,
        event.provider_id.as_deref().unwrap_or("-"),
        event.session_sha256.as_deref().unwrap_or("-").to_ascii_lowercase(),
        "redacted"
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AiModePolicy {
        AiModePolicy {
            trace: TracePolicy {
                required: true,
                include_ai_metadata: false,
                require_signature: true,
            },
            vault: VaultPolicy::deny_all(),
            proof: ProofPolicy {
                required: [ProofRequirement::TypeSafety, ProofRequirement::EffectPolicy]
                    .into_iter()
                    .collect(),
                on_unknown: UnknownProofPolicy::Reject,
            },
        }
    }

    #[test]
    fn vault_denies_undeclared_effects_and_accepts_explicit_grants() {
        let mut vault = VaultPolicy::deny_all();
        assert!(vault.validate_effects([Capability::Network]).is_err());
        vault.allowed_capabilities.insert(Capability::Clock);
        assert!(vault.validate_effects([Capability::Clock]).is_ok());
        assert!(vault.validate_effects([Capability::Clock, Capability::Network]).is_err());
    }

    #[test]
    fn required_proofs_fail_closed_when_missing_failed_or_unknown() {
        let policy = policy();
        assert!(policy.validate_proofs(&BTreeMap::new()).is_err());

        let results = [
            (ProofRequirement::TypeSafety, ProofStatus::Proved),
            (ProofRequirement::EffectPolicy, ProofStatus::Failed),
        ]
        .into_iter()
        .collect();
        assert!(policy.validate_proofs(&results).is_err());

        let results = [
            (ProofRequirement::TypeSafety, ProofStatus::Proved),
            (ProofRequirement::EffectPolicy, ProofStatus::Unknown),
        ]
        .into_iter()
        .collect();
        assert!(policy.validate_proofs(&results).is_err());
    }

    #[test]
    fn runtime_checked_is_not_mislabelled_as_formally_proved() {
        let policy = policy();
        let results = [
            (ProofRequirement::TypeSafety, ProofStatus::RuntimeChecked),
            (ProofRequirement::EffectPolicy, ProofStatus::Proved),
        ]
        .into_iter()
        .collect();
        assert!(policy.validate_proofs(&results).is_ok());
        assert_eq!(results[&ProofRequirement::TypeSafety], ProofStatus::RuntimeChecked);
    }

    #[test]
    fn trace_policy_requires_presence_and_configured_signature() {
        let policy = policy();
        assert_eq!(
            policy.validate_trace(false, false),
            Err(PolicyViolation::MissingTrace)
        );
        assert_eq!(
            policy.validate_trace(true, false),
            Err(PolicyViolation::MissingSignature)
        );
        assert!(policy.validate_trace(true, true).is_ok());
    }

    #[test]
    fn optional_trace_does_not_require_an_attestation() {
        let mut policy = policy();
        policy.trace.required = false;
        assert!(policy.validate_trace(false, false).is_ok());
    }

    #[test]
    fn phase_machine_accepts_only_declared_edges() {
        let mut machine = PhaseMachine::new(
            "draft",
            vec![
                ("draft".to_string(), "review".to_string()),
                ("review".to_string(), "release".to_string()),
            ],
        ).expect("valid graph");

        assert!(machine.can_transition_to("review"));
        machine.transition("review").expect("declared edge");
        assert_eq!(machine.current(), "review");

        assert_eq!(
            machine.transition("release"),
            Ok(()),
            "review -> release is declared"
        );
        assert_eq!(machine.current(), "release");
        assert_eq!(
            machine.transition("draft"),
            Err(PhaseMachineError::InvalidTransition {
                from: "release".to_string(),
                to: "draft".to_string(),
            })
        );
        assert_eq!(machine.current(), "release", "rejected transitions must not mutate state");
    }

    #[test]
    fn phase_machine_rejects_invalid_graphs_before_runtime() {
        assert_eq!(
            PhaseMachine::new("", vec![("a".to_string(), "b".to_string())]),
            Err(PhaseMachineError::EmptyState)
        );
        assert_eq!(
            PhaseMachine::new("a", Vec::new()),
            Err(PhaseMachineError::EmptyTransitions)
        );
        assert_eq!(
            PhaseMachine::new("a", vec![("a".to_string(), "a".to_string())]),
            Err(PhaseMachineError::SelfTransition("a".to_string()))
        );
        assert_eq!(
            PhaseMachine::new("a", vec![("a".to_string(), "b".to_string()), ("a".to_string(), "b".to_string())]),
            Err(PhaseMachineError::DuplicateTransition { from: "a".to_string(), to: "b".to_string() })
        );
        assert_eq!(
            PhaseMachine::new("unknown", vec![("a".to_string(), "b".to_string())]),
            Err(PhaseMachineError::InitialStateUnknown("unknown".to_string()))
        );
    }
}
#[cfg(test)]
mod invariant_cell_tests {
    use super::*;

    #[test]
    fn invariant_cell_rejects_invalid_initial_value_and_update() {
        let initial = InvariantCell::new(3_i64, |value| *value >= 0)
            .expect("non-negative initial value should satisfy invariant");
        assert_eq!(*initial.get(), 3);
        assert_eq!(
            InvariantCell::new(-1_i64, |value| *value >= 0).err(),
            Some(CellInvariantError::InitialValueRejected)
        );

        let mut cell = initial;
        assert_eq!(cell.set(-5), Err(CellInvariantError::InvariantViolation));
        assert_eq!(*cell.get(), 3, "rejected update must preserve previous state");
    }

    #[test]
    fn invariant_cell_update_is_transactional() {
        let mut cell = InvariantCell::new(vec![1, 2], |values: &Vec<i32>| values.len() <= 3)
            .expect("initial vector is within bound");
        cell.try_update(|values| values.push(3)).expect("within-bound update");
        assert_eq!(cell.get(), &vec![1, 2, 3]);

        assert_eq!(
            cell.try_update(|values| values.push(4)),
            Err(CellInvariantError::InvariantViolation)
        );
        assert_eq!(
            cell.get(),
            &vec![1, 2, 3],
            "failed candidate update must not leak partial mutation"
        );
    }
}
