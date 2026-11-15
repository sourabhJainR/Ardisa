//! Explicit trust registry for Trace receipt verification.
//!
//! The registry maps operator-configured key identifiers to verifier instances.
//! It never trusts a key identifier supplied by a receipt or event. Callers must
//! provision verifier instances from an authenticated configuration source.
//! This is a host-side trust boundary, not a key store or cryptographic algorithm.

use std::collections::BTreeMap;

use crate::ai_mode::{TraceError, TraceLedger, TracePolicy, TraceVerifier};

pub const MAX_TRACE_TRUSTED_KEYS: usize = 1024;
pub const MAX_TRACE_KEY_ID_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceTrustError {
    InvalidKeyId,
    DuplicateKeyId,
    UnknownKeyId,
    UnattributedReceipt,
    KeyRevoked,
    RegistryLimitExceeded,
    ReceiptVerification(TraceError),
}

struct TraceTrustedKey<V> {
    verifier: V,
    /// First receipt sequence that must not be trusted under historical policy.
    /// None means active. A cutoff of zero invalidates every receipt for this key.
    revoked_from_sequence: Option<u64>,
}

/// Operator-configured mapping from stable key IDs to verifier instances.
///
/// Revocation retains the verifier so historical receipts can be checked under
/// an explicit cutoff policy. Current-trust verification rejects every revoked
/// key. Registration remains an explicit trust decision.
pub struct TraceKeyRegistry<V> {
    verifiers: BTreeMap<String, TraceTrustedKey<V>>,
}

impl<V: TraceVerifier> Default for TraceKeyRegistry<V> {
    fn default() -> Self {
        Self { verifiers: BTreeMap::new() }
    }
}

impl<V: TraceVerifier> TraceKeyRegistry<V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a trusted verifier. Existing IDs cannot be silently replaced.
    pub fn register(&mut self, key_id: impl Into<String>, verifier: V) -> Result<(), TraceTrustError> {
        let key_id = key_id.into();
        validate_key_id(&key_id)?;
        if self.verifiers.contains_key(&key_id) {
            return Err(TraceTrustError::DuplicateKeyId);
        }
        if self.verifiers.len() >= MAX_TRACE_TRUSTED_KEYS {
            return Err(TraceTrustError::RegistryLimitExceeded);
        }
        self.verifiers.insert(key_id, TraceTrustedKey {
            verifier,
            revoked_from_sequence: None,
        });
        Ok(())
    }

    /// Revoke a key immediately. Its verifier is retained only for audit
    /// inspection; no receipt signed by it is trusted by either verification API.
    pub fn revoke(&mut self, key_id: &str) -> Result<(), TraceTrustError> {
        self.revoke_from_sequence(key_id, 0)
    }

    /// Revoke a key from the first untrusted receipt sequence onward.
    ///
    /// Historical verification may accept earlier receipts from this key, but
    /// current-trust verification rejects the key in full. The caller must
    /// choose the cutoff from an independently trusted incident/configuration
    /// record; this method does not establish when compromise occurred.
    pub fn revoke_from_sequence(
        &mut self,
        key_id: &str,
        first_untrusted_sequence: u64,
    ) -> Result<(), TraceTrustError> {
        validate_key_id(key_id)?;
        let entry = self.verifiers.get_mut(key_id).ok_or(TraceTrustError::UnknownKeyId)?;
        entry.revoked_from_sequence = Some(
            entry.revoked_from_sequence
                .map_or(first_untrusted_sequence, |existing| existing.min(first_untrusted_sequence)),
        );
        Ok(())
    }

    /// Re-activation is deliberately unsupported. Register a new key ID for
    /// rotation rather than silently restoring a revoked identity.
    pub fn contains(&self, key_id: &str) -> bool {
        self.verifiers.get(key_id).is_some_and(|entry| entry.revoked_from_sequence.is_none())
    }

    pub fn trusted_key_ids(&self) -> impl Iterator<Item = &str> {
        self.verifiers.iter()
            .filter(|(_, entry)| entry.revoked_from_sequence.is_none())
            .map(|(key_id, _)| key_id.as_str())
    }

    /// Verify a complete ledger using only currently active signer identities.
    /// Legacy V1 receipts without an identity and all revoked identities fail closed.
    pub fn verify_chain(
        &self,
        ledger: &TraceLedger,
        policy: &TracePolicy,
    ) -> Result<(), TraceTrustError> {
        self.verify_chain_internal(ledger, policy, false)
    }

    /// Verify historical receipts against the revocation cutoff recorded for
    /// each key. A receipt is accepted only when its sequence is strictly below
    /// that key's first-untrusted sequence. This is an audit policy, not proof
    /// that the key was uncompromised at signing time.
    pub fn verify_historical_chain(
        &self,
        ledger: &TraceLedger,
        policy: &TracePolicy,
    ) -> Result<(), TraceTrustError> {
        self.verify_chain_internal(ledger, policy, true)
    }

    fn verify_chain_internal(
        &self,
        ledger: &TraceLedger,
        policy: &TracePolicy,
        allow_pre_revocation_history: bool,
    ) -> Result<(), TraceTrustError> {
        for (sequence, receipt) in ledger.receipts().iter().enumerate() {
            let key_id = receipt.key_id.as_deref().ok_or(TraceTrustError::UnattributedReceipt)?;
            validate_key_id(key_id)?;
            let entry = self.verifiers.get(key_id).ok_or(TraceTrustError::UnknownKeyId)?;
            if let Some(cutoff) = entry.revoked_from_sequence {
                if !allow_pre_revocation_history || (sequence as u64) >= cutoff {
                    return Err(TraceTrustError::KeyRevoked);
                }
            }
        }

        ledger
            .verify_chain_with(policy, |key_id, payload, signature| {
                let key_id = key_id.ok_or_else(|| "legacy receipt has no signer identity".to_string())?;
                let entry = self.verifiers.get(key_id)
                    .ok_or_else(|| format!("untrusted Trace signer: {key_id}"))?;
                entry.verifier.verify(payload, signature)
            })
            .map_err(TraceTrustError::ReceiptVerification)
    }

    /// Compatibility helper for callers verifying a legacy V1 ledger against
    /// one explicitly selected active key. It does not establish receipt-level identity.
    pub fn verify_legacy_chain(
        &self,
        key_id: &str,
        ledger: &TraceLedger,
        policy: &TracePolicy,
    ) -> Result<(), TraceTrustError> {
        validate_key_id(key_id)?;
        if ledger.receipts().iter().any(|receipt| receipt.key_id.is_some()) {
            return Err(TraceTrustError::ReceiptVerification(TraceError::InvalidSignature));
        }
        let entry = self.verifiers.get(key_id).ok_or(TraceTrustError::UnknownKeyId)?;
        if entry.revoked_from_sequence.is_some() {
            return Err(TraceTrustError::KeyRevoked);
        }
        ledger.verify_chain(policy, &entry.verifier).map_err(TraceTrustError::ReceiptVerification)
    }
}

fn validate_key_id(key_id: &str) -> Result<(), TraceTrustError> {
    if key_id.is_empty()
        || key_id.len() > MAX_TRACE_KEY_ID_BYTES
        || !key_id.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':'))
    {
        return Err(TraceTrustError::InvalidKeyId);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_mode::{TraceEventInput, TraceSigner};

    struct KeyedTestVerifier(u8);

    impl TraceVerifier for KeyedTestVerifier {
        fn verify(&self, payload: &[u8], signature: &[u8]) -> Result<bool, String> {
            let expected = vec![payload.iter().fold(self.0, |acc, byte| acc.wrapping_add(*byte))];
            Ok(expected.as_slice() == signature)
        }
    }

    struct KeyedTestSigner(u8);

    impl TraceSigner for KeyedTestSigner {
        fn sign(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
            Ok(vec![payload.iter().fold(self.0, |acc, byte| acc.wrapping_add(*byte))])
        }
    }

    fn policy() -> TracePolicy {
        TracePolicy { required: true, include_ai_metadata: false, require_signature: true }
    }

    fn event() -> TraceEventInput {
        TraceEventInput {
            event_kind: "compile".into(),
            actor: "compiler.v1".into(),
            subject_sha256: "a".repeat(64),
            outcome: "success".into(),
            provider_id: None,
            session_sha256: None,
        }
    }

    #[test]
    fn configured_key_verifies_and_unknown_key_fails_closed() {
        let mut ledger = TraceLedger::new();
        ledger.record_with_key_id(&policy(), event(), "build-key-2026", Some(&KeyedTestSigner(7))).unwrap();

        let mut registry = TraceKeyRegistry::new();
        registry.register("build-key-2026", KeyedTestVerifier(7)).unwrap();
        assert!(registry.verify_chain(&ledger, &policy()).is_ok());
        let mut unknown = TraceLedger::new();
        unknown.record_with_key_id(&policy(), event(), "unconfigured-key", Some(&KeyedTestSigner(7))).unwrap();
        assert_eq!(registry.verify_chain(&unknown, &policy()), Err(TraceTrustError::UnknownKeyId));
    }

    #[test]
    fn key_registry_rejects_invalid_duplicate_and_revoked_ids() {
        let mut registry = TraceKeyRegistry::new();
        assert_eq!(registry.register("contains spaces", KeyedTestVerifier(1)), Err(TraceTrustError::InvalidKeyId));
        registry.register("release-key", KeyedTestVerifier(1)).unwrap();
        assert_eq!(registry.register("release-key", KeyedTestVerifier(2)), Err(TraceTrustError::DuplicateKeyId));
        assert!(registry.contains("release-key"));
        registry.revoke("release-key").unwrap();
        assert!(!registry.contains("release-key"));
        assert_eq!(registry.revoke("release-key"), Ok(()));
        assert_eq!(registry.verify_chain(&TraceLedger::new(), &policy()), Ok(()));
    }

    #[test]
    fn mixed_key_chain_supports_rotation_and_rejects_untrusted_or_tampered_ids() {
        let mut ledger = TraceLedger::new();
        ledger.record_with_key_id(&policy(), event(), "old-key", Some(&KeyedTestSigner(7))).unwrap();
        ledger.record_with_key_id(&policy(), event(), "new-key", Some(&KeyedTestSigner(9))).unwrap();

        let mut registry = TraceKeyRegistry::new();
        registry.register("old-key", KeyedTestVerifier(7)).unwrap();
        registry.register("new-key", KeyedTestVerifier(9)).unwrap();
        assert!(registry.verify_chain(&ledger, &policy()).is_ok());

        let mut tampered_id = ledger.clone();
        tampered_id.receipts_mut_for_test()[1].key_id = Some("old-key".into());
        assert!(matches!(registry.verify_chain(&tampered_id, &policy()),
            Err(TraceTrustError::ReceiptVerification(TraceError::InvalidSignature))));

        let mut unknown = TraceKeyRegistry::new();
        unknown.register("old-key", KeyedTestVerifier(7)).unwrap();
        assert_eq!(unknown.verify_chain(&ledger, &policy()), Err(TraceTrustError::UnknownKeyId));
    }

    #[test]
    fn legacy_receipts_are_not_attributed_by_the_new_registry_api() {
        let mut ledger = TraceLedger::new();
        ledger.record(&policy(), event(), Some(&KeyedTestSigner(7))).unwrap();
        let mut registry = TraceKeyRegistry::new();
        registry.register("legacy-key", KeyedTestVerifier(7)).unwrap();
        assert_eq!(registry.verify_chain(&ledger, &policy()), Err(TraceTrustError::UnattributedReceipt));
        assert!(registry.verify_legacy_chain("legacy-key", &ledger, &policy()).is_ok());
    }

    #[test]
    fn wrong_configured_key_cannot_verify_receipts() {
        let mut ledger = TraceLedger::new();
        ledger.record_with_key_id(&policy(), event(), "wrong-key", Some(&KeyedTestSigner(7))).unwrap();
        let mut registry = TraceKeyRegistry::new();
        registry.register("wrong-key", KeyedTestVerifier(9)).unwrap();
        assert_eq!(registry.verify_chain(&ledger, &policy()),
            Err(TraceTrustError::ReceiptVerification(TraceError::InvalidSignature)));
    }

    #[test]
    fn revocation_cutoff_preserves_only_explicitly_accepted_history() {
        let mut ledger = TraceLedger::new();
        ledger.record_with_key_id(&policy(), event(), "rotating-key", Some(&KeyedTestSigner(7))).unwrap();
        ledger.record_with_key_id(&policy(), event(), "rotating-key", Some(&KeyedTestSigner(7))).unwrap();

        let mut registry = TraceKeyRegistry::new();
        registry.register("rotating-key", KeyedTestVerifier(7)).unwrap();
        registry.revoke_from_sequence("rotating-key", 1).unwrap();

        assert!(!registry.contains("rotating-key"));
        assert_eq!(registry.verify_chain(&ledger, &policy()), Err(TraceTrustError::KeyRevoked));
        assert!(registry.verify_historical_chain(&ledger, &policy()).is_err(),
            "the second receipt is at the first-untrusted sequence and must fail");
        
        let mut earlier_only = TraceLedger::new();
        earlier_only.record_with_key_id(&policy(), event(), "rotating-key", Some(&KeyedTestSigner(7))).unwrap();
        assert!(registry.verify_historical_chain(&earlier_only, &policy()).is_ok());
    }

    #[test]
    fn revocation_cutoff_is_monotonic_and_immediate_revoke_invalidates_all_history() {
        let mut registry = TraceKeyRegistry::new();
        registry.register("key-a", KeyedTestVerifier(1)).unwrap();
        registry.revoke_from_sequence("key-a", 10).unwrap();
        registry.revoke_from_sequence("key-a", 20).unwrap();
        assert!(!registry.contains("key-a"));
        registry.revoke("key-a").unwrap();
        
        let mut ledger = TraceLedger::new();
        ledger.record_with_key_id(&policy(), event(), "key-a", Some(&KeyedTestSigner(1))).unwrap();
        assert_eq!(registry.verify_historical_chain(&ledger, &policy()), Err(TraceTrustError::KeyRevoked));
    }
}
