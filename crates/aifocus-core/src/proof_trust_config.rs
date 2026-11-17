//! Authenticated, versioned Proof producer trust configuration.
//!
//! The store is a security boundary: production implementations must durably
//! compare-and-store against independently protected monotonic state. A plain
//! file, database snapshot, or in-memory store is not rollback-resistant.
use crate::proof_evidence::{AdmissionError, SignatureVerifier, TrustRegistry};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

const DOMAIN: &[u8] = b"ARDISA-PROOF-TRUST-CONFIG-V1";
const ZERO_DIGEST: [u8; 32] = [0; 32];
const MAX_ENTRIES: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KeyState {
    Active,
    Revoked,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrustEntry {
    pub producer_id: String,
    pub key_id: String,
    pub state: KeyState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedTrustConfig {
    pub generation: u64,
    /// Digest of the previous signed configuration. The first generation uses zero.
    pub previous_digest: [u8; 32],
    pub entries: Vec<TrustEntry>,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustConfigVersion {
    pub generation: u64,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustConfigError {
    InvalidId,
    EmptyEntries,
    TooManyEntries,
    DuplicateEntry,
    InvalidSignature,
    SignatureVerifierUnavailable(String),
    InvalidGeneration,
    PreviousDigestMismatch,
    KeyRemoved,
    RevokedKeyReactivated,
    StoreUnavailable(String),
    StoreConflict,
    UnknownProducerKey,
    Admission(AdmissionError),
}

pub trait TrustConfigSignatureVerifier: Send + Sync {
    fn verify(&self, canonical_payload: &[u8], signature: &[u8]) -> Result<bool, String>;
}

pub trait ProducerVerifierResolver: Send + Sync {
    fn resolve(&self, producer_id: &str, key_id: &str)
        -> Result<Arc<dyn SignatureVerifier>, String>;
}

/// Implementations must provide durable atomic compare-and-store and protect
/// the current version against rollback independently of the stored config.
/// The signed config must be durably persisted before the monotonic checkpoint
/// advances. Restoring an old config snapshot must be detected and rejected.
pub trait TrustConfigStore: Send + Sync {
    fn load(&self) -> Result<Option<SignedTrustConfig>, String>;
    fn compare_and_store(
        &self,
        expected: Option<TrustConfigVersion>,
        next: SignedTrustConfig,
    ) -> Result<(), String>;
}

impl SignedTrustConfig {
    /// Canonical, deterministic bytes covered by the root signature.
    pub fn canonical_payload(&self) -> Result<Vec<u8>, TrustConfigError> {
        validate_entries(&self.entries)?;
        let mut entries = self.entries.clone();
        entries.sort();
        let mut out = DOMAIN.to_vec();
        out.extend_from_slice(&self.generation.to_be_bytes());
        out.extend_from_slice(&self.previous_digest);
        out.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for entry in entries {
            put_field(&mut out, entry.producer_id.as_bytes())?;
            put_field(&mut out, entry.key_id.as_bytes())?;
            out.push(match entry.state { KeyState::Active => 1, KeyState::Revoked => 2 });
        }
        Ok(out)
    }

    pub fn version(&self) -> Result<TrustConfigVersion, TrustConfigError> {
        let payload = self.canonical_payload()?;
        let mut hash = Sha256::new();
        hash.update(&payload);
        hash.update(&(self.signature.len() as u64).to_be_bytes());
        hash.update(&self.signature);
        Ok(TrustConfigVersion { generation: self.generation, digest: hash.finalize().into() })
    }
}

/// Verify the current signed configuration and resolve only active producer keys.
/// A production store must reject stale snapshots even after backup restoration.
pub fn load_authenticated_registry(
    store: &dyn TrustConfigStore,
    root: &dyn TrustConfigSignatureVerifier,
    resolver: &dyn ProducerVerifierResolver,
) -> Result<TrustRegistry, TrustConfigError> {
    let config = store.load().map_err(TrustConfigError::StoreUnavailable)?
        .ok_or(TrustConfigError::InvalidGeneration)?;
    verify_signature(&config, root)?;
    if config.generation == 0 { return Err(TrustConfigError::InvalidGeneration); }
    validate_entries(&config.entries)?;
    build_registry(&config, resolver)
}

/// Commit a new signed trust configuration. The root signature is checked before
/// storage mutation; generation, previous digest, and lifecycle transitions are
/// checked against the currently stored configuration.
pub fn commit_authenticated_config(
    store: &dyn TrustConfigStore,
    root: &dyn TrustConfigSignatureVerifier,
    next: SignedTrustConfig,
) -> Result<TrustConfigVersion, TrustConfigError> {
    verify_signature(&next, root)?;
    let current = store.load().map_err(TrustConfigError::StoreUnavailable)?;
    let expected = match current.as_ref() {
        None => {
            if next.generation != 1 || next.previous_digest != ZERO_DIGEST {
                return Err(TrustConfigError::InvalidGeneration);
            }
            None
        }
        Some(previous) => {
            verify_signature(previous, root)?;
            validate_config_chain(&next, Some(previous))?;
            Some(previous.version()?)
        }
    };
    let version = next.version()?;
    store.compare_and_store(expected, next).map_err(|e| {
        if e == "conflict" { TrustConfigError::StoreConflict }
        else { TrustConfigError::StoreUnavailable(e) }
    })?;
    Ok(version)
}

fn verify_signature(
    config: &SignedTrustConfig,
    root: &dyn TrustConfigSignatureVerifier,
) -> Result<(), TrustConfigError> {
    if config.signature.is_empty() { return Err(TrustConfigError::InvalidSignature); }
    let payload = config.canonical_payload()?;
    match root.verify(&payload, &config.signature) {
        Ok(true) => Ok(()),
        Ok(false) => Err(TrustConfigError::InvalidSignature),
        Err(error) => Err(TrustConfigError::SignatureVerifierUnavailable(error)),
    }
}

fn validate_config_chain(
    next: &SignedTrustConfig,
    previous: Option<&SignedTrustConfig>,
) -> Result<(), TrustConfigError> {
    validate_entries(&next.entries)?;
    match previous {
        None => {
            if next.generation != 1 || next.previous_digest != ZERO_DIGEST {
                return Err(TrustConfigError::InvalidGeneration);
            }
        }
        Some(old) => {
            if old.generation.checked_add(1) != Some(next.generation) {
                return Err(TrustConfigError::InvalidGeneration);
            }
            if old.version()?.digest != next.previous_digest {
                return Err(TrustConfigError::PreviousDigestMismatch);
            }
            let old_entries: BTreeMap<_, _> = old.entries.iter()
                .map(|e| ((e.producer_id.as_str(), e.key_id.as_str()), e.state)).collect();
            let new_entries: BTreeMap<_, _> = next.entries.iter()
                .map(|e| ((e.producer_id.as_str(), e.key_id.as_str()), e.state)).collect();
            for (id, state) in old_entries {
                let new_state = new_entries.get(&id).ok_or(TrustConfigError::KeyRemoved)?;
                if state == KeyState::Revoked && *new_state != KeyState::Revoked {
                    return Err(TrustConfigError::RevokedKeyReactivated);
                }
            }
        }
    }
    Ok(())
}

fn build_registry(
    config: &SignedTrustConfig,
    resolver: &dyn ProducerVerifierResolver,
) -> Result<TrustRegistry, TrustConfigError> {
    let mut registry = TrustRegistry::new();
    for entry in &config.entries {
        match entry.state {
            KeyState::Active => {
                let verifier = resolver.resolve(&entry.producer_id, &entry.key_id)
                    .map_err(TrustConfigError::SignatureVerifierUnavailable)?;
                registry.register(&entry.producer_id, &entry.key_id, verifier)
                    .map_err(TrustConfigError::Admission)?;
            }
            KeyState::Revoked => {
                // Preserve revoked IDs as tombstones. A temporary verifier is not
                // needed because the registry rejects this identity before verify.
                let tombstone = Arc::new(RejectAllVerifier);
                registry.register(&entry.producer_id, &entry.key_id, tombstone)
                    .map_err(TrustConfigError::Admission)?;
                registry.revoke(&entry.producer_id, &entry.key_id)
                    .map_err(TrustConfigError::Admission)?;
            }
        }
    }
    Ok(registry)
}

struct RejectAllVerifier;
impl SignatureVerifier for RejectAllVerifier {
    fn verify(&self, _: &str, _: &[u8], _: &[u8]) -> Result<bool, String> { Ok(false) }
}

fn validate_entries(entries: &[TrustEntry]) -> Result<(), TrustConfigError> {
    if entries.is_empty() { return Err(TrustConfigError::EmptyEntries); }
    if entries.len() > MAX_ENTRIES { return Err(TrustConfigError::TooManyEntries); }
    let mut seen = BTreeMap::new();
    for entry in entries {
        valid_id(&entry.producer_id)?;
        valid_id(&entry.key_id)?;
        if seen.insert((&entry.producer_id, &entry.key_id), entry.state).is_some() {
            return Err(TrustConfigError::DuplicateEntry);
        }
    }
    Ok(())
}
fn valid_id(value: &str) -> Result<(), TrustConfigError> {
    if value.is_empty() || value.len() > 128 || value.trim() != value
        || !value.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')) {
        return Err(TrustConfigError::InvalidId);
    }
    Ok(())
}
fn put_field(out: &mut Vec<u8>, value: &[u8]) -> Result<(), TrustConfigError> {
    let length = u16::try_from(value.len()).map_err(|_| TrustConfigError::InvalidId)?;
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct EchoRoot;
    impl TrustConfigSignatureVerifier for EchoRoot {
        fn verify(&self, payload: &[u8], signature: &[u8]) -> Result<bool, String> {
            Ok(payload == signature)
        }
    }
    struct EchoProducer;
    impl SignatureVerifier for EchoProducer {
        fn verify(&self, _: &str, payload: &[u8], signature: &[u8]) -> Result<bool, String> {
            Ok(payload == signature)
        }
    }
    struct Resolver;
    impl ProducerVerifierResolver for Resolver {
        fn resolve(&self, _: &str, _: &str) -> Result<Arc<dyn SignatureVerifier>, String> {
            Ok(Arc::new(EchoProducer))
        }
    }
    #[derive(Default)]
    struct TestStore(Mutex<Option<SignedTrustConfig>>);
    impl TrustConfigStore for TestStore {
        fn load(&self) -> Result<Option<SignedTrustConfig>, String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn compare_and_store(&self, expected: Option<TrustConfigVersion>, next: SignedTrustConfig) -> Result<(), String> {
            let mut state = self.0.lock().unwrap();
            let actual = state.as_ref().map(SignedTrustConfig::version).transpose().map_err(|e| format!("{e:?}"))?;
            if actual != expected { return Err("conflict".into()); }
            *state = Some(next);
            Ok(())
        }
    }
    fn signed(generation: u64, previous_digest: [u8; 32], entries: Vec<TrustEntry>) -> SignedTrustConfig {
        let mut config = SignedTrustConfig { generation, previous_digest, entries, signature: vec![] };
        config.signature = config.canonical_payload().unwrap();
        config
    }
    fn active() -> TrustEntry {
        TrustEntry { producer_id: "ci".into(), key_id: "key-v1".into(), state: KeyState::Active }
    }

    #[test]
    fn accepts_signed_config_and_builds_active_registry() {
        let store = TestStore::default();
        let first = signed(1, ZERO_DIGEST, vec![active()]);
        commit_authenticated_config(&store, &EchoRoot, first).unwrap();
        let registry = load_authenticated_registry(&store, &EchoRoot, &Resolver).unwrap();
        assert!(registry.is_active("ci", "key-v1"));
    }
    #[test]
    fn persists_revocation_as_non_reusable_tombstone() {
        let store = TestStore::default();
        let first = signed(1, ZERO_DIGEST, vec![active()]);
        let first_version = commit_authenticated_config(&store, &EchoRoot, first).unwrap();
        let second = signed(2, first_version.digest, vec![
            TrustEntry { state: KeyState::Revoked, ..active() },
            TrustEntry { producer_id: "ci".into(), key_id: "key-v2".into(), state: KeyState::Active },
        ]);
        commit_authenticated_config(&store, &EchoRoot, second).unwrap();
        let registry = load_authenticated_registry(&store, &EchoRoot, &Resolver).unwrap();
        assert!(!registry.is_active("ci", "key-v1"));
        assert!(registry.is_active("ci", "key-v2"));
    }
    #[test]
    fn rejects_removed_or_reactivated_revoked_keys() {
        let store = TestStore::default();
        let first = signed(1, ZERO_DIGEST, vec![active()]);
        let first_version = commit_authenticated_config(&store, &EchoRoot, first).unwrap();
        let revoked = signed(2, first_version.digest, vec![TrustEntry { state: KeyState::Revoked, ..active() }]);
        let revoked_version = commit_authenticated_config(&store, &EchoRoot, revoked).unwrap();
        let reactivated = signed(3, revoked_version.digest, vec![active()]);
        assert_eq!(commit_authenticated_config(&store, &EchoRoot, reactivated), Err(TrustConfigError::RevokedKeyReactivated));
        let removed = signed(3, revoked_version.digest, vec![TrustEntry {
            producer_id: "ci".into(), key_id: "key-v2".into(), state: KeyState::Active
        }]);
        assert_eq!(commit_authenticated_config(&store, &EchoRoot, removed), Err(TrustConfigError::KeyRemoved));
    }
    #[test]
    fn rejects_bad_signature_and_stale_generation() {
        let store = TestStore::default();
        let mut bad = signed(1, ZERO_DIGEST, vec![active()]);
        bad.signature[0] ^= 0xff;
        assert_eq!(commit_authenticated_config(&store, &EchoRoot, bad), Err(TrustConfigError::InvalidSignature));
        let first = signed(1, ZERO_DIGEST, vec![active()]);
        let version = commit_authenticated_config(&store, &EchoRoot, first).unwrap();
        let stale = signed(3, version.digest, vec![active()]);
        assert_eq!(commit_authenticated_config(&store, &EchoRoot, stale), Err(TrustConfigError::InvalidGeneration));
    }
}
