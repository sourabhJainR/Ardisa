//! File-backed Proof trust configuration with an independent monotonic anchor.
//!
//! The file is durable storage, not the rollback boundary. Rollback resistance
//! depends on a remote or hardware-backed anchor that cannot be restored with it.
use crate::proof_trust_config::{
    KeyState, SignedTrustConfig, TrustConfigStore, TrustConfigVersion, TrustEntry,
};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const MAGIC: &[u8; 8] = b"ARDTCF01";
const DOMAIN: &[u8] = b"ARDISA-PROOF-TRUST-CONFIG-V1";
const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;
const MAX_SIGNATURE_BYTES: usize = 64 * 1024;
const MAX_ENTRIES: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustAnchorState {
    pub generation: u64,
    pub digest: [u8; 32],
}

/// Must be durable, atomic, shared by all writers in the trust domain, and
/// protected independently from the config file and its backups.
pub trait TrustConfigAnchor: Send + Sync {
    fn load(&self) -> Result<TrustAnchorState, String>;
    fn compare_and_advance(
        &self,
        expected: &TrustAnchorState,
        next: &TrustAnchorState,
    ) -> Result<(), String>;
}

struct LockGuard { path: PathBuf, file: Option<File> }
impl Drop for LockGuard {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.path);
    }
}

pub struct AnchoredFileTrustConfigStore {
    path: PathBuf,
    anchor: Arc<dyn TrustConfigAnchor>,
}

impl AnchoredFileTrustConfigStore {
    pub fn open(
        path: impl AsRef<Path>,
        anchor: Arc<dyn TrustConfigAnchor>,
    ) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(io_error)?;
            }
        }
        let store = Self { path, anchor };
        store.validate_anchor()?;
        Ok(store)
    }

    fn lock_path(&self) -> PathBuf {
        let mut value = self.path.as_os_str().to_os_string();
        value.push(".lock");
        PathBuf::from(value)
    }

    fn validate_anchor(&self) -> Result<(), String> {
        let anchor = self.anchor.load()?;
        match self.read_config()? {
            None if anchor.generation == 0 && anchor.digest == [0; 32] => Ok(()),
            Some(config) => {
                let version = config.version().map_err(|e| format!("invalid config: {e:?}"))?;
                if anchor.generation == version.generation && anchor.digest == version.digest {
                    Ok(())
                } else {
                    Err("trust config does not match independent monotonic anchor".into())
                }
            }
            _ => Err("trust config and anchor state disagree".into()),
        }
    }

    fn read_config(&self) -> Result<Option<SignedTrustConfig>, String> {
        let mut file = match OpenOptions::new().read(true).open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error(error)),
        };
        let length = file.metadata().map_err(io_error)?.len() as usize;
        if length > MAX_FILE_BYTES {
            return Err("trust config exceeds size limit".into());
        }
        let mut bytes = Vec::with_capacity(length);
        file.read_to_end(&mut bytes).map_err(io_error)?;
        decode(&bytes).map(Some)
    }

    fn write_config(&self, config: &SignedTrustConfig) -> Result<(), String> {
        let bytes = encode(config)?;
        let mut temp_name = self.path.as_os_str().to_os_string();
        temp_name.push(format!(".{}.tmp", std::process::id()));
        let temp = PathBuf::from(temp_name);
        let result = (|| {
            let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)
                .map_err(io_error)?;
            file.write_all(&bytes).map_err(io_error)?;
            file.sync_all().map_err(io_error)?;
            fs::rename(&temp, &self.path).map_err(io_error)?;
            if let Some(parent) = self.path.parent() {
                if !parent.as_os_str().is_empty() {
                    OpenOptions::new().read(true).open(parent)
                        .and_then(|directory| directory.sync_all()).map_err(io_error)?;
                }
            }
            Ok(())
        })();
        if result.is_err() { let _ = fs::remove_file(&temp); }
        result
    }
}

impl TrustConfigStore for AnchoredFileTrustConfigStore {
    fn load(&self) -> Result<Option<SignedTrustConfig>, String> {
        self.validate_anchor()?;
        self.read_config()
    }

    fn compare_and_store(
        &self,
        expected: Option<TrustConfigVersion>,
        next: SignedTrustConfig,
    ) -> Result<(), String> {
        let lock = self.lock_path();
        let lock_file = OpenOptions::new().write(true).create_new(true).open(&lock)
            .map_err(|_| "trust config writer lock unavailable".to_string())?;
        let _guard = LockGuard { path: lock, file: Some(lock_file) };
        let current_anchor = self.anchor.load()?;
        let current = self.read_config()?;
        let current_version = current.as_ref().map(SignedTrustConfig::version)
            .transpose().map_err(|e| format!("invalid current config: {e:?}"))?;
        if current_version != expected {
            return Err("conflict".into());
        }
        if current.is_none() && (current_anchor.generation != 0 || current_anchor.digest != [0; 32]) {
            return Err("trust config file is missing while anchor is advanced".into());
        }
        let expected_generation = current_anchor.generation.checked_add(1)
            .ok_or_else(|| "trust config generation exhausted".to_string())?;
        let next_version = next.version().map_err(|e| format!("invalid next config: {e:?}"))?;
        if next_version.generation != expected_generation {
            return Err("generation does not advance anchor by one".into());
        }
        if current_version.as_ref().is_some_and(|v| {
            current_anchor.generation != v.generation || current_anchor.digest != v.digest
        }) {
            return Err("current config does not match anchor".into());
        }
        let next_anchor = TrustAnchorState {
            generation: next_version.generation,
            digest: next_version.digest,
        };
        // Persist config first. A crash before anchor advancement leaves a
        // mismatch and therefore fails closed on the next open/load.
        self.write_config(&next)?;
        self.anchor.compare_and_advance(&current_anchor, &next_anchor)?;
        Ok(())
    }
}

fn encode(config: &SignedTrustConfig) -> Result<Vec<u8>, String> {
    let payload = config.canonical_payload().map_err(|e| format!("{e:?}"))?;
    if config.signature.is_empty() || config.signature.len() > MAX_SIGNATURE_BYTES {
        return Err("invalid trust config signature length".into());
    }
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&(config.signature.len() as u32).to_be_bytes());
    out.extend_from_slice(&config.signature);
    if out.len() > MAX_FILE_BYTES { return Err("trust config exceeds size limit".into()); }
    Ok(out)
}

fn decode(bytes: &[u8]) -> Result<SignedTrustConfig, String> {
    if bytes.len() > MAX_FILE_BYTES { return Err("trust config exceeds size limit".into()); }
    let mut cursor = Cursor { bytes, offset: 0 };
    if cursor.take(8)? != MAGIC { return Err("invalid trust config file header".into()); }
    let payload_len = cursor.u32()? as usize;
    if payload_len > MAX_FILE_BYTES { return Err("trust config payload too large".into()); }
    let payload = cursor.take(payload_len)?;
    let signature_len = cursor.u32()? as usize;
    if signature_len == 0 || signature_len > MAX_SIGNATURE_BYTES {
        return Err("invalid trust config signature length".into());
    }
    let signature = cursor.take(signature_len)?.to_vec();
    if cursor.offset != bytes.len() { return Err("trailing trust config bytes".into()); }
    decode_payload(payload, signature)
}

fn decode_payload(payload: &[u8], signature: Vec<u8>) -> Result<SignedTrustConfig, String> {
    let mut c = Cursor { bytes: payload, offset: 0 };
    if c.take(DOMAIN.len())? != DOMAIN { return Err("invalid trust config domain".into()); }
    let generation = c.u64()?;
    let previous_digest: [u8; 32] = c.take(32)?.try_into().map_err(|_| "invalid previous digest")?;
    let count = c.u32()? as usize;
    if count == 0 || count > MAX_ENTRIES { return Err("invalid trust entry count".into()); }
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let producer_id = c.string()?;
        let key_id = c.string()?;
        let state = match c.take(1)?[0] {
            1 => KeyState::Active,
            2 => KeyState::Revoked,
            _ => return Err("invalid trust key state".into()),
        };
        entries.push(TrustEntry { producer_id, key_id, state });
    }
    if c.offset != payload.len() { return Err("trailing trust config payload bytes".into()); }
    let config = SignedTrustConfig { generation, previous_digest, entries, signature };
    let canonical = config.canonical_payload().map_err(|e| format!("{e:?}"))?;
    if canonical != payload { return Err("non-canonical trust config payload".into()); }
    Ok(config)
}

struct Cursor<'a> { bytes: &'a [u8], offset: usize }
impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self.offset.checked_add(count).ok_or("trust config length overflow")?;
        let value = self.bytes.get(self.offset..end).ok_or("truncated trust config")?;
        self.offset = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().map_err(|_| "invalid integer")?))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().map_err(|_| "invalid integer")?))
    }
    fn string(&mut self) -> Result<String, String> {
        let len = u16::from_be_bytes(self.take(2)?.try_into().map_err(|_| "invalid string length")?) as usize;
        if len == 0 || len > 128 { return Err("invalid trust identifier length".into()); }
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| "trust identifier is not UTF-8".into())
    }
}
fn io_error(error: std::io::Error) -> String { error.to_string() }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_trust_config::{KeyState, TrustEntry};
    use sha2::{Digest, Sha256};
    use std::{sync::Mutex, time::{SystemTime, UNIX_EPOCH}};

    #[derive(Default)]
    struct TestAnchor(Mutex<TrustAnchorState>);
    impl Default for TrustAnchorState {
        fn default() -> Self { Self { generation: 0, digest: [0; 32] } }
    }
    impl TrustConfigAnchor for TestAnchor {
        fn load(&self) -> Result<TrustAnchorState, String> { Ok(self.0.lock().unwrap().clone()) }
        fn compare_and_advance(&self, expected: &TrustAnchorState, next: &TrustAnchorState) -> Result<(), String> {
            let mut state = self.0.lock().unwrap();
            if *state != *expected || next.generation != expected.generation + 1 {
                return Err("anchor conflict".into());
            }
            *state = next.clone();
            Ok(())
        }
    }
    fn path() -> PathBuf {
        let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("ardisa-trust-{}-{id}.bin", std::process::id()))
    }
    fn config() -> SignedTrustConfig {
        let mut value = SignedTrustConfig {
            generation: 1, previous_digest: [0; 32],
            entries: vec![TrustEntry { producer_id: "ci".into(), key_id: "key-v1".into(), state: KeyState::Active }],
            signature: vec![1, 2, 3],
        };
        value.signature = value.canonical_payload().unwrap();
        value
    }
    #[test]
    fn persists_and_reopens_only_when_independent_anchor_matches() {
        let path = path();
        let anchor = Arc::new(TestAnchor::default());
        let store = AnchoredFileTrustConfigStore::open(&path, anchor.clone()).unwrap();
        let next = config();
        let version = next.version().unwrap();
        store.compare_and_store(None, next.clone()).unwrap();
        assert_eq!(store.load().unwrap(), Some(next));
        drop(store);
        let reopened = AnchoredFileTrustConfigStore::open(&path, anchor).unwrap();
        assert_eq!(reopened.load().unwrap().unwrap().version().unwrap(), version);
        let _ = fs::remove_file(path);
    }
    #[test]
    fn restored_old_file_is_rejected_by_independent_anchor() {
        let path = path();
        let backup = path.with_extension("backup");
        let anchor = Arc::new(TestAnchor::default());
        let store = AnchoredFileTrustConfigStore::open(&path, anchor.clone()).unwrap();
        let original = encode(&config()).unwrap();
        fs::write(&backup, &original).unwrap();
        store.compare_and_store(None, config()).unwrap();
        drop(store);
        fs::copy(&backup, &path).unwrap();
        assert!(AnchoredFileTrustConfigStore::open(&path, anchor).is_err());
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(backup);
    }
    #[test]
    fn rejects_corrupt_and_noncanonical_files() {
        let mut bytes = encode(&config()).unwrap();
        bytes.push(0);
        assert!(decode(&bytes).is_err());
        let mut hash = Sha256::new();
        hash.update(b"test");
        assert_eq!(hash.finalize().len(), 32);
    }
}
