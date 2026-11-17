//! Durable Proof replay journal with an independently protected monotonic anchor.
//!
//! The journal provides crash durability and tamper detection. Rollback resistance
//! exists only when the supplied anchor is backed by state that cannot be restored
//! with this journal (for example, a remote monotonic service or hardware counter).
use crate::proof_evidence::{AdmissionError, ReplayStore};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const MAGIC: &[u8; 8] = b"ARDPRP01";
const ZERO_HASH: [u8; 32] = [0; 32];
const MAX_RECORD_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorState {
    pub generation: u64,
    pub head_sha256: [u8; 32],
}

/// Implementations must provide durable, atomic compare-and-advance semantics.
/// The anchor must be stored independently of the journal and must not permit
/// restoring an older state. A normal database backup is not such an anchor.
pub trait MonotonicAnchor: Send + Sync {
    fn load(&self) -> Result<AnchorState, String>;
    fn compare_and_advance(&self, expected: &AnchorState, next: &AnchorState) -> Result<(), String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnchoredReplayError {
    Io(String),
    Anchor(String),
    AnchorMismatch,
    CorruptJournal,
    InvalidBatch,
    LockUnavailable,
}
impl From<AnchoredReplayError> for AdmissionError {
    fn from(error: AnchoredReplayError) -> Self {
        match error {
            AnchoredReplayError::AnchorMismatch
            | AnchoredReplayError::CorruptJournal
            | AnchoredReplayError::InvalidBatch => AdmissionError::ReplayOrRollback,
            _ => AdmissionError::ReplayStoreUnavailable,
        }
    }
}

struct LockGuard(PathBuf);
impl Drop for LockGuard {
    fn drop(&mut self) { let _ = fs::remove_file(&self.0); }
}

struct Snapshot {
    state: AnchorState,
    end: u64,
    counters: BTreeMap<(String, String), u64>,
}

/// File-backed journal. Use a different, independently protected anchor for
/// each logical replay domain. Concurrent writers fail closed if the lock exists.
pub struct AnchoredFileReplayStore {
    path: PathBuf,
    anchor: Arc<dyn MonotonicAnchor>,
}
impl AnchoredFileReplayStore {
    pub fn open(path: impl AsRef<Path>, anchor: Arc<dyn MonotonicAnchor>) -> Result<Self, AnchoredReplayError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(io_error)?;
            }
        }
        OpenOptions::new().create(true).append(true).open(&path).map_err(io_error)?;
        let store = Self { path, anchor };
        store.with_lock(|file| { store.recover(file).map(|_| ()) })?;
        Ok(store)
    }

    fn lock_path(&self) -> PathBuf {
        let mut name = self.path.as_os_str().to_os_string();
        name.push(".lock");
        PathBuf::from(name)
    }

    fn with_lock<T>(&self, action: impl FnOnce(&mut File) -> Result<T, AnchoredReplayError>) -> Result<T, AnchoredReplayError> {
        let lock_path = self.lock_path();
        let _guard = OpenOptions::new().write(true).create_new(true).open(&lock_path)
            .map_err(|_| AnchoredReplayError::LockUnavailable)?;
        let guard = LockGuard(lock_path);
        let mut file = OpenOptions::new().read(true).write(true).create(true).open(&self.path).map_err(io_error)?;
        let result = action(&mut file);
        drop(guard);
        result
    }

    fn recover(&self, file: &mut File) -> Result<Snapshot, AnchoredReplayError> {
        let anchor = self.anchor.load().map_err(AnchoredReplayError::Anchor)?;
        let mut bytes = Vec::new();
        file.seek(SeekFrom::Start(0)).map_err(io_error)?;
        file.read_to_end(&mut bytes).map_err(io_error)?;
        let snapshots = scan(&bytes)?;
        let matched = snapshots.iter().find(|s| s.state == anchor).ok_or(AnchoredReplayError::AnchorMismatch)?;
        // An unanchored tail can only be an interrupted append. Discard it.
        if matched.end as usize != bytes.len() {
            file.set_len(matched.end).map_err(io_error)?;
            file.sync_all().map_err(io_error)?;
        }
        file.seek(SeekFrom::End(0)).map_err(io_error)?;
        Ok(Snapshot { state: matched.state.clone(), end: matched.end, counters: matched.counters.clone() })
    }
}

impl ReplayStore for AnchoredFileReplayStore {
    fn commit_batch(&self, advances: &[(String, String, u64)]) -> Result<(), AdmissionError> {
        self.with_lock(|file| {
            let current = self.recover(file)?;
            if advances.is_empty() { return Err(AnchoredReplayError::InvalidBatch); }
            let mut sorted = advances.to_vec();
            sorted.sort_by(|a, b| (&a.0, &a.1, a.2).cmp(&(&b.0, &b.1, b.2)));
            let mut pending = current.counters.clone();
            let mut previous: Option<(String, String, u64)> = None;
            for (producer, key, sequence) in &sorted {
                if producer.is_empty() || key.is_empty() || producer.len() > 128 || key.len() > 128 {
                    return Err(AnchoredReplayError::InvalidBatch);
                }
                let id = (producer.clone(), key.clone());
                if previous.as_ref().is_some_and(|p| p.0 == *producer && p.1 == *key && p.2 == *sequence) {
                    return Err(AnchoredReplayError::InvalidBatch);
                }
                let expected = pending.get(&id).copied().map_or(Some(1), |v| v.checked_add(1));
                if expected != Some(*sequence) { return Err(AnchoredReplayError::InvalidBatch); }
                pending.insert(id, *sequence);
                previous = Some((producer.clone(), key.clone(), *sequence));
            }
            let payload = encode_batch(&sorted)?;
            let next = AnchorState {
                generation: current.state.generation.checked_add(1).ok_or(AnchoredReplayError::InvalidBatch)?,
                head_sha256: record_hash(&current.state.head_sha256, &payload),
            };
            let mut record = Vec::with_capacity(4 + 32 + payload.len() + 32);
            let body_len = 32usize.checked_add(payload.len()).and_then(|v| v.checked_add(32))
                .ok_or(AnchoredReplayError::InvalidBatch)?;
            if body_len > MAX_RECORD_BYTES { return Err(AnchoredReplayError::InvalidBatch); }
            record.extend_from_slice(&(body_len as u32).to_be_bytes());
            record.extend_from_slice(&current.state.head_sha256);
            record.extend_from_slice(&payload);
            record.extend_from_slice(&next.head_sha256);
            file.seek(SeekFrom::End(0)).map_err(io_error)?;
            file.write_all(&record).map_err(io_error)?;
            file.sync_all().map_err(io_error)?;
            self.anchor.compare_and_advance(&current.state, &next).map_err(AnchoredReplayError::Anchor)?;
            Ok(())
        }).map_err(AdmissionError::from)
    }
}

fn io_error(error: std::io::Error) -> AnchoredReplayError {
    AnchoredReplayError::Io(error.to_string())
}
fn record_hash(previous: &[u8; 32], payload: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(previous);
    hash.update(payload);
    hash.finalize().into()
}
fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), AnchoredReplayError> {
    let len = u16::try_from(value.len()).map_err(|_| AnchoredReplayError::InvalidBatch)?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    Ok(())
}
fn encode_batch(items: &[(String, String, u64)]) -> Result<Vec<u8>, AnchoredReplayError> {
    let count = u32::try_from(items.len()).map_err(|_| AnchoredReplayError::InvalidBatch)?;
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&count.to_be_bytes());
    for (producer, key, sequence) in items {
        put_string(&mut out, producer)?;
        put_string(&mut out, key)?;
        out.extend_from_slice(&sequence.to_be_bytes());
    }
    Ok(out)
}
fn take<'a>(bytes: &'a [u8], offset: &mut usize, count: usize) -> Result<&'a [u8], AnchoredReplayError> {
    let end = offset.checked_add(count).ok_or(AnchoredReplayError::CorruptJournal)?;
    let value = bytes.get(*offset..end).ok_or(AnchoredReplayError::CorruptJournal)?;
    *offset = end;
    Ok(value)
}
fn read_u16(bytes: &[u8], offset: &mut usize) -> Result<usize, AnchoredReplayError> {
    Ok(u16::from_be_bytes(take(bytes, offset, 2)?.try_into().unwrap()) as usize)
}
fn decode_batch(payload: &[u8], counters: &mut BTreeMap<(String, String), u64>) -> Result<(), AnchoredReplayError> {
    let mut offset = 0;
    if take(payload, &mut offset, 8)? != MAGIC { return Err(AnchoredReplayError::CorruptJournal); }
    let count = u32::from_be_bytes(take(payload, &mut offset, 4)?.try_into().unwrap()) as usize;
    if count == 0 || count > 65_536 { return Err(AnchoredReplayError::CorruptJournal); }
    for _ in 0..count {
        let producer_len = read_u16(payload, &mut offset)?;
        if producer_len == 0 || producer_len > 128 { return Err(AnchoredReplayError::CorruptJournal); }
        let producer = std::str::from_utf8(take(payload, &mut offset, producer_len)?)
            .map_err(|_| AnchoredReplayError::CorruptJournal)?.to_owned();
        let key_len = read_u16(payload, &mut offset)?;
        if key_len == 0 || key_len > 128 { return Err(AnchoredReplayError::CorruptJournal); }
        let key = std::str::from_utf8(take(payload, &mut offset, key_len)?)
            .map_err(|_| AnchoredReplayError::CorruptJournal)?.to_owned();
        let sequence = u64::from_be_bytes(take(payload, &mut offset, 8)?.try_into().unwrap());
        if producer.is_empty() || key.is_empty() { return Err(AnchoredReplayError::CorruptJournal); }
        let id = (producer, key);
        let expected = counters.get(&id).and_then(|v| v.checked_add(1)).unwrap_or(1);
        if sequence != expected { return Err(AnchoredReplayError::CorruptJournal); }
        counters.insert(id, sequence);
    }
    if offset != payload.len() { return Err(AnchoredReplayError::CorruptJournal); }
    Ok(())
}
fn scan(bytes: &[u8]) -> Result<Vec<Snapshot>, AnchoredReplayError> {
    let genesis = AnchorState { generation: 0, head_sha256: ZERO_HASH };
    let mut result = vec![Snapshot { state: genesis, end: 0, counters: BTreeMap::new() }];
    let mut offset = 0usize;
    while bytes.len().saturating_sub(offset) >= 4 {
        let len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        if len < 64 || len > MAX_RECORD_BYTES { return Err(AnchoredReplayError::CorruptJournal); }
        let end = offset.checked_add(4).and_then(|v| v.checked_add(len)).ok_or(AnchoredReplayError::CorruptJournal)?;
        if end > bytes.len() { break; } // interrupted append; recover to the external anchor
        let body = &bytes[offset + 4..end];
        let previous: [u8; 32] = body[..32].try_into().unwrap();
        let payload = &body[32..body.len() - 32];
        let stored: [u8; 32] = body[body.len() - 32..].try_into().unwrap();
        let last = result.last().unwrap();
        if previous != last.state.head_sha256 || record_hash(&previous, payload) != stored {
            return Err(AnchoredReplayError::CorruptJournal);
        }
        let mut counters = last.counters.clone();
        decode_batch(payload, &mut counters)?;
        result.push(Snapshot {
            state: AnchorState { generation: last.state.generation.checked_add(1).ok_or(AnchoredReplayError::CorruptJournal)?, head_sha256: stored },
            end: end as u64,
            counters,
        });
        offset = end;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::Mutex, time::{SystemTime, UNIX_EPOCH}};

    #[derive(Default)]
    struct TestAnchor(Mutex<AnchorState>);
    impl Default for AnchorState {
        fn default() -> Self { Self { generation: 0, head_sha256: ZERO_HASH } }
    }
    impl MonotonicAnchor for TestAnchor {
        fn load(&self) -> Result<AnchorState, String> { Ok(self.0.lock().unwrap().clone()) }
        fn compare_and_advance(&self, expected: &AnchorState, next: &AnchorState) -> Result<(), String> {
            let mut state = self.0.lock().unwrap();
            if *state != *expected || next.generation != expected.generation + 1 { return Err("anchor compare failed".into()); }
            *state = next.clone(); Ok(())
        }
    }
    fn path() -> PathBuf {
        let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        std::env::temp_dir().join(format!("ardisa-replay-{}-{id}.log", std::process::id()))
    }

    #[test]
    fn persists_atomic_sequence_advances_across_reopen() {
        let path = path();
        let anchor = Arc::new(TestAnchor::default());
        let store = AnchoredFileReplayStore::open(&path, anchor.clone()).unwrap();
        store.commit_batch(&[("ci".into(), "key".into(), 1), ("ci".into(), "key".into(), 2)]).unwrap();
        drop(store);
        let reopened = AnchoredFileReplayStore::open(&path, anchor).unwrap();
        assert_eq!(reopened.commit_batch(&[("ci".into(), "key".into(), 2)]), Err(AdmissionError::ReplayOrRollback));
        assert!(reopened.commit_batch(&[("ci".into(), "key".into(), 3)]).is_ok());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn restored_old_journal_is_rejected_by_independent_anchor() {
        let path = path();
        let backup = path.with_extension("backup");
        let anchor = Arc::new(TestAnchor::default());
        let store = AnchoredFileReplayStore::open(&path, anchor.clone()).unwrap();
        fs::copy(&path, &backup).unwrap();
        store.commit_batch(&[("ci".into(), "key".into(), 1)]).unwrap();
        drop(store);
        fs::copy(&backup, &path).unwrap();
        assert!(matches!(AnchoredFileReplayStore::open(&path, anchor), Err(AnchoredReplayError::AnchorMismatch)));
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(backup);
    }

    #[test]
    fn rejects_gaps_without_writing_a_record() {
        let path = path();
        let anchor = Arc::new(TestAnchor::default());
        let store = AnchoredFileReplayStore::open(&path, anchor).unwrap();
        assert_eq!(store.commit_batch(&[("ci".into(), "key".into(), 2)]), Err(AdmissionError::ReplayOrRollback));
        assert!(store.commit_batch(&[("ci".into(), "key".into(), 1)]).is_ok());
        let _ = fs::remove_file(path);
    }
}
