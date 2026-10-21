//! Bounded incremental UTF-8 tokenization over an explicit vocabulary.
//!
//! This is a runtime helper, not yet a first-class .ardisa source construct.
//! Input chunks are transactional: invalid UTF-8, unknown tokens, or resource
//! exhaustion leaves the state unchanged.

use std::collections::BTreeMap;
use crate::ai::RuntimeError;

pub const MAX_TOKENIZER_VOCABULARY: usize = 65_536;
pub const MAX_TOKENIZER_PENDING_BYTES: usize = 1_048_576;
pub const MAX_TOKENIZER_EMITTED_IDS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerState {
    vocabulary: BTreeMap<String, u32>,
    pending_utf8: Vec<u8>,
    pending_token: String,
    max_pending_bytes: usize,
    max_emitted_ids: usize,
    emitted_ids: usize,
}

impl TokenizerState {
    pub fn new(
        vocabulary: impl IntoIterator<Item = (String, u32)>,
        max_pending_bytes: usize,
        max_emitted_ids: usize,
    ) -> Result<Self, RuntimeError> {
        if max_pending_bytes == 0 || max_pending_bytes > MAX_TOKENIZER_PENDING_BYTES {
            return Err(RuntimeError::new("AIT030", format!(
                "tokenizer pending-byte budget must be between 1 and {}", MAX_TOKENIZER_PENDING_BYTES
            )));
        }
        if max_emitted_ids == 0 || max_emitted_ids > MAX_TOKENIZER_EMITTED_IDS {
            return Err(RuntimeError::new("AIT030", format!(
                "tokenizer output budget must be between 1 and {}", MAX_TOKENIZER_EMITTED_IDS
            )));
        }
        let mut map = BTreeMap::new();
        for (token, id) in vocabulary {
            if token.is_empty() || token.chars().any(char::is_whitespace) {
                return Err(RuntimeError::new("AIT031", "vocabulary entries must be non-empty single tokens"));
            }
            if map.contains_key(&token) {
                return Err(RuntimeError::new("AIT031", format!("duplicate vocabulary token '{}'", token)));
            }
            map.insert(token, id);
            if map.len() > MAX_TOKENIZER_VOCABULARY {
                return Err(RuntimeError::new("AIT031", "tokenizer vocabulary capacity exceeded"));
            }
        }
        if map.is_empty() {
            return Err(RuntimeError::new("AIT031", "tokenizer vocabulary must not be empty"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for id in map.values() {
            if !ids.insert(*id) {
                return Err(RuntimeError::new("AIT031", format!("duplicate vocabulary id {}", id)));
            }
        }
        Ok(Self {
            vocabulary: map,
            pending_utf8: Vec::new(),
            pending_token: String::new(),
            max_pending_bytes,
            max_emitted_ids,
            emitted_ids: 0,
        })
    }

    /// Feed an arbitrary byte chunk. A trailing incomplete UTF-8 code point is
    /// retained until the next chunk; invalid UTF-8 and unknown complete tokens
    /// are rejected without changing this state.
    pub fn push_chunk(&mut self, chunk: &[u8]) -> Result<Vec<u32>, RuntimeError> {
        let mut next = self.clone();
        let total = next.pending_utf8.len().checked_add(chunk.len())
            .ok_or_else(|| RuntimeError::new("AIT032", "tokenizer pending-byte budget overflow"))?;
        if total > next.max_pending_bytes {
            return Err(RuntimeError::new("AIT032", "tokenizer pending-byte budget exceeded"));
        }
        next.pending_utf8.extend_from_slice(chunk);
        let valid_len = match std::str::from_utf8(&next.pending_utf8) {
            Ok(_) => next.pending_utf8.len(),
            Err(error) if error.error_len().is_none() => error.valid_up_to(),
            Err(_) => return Err(RuntimeError::new("AIT033", "tokenizer input contains invalid UTF-8")),
        };
        let complete = std::str::from_utf8(&next.pending_utf8[..valid_len])
            .map_err(|_| RuntimeError::new("AIT033", "tokenizer input contains invalid UTF-8"))?
            .to_owned();
        let suffix = next.pending_utf8[valid_len..].to_vec();
        next.pending_utf8 = suffix;
        let mut output = Vec::new();
        for ch in complete.chars() {
            if ch.is_whitespace() {
                if !next.pending_token.is_empty() {
                    let token = std::mem::take(&mut next.pending_token);
                    let id = next.vocabulary.get(&token).copied().ok_or_else(|| {
                        RuntimeError::new("AIT034", format!("token '{}' is not in the explicit vocabulary", token))
                    })?;
                    next.emit(id, &mut output)?;
                }
            } else {
                next.pending_token.push(ch);
                if next.pending_token.len() > next.max_pending_bytes {
                    return Err(RuntimeError::new("AIT032", "tokenizer token-byte budget exceeded"));
                }
            }
        }
        *self = next;
        Ok(output)
    }

    /// Flush the final token. Incomplete UTF-8 and unknown tokens fail atomically.
    pub fn finish(&mut self) -> Result<Vec<u32>, RuntimeError> {
        if !self.pending_utf8.is_empty() {
            return Err(RuntimeError::new("AIT033", "cannot finish tokenizer with incomplete UTF-8 sequence"));
        }
        let mut next = self.clone();
        let mut output = Vec::new();
        if !next.pending_token.is_empty() {
            let token = std::mem::take(&mut next.pending_token);
            let id = next.vocabulary.get(&token).copied().ok_or_else(|| {
                RuntimeError::new("AIT034", format!("token '{}' is not in the explicit vocabulary", token))
            })?;
            next.emit(id, &mut output)?;
        }
        *self = next;
        Ok(output)
    }

    fn emit(&mut self, id: u32, output: &mut Vec<u32>) -> Result<(), RuntimeError> {
        if self.emitted_ids >= self.max_emitted_ids {
            return Err(RuntimeError::new("AIT032", "tokenizer output budget exceeded"));
        }
        self.emitted_ids += 1;
        output.push(id);
        Ok(())
    }

    pub fn emitted_ids(&self) -> usize { self.emitted_ids }
    pub fn pending_token(&self) -> &str { &self.pending_token }
    pub fn has_incomplete_utf8(&self) -> bool { !self.pending_utf8.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(max_pending: usize, max_ids: usize) -> TokenizerState {
        TokenizerState::new(
            vec![("hello".to_owned(), 1), ("世界".to_owned(), 2), ("ok".to_owned(), 3)],
            max_pending,
            max_ids,
        ).unwrap()
    }

    #[test]
    fn incremental_utf8_chunks_produce_deterministic_vocabulary_ids() {
        let mut state = tokenizer(64, 8);
        assert_eq!(state.push_chunk(b"hello ").unwrap(), vec![1]);
        let bytes = "世".as_bytes();
        assert!(state.push_chunk(&bytes[..2]).unwrap().is_empty());
        assert!(state.has_incomplete_utf8());
        let mut remainder = bytes[2..].to_vec();
        remainder.extend_from_slice("界 ".as_bytes());
        assert_eq!(state.push_chunk(&remainder).unwrap(), vec![2]);
        assert_eq!(state.finish().unwrap(), Vec::<u32>::new());
        assert_eq!(state.emitted_ids(), 2);
    }

    #[test]
    fn invalid_utf8_unknown_tokens_and_budget_fail_without_mutation() {
        let mut state = tokenizer(8, 1);
        assert_eq!(state.push_chunk(&[0xff]).unwrap_err().code, "AIT033");
        assert_eq!(state.pending_token(), "");
        assert!(state.push_chunk(b"hello ").unwrap().is_empty());
        let before = state.clone();
        assert_eq!(state.push_chunk(b"unknown ").unwrap_err().code, "AIT034");
        assert_eq!(state, before);
        assert_eq!(state.finish().unwrap(), vec![1]);
        let before = state.clone();
        assert_eq!(state.push_chunk(b"ok ").unwrap_err().code, "AIT032");
        assert_eq!(state, before);
    }

    #[test]
    fn incomplete_utf8_cannot_be_finished_and_vocabulary_is_validated() {
        let mut state = tokenizer(8, 4);
        assert!(state.push_chunk(&[0xe4, 0xb8]).unwrap().is_empty());
        assert_eq!(state.finish().unwrap_err().code, "AIT033");
        assert!(TokenizerState::new(vec![("two words".to_owned(), 1)], 8, 4).is_err());
        assert!(TokenizerState::new(vec![("a".to_owned(), 1), ("b".to_owned(), 1)], 8, 4).is_err());
    }
}
