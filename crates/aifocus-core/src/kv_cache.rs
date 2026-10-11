//! Bounded runtime foundation for attention key/value caching.
//!
//! This module is deliberately a runtime primitive only. It is not yet a
//! first-class `.ardisa` type and is not emitted by the native compiler.
//! Source-language integration must add parser/type/IR/instruction support
//! before this capability can be advertised to Ardisa programs.

use std::collections::VecDeque;

/// Hard upper bound for a single cache's entry count. This bounds both runtime
/// work and the amount of state accepted from untrusted program inputs.
pub const MAX_KV_CACHE_ENTRIES: usize = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheFullPolicy {
    /// Preserve existing entries and reject an append when capacity is reached.
    Reject,
    /// Evict exactly the oldest entry before appending a new one.
    EvictOldest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvEntry {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    ZeroCapacity,
    CapacityTooLarge { requested: usize, maximum: usize },
    EmptyVector,
    KeyDimension { expected: usize, actual: usize },
    ValueDimension { expected: usize, actual: usize },
    NonFiniteKey { index: usize },
    NonFiniteValue { index: usize },
    CapacityExceeded { capacity: usize },
    InvalidWindow { start: usize, end: usize, len: usize },
}

/// A bounded, deterministic key/value cache for one attention head.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    capacity: usize,
    key_dim: Option<usize>,
    value_dim: Option<usize>,
    entries: VecDeque<KvEntry>,
}

impl KvCache {
    pub fn new(capacity: usize) -> Result<Self, KvCacheError> {
        if capacity == 0 {
            return Err(KvCacheError::ZeroCapacity);
        }
        if capacity > MAX_KV_CACHE_ENTRIES {
            return Err(KvCacheError::CapacityTooLarge { requested: capacity, maximum: MAX_KV_CACHE_ENTRIES });
        }
        Ok(Self {
            capacity,
            key_dim: None,
            value_dim: None,
            entries: VecDeque::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Append after validating dimensions and finite numeric payloads.
    /// Cache dimensions are fixed by the first successful append.
    pub fn append(
        &mut self,
        key: Vec<f32>,
        value: Vec<f32>,
        policy: CacheFullPolicy,
    ) -> Result<(), KvCacheError> {
        if key.is_empty() || value.is_empty() {
            return Err(KvCacheError::EmptyVector);
        }
        for (index, item) in key.iter().enumerate() {
            if !item.is_finite() {
                return Err(KvCacheError::NonFiniteKey { index });
            }
        }
        for (index, item) in value.iter().enumerate() {
            if !item.is_finite() {
                return Err(KvCacheError::NonFiniteValue { index });
            }
        }
        if let Some(expected) = self.key_dim {
            if key.len() != expected {
                return Err(KvCacheError::KeyDimension { expected, actual: key.len() });
            }
        }
        if let Some(expected) = self.value_dim {
            if value.len() != expected {
                return Err(KvCacheError::ValueDimension { expected, actual: value.len() });
            }
        }
        if self.entries.len() == self.capacity {
            match policy {
                CacheFullPolicy::Reject => {
                    return Err(KvCacheError::CapacityExceeded { capacity: self.capacity });
                }
                CacheFullPolicy::EvictOldest => {
                    self.entries.pop_front();
                }
            }
        }
        self.key_dim.get_or_insert(key.len());
        self.value_dim.get_or_insert(value.len());
        self.entries.push_back(KvEntry { key, value });
        Ok(())
    }

    /// Return a cloned half-open window [start, end) in insertion order.
    pub fn window(&self, start: usize, end: usize) -> Result<Vec<KvEntry>, KvCacheError> {
        if start > end || end > self.entries.len() {
            return Err(KvCacheError::InvalidWindow { start, end, len: self.entries.len() });
        }
        Ok(self.entries.iter().skip(start).take(end - start).cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_capacity_and_empty_vectors() {
        assert_eq!(KvCache::new(0).unwrap_err(), KvCacheError::ZeroCapacity);
        assert_eq!(KvCache::new(MAX_KV_CACHE_ENTRIES + 1).unwrap_err(), KvCacheError::CapacityTooLarge { requested: MAX_KV_CACHE_ENTRIES + 1, maximum: MAX_KV_CACHE_ENTRIES });
        let mut cache = KvCache::new(2).unwrap();
        assert_eq!(cache.append(vec![], vec![1.0], CacheFullPolicy::Reject), Err(KvCacheError::EmptyVector));
    }

    #[test]
    fn enforces_shape_and_finite_values_without_mutating_cache() {
        let mut cache = KvCache::new(2).unwrap();
        cache.append(vec![1.0, 2.0], vec![3.0], CacheFullPolicy::Reject).unwrap();
        assert_eq!(cache.append(vec![1.0], vec![4.0], CacheFullPolicy::Reject), Err(KvCacheError::KeyDimension { expected: 2, actual: 1 }));
        assert_eq!(cache.append(vec![f32::NAN, 2.0], vec![4.0], CacheFullPolicy::Reject), Err(KvCacheError::NonFiniteKey { index: 0 }));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn full_policy_is_explicit_and_window_order_is_stable() {
        let mut cache = KvCache::new(2).unwrap();
        cache.append(vec![1.0], vec![10.0], CacheFullPolicy::Reject).unwrap();
        cache.append(vec![2.0], vec![20.0], CacheFullPolicy::Reject).unwrap();
        assert_eq!(cache.append(vec![3.0], vec![30.0], CacheFullPolicy::Reject), Err(KvCacheError::CapacityExceeded { capacity: 2 }));
        cache.append(vec![3.0], vec![30.0], CacheFullPolicy::EvictOldest).unwrap();
        assert_eq!(cache.window(0, 2).unwrap().iter().map(|entry| entry.key[0]).collect::<Vec<_>>(), vec![2.0, 3.0]);
        assert!(matches!(cache.window(0, 3), Err(KvCacheError::InvalidWindow { .. })));
    }

    #[test]
    fn rejects_value_dimension_and_non_finite_value() {
        let mut cache = KvCache::new(3).unwrap();
        cache.append(vec![1.0], vec![2.0, 3.0], CacheFullPolicy::Reject).unwrap();
        assert_eq!(cache.append(vec![2.0], vec![4.0], CacheFullPolicy::Reject), Err(KvCacheError::ValueDimension { expected: 2, actual: 1 }));
        assert_eq!(cache.append(vec![2.0], vec![f32::INFINITY, 1.0], CacheFullPolicy::Reject), Err(KvCacheError::NonFiniteValue { index: 0 }));
        assert_eq!(cache.len(), 1);
    }
}
