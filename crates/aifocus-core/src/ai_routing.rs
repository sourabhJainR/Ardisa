//! Bounded, deterministic reference algorithms for AI-native routing and vector search.
//!
//! These APIs are an implementation substrate, not a claim that source-level
//! Ardisa syntax or ARDISA-EXEC-V1 opcodes are wired yet. Callers must enforce
//! the same bounds at the compiler/verifier boundary before exposing them to
//! untrusted programs.

use std::cmp::Ordering;

pub const MAX_EXPERTS: usize = 256;
pub const MAX_TOP_K: usize = 32;
pub const MAX_VECTOR_DIMENSIONS: usize = 16_384;
pub const MAX_INDEX_ENTRIES: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiRoutingError {
    EmptyExperts,
    TooManyExperts,
    InvalidTopK,
    LengthMismatch,
    NonFiniteLogit,
    AllExpertsMasked,
    InvalidDimension,
    NonFiniteCoordinate,
    DomainMismatch,
    IndexFull,
    DuplicateId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertWeight {
    pub expert: usize,
    pub weight: f32,
}

/// Select top-k unmasked experts using stable descending-logit ordering.
/// Equal logits break ties by ascending expert index. Softmax is computed only
/// over selected experts with max subtraction to avoid overflow.
pub fn route_top_k(logits: &[f32], masked: &[bool], top_k: usize) -> Result<Vec<ExpertWeight>, AiRoutingError> {
    if logits.is_empty() { return Err(AiRoutingError::EmptyExperts); }
    if logits.len() > MAX_EXPERTS { return Err(AiRoutingError::TooManyExperts); }
    if logits.len() != masked.len() { return Err(AiRoutingError::LengthMismatch); }
    if top_k == 0 || top_k > MAX_TOP_K || top_k > logits.len() { return Err(AiRoutingError::InvalidTopK); }
    if logits.iter().any(|v| !v.is_finite()) { return Err(AiRoutingError::NonFiniteLogit); }

    let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate()
        .filter(|(i, _)| !masked[*i]).collect();
    if candidates.is_empty() { return Err(AiRoutingError::AllExpertsMasked); }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    candidates.truncate(top_k.min(candidates.len()));

    let max_logit = candidates[0].1;
    let exps: Vec<f64> = candidates.iter().map(|(_, v)| (f64::from(*v) - f64::from(max_logit)).exp()).collect();
    let total: f64 = exps.iter().sum();
    if !total.is_finite() || total <= 0.0 { return Err(AiRoutingError::NonFiniteLogit); }
    Ok(candidates.into_iter().zip(exps).map(|((expert, _), exp)| ExpertWeight {
        expert, weight: (exp / total) as f32,
    }).collect())
}

#[derive(Debug, Clone, PartialEq)]
pub struct DomainVector {
    pub domain: String,
    pub coordinates: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    pub id: u64,
    /// Squared Euclidean distance, ordered ascending.
    pub distance: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    id: u64,
    vector: DomainVector,
}

/// A bounded in-memory reference index. Domain labels are exact-match isolated;
/// ids are unique and search ties are resolved by ascending id.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIndex {
    domain: String,
    dimensions: usize,
    entries: Vec<Entry>,
}

impl VectorIndex {
    pub fn new(domain: impl Into<String>, dimensions: usize) -> Result<Self, AiRoutingError> {
        if dimensions == 0 || dimensions > MAX_VECTOR_DIMENSIONS {
            return Err(AiRoutingError::InvalidDimension);
        }
        let domain = domain.into();
        if domain.is_empty() { return Err(AiRoutingError::DomainMismatch); }
        Ok(Self { domain, dimensions, entries: Vec::new() })
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    pub fn insert(&mut self, id: u64, vector: DomainVector) -> Result<(), AiRoutingError> {
        self.validate(&vector)?;
        if self.entries.len() >= MAX_INDEX_ENTRIES { return Err(AiRoutingError::IndexFull); }
        if self.entries.iter().any(|entry| entry.id == id) {
            return Err(AiRoutingError::DuplicateId);
        }
        self.entries.push(Entry { id, vector });
        Ok(())
    }

    pub fn search(&self, query: &DomainVector, limit: usize) -> Result<Vec<VectorHit>, AiRoutingError> {
        self.validate(query)?;
        if limit > MAX_INDEX_ENTRIES { return Err(AiRoutingError::IndexFull); }
        let mut hits: Vec<VectorHit> = self.entries.iter().map(|entry| {
            let distance = entry.vector.coordinates.iter().zip(&query.coordinates)
                .map(|(a, b)| {
                    let delta = f64::from(*a) - f64::from(*b);
                    delta * delta
                }).sum();
            VectorHit { id: entry.id, distance }
        }).collect();
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(limit);
        Ok(hits)
    }

    fn validate(&self, vector: &DomainVector) -> Result<(), AiRoutingError> {
        if vector.domain != self.domain { return Err(AiRoutingError::DomainMismatch); }
        if vector.coordinates.len() != self.dimensions { return Err(AiRoutingError::InvalidDimension); }
        if vector.coordinates.iter().any(|v| !v.is_finite()) { return Err(AiRoutingError::NonFiniteCoordinate); }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_is_deterministic_and_normalized() {
        let result = route_top_k(&[2.0, 2.0, 1.0], &[false, false, false], 2).unwrap();
        assert_eq!(result.iter().map(|v| v.expert).collect::<Vec<_>>(), vec![0, 1]);
        assert!((result.iter().map(|v| v.weight).sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn routing_rejects_invalid_and_all_masked_inputs() {
        assert_eq!(route_top_k(&[1.0], &[true], 1), Err(AiRoutingError::AllExpertsMasked));
        assert_eq!(route_top_k(&[f32::NAN], &[false], 1), Err(AiRoutingError::NonFiniteLogit));
        assert_eq!(route_top_k(&[1.0], &[false, false], 1), Err(AiRoutingError::LengthMismatch));
    }

    #[test]
    fn vector_search_is_domain_checked_and_ties_use_id() {
        let mut index = VectorIndex::new("text", 2).unwrap();
        index.insert(9, DomainVector { domain: "text".into(), coordinates: vec![1.0, 0.0] }).unwrap();
        index.insert(3, DomainVector { domain: "text".into(), coordinates: vec![-1.0, 0.0] }).unwrap();
        let hits = index.search(&DomainVector { domain: "text".into(), coordinates: vec![0.0, 0.0] }, 2).unwrap();
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![3, 9]);
        assert_eq!(index.search(&DomainVector { domain: "image".into(), coordinates: vec![0.0, 0.0] }, 1), Err(AiRoutingError::DomainMismatch));
    }

    #[test]
    fn vector_index_rejects_dimension_and_non_finite_values() {
        let index = VectorIndex::new("text", 2).unwrap();
        assert_eq!(index.search(&DomainVector { domain: "text".into(), coordinates: vec![0.0] }, 1), Err(AiRoutingError::InvalidDimension));
        assert_eq!(index.search(&DomainVector { domain: "text".into(), coordinates: vec![0.0, f32::INFINITY] }, 1), Err(AiRoutingError::NonFiniteCoordinate));
    }

    #[test]
    fn adversarial_routing_bounds_and_masks_fail_closed() {
        assert_eq!(route_top_k(&[], &[], 1), Err(AiRoutingError::EmptyExperts));
        assert_eq!(route_top_k(&[1.0], &[], 1), Err(AiRoutingError::LengthMismatch));
        assert_eq!(route_top_k(&[1.0], &[false], 0), Err(AiRoutingError::InvalidTopK));
        assert_eq!(route_top_k(&[1.0], &[false], MAX_TOP_K + 1), Err(AiRoutingError::InvalidTopK));
        assert_eq!(route_top_k(&vec![0.0; MAX_EXPERTS + 1], &vec![false; MAX_EXPERTS + 1], 1), Err(AiRoutingError::TooManyExperts));
        assert_eq!(route_top_k(&[f32::NEG_INFINITY], &[false], 1), Err(AiRoutingError::NonFiniteLogit));
        assert_eq!(route_top_k(&[f32::MAX, f32::MAX], &[false, false], 2).unwrap().len(), 2);
        assert_eq!(route_top_k(&[1.0, 2.0], &[true, false], 2).unwrap().iter().map(|x| x.expert).collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn adversarial_vector_index_rejects_duplicate_ids_and_invalid_limits() {
        assert_eq!(VectorIndex::new("text", 0), Err(AiRoutingError::InvalidDimension));
        assert_eq!(VectorIndex::new("text", MAX_VECTOR_DIMENSIONS + 1), Err(AiRoutingError::InvalidDimension));
        assert_eq!(VectorIndex::new("", 1), Err(AiRoutingError::DomainMismatch));
        let mut index = VectorIndex::new("text", 1).unwrap();
        let first = DomainVector { domain: "text".into(), coordinates: vec![0.0] };
        index.insert(7, first.clone()).unwrap();
        assert_eq!(index.insert(7, first), Err(AiRoutingError::DuplicateId));
        assert_eq!(index.search(&DomainVector { domain: "text".into(), coordinates: vec![0.0] }, MAX_INDEX_ENTRIES + 1), Err(AiRoutingError::IndexFull));
        assert_eq!(index.search(&DomainVector { domain: "text".into(), coordinates: vec![f32::NAN] }, 1), Err(AiRoutingError::NonFiniteCoordinate));
    }

    #[test]
    fn vector_index_keeps_domain_and_distance_ordering_deterministic() {
        let mut index = VectorIndex::new("text", 1).unwrap();
        index.insert(4, DomainVector { domain: "text".into(), coordinates: vec![f32::MAX] }).unwrap();
        index.insert(2, DomainVector { domain: "text".into(), coordinates: vec![-f32::MAX] }).unwrap();
        let hits = index.search(&DomainVector { domain: "text".into(), coordinates: vec![0.0] }, 2).unwrap();
        assert_eq!(hits.iter().map(|h| h.id).collect::<Vec<_>>(), vec![2, 4]);
        assert!(hits.iter().all(|h| h.distance.is_finite()));
    }

}
