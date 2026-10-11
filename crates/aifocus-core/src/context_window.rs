//! Bounded context-window runtime foundation.
//!
//! This primitive is not yet a first-class `.ardisa` type or native opcode.
//! Token counts are supplied by the caller; this module does not pretend to
//! implement tokenization or multimodal device paging.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SegmentModality {
    Text,
    Image,
    Audio,
    Video,
    Structured,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextSegment {
    pub modality: SegmentModality,
    pub payload: Vec<u8>,
    /// Token count from an explicit upstream tokenizer/estimator.
    pub token_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextBudget {
    pub max_segments: usize,
    pub max_bytes: usize,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextWindowError {
    ZeroBudget,
    EmptyPayload,
    SegmentExceedsBudget { bytes: usize, tokens: usize },
    ArithmeticOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextWindow {
    budget: ContextBudget,
    segments: Vec<ContextSegment>,
    bytes: usize,
    tokens: usize,
}

impl ContextWindow {
    pub fn new(budget: ContextBudget) -> Result<Self, ContextWindowError> {
        if budget.max_segments == 0 || budget.max_bytes == 0 || budget.max_tokens == 0 {
            return Err(ContextWindowError::ZeroBudget);
        }
        Ok(Self { budget, segments: Vec::new(), bytes: 0, tokens: 0 })
    }

    pub fn len(&self) -> usize { self.segments.len() }
    pub fn is_empty(&self) -> bool { self.segments.is_empty() }
    pub fn bytes(&self) -> usize { self.bytes }
    pub fn tokens(&self) -> usize { self.tokens }
    pub fn segments(&self) -> &[ContextSegment] { &self.segments }

    /// Append a segment, evicting oldest segments deterministically until all
    /// three budgets are satisfied. A single segment that cannot fit is
    /// rejected before any existing content is mutated.
    pub fn append(&mut self, segment: ContextSegment) -> Result<(), ContextWindowError> {
        if segment.payload.is_empty() { return Err(ContextWindowError::EmptyPayload); }
        let segment_bytes = segment.payload.len();
        if segment_bytes > self.budget.max_bytes || segment.token_count > self.budget.max_tokens {
            return Err(ContextWindowError::SegmentExceedsBudget { bytes: segment_bytes, tokens: segment.token_count });
        }
        let _ = self.bytes.checked_add(segment_bytes).ok_or(ContextWindowError::ArithmeticOverflow)?;
        let _ = self.tokens.checked_add(segment.token_count).ok_or(ContextWindowError::ArithmeticOverflow)?;

        while self.segments.len() >= self.budget.max_segments
            || self.bytes + segment_bytes > self.budget.max_bytes
            || self.tokens + segment.token_count > self.budget.max_tokens
        {
            let removed = self.segments.remove(0).expect("budget loop only runs when at least one segment exists");
            self.bytes -= removed.payload.len();
            self.tokens -= removed.token_count;
        }
        self.bytes += segment_bytes;
        self.tokens += segment.token_count;
        self.segments.push(segment);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(modality: SegmentModality, payload: &[u8], tokens: usize) -> ContextSegment {
        ContextSegment { modality, payload: payload.to_vec(), token_count: tokens }
    }
    fn budget() -> ContextBudget {
        ContextBudget { max_segments: 3, max_bytes: 6, max_tokens: 5 }
    }

    #[test]
    fn rejects_zero_budget_and_empty_payload() {
        assert_eq!(ContextWindow::new(ContextBudget { max_segments: 0, max_bytes: 1, max_tokens: 1 }).unwrap_err(), ContextWindowError::ZeroBudget);
        let mut window = ContextWindow::new(budget()).unwrap();
        assert_eq!(window.append(segment(SegmentModality::Text, b"", 0)), Err(ContextWindowError::EmptyPayload));
    }

    #[test]
    fn evicts_oldest_deterministically_to_meet_all_budgets() {
        let mut window = ContextWindow::new(budget()).unwrap();
        window.append(segment(SegmentModality::Text, b"ab", 2)).unwrap();
        window.append(segment(SegmentModality::Image, b"cd", 1)).unwrap();
        window.append(segment(SegmentModality::Audio, b"ef", 2)).unwrap();
        assert_eq!(window.segments().iter().map(|s| s.modality).collect::<Vec<_>>(), vec![SegmentModality::Image, SegmentModality::Audio]);
        assert_eq!((window.bytes(), window.tokens()), (4, 3));
    }

    #[test]
    fn oversized_segment_does_not_mutate_existing_window() {
        let mut window = ContextWindow::new(budget()).unwrap();
        window.append(segment(SegmentModality::Text, b"ab", 1)).unwrap();
        assert_eq!(window.append(segment(SegmentModality::Video, b"1234567", 1)), Err(ContextWindowError::SegmentExceedsBudget { bytes: 7, tokens: 1 }));
        assert_eq!(window.len(), 1);
        assert_eq!(window.bytes(), 2);
    }

    #[test]
    fn token_budget_also_drives_eviction_and_modality_is_preserved() {
        let mut window = ContextWindow::new(budget()).unwrap();
        window.append(segment(SegmentModality::Structured, b"a", 3)).unwrap();
        window.append(segment(SegmentModality::Text, b"b", 2)).unwrap();
        assert_eq!(window.segments().len(), 1);
        assert_eq!(window.segments()[0].modality, SegmentModality::Text);
        assert_eq!(window.tokens(), 2);
    }
}
