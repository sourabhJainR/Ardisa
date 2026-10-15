//! Bounded ownership-taking iterator semantics for Ardisa's first iterator subset.
//!
//! This module is the reference semantics for owned `List<T>` iteration. It is
//! deliberately not wired into source compilation yet: `for/in` remains rejected
//! until the typed IR and native artifact/runtime paths can enforce the same rules.
//! Keeping the executable surface small prevents parser acceptance from overstating
//! language support.

use std::vec::IntoIter;

/// Failure to create an iterator without violating the collection item budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IteratorError {
    /// The source collection exceeds the runtime collection-item budget.
    ItemLimitExceeded { actual: usize, limit: usize },
}

/// A single-pass iterator that owns its source collection.
///
/// The source list is consumed at construction time, so callers cannot retain a
/// second mutable alias to the collection through this API. Elements are yielded
/// in insertion order and are moved out one at a time; no clone bound is required.
#[derive(Debug)]
pub struct OwnedListIterator<T> {
    items: IntoIter<T>,
    remaining: usize,
}

impl<T> OwnedListIterator<T> {
    /// Consume an owned list and enforce the configured item budget before
    /// iteration begins. An over-budget collection is rejected atomically.
    pub fn new(items: Vec<T>, item_limit: usize) -> Result<Self, IteratorError> {
        let actual = items.len();
        if actual > item_limit {
            return Err(IteratorError::ItemLimitExceeded {
                actual,
                limit: item_limit,
            });
        }

        Ok(Self {
            items: items.into_iter(),
            remaining: actual,
        })
    }

    /// Number of elements not yet yielded.
    pub fn remaining(&self) -> usize {
        self.remaining
    }

    /// Yield the next owned element in source order.
    pub fn next_owned(&mut self) -> Option<T> {
        let next = self.items.next();
        if next.is_some() {
            // The iterator's internal count and underlying iterator must agree.
            // A mismatch is an internal invariant violation, not a user error.
            debug_assert!(self.remaining > 0);
            self.remaining = self.remaining.saturating_sub(1);
        } else {
            debug_assert_eq!(self.remaining, 0);
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_collection_yields_no_items() {
        let mut iter = OwnedListIterator::<i64>::new(vec![], 0).unwrap();
        assert_eq!(iter.remaining(), 0);
        assert_eq!(iter.next_owned(), None);
        assert_eq!(iter.remaining(), 0);
    }

    #[test]
    fn yields_elements_once_in_source_order() {
        let mut iter = OwnedListIterator::new(vec![3, 1, 4, 1, 5], 5).unwrap();
        assert_eq!(iter.remaining(), 5);
        assert_eq!(iter.next_owned(), Some(3));
        assert_eq!(iter.remaining(), 4);
        assert_eq!(iter.next_owned(), Some(1));
        assert_eq!(iter.next_owned(), Some(4));
        assert_eq!(iter.next_owned(), Some(1));
        assert_eq!(iter.next_owned(), Some(5));
        assert_eq!(iter.remaining(), 0);
        assert_eq!(iter.next_owned(), None);
    }

    #[test]
    fn singleton_collection_yields_exactly_once() {
        let mut iter = OwnedListIterator::new(vec![42], 1).unwrap();
        assert_eq!(iter.next_owned(), Some(42));
        assert_eq!(iter.next_owned(), None);
    }

    #[test]
    fn rejects_over_budget_collection_before_yielding_any_element() {
        assert_eq!(
            OwnedListIterator::new(vec![1, 2, 3], 2).err(),
            Some(IteratorError::ItemLimitExceeded { actual: 3, limit: 2 })
        );
    }

    #[test]
    fn permits_exact_budget_and_zero_budget() {
        assert!(OwnedListIterator::new(vec![1, 2], 2).is_ok());
        assert!(OwnedListIterator::<i32>::new(vec![], 0).is_ok());
        assert_eq!(
            OwnedListIterator::new(vec![1], 0).err(),
            Some(IteratorError::ItemLimitExceeded { actual: 1, limit: 0 })
        );
    }

    #[test]
    fn moves_non_clone_elements_without_cloning() {
        #[derive(Debug, PartialEq, Eq)]
        struct NonClone(String);

        let mut iter = OwnedListIterator::new(
            vec![NonClone("first".into()), NonClone("second".into())],
            2,
        ).unwrap();
        assert_eq!(iter.next_owned(), Some(NonClone("first".into())));
        assert_eq!(iter.next_owned(), Some(NonClone("second".into())));
        assert_eq!(iter.next_owned(), None);
    }
}
