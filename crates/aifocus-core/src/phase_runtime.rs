//! Fail-closed runtime primitive for explicitly declared phase transitions.
//!
//! This is a runtime building block, not source-level `phase` execution: the
//! compiler must still lower declarations into a validated transition table.

use std::collections::BTreeSet;
use std::fmt;

pub const MAX_PHASE_STATES: usize = 4096;
pub const MAX_PHASE_TRANSITIONS: usize = 16_384;
pub const MAX_PHASE_HISTORY: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseError {
    EmptyState,
    EmptyTransition,
    NoTransitions,
    TooManyStates,
    TooManyTransitions,
    DuplicateTransition { from: String, to: String },
    UnknownInitialState(String),
    UnknownState(String),
    TransitionNotAllowed { from: String, to: String },
    HistoryLimitExceeded,
}

impl fmt::Display for PhaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyState => write!(f, "phase state must not be empty"),
            Self::EmptyTransition => write!(f, "phase transition endpoints must not be empty"),
            Self::NoTransitions => write!(f, "phase machine requires at least one transition"),
            Self::TooManyStates => write!(f, "phase state limit exceeded"),
            Self::TooManyTransitions => write!(f, "phase transition limit exceeded"),
            Self::DuplicateTransition { from, to } => write!(f, "duplicate phase transition '{from} -> {to}'"),
            Self::UnknownInitialState(state) => write!(f, "initial phase '{state}' is not declared"),
            Self::UnknownState(state) => write!(f, "unknown phase '{state}'"),
            Self::TransitionNotAllowed { from, to } => write!(f, "phase transition '{from} -> {to}' is not declared"),
            Self::HistoryLimitExceeded => write!(f, "phase transition history limit exceeded"),
        }
    }
}

impl std::error::Error for PhaseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseTransition {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseMachine {
    current: String,
    states: BTreeSet<String>,
    transitions: BTreeSet<(String, String)>,
    history: Vec<PhaseTransition>,
}

impl PhaseMachine {
    pub fn new<I, F, T>(initial: impl Into<String>, transitions: I) -> Result<Self, PhaseError>
    where
        I: IntoIterator<Item = (F, T)>,
        F: Into<String>,
        T: Into<String>,
    {
        let initial = initial.into();
        if initial.trim().is_empty() {
            return Err(PhaseError::EmptyState);
        }

        let mut states = BTreeSet::new();
        let mut edges = BTreeSet::new();
        let mut saw_transition = false;
        for (from, to) in transitions {
            saw_transition = true;
            let from = from.into();
            let to = to.into();
            if from.trim().is_empty() || to.trim().is_empty() {
                return Err(PhaseError::EmptyTransition);
            }
            if from == to {
                return Err(PhaseError::TransitionNotAllowed { from, to });
            }
            if !edges.insert((from.clone(), to.clone())) {
                return Err(PhaseError::DuplicateTransition { from, to });
            }
            states.insert(from);
            states.insert(to);
            if edges.len() > MAX_PHASE_TRANSITIONS {
                return Err(PhaseError::TooManyTransitions);
            }
            if states.len() > MAX_PHASE_STATES {
                return Err(PhaseError::TooManyStates);
            }
        }

        if !saw_transition {
            return Err(PhaseError::NoTransitions);
        }
        if !states.contains(&initial) {
            return Err(PhaseError::UnknownInitialState(initial));
        }

        Ok(Self {
            current: initial.clone(),
            states,
            transitions: edges,
            history: Vec::new(),
        })
    }

    pub fn current(&self) -> &str {
        &self.current
    }

    pub fn is_declared(&self, state: &str) -> bool {
        self.states.contains(state)
    }

    pub fn can_transition_to(&self, target: &str) -> bool {
        self.transitions.contains(&(self.current.clone(), target.to_string()))
    }

    /// Apply only a declared edge. On any error, current state and history stay unchanged.
    pub fn transition_to(&mut self, target: impl Into<String>) -> Result<(), PhaseError> {
        let target = target.into();
        if !self.states.contains(&target) {
            return Err(PhaseError::UnknownState(target));
        }
        let from = self.current.clone();
        if !self.transitions.contains(&(from.clone(), target.clone())) {
            return Err(PhaseError::TransitionNotAllowed { from, to: target });
        }
        if self.history.len() >= MAX_PHASE_HISTORY {
            return Err(PhaseError::HistoryLimitExceeded);
        }

        self.history.push(PhaseTransition { from, to: target.clone() });
        self.current = target;
        Ok(())
    }

    pub fn history(&self) -> &[PhaseTransition] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> PhaseMachine {
        PhaseMachine::new("draft", [
            ("draft", "review"),
            ("review", "approved"),
            ("review", "rejected"),
            ("rejected", "draft"),
        ]).expect("valid transition graph")
    }

    #[test]
    fn applies_only_declared_transitions_and_records_history() {
        let mut machine = machine();
        assert!(machine.can_transition_to("review"));
        machine.transition_to("review").unwrap();
        machine.transition_to("approved").unwrap();
        assert_eq!(machine.current(), "approved");
        assert_eq!(machine.history(), &[
            PhaseTransition { from: "draft".into(), to: "review".into() },
            PhaseTransition { from: "review".into(), to: "approved".into() },
        ]);
    }

    #[test]
    fn rejected_transition_preserves_state_and_history() {
        let mut machine = machine();
        let error = machine.transition_to("approved").unwrap_err();
        assert_eq!(error, PhaseError::TransitionNotAllowed {
            from: "draft".into(), to: "approved".into()
        });
        assert_eq!(machine.current(), "draft");
        assert!(machine.history().is_empty());
    }

    #[test]
    fn unknown_state_is_rejected_without_mutation() {
        let mut machine = machine();
        assert_eq!(machine.transition_to("missing"), Err(PhaseError::UnknownState("missing".into())));
        assert_eq!(machine.current(), "draft");
        assert!(machine.history().is_empty());
    }

    #[test]
    fn construction_rejects_invalid_or_ambiguous_graphs() {
        assert_eq!(PhaseMachine::new("draft", Vec::<(&str, &str)>::new()).unwrap_err(), PhaseError::NoTransitions);
        assert_eq!(PhaseMachine::new("missing", [("draft", "review")]).unwrap_err(), PhaseError::UnknownInitialState("missing".into()));
        assert_eq!(
            PhaseMachine::new("draft", [("draft", "review"), ("draft", "review")]).unwrap_err(),
            PhaseError::DuplicateTransition { from: "draft".into(), to: "review".into() }
        );
        assert!(matches!(
            PhaseMachine::new("draft", [("draft", "draft")]),
            Err(PhaseError::TransitionNotAllowed { .. })
        ));
    }

    #[test]
    fn transition_graph_and_history_are_bounded() {
        assert!(MAX_PHASE_STATES > 0 && MAX_PHASE_TRANSITIONS > 0 && MAX_PHASE_HISTORY > 0);
        let mut machine = machine();
        assert!(machine.is_declared("approved"));
        assert!(!machine.is_declared("missing"));
        machine.transition_to("review").unwrap();
        machine.transition_to("rejected").unwrap();
        machine.transition_to("draft").unwrap();
        assert_eq!(machine.current(), "draft");
    }
}
