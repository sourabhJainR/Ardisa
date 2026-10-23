//! Source-to-runtime bridge for validated AI-native construct declarations.
//!
//! This module deliberately does not make constructs executable through
//! ARDISA-EXEC-V1. It provides a checked bridge from source text through typed
//! construct IR to the existing runtime enforcement APIs. The normal compiler
//! remains fail-closed until each construct has native instruction semantics.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ai_mode::{PhaseMachine, PhaseMachineError, PhaseTransition},
    ast::ConstructKind,
    typed_ir::{self, TypedIrConstruct},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructRuntimeError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    DuplicateInitialState(String),
    MissingInitialState(String),
    InitialStateMismatch { phase: String, source: String, supplied: String },
    UnknownInitialState(String),
    UnknownPhase(String),
    Phase { name: String, source: PhaseMachineError },
}

/// Runtime view of construct declarations, with Phase declarations backed by
/// validated state machines. Other construct kinds remain typed metadata only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructRuntime {
    constructs: Vec<TypedIrConstruct>,
    phases: BTreeMap<String, PhaseMachine>,
}

impl ConstructRuntime {
    /// Parse source, lower declarations to typed IR, then construct each Phase
    /// machine using an explicit initial state supplied by the caller.
    ///
    /// Initial states are never inferred from declaration order. Duplicate,
    /// missing, or extraneous initial-state entries are rejected.
    pub fn from_source(
        source: &str,
        initial_states: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Self, ConstructRuntimeError> {
        let module = crate::parse(source).map_err(ConstructRuntimeError::Parse)?;
        let constructs = typed_ir::lower_constructs(&module)
            .map_err(ConstructRuntimeError::TypedIr)?;

        let mut supplied = BTreeMap::new();
        for (name, state) in initial_states {
            if supplied.insert(name.clone(), state).is_some() {
                return Err(ConstructRuntimeError::DuplicateInitialState(name));
            }
        }

        let phase_names: BTreeSet<String> = constructs
            .iter()
            .filter(|construct| construct.kind == ConstructKind::Phase)
            .map(|construct| construct.name.clone())
            .collect();

        if let Some(name) = supplied.keys().find(|name| !phase_names.contains(*name)) {
            return Err(ConstructRuntimeError::UnknownInitialState(name.clone()));
        }

        let mut phases = BTreeMap::new();
        for construct in constructs.iter().filter(|c| c.kind == ConstructKind::Phase) {
            let declared_initial = construct.members.iter().find_map(|member| {
                if let typed_ir::TypedIrConstructMember::Clause { name, value, .. } = member {
                    (name == "initial").then(|| value.trim().to_string())
                } else {
                    None
                }
            });
            let supplied_initial = supplied.remove(&construct.name);
            let initial = match (declared_initial, supplied_initial) {
                (Some(source), Some(supplied)) if source != supplied => {
                    return Err(ConstructRuntimeError::InitialStateMismatch {
                        phase: construct.name.clone(), source, supplied,
                    });
                }
                (Some(source), _) => source,
                (None, Some(supplied)) => supplied,
                (None, None) => return Err(ConstructRuntimeError::MissingInitialState(construct.name.clone())),
            };
            let machine = PhaseMachine::from_typed_construct(construct, initial)
                .map_err(|source| ConstructRuntimeError::Phase {
                    name: construct.name.clone(),
                    source,
                })?;
            phases.insert(construct.name.clone(), machine);
        }

        Ok(Self { constructs, phases })
    }

    /// Typed declaration metadata is retained for evidence and downstream
    /// compiler stages; metadata alone is not executable semantics.
    pub fn constructs(&self) -> &[TypedIrConstruct] {
        &self.constructs
    }

    pub fn phase_current(&self, name: &str) -> Result<&str, ConstructRuntimeError> {
        self.phases
            .get(name)
            .map(PhaseMachine::current)
            .ok_or_else(|| ConstructRuntimeError::UnknownPhase(name.to_string()))
    }

    /// Execute one declared transition. A rejected transition leaves both the
    /// current state and history unchanged.
    pub fn transition_phase(
        &mut self,
        name: &str,
        next: &str,
    ) -> Result<(), ConstructRuntimeError> {
        let machine = self
            .phases
            .get_mut(name)
            .ok_or_else(|| ConstructRuntimeError::UnknownPhase(name.to_string()))?;
        machine.transition(next).map_err(|source| ConstructRuntimeError::Phase {
            name: name.to_string(),
            source,
        })
    }

    pub fn phase_history(
        &self,
        name: &str,
    ) -> Result<&[PhaseTransition], ConstructRuntimeError> {
        self.phases
            .get(name)
            .map(PhaseMachine::history)
            .ok_or_else(|| ConstructRuntimeError::UnknownPhase(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module phase_runtime\nphase Payment\n  Pending -> Authorized\n  Pending -> Cancelled\n  Authorized -> Settled\n";

    #[test]
    fn source_declared_initial_state_is_used_when_no_external_value_is_supplied() {
        let source = "module phase_runtime\nphase Payment\n  initial: Pending\n  Pending -> Authorized\n  Pending -> Cancelled\n  Authorized -> Settled\n";
        let mut runtime = ConstructRuntime::from_source(source, std::iter::empty())
            .expect("source initial state should initialize the machine");
        assert_eq!(runtime.phase_current("Payment").unwrap(), "Pending");
        runtime.transition_phase("Payment", "Authorized").unwrap();
        assert_eq!(runtime.phase_current("Payment").unwrap(), "Authorized");
    }

    #[test]
    fn source_and_external_initial_states_must_match() {
        let source = "module phase_runtime\nphase Payment\n  initial: Pending\n  Pending -> Authorized\n";
        assert_eq!(
            ConstructRuntime::from_source(source, [("Payment".into(), "Authorized".into())]),
            Err(ConstructRuntimeError::InitialStateMismatch {
                phase: "Payment".into(),
                source: "Pending".into(),
                supplied: "Authorized".into(),
            })
        );
    }

    #[test]
    fn invalid_or_duplicate_source_initial_state_fails_closed() {
        let invalid = "module phase_runtime\nphase Payment\n  initial: Unknown\n  Pending -> Authorized\n";
        assert!(matches!(
            ConstructRuntime::from_source(invalid, std::iter::empty()),
            Err(ConstructRuntimeError::TypedIr(_))
        ));
        let duplicate = "module phase_runtime\nphase Payment\n  initial: Pending\n  initial: Pending\n  Pending -> Authorized\n";
        assert!(matches!(
            ConstructRuntime::from_source(duplicate, std::iter::empty()),
            Err(ConstructRuntimeError::TypedIr(_))
        ));
    }

    #[test]
    fn source_phase_lowers_to_typed_ir_and_executes_only_declared_edges() {
        let mut runtime = ConstructRuntime::from_source(
            SOURCE,
            [("Payment".to_string(), "Pending".to_string())],
        )
        .expect("source declaration should produce a checked runtime");

        assert!(runtime.constructs().iter().any(|c| {
            c.kind == ConstructKind::Phase
                && c.name == "Payment"
                && c.members.len() == 3
        }));
        assert_eq!(runtime.phase_current("Payment"), Ok("Pending"));

        runtime.transition_phase("Payment", "Authorized").unwrap();
        assert_eq!(runtime.phase_current("Payment"), Ok("Authorized"));
        runtime.transition_phase("Payment", "Settled").unwrap();
        assert_eq!(runtime.phase_current("Payment"), Ok("Settled"));
        assert_eq!(runtime.phase_history("Payment").unwrap().len(), 2);
    }

    #[test]
    fn rejected_transition_is_atomic_and_does_not_extend_history() {
        let mut runtime = ConstructRuntime::from_source(
            SOURCE,
            [("Payment".to_string(), "Pending".to_string())],
        )
        .unwrap();

        let error = runtime.transition_phase("Payment", "Settled");
        assert!(matches!(
            error,
            Err(ConstructRuntimeError::Phase {
                source: PhaseMachineError::InvalidTransition { .. },
                ..
            })
        ));
        assert_eq!(runtime.phase_current("Payment"), Ok("Pending"));
        assert!(runtime.phase_history("Payment").unwrap().is_empty());
    }

    #[test]
    fn phase_initial_state_must_be_explicit_and_declared() {
        assert_eq!(
            ConstructRuntime::from_source(SOURCE, std::iter::empty()),
            Err(ConstructRuntimeError::MissingInitialState("Payment".into()))
        );
        assert!(matches!(
            ConstructRuntime::from_source(
                SOURCE,
                [("Payment".to_string(), "NotAState".to_string())]
            ),
            Err(ConstructRuntimeError::Phase {
                source: PhaseMachineError::InitialStateUnknown(_),
                ..
            })
        ));
    }

    #[test]
    fn duplicate_and_unrelated_initial_states_are_rejected() {
        assert_eq!(
            ConstructRuntime::from_source(
                SOURCE,
                [
                    ("Payment".to_string(), "Pending".to_string()),
                    ("Payment".to_string(), "Pending".to_string())
                ]
            ),
            Err(ConstructRuntimeError::DuplicateInitialState("Payment".into()))
        );
        assert_eq!(
            ConstructRuntime::from_source(
                SOURCE,
                [
                    ("Payment".to_string(), "Pending".to_string()),
                    ("Unknown".to_string(), "Pending".to_string())
                ]
            ),
            Err(ConstructRuntimeError::UnknownInitialState("Unknown".into()))
        );
    }

    #[test]
    fn runtime_does_not_infer_phase_semantics_from_other_constructs() {
        let source = "module metadata_only\ncell Counter\n  value: Int\n  invariant: value >= 0\n";
        let runtime = ConstructRuntime::from_source(source, std::iter::empty()).unwrap();
        assert!(runtime.phases.is_empty());
        assert_eq!(runtime.constructs()[0].kind, ConstructKind::Cell);
    }
}
