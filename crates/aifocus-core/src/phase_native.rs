//! Narrow native execution path for source Phase declarations.
//!
//! This deliberately does not change the general compiler's fail-closed AIF610
//! policy. It accepts exactly one Phase declaration, requires an explicit
//! initial state, lowers declared edges to ARDISA-EXEC-V2 instructions, and
//! executes the resulting artifact through the normal native interpreter.
use std::collections::BTreeMap;

use crate::{
    ast::ConstructKind,
    native::{self, NativeFunction, NativeInstr, NativePhaseGraph, NativeProgram, NativeValue},
    typed_ir::{self, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativePhaseCompileError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    ExpectedSinglePhase,
    UnknownPhase(String),
    UnsupportedConstruct(String),
    MissingInitialState(String),
    InvalidInitialState { phase: String, state: String },
    TooManyOperations,
    Native(native::NativeError),
    InvalidOperation(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePhaseArtifact {
    pub artifact: String,
    pub phase_name: String,
    pub initial_state: String,
    pub operation_count: usize,
}

/// Compile a Phase declaration plus an explicit sequence of transition targets
/// into a deterministic executable artifact. Each operation is checked against
/// the declaration graph by the native interpreter, not by this source adapter.
pub fn compile_phase_source(
    source: &str,
    phase_name: &str,
    initial_state: &str,
    transition_targets: &[String],
) -> Result<NativePhaseArtifact, NativePhaseCompileError> {
    if transition_targets.len() > crate::ai_mode::MAX_PHASE_TRANSITIONS {
        return Err(NativePhaseCompileError::TooManyOperations);
    }
    let module = crate::parse(source).map_err(NativePhaseCompileError::Parse)?;
    let constructs = typed_ir::lower_constructs(&module).map_err(NativePhaseCompileError::TypedIr)?;
    if constructs.len() != 1 || constructs[0].kind != ConstructKind::Phase {
        return Err(NativePhaseCompileError::ExpectedSinglePhase);
    }
    let construct = constructs.iter().find(|item| item.name == phase_name)
        .ok_or_else(|| NativePhaseCompileError::UnknownPhase(phase_name.to_string()))?;

    let mut transitions = Vec::new();
    for member in &construct.members {
        match member {
            TypedIrConstructMember::Transition { from, to, .. } => {
                transitions.push((from.clone(), to.clone()));
            }
            TypedIrConstructMember::Field { name, .. } => {
                return Err(NativePhaseCompileError::UnsupportedConstruct(format!("field '{name}'")));
            }
            TypedIrConstructMember::Clause { name, value, .. } if name == "initial" => {
                if value.trim() != initial_state {
                    return Err(NativePhaseCompileError::InvalidInitialState {
                        phase: phase_name.to_string(),
                        state: value.trim().to_string(),
                    });
                }
            }
            TypedIrConstructMember::Clause { name, .. } => {
                return Err(NativePhaseCompileError::UnsupportedConstruct(format!("clause '{name}'")));
            }
        }
    }
    let known_states: std::collections::BTreeSet<&str> = transitions.iter()
        .flat_map(|(from, to)| [from.as_str(), to.as_str()])
        .collect();
    if !known_states.contains(initial_state) {
        return Err(NativePhaseCompileError::InvalidInitialState {
            phase: phase_name.to_string(),
            state: initial_state.to_string(),
        });
    }
    let graph = NativePhaseGraph {
        name: phase_name.to_string(),
        initial: initial_state.to_string(),
        transitions,
    };
    // Build, serialize, decode, and validate the exact artifact users will execute.
    let mut code = vec![NativeInstr::PhaseStart { graph }];
    for target in transition_targets {
        code.push(NativeInstr::PushString(target.clone()));
        code.push(NativeInstr::PhaseTransition);
    }
    code.push(NativeInstr::PhaseCurrent);
    code.push(NativeInstr::Return);
    let mut functions = BTreeMap::new();
    functions.insert("main".to_string(), NativeFunction { params: Vec::new(), code });
    let program = NativeProgram { functions };
    let artifact = native::encode_program(&program);
    let decoded = native::decode_program(&artifact).map_err(NativePhaseCompileError::Native)?;
    native::ValidatedNativeProgram::try_new(decoded).map_err(NativePhaseCompileError::Native)?;
    Ok(NativePhaseArtifact {
        artifact,
        phase_name: phase_name.to_string(),
        initial_state: initial_state.to_string(),
        operation_count: transition_targets.len(),
    })
}


/// Compile a Phase using its declared `initial:` state instead of a host-supplied
/// initial value. This is the preferred source-backed API for declarations that
/// include an initial state. The operation targets remain explicit runtime inputs.
pub fn compile_declared_phase_source(
    source: &str,
    phase_name: &str,
    transition_targets: &[String],
) -> Result<NativePhaseArtifact, NativePhaseCompileError> {
    let module = crate::parse(source).map_err(NativePhaseCompileError::Parse)?;
    let constructs = typed_ir::lower_constructs(&module).map_err(NativePhaseCompileError::TypedIr)?;
    if constructs.len() != 1 || constructs[0].kind != ConstructKind::Phase {
        return Err(NativePhaseCompileError::ExpectedSinglePhase);
    }
    let construct = constructs.iter().find(|item| item.name == phase_name)
        .ok_or_else(|| NativePhaseCompileError::UnknownPhase(phase_name.to_string()))?;
    let initial_state = construct.members.iter().find_map(|member| match member {
        TypedIrConstructMember::Clause { name, value, .. } if name == "initial" => Some(value.trim().to_string()),
        _ => None,
    }).filter(|value| !value.is_empty())
      .ok_or_else(|| NativePhaseCompileError::MissingInitialState(phase_name.to_string()))?;
    compile_phase_source(source, phase_name, &initial_state, transition_targets)
}

/// Execute the exact serialized artifact emitted by compile_phase_source.
pub fn execute_phase_artifact(artifact: &NativePhaseArtifact) -> Result<String, NativePhaseCompileError> {
    let program = native::decode_program(&artifact.artifact).map_err(NativePhaseCompileError::Native)?;
    match native::run_program(&program, "main", &[]) {
        Ok(NativeValue::String(state)) => Ok(state),
        Ok(_) => Err(NativePhaseCompileError::Native(native::NativeError::InvalidProgram(
            "Phase entry point did not return a String state".into(),
        ))),
        Err(error) => Err(NativePhaseCompileError::Native(error)),
    }
}


/// Execute one validated PhaseTransition operation through the native artifact.
/// This narrow adapter accepts a transition only when its declared `from` state
/// equals the Phase's declared initial state; it never assumes hidden runtime state.
pub fn execute_phase_construct_operation(
    source: &str,
    operation: &crate::construct_operations::ConstructOperation,
) -> Result<String, NativePhaseCompileError> {
    let module = crate::parse(source).map_err(NativePhaseCompileError::Parse)?;
    crate::construct_operations::validate_module_operations(
        &module,
        std::slice::from_ref(operation),
    ).map_err(|errors| NativePhaseCompileError::InvalidOperation(errors.join("; ")))?;

    let (phase_name, from, to) = match &operation.kind {
        crate::construct_operations::ConstructOperationKind::PhaseTransition { phase, from, to } =>
            (phase.as_str(), from.as_str(), to.as_str()),
        _ => return Err(NativePhaseCompileError::InvalidOperation(
            "Phase runtime bridge requires a PhaseTransition operation".into(),
        )),
    };
    let artifact = compile_declared_phase_source(source, phase_name, &[to.to_string()])?;
    if artifact.initial_state != from {
        return Err(NativePhaseCompileError::InvalidOperation(format!(
            "transition starts at '{from}', but the declared initial state is '{}'",
            artifact.initial_state
        )));
    }
    execute_phase_artifact(&artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module lifecycle\nphase Lifecycle\n  Ready -> Running\n  Running -> Done\n";


    #[test]
    fn phase_operation_bridge_executes_only_declared_initial_edge() {
        use crate::construct_operations::{ConstructOperation, ConstructOperationKind};
        let source = "module lifecycle\nphase Lifecycle\n  initial: Ready\n  Ready -> Running\n  Running -> Done\nfn main() -> Int\n  1\n";
        let operation = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::PhaseTransition {
                phase: "Lifecycle".into(), from: "Ready".into(), to: "Running".into(),
            },
        };
        assert_eq!(execute_phase_construct_operation(source, &operation).unwrap(), "Running");
        let wrong_start = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::PhaseTransition {
                phase: "Lifecycle".into(), from: "Running".into(), to: "Done".into(),
            },
        };
        assert!(matches!(
            execute_phase_construct_operation(source, &wrong_start),
            Err(NativePhaseCompileError::InvalidOperation(_))
        ));
    }

    #[test]
    fn phase_operation_bridge_rejects_wrong_kind_and_unresolved_function() {
        use crate::construct_operations::{ConstructOperation, ConstructOperationKind};
        let source = "module lifecycle\nphase Lifecycle\n  initial: Ready\n  Ready -> Running\nfn main() -> Int\n  1\n";
        let wrong_kind = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::ProofEvaluate { proof: "P".into() },
        };
        assert!(matches!(
            execute_phase_construct_operation(source, &wrong_kind),
            Err(NativePhaseCompileError::InvalidOperation(_))
        ));
        let unresolved = ConstructOperation {
            function: "missing".into(),
            kind: ConstructOperationKind::PhaseTransition {
                phase: "Lifecycle".into(), from: "Ready".into(), to: "Running".into(),
            },
        };
        assert!(matches!(
            execute_phase_construct_operation(source, &unresolved),
            Err(NativePhaseCompileError::InvalidOperation(_))
        ));
    }

    #[test]
    fn source_phase_lowers_to_deterministic_native_artifact_and_executes() {
        let operations = vec!["Running".to_string(), "Done".to_string()];
        let first = compile_phase_source(SOURCE, "Lifecycle", "Ready", &operations).unwrap();
        let second = compile_phase_source(SOURCE, "Lifecycle", "Ready", &operations).unwrap();
        assert_eq!(first.artifact, second.artifact);
        assert_eq!(execute_phase_artifact(&first).unwrap(), "Done");
        assert!(first.artifact.starts_with(native::ARTIFACT_MAGIC));
    }

    #[test]
    fn declared_initial_state_lowers_into_the_native_artifact() {
        let source = "module lifecycle\nphase Lifecycle\n  initial: Ready\n  Ready -> Running\n  Running -> Done\n";
        let artifact = compile_declared_phase_source(source, "Lifecycle", &["Running".into(), "Done".into()]).unwrap();
        assert_eq!(artifact.initial_state, "Ready");
        assert_eq!(execute_phase_artifact(&artifact).unwrap(), "Done");
    }

    #[test]
    fn declared_phase_requires_a_valid_explicit_initial_state() {
        let missing = "module lifecycle\nphase Lifecycle\n  Ready -> Done\n";
        assert!(matches!(
            compile_declared_phase_source(missing, "Lifecycle", &[]),
            Err(NativePhaseCompileError::MissingInitialState(name)) if name == "Lifecycle"
        ));
        let invalid = "module lifecycle\nphase Lifecycle\n  initial: Missing\n  Ready -> Done\n";
        assert!(compile_declared_phase_source(invalid, "Lifecycle", &[]).is_err());
        let mismatched = "module lifecycle\nphase Lifecycle\n  initial: Ready\n  Ready -> Done\n";
        assert!(matches!(
            compile_phase_source(mismatched, "Lifecycle", "Done", &[]),
            Err(NativePhaseCompileError::InvalidInitialState { .. })
        ));
    }

    #[test]
    fn native_interpreter_rejects_undeclared_transition() {
        let operations = vec!["Done".to_string()];
        let artifact = compile_phase_source(SOURCE, "Lifecycle", "Ready", &operations).unwrap();
        assert!(matches!(
            execute_phase_artifact(&artifact),
            Err(NativePhaseCompileError::Native(native::NativeError::Type(_)))
        ));
    }

    #[test]
    fn source_adapter_rejects_unknown_phase_initial_state_and_other_constructs() {
        assert!(matches!(
            compile_phase_source(SOURCE, "Missing", "Ready", &[]),
            Err(NativePhaseCompileError::UnknownPhase(_))
        ));
        assert!(matches!(
            compile_phase_source(SOURCE, "Lifecycle", "Unknown", &[]),
            Err(NativePhaseCompileError::InvalidInitialState { .. })
        ));
        let mixed = format!("{SOURCE}cell Counter\n  value: Int\n  invariant: value >= 0\n");
        assert_eq!(
            compile_phase_source(&mixed, "Lifecycle", "Ready", &[]),
            Err(NativePhaseCompileError::ExpectedSinglePhase)
        );
    }

    #[test]
    fn native_artifact_rejects_bad_transition_without_returning_a_success_state() {
        let operations = vec!["Running".to_string(), "Ready".to_string()];
        let artifact = compile_phase_source(SOURCE, "Lifecycle", "Ready", &operations).unwrap();
        assert!(execute_phase_artifact(&artifact).is_err());
    }
}


/// Execute an ordered sequence of source-bound PhaseTransition operations as one
/// native state-machine invocation. Every operation must target the same Phase,
/// each from state must equal the state produced by the preceding transition, and
/// the initial state comes from the source declaration rather than caller inference.
pub fn execute_phase_construct_operations(
    source: &str,
    operations: &[crate::construct_operations::ConstructOperation],
) -> Result<String, NativePhaseCompileError> {
    if operations.is_empty() {
        return Err(NativePhaseCompileError::InvalidOperation(
            "Phase operation sequence must not be empty".into(),
        ));
    }
    let module = crate::parse(source).map_err(NativePhaseCompileError::Parse)?;
    crate::construct_operations::validate_module_operations(&module, operations)
        .map_err(|errors| NativePhaseCompileError::InvalidOperation(errors.join("; ")))?;

    let mut phase_name: Option<&str> = None;
    let mut targets = Vec::with_capacity(operations.len());
    let mut expected_from: Option<String> = None;

    for (index, operation) in operations.iter().enumerate() {
        let (phase, from, to) = match &operation.kind {
            crate::construct_operations::ConstructOperationKind::PhaseTransition { phase, from, to } =>
                (phase.as_str(), from.as_str(), to.as_str()),
            _ => return Err(NativePhaseCompileError::InvalidOperation(format!(
                "operation {index} is not a PhaseTransition"
            ))),
        };
        if let Some(first_phase) = phase_name {
            if phase != first_phase {
                return Err(NativePhaseCompileError::InvalidOperation(format!(
                    "operation {index} targets Phase '{phase}', expected '{first_phase}'"
                )));
            }
        } else {
            phase_name = Some(phase);
        }

        if let Some(expected) = expected_from.as_deref() {
            if from != expected {
                return Err(NativePhaseCompileError::InvalidOperation(format!(
                    "operation {index} starts at '{from}', but the preceding transition ends at '{expected}'"
                )));
            }
        } else {
            expected_from = Some(from.to_string());
        }
        expected_from = Some(to.to_string());
        targets.push(to.to_string());
    }

    let phase_name = phase_name.expect("non-empty operations established a Phase name");
    let artifact = compile_declared_phase_source(source, phase_name, &targets)?;
    let declared_initial = artifact.initial_state.as_str();
    let first_from = match &operations[0].kind {
        crate::construct_operations::ConstructOperationKind::PhaseTransition { from, .. } => from.as_str(),
        _ => unreachable!("operation kind checked above"),
    };
    if first_from != declared_initial {
        return Err(NativePhaseCompileError::InvalidOperation(format!(
            "first transition starts at '{first_from}', but the declared initial state is '{declared_initial}'"
        )));
    }
    execute_phase_artifact(&artifact)
}


#[cfg(test)]
mod phase_operation_sequence_tests {
    use super::*;
    use crate::construct_operations::{ConstructOperation, ConstructOperationKind};

    const SOURCE: &str = "module lifecycle\nphase Lifecycle\n  initial: Ready\n  Ready -> Running\n  Running -> Done\n  Ready -> Failed\nfn main() -> Int\n  1\n";

    fn transition(from: &str, to: &str) -> ConstructOperation {
        ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::PhaseTransition {
                phase: "Lifecycle".into(),
                from: from.into(),
                to: to.into(),
            },
        }
    }

    #[test]
    fn ordered_phase_operations_execute_as_one_state_machine_run() {
        let result = execute_phase_construct_operations(
            SOURCE,
            &[transition("Ready", "Running"), transition("Running", "Done")],
        ).unwrap();
        assert_eq!(result, "Done");
    }

    #[test]
    fn phase_operation_sequence_rejects_discontinuous_or_wrong_initial_state() {
        assert!(execute_phase_construct_operations(
            SOURCE,
            &[transition("Ready", "Running"), transition("Ready", "Failed")],
        ).is_err());
        assert!(execute_phase_construct_operations(
            SOURCE,
            &[transition("Running", "Done")],
        ).is_err());
        assert!(execute_phase_construct_operations(SOURCE, &[]).is_err());
    }

    #[test]
    fn phase_operation_sequence_rejects_mixed_phase_and_wrong_operation_kind() {
        let mixed = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::PhaseTransition {
                phase: "Other".into(), from: "Ready".into(), to: "Done".into(),
            },
        };
        assert!(execute_phase_construct_operations(
            SOURCE,
            &[transition("Ready", "Running"), mixed],
        ).is_err());
        let wrong = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::ProofEvaluate { proof: "P".into() },
        };
        assert!(execute_phase_construct_operations(SOURCE, &[wrong]).is_err());
    }
}
