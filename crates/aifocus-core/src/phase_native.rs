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

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module lifecycle\nphase Lifecycle\n  Ready -> Running\n  Running -> Done\n";

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
        let source = "module lifecycle\\nphase Lifecycle\\n  initial: Ready\\n  Ready -> Running\\n  Running -> Done\\n";
        let artifact = compile_declared_phase_source(source, "Lifecycle", &["Running".into(), "Done".into()]).unwrap();
        assert_eq!(artifact.initial_state, "Ready");
        assert_eq!(execute_phase_artifact(&artifact).unwrap(), "Done");
    }

    #[test]
    fn declared_phase_requires_a_valid_explicit_initial_state() {
        let missing = "module lifecycle\\nphase Lifecycle\\n  Ready -> Done\\n";
        assert!(matches!(
            compile_declared_phase_source(missing, "Lifecycle", &[]),
            Err(NativePhaseCompileError::MissingInitialState(name)) if name == "Lifecycle"
        ));
        let invalid = "module lifecycle\\nphase Lifecycle\\n  initial: Missing\\n  Ready -> Done\\n";
        assert!(compile_declared_phase_source(invalid, "Lifecycle", &[]).is_err());
        let mismatched = "module lifecycle\\nphase Lifecycle\\n  initial: Ready\\n  Ready -> Done\\n";
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
