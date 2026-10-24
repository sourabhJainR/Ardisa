//! Source-to-native artifact path for required Proof outcomes.
//!
//! The native instruction validates runtime-supplied outcome records. It does not
//! produce proofs, authenticate the producer, or claim that a supplied result is
//! trustworthy. Trusted proof generation and evidence provenance remain separate gates.
use std::collections::BTreeMap;

use crate::{
    ast::ConstructKind,
    native::{self, NativeFunction, NativeInstr, NativeProgram, NativeValue},
    typed_ir,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeProofCompileError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    ExpectedSingleProof,
    UnknownProof(String),
    Native(native::NativeError),
    InvalidOperation(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeProofArtifact {
    pub artifact: String,
    pub proof_name: String,
}

/// Compile one source Proof policy to a native artifact. Runtime outcomes are
/// supplied as requirement=status strings, e.g. type_safety=proved.
pub fn compile_proof_source(
    source: &str,
    proof_name: &str,
) -> Result<NativeProofArtifact, NativeProofCompileError> {
    let module = crate::parse(source).map_err(NativeProofCompileError::Parse)?;
    let constructs = typed_ir::lower_constructs(&module).map_err(NativeProofCompileError::TypedIr)?;
    if constructs.len() != 1 || constructs[0].kind != ConstructKind::Proof {
        return Err(NativeProofCompileError::ExpectedSingleProof);
    }
    let construct = constructs.iter().find(|item| item.name == proof_name)
        .ok_or_else(|| NativeProofCompileError::UnknownProof(proof_name.to_string()))?;
    let check = native::proof_check_instruction(construct).map_err(NativeProofCompileError::Native)?;
    let mut functions = BTreeMap::new();
    functions.insert("main".to_string(), NativeFunction {
        params: vec!["results".into()],
        code: vec![NativeInstr::Load("results".into()), check, NativeInstr::Return],
    });
    let artifact = native::encode_program(&NativeProgram { functions });
    let decoded = native::decode_program(&artifact).map_err(NativeProofCompileError::Native)?;
    native::ValidatedNativeProgram::try_new(decoded).map_err(NativeProofCompileError::Native)?;
    Ok(NativeProofArtifact { artifact, proof_name: proof_name.to_string() })
}

/// Validate and execute a named ProofEvaluate operation through the native policy.
/// Outcomes remain explicit runtime evidence supplied by the caller; this bridge does
/// not attest to the provenance or truthfulness of the outcome producer.
pub fn execute_proof_construct_operation(
    source: &str,
    operation: &crate::construct_operations::ConstructOperation,
    outcomes: &[String],
) -> Result<(), NativeProofCompileError> {
    let module = crate::parse(source).map_err(NativeProofCompileError::Parse)?;
    crate::construct_operations::validate_module_operations(
        &module,
        std::slice::from_ref(operation),
    ).map_err(|errors| NativeProofCompileError::InvalidOperation(errors.join("; ")))?;
    let proof_name = match &operation.kind {
        crate::construct_operations::ConstructOperationKind::ProofEvaluate { proof } => proof.as_str(),
        _ => return Err(NativeProofCompileError::InvalidOperation(
            "Proof runtime bridge requires a ProofEvaluate operation".into(),
        )),
    };
    let artifact = compile_proof_source(source, proof_name)?;
    execute_proof_artifact(&artifact, outcomes)
}

/// Execute the artifact with explicit runtime outcome records.
pub fn execute_proof_artifact(
    artifact: &NativeProofArtifact,
    outcomes: &[String],
) -> Result<(), NativeProofCompileError> {
    let program = native::decode_program(&artifact.artifact).map_err(NativeProofCompileError::Native)?;
    let args = [NativeValue::List(outcomes.iter().cloned().map(NativeValue::String).collect())];
    match native::run_program(&program, "main", &args) {
        Ok(NativeValue::Unit) => Ok(()),
        Ok(_) => Err(NativeProofCompileError::Native(native::NativeError::InvalidProgram(
            "Proof entry point did not return Unit".into(),
        ))),
        Err(error) => Err(NativeProofCompileError::Native(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module proofs\nproof ReleaseGate\n  requires: type_safety, bounds\n  ensures: true\n  on_unknown: reject\n  required: true\n";

    #[test]
    fn native_proof_accepts_complete_proved_or_runtime_checked_results() {
        let artifact = compile_proof_source(SOURCE, "ReleaseGate").unwrap();
        let results = vec!["type_safety=proved".into(), "bounds=runtime_checked".into()];
        execute_proof_artifact(&artifact, &results).unwrap();
    }

    #[test]
    fn proof_operation_bridge_validates_declaration_function_and_outcomes() {
        use crate::construct_operations::{ConstructOperation, ConstructOperationKind};
        let source = format!("{SOURCE}fn main() -> Int\\n  1\\n");
        let operation = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::ProofEvaluate { proof: "ReleaseGate".into() },
        };
        execute_proof_construct_operation(&source, &operation, &[
            "type_safety=proved".into(), "bounds=runtime_checked".into(),
        ]).unwrap();
        assert!(execute_proof_construct_operation(&source, &operation, &["type_safety=proved".into()]).is_err());
        let unresolved = ConstructOperation { function: "missing".into(), kind: operation.kind.clone() };
        assert!(matches!(execute_proof_construct_operation(&source, &unresolved, &[
            "type_safety=proved".into(), "bounds=proved".into(),
        ]), Err(NativeProofCompileError::InvalidOperation(_))));
    }

    #[test]
    fn proof_operation_bridge_rejects_wrong_operation_kind_and_unresolved_policy() {
        use crate::construct_operations::{ConstructOperation, ConstructOperationKind};
        let source = format!("{SOURCE}fn main() -> Int\\n  1\\n");
        let wrong_kind = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::TraceEmit { trace: "T".into(), event: "build".into() },
        };
        assert!(matches!(execute_proof_construct_operation(&source, &wrong_kind, &[]),
            Err(NativeProofCompileError::InvalidOperation(_))));
        let unresolved = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::ProofEvaluate { proof: "Missing".into() },
        };
        assert!(execute_proof_construct_operation(&source, &unresolved, &[]).is_err());
    }

    #[test]
    fn missing_failed_unknown_and_malformed_results_reject() {
        let artifact = compile_proof_source(SOURCE, "ReleaseGate").unwrap();
        for results in [
            vec!["type_safety=proved".into()],
            vec!["type_safety=failed".into(), "bounds=proved".into()],
            vec!["type_safety=unknown".into(), "bounds=proved".into()],
            vec!["type_safety=proved".into(), "bounds=maybe".into()],
        ] {
            assert!(execute_proof_artifact(&artifact, &results).is_err(), "accepted {results:?}");
        }
    }

    #[test]
    fn native_proof_rejects_unknown_duplicate_and_optional_policy() {
        assert!(matches!(
            compile_proof_source(
                "module p\nproof P\n  requires: invented\n  ensures: true\n  on_unknown: reject\n  required: true\n",
                "P"
            ),
            Err(NativeProofCompileError::Native(native::NativeError::InvalidProgram(_)))
        ));
        assert!(compile_proof_source(
            "module p\nproof P\n  requires: bounds\n  ensures: true\n  on_unknown: record\n  required: true\n",
            "P"
        ).is_err());
        let artifact = compile_proof_source(SOURCE, "ReleaseGate").unwrap();
        assert!(matches!(
            execute_proof_artifact(&artifact, &["bounds=proved".into(), "bounds=runtime_checked".into(), "type_safety=proved".into()]),
            Err(NativeProofCompileError::Native(native::NativeError::Type(_)))
        ));
    }

    #[test]
    fn native_proof_artifact_is_deterministic_and_rejects_mixed_constructs() {
        let first = compile_proof_source(SOURCE, "ReleaseGate").unwrap();
        let second = compile_proof_source(SOURCE, "ReleaseGate").unwrap();
        assert_eq!(first.artifact, second.artifact);
        let mixed = format!("{SOURCE}cell Counter\n  value: Int\n  invariant: value >= 0\n");
        assert_eq!(
            compile_proof_source(&mixed, "ReleaseGate"),
            Err(NativeProofCompileError::ExpectedSingleProof)
        );
    }
}
