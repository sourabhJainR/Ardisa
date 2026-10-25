//! Source-to-native event validation followed by signed, atomic Trace recording.
//!
//! The VM validates the source event shape. Signature generation, verification,
//! and evidence trust remain explicit host interfaces; a receipt is committed only
//! after trusted verification accepts the candidate chain.
use std::collections::BTreeMap;

use crate::{
    ai_mode::{TraceEventInput, TraceSigner, TraceVerifier},
    ast::ConstructKind,
    native::{self, NativeFunction, NativeInstr, NativeProgram, NativeValue},
    trace_policy_runtime::{TraceRuntimeError, TraceRuntimeSet},
    typed_ir::{self, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeTraceCompileError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    ExpectedSingleTrace,
    UnknownTrace(String),
    UnsupportedConfiguration { trace: String, detail: String },
    Native(native::NativeError),
    Runtime(TraceRuntimeError),
    InvalidOperation(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTraceArtifact {
    pub artifact: String,
    pub trace_name: String,
}

/// Compile one required-signature Trace declaration to a deterministic native artifact.
pub fn compile_trace_source(
    source: &str,
    trace_name: &str,
) -> Result<NativeTraceArtifact, NativeTraceCompileError> {
    let module = crate::parse(source).map_err(NativeTraceCompileError::Parse)?;
    let constructs = typed_ir::lower_constructs(&module).map_err(NativeTraceCompileError::TypedIr)?;
    // This focused artifact path may consume a combined module, but it requires
    // exactly one Trace declaration. Other construct metadata is validated and
    // retained by the source pipeline; this artifact executes only Trace policy.
    let traces: Vec<_> = constructs.iter()
        .filter(|item| item.kind == ConstructKind::Trace)
        .collect();
    if traces.len() != 1 {
        return Err(NativeTraceCompileError::ExpectedSingleTrace);
    }
    let construct = traces.iter().copied().find(|item| item.name == trace_name)
        .ok_or_else(|| NativeTraceCompileError::UnknownTrace(trace_name.to_string()))?;
    let mut signature = None;
    for member in &construct.members {
        match member {
            TypedIrConstructMember::Clause { name, value, .. } if name == "signature" => {
                signature = Some(value.trim());
            }
            _ => return Err(NativeTraceCompileError::UnsupportedConfiguration {
                trace: trace_name.to_string(),
                detail: "only the signature clause is supported".into(),
            }),
        }
    }
    if signature != Some("required") {
        return Err(NativeTraceCompileError::UnsupportedConfiguration {
            trace: trace_name.to_string(),
            detail: "signature: required is mandatory".into(),
        });
    }
    let mut functions = BTreeMap::new();
    functions.insert("main".to_string(), NativeFunction {
        params: vec!["event".into()],
        code: vec![
            NativeInstr::Load("event".into()),
            NativeInstr::TracePrepare { name: trace_name.to_string() },
            NativeInstr::Return,
        ],
    });
    let artifact = native::encode_program(&NativeProgram { functions });
    let decoded = native::decode_program(&artifact).map_err(NativeTraceCompileError::Native)?;
    native::ValidatedNativeProgram::try_new(decoded).map_err(NativeTraceCompileError::Native)?;
    Ok(NativeTraceArtifact { artifact, trace_name: trace_name.to_string() })
}

/// Validate an event in the VM, then atomically append and verify its signed receipt.
pub fn execute_trace_event<S: TraceSigner, V: TraceVerifier>(
    artifact: &NativeTraceArtifact,
    runtime: &mut TraceRuntimeSet,
    event: TraceEventInput,
    signer: &S,
    verifier: &V,
) -> Result<(), NativeTraceCompileError> {
    let event_fields = vec![
        event.event_kind.clone(),
        event.actor.clone(),
        event.subject_sha256.clone(),
        event.outcome.clone(),
        event.provider_id.clone().unwrap_or_default(),
        event.session_sha256.clone().unwrap_or_default(),
    ];
    let program = native::decode_program(&artifact.artifact).map_err(NativeTraceCompileError::Native)?;
    let embedded_trace = program.functions.get("main").and_then(|function| {
        function.code.iter().find_map(|instruction| match instruction {
            NativeInstr::TracePrepare { name } => Some(name.as_str()),
            _ => None,
        })
    });
    if embedded_trace != Some(artifact.trace_name.as_str()) {
        return Err(NativeTraceCompileError::Native(native::NativeError::InvalidProgram(
            "Trace artifact name does not match its encoded policy".into(),
        )));
    }
    let args = [NativeValue::List(event_fields.into_iter().map(NativeValue::String).collect())];
    match native::run_program(&program, "main", &args) {
        Ok(NativeValue::List(_)) => {}
        Ok(_) => return Err(NativeTraceCompileError::Native(native::NativeError::InvalidProgram(
            "Trace entry point did not return its validated event".into(),
        ))),
        Err(error) => return Err(NativeTraceCompileError::Native(error)),
    }
    runtime.record_verified(&artifact.trace_name, event, signer, verifier)
        .map_err(NativeTraceCompileError::Runtime)
}

/// Execute a validated TraceEmit operation through the native event validator and
/// the trusted host signing/verifying boundary. The operation's event label must
/// match the structured runtime event; this adapter never fabricates missing fields.
pub fn execute_trace_construct_operation<S: TraceSigner, V: TraceVerifier>(
    source: &str,
    operation: &crate::construct_operations::ConstructOperation,
    runtime: &mut TraceRuntimeSet,
    event: TraceEventInput,
    signer: &S,
    verifier: &V,
) -> Result<(), NativeTraceCompileError> {
    let module = crate::parse(source).map_err(NativeTraceCompileError::Parse)?;
    crate::construct_operations::validate_module_operations(
        &module,
        std::slice::from_ref(operation),
    ).map_err(|errors| NativeTraceCompileError::InvalidOperation(errors.join("; ")))?;

    let trace_name = match &operation.kind {
        crate::construct_operations::ConstructOperationKind::TraceEmit { trace, event: declared_event } => {
            if declared_event.trim().is_empty() {
                return Err(NativeTraceCompileError::InvalidOperation("Trace event label must not be empty".into()));
            }
            if declared_event != &event.event_kind {
                return Err(NativeTraceCompileError::InvalidOperation(
                    "Trace operation event label does not match the structured event kind".into(),
                ));
            }
            trace.as_str()
        }
        _ => return Err(NativeTraceCompileError::InvalidOperation(
            "Trace runtime bridge requires a TraceEmit operation".into(),
        )),
    };
    let artifact = compile_trace_source(source, trace_name)?;
    execute_trace_event(&artifact, runtime, event, signer, verifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module traces\ntrace Build\n  signature: required\nfn main() -> Int\n  1\n";

    struct TestSigner;
    impl TraceSigner for TestSigner {
        fn sign(&self, payload: &[u8]) -> Result<Vec<u8>, String> {
            Ok(payload.iter().map(|byte| byte ^ 0x5a).collect())
        }
    }
    impl TraceVerifier for TestSigner {
        fn verify(&self, payload: &[u8], signature: &[u8]) -> Result<bool, String> {
            Ok(payload.iter().map(|byte| byte ^ 0x5a).eq(signature.iter().copied()))
        }
    }

    struct BadSigner;
    impl TraceSigner for BadSigner {
        fn sign(&self, _payload: &[u8]) -> Result<Vec<u8>, String> {
            Ok(b"bad signature".to_vec())
        }
    }

    fn event() -> TraceEventInput {
        TraceEventInput {
            event_kind: "build".into(),
            actor: "compiler".into(),
            subject_sha256: "a".repeat(64),
            outcome: "passed".into(),
            provider_id: None,
            session_sha256: None,
        }
    }

    #[test]
    fn source_trace_round_trips_and_commits_verified_signed_receipts() {
        let artifact = compile_trace_source(SOURCE, "Build").unwrap();
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        execute_trace_event(&artifact, &mut runtime, event(), &TestSigner, &TestSigner).unwrap();
        runtime.verify("Build", &TestSigner).unwrap();
        assert_eq!(runtime.receipts("Build").unwrap().len(), 1);
        assert_eq!(artifact.artifact, compile_trace_source(SOURCE, "Build").unwrap().artifact);
    }

    #[test]
    fn trace_construct_operation_binds_event_label_and_commits_only_verified_receipt() {
        use crate::construct_operations::{ConstructOperation, ConstructOperationKind};
        let operation = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::TraceEmit { trace: "Build".into(), event: "build".into() },
        };
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        execute_trace_construct_operation(SOURCE, &operation, &mut runtime, event(), &TestSigner, &TestSigner).unwrap();
        assert_eq!(runtime.receipts("Build").unwrap().len(), 1);
        runtime.verify("Build", &TestSigner).unwrap();

        let mismatch = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::TraceEmit { trace: "Build".into(), event: "release".into() },
        };
        assert!(matches!(
            execute_trace_construct_operation(SOURCE, &mismatch, &mut runtime, event(), &TestSigner, &TestSigner),
            Err(NativeTraceCompileError::InvalidOperation(message)) if message.contains("does not match")
        ));
        assert_eq!(runtime.receipts("Build").unwrap().len(), 1, "mismatched event must not append a receipt");
    }

    #[test]
    fn trace_operation_rejects_wrong_operation_kind_and_unresolved_function() {
        use crate::construct_operations::{ConstructOperation, ConstructOperationKind};
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        let wrong_kind = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::ProofEvaluate { proof: "P".into() },
        };
        assert!(matches!(
            execute_trace_construct_operation(SOURCE, &wrong_kind, &mut runtime, event(), &TestSigner, &TestSigner),
            Err(NativeTraceCompileError::InvalidOperation(_))
        ));
        let unresolved = ConstructOperation {
            function: "missing".into(),
            kind: ConstructOperationKind::TraceEmit { trace: "Build".into(), event: "build".into() },
        };
        assert!(matches!(
            execute_trace_construct_operation(SOURCE, &unresolved, &mut runtime, event(), &TestSigner, &TestSigner),
            Err(NativeTraceCompileError::InvalidOperation(_))
        ));
        assert!(runtime.receipts("Build").unwrap().is_empty());
    }

    #[test]
    fn malformed_events_and_untrusted_signatures_do_not_commit_receipts() {
        let artifact = compile_trace_source(SOURCE, "Build").unwrap();
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        let mut malformed = event();
        malformed.subject_sha256 = "not-a-digest".into();
        assert!(execute_trace_event(&artifact, &mut runtime, malformed, &TestSigner, &TestSigner).is_err());
        assert!(runtime.receipts("Build").unwrap().is_empty());
        let mut metadata = event();
        metadata.provider_id = Some("provider".into());
        assert!(execute_trace_event(&artifact, &mut runtime, metadata, &TestSigner, &TestSigner).is_err());
        assert!(runtime.receipts("Build").unwrap().is_empty());
        assert!(execute_trace_event(&artifact, &mut runtime, event(), &BadSigner, &TestSigner).is_err());
        assert!(runtime.receipts("Build").unwrap().is_empty());
    }

    #[test]
    fn modified_artifact_metadata_cannot_redirect_receipts_to_another_trace() {
        let mut artifact = compile_trace_source(SOURCE, "Build").unwrap();
        artifact.trace_name = "OtherTrace".into();
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        assert!(execute_trace_event(&artifact, &mut runtime, event(), &TestSigner, &TestSigner).is_err());
        assert!(runtime.receipts("Build").unwrap().is_empty());
    }

    #[test]
    fn trace_compilation_fails_closed_for_optional_policy_and_duplicate_traces() {
        assert!(compile_trace_source(
            "module t\ntrace T\n  signature: optional\n", "T"
        ).is_err());
        let mixed = "module t\ntrace T\n  signature: required\ncell C\n  value: Int\n  invariant: value >= 0\nfn main() -> Int\n  1\n";
        assert!(compile_trace_source(mixed, "T").is_ok());
        let duplicate = format!("{mixed}trace Other\n  signature: required\n");
        assert_eq!(
            compile_trace_source(&duplicate, "T"),
            Err(NativeTraceCompileError::ExpectedSingleTrace)
        );
    }
}
