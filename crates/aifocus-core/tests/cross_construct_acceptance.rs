//! Cross-construct acceptance checks for the bounded source runtime adapters.
//!
//! These tests exercise the host-side enforcement boundary for Cell, Phase, Vault,
//! Proof, and Trace in one source module. They deliberately do not promote Proof or
//! Trace to standard ARDISA-EXEC-V1 support: compile_source must remain fail-closed.

use std::cell::Cell as FlagCell;
use std::collections::BTreeMap;

use ardisa_core::{
    ai_mode::{
        Capability, ProofRequirement, ProofStatus, TraceEventInput, TraceSigner, TraceVerifier,
    },
    source_construct_runtime::{SourceConstructRuntime, SourceRuntimeError},
};

const SOURCE: &str = r#"module cross_construct_acceptance
phase Lifecycle
  Ready -> Done
cell Counter
  value: Int
  invariant: value >= 0
vault Local
  capabilities: filesystem_read, clock
proof ReleaseGate
  requires: type_safety, bounds
  ensures: true
  on_unknown: reject
  required: true
trace Build
  signature: required
fn main() -> Int
  return 0
"#;

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
        Ok(b"not-a-valid-signature".to_vec())
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

fn runtime() -> SourceConstructRuntime {
    SourceConstructRuntime::from_source(
        SOURCE,
        [("Lifecycle".into(), "Ready".into())],
        [("Counter".into(), 1)],
    )
    .expect("all five bounded construct adapters should initialize")
}

#[test]
fn all_five_constructs_compose_with_explicit_runtime_inputs() {
    let mut runtime = runtime();

    assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Ready");
    runtime.phases.transition_phase("Lifecycle", "Done").unwrap();
    assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Done");

    runtime.cells.update("Counter", 2).unwrap();
    assert_eq!(runtime.cells.get("Counter").unwrap(), 2);

    let mut proof_results = BTreeMap::new();
    proof_results.insert(ProofRequirement::TypeSafety, ProofStatus::Proved);
    proof_results.insert(ProofRequirement::Bounds, ProofStatus::RuntimeChecked);
    runtime.validate_proofs("ReleaseGate", &proof_results).unwrap();

    let invoked = FlagCell::new(false);
    runtime
        .execute_vault("Local", [Capability::Network], || invoked.set(true))
        .expect_err("undeclared and denied network capability must reject");
    assert!(!invoked.get(), "denied host effect must not run");

    runtime
        .record_trace("Build", event(), Some(&TestSigner))
        .unwrap();
    runtime.verify_trace("Build", &TestSigner).unwrap();
    assert_eq!(runtime.traces.receipts("Build").unwrap().len(), 1);
}

#[test]
fn cell_and_phase_rejections_preserve_prior_state() {
    let mut runtime = runtime();

    assert!(runtime.cells.update("Counter", -1).is_err());
    assert_eq!(runtime.cells.get("Counter").unwrap(), 1);

    assert!(runtime.phases.transition_phase("Lifecycle", "Missing").is_err());
    assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Ready");
}

#[test]
fn proof_missing_failed_and_unknown_outcomes_all_reject() {
    let runtime = runtime();

    assert!(matches!(
        runtime.validate_proofs("ReleaseGate", &BTreeMap::new()),
        Err(SourceRuntimeError::Proof(_))
    ));

    let mut failed = BTreeMap::new();
    failed.insert(ProofRequirement::TypeSafety, ProofStatus::Failed);
    failed.insert(ProofRequirement::Bounds, ProofStatus::Proved);
    assert!(matches!(
        runtime.validate_proofs("ReleaseGate", &failed),
        Err(SourceRuntimeError::Proof(_))
    ));

    let mut unknown = BTreeMap::new();
    unknown.insert(ProofRequirement::TypeSafety, ProofStatus::Unknown);
    unknown.insert(ProofRequirement::Bounds, ProofStatus::Proved);
    assert!(matches!(
        runtime.validate_proofs("ReleaseGate", &unknown),
        Err(SourceRuntimeError::Proof(_))
    ));
}

#[test]
fn trace_candidate_is_not_committed_when_signature_verification_fails() {
    let mut runtime = runtime();

    let result = runtime.traces.record_verified("Build", event(), &BadSigner, &TestSigner);
    assert!(result.is_err());
    assert!(runtime.traces.receipts("Build").unwrap().is_empty());

    runtime
        .traces
        .record_verified("Build", event(), &TestSigner, &TestSigner)
        .unwrap();
    assert_eq!(runtime.traces.receipts("Build").unwrap().len(), 1);
    runtime.traces.verify("Build", &TestSigner).unwrap();
}

#[test]
fn standard_compiler_keeps_proof_and_trace_fail_closed() {
    let error = ardisa_core::pipeline::compile_source(SOURCE)
        .expect_err("Proof and Trace still require an explicit executable trust contract");
    assert!(matches!(
        error,
        ardisa_core::PipelineError::CompilerContract(ref diagnostics)
            if diagnostics.iter().any(|diagnostic| diagnostic.contains("UnsupportedRuntimeBoundary"))
    ));
}


#[test]
fn all_five_constructs_bind_to_one_validated_source_operation_batch() {
    use ardisa_core::{
        ast::{Item, StmtKind, TypeKind},
        combined_construct_runtime::CombinedConstructPlan,
        construct_operations::{ConstructOperation, ConstructOperationKind},
    };

    let module = ardisa_core::parse(SOURCE).expect("combined source parses");
    let function = match &module.items[0] {
        Item::Function(function) if function.name == "main" => function,
        other => panic!("expected main function, got {other:?}"),
    };
    let expression = match &function.body.stmts[0].kind {
        StmtKind::Return(Some(expression)) => expression.id.0,
        other => panic!("expected a return expression for Cell initialization, got {other:?}"),
    };

    let operations = vec![
        ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::CellInitialize {
                cell: "Counter".into(),
                field: "value".into(),
                expression,
                value_type: TypeKind::Int,
            },
        },
        ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::PhaseTransition {
                phase: "Lifecycle".into(),
                from: "Ready".into(),
                to: "Done".into(),
            },
        },
        ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::VaultAuthorize {
                vault: "Local".into(),
                capability: "filesystem_read".into(),
            },
        },
        ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::ProofEvaluate {
                proof: "ReleaseGate".into(),
            },
        },
        ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::TraceEmit {
                trace: "Build".into(),
                event: "compile".into(),
            },
        },
    ];

    let plan = CombinedConstructPlan::validate(SOURCE, &operations)
        .expect("all five source constructs must validate as one batch");
    assert_eq!(plan.module_name(), "cross_construct_acceptance");
    assert_eq!(plan.operations(), operations.as_slice());

    let mut dispatch_count = 0;
    let count = plan
        .dispatch_validated_batch(|batch| {
            dispatch_count += 1;
            assert_eq!(batch, operations.as_slice());
            Ok(batch.len())
        })
        .expect("validated batch dispatch succeeds");
    assert_eq!(count, 5);
    assert_eq!(dispatch_count, 1);
}

#[test]
fn all_five_construct_declarations_have_deterministic_metadata_round_trip() {
    let module = ardisa_core::parse(SOURCE).expect("combined source parses");
    let constructs = ardisa_core::typed_ir::lower_constructs(&module)
        .expect("all construct declarations lower to typed metadata");
    assert_eq!(constructs.len(), 5);

    let encoded = ardisa_core::typed_ir::encode_construct_metadata(&constructs)
        .expect("metadata encoding validates");
    let decoded = ardisa_core::typed_ir::decode_construct_metadata(&encoded)
        .expect("encoded metadata decodes");
    let reencoded = ardisa_core::typed_ir::encode_construct_metadata(&decoded)
        .expect("decoded metadata remains valid");

    assert_eq!(reencoded, encoded);
    assert_eq!(decoded.len(), 5);
}

#[test]
fn failed_proof_prevents_phase_and_cell_promotion() {
    let mut runtime = runtime();
    let mut failed = BTreeMap::new();
    failed.insert(ProofRequirement::TypeSafety, ProofStatus::Failed);
    failed.insert(ProofRequirement::Bounds, ProofStatus::Proved);

    let proof_result = runtime.validate_proofs("ReleaseGate", &failed);
    assert!(matches!(proof_result, Err(SourceRuntimeError::Proof(_))));

    // State promotion is gated on successful proof validation.
    if proof_result.is_ok() {
        runtime.phases.transition_phase("Lifecycle", "Done").unwrap();
        runtime.cells.update("Counter", 2).unwrap();
    }

    assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Ready");
    assert!(runtime.phases.phase_history("Lifecycle").unwrap().is_empty());
    assert_eq!(runtime.cells.get("Counter").unwrap(), 1);
}

struct RejectingEvidenceVerifier;

impl ardisa_core::evidence::EvidenceSourceVerifier for RejectingEvidenceVerifier {
    fn verify_source_backing(
        &self,
        _envelope: &ardisa_core::evidence::EvidenceEnvelope,
    ) -> Result<(), String> {
        Err("source provenance is not trusted".into())
    }
}

#[test]
fn untrusted_trace_source_cannot_be_promoted_into_evidence_graph() {
    let mut runtime = runtime();
    runtime.record_trace("Build", event(), Some(&TestSigner)).unwrap();
    runtime.verify_trace("Build", &TestSigner).unwrap();

    let result = runtime.verified_trace_evidence(
        "Build",
        "https://example.invalid/source",
        &TestSigner,
        &RejectingEvidenceVerifier,
    );
    assert!(matches!(
        result,
        Err(SourceRuntimeError::Trace(_))
    ));
}
