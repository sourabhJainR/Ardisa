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
