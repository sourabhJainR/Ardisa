//! Authenticated Proof admission for combined Proof + Trace native execution.
//!
//! This is the strict entry point for callers that need producer-authenticated Proof
//! evidence and a source-bound Trace receipt in the same invocation. Lower-level
//! adapters remain available for focused testing and host integrations.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ast::ConstructKind,
    native::{self, ExecutionLimits, NativeValue},
    proof_evidence::{self, ProofEvidence, ReplayStore, TrustRegistry},
    proof_trust_config::{self, ProducerVerifierResolver, TrustConfigSignatureVerifier, TrustConfigStore},
    source_construct_executable::{self, SourceConstructExecutableError},
    trace_policy_runtime::TraceRuntimeSet,
    ai_mode::{TraceEventInput, TraceSigner, TraceVerifier},
};

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Admit a complete authenticated Proof bundle, then execute the combined native
/// artifact and commit its signed Trace receipt.
///
/// All source Proof declarations are included in admission, whether or not their
/// operation appears before the Trace operation. The artifact digest is calculated
/// from the exact encoded artifact string. The Trace subject digest must identify
/// the exact source bytes. Event-label and source/artifact checks happen before
/// replay counters are advanced.
pub fn run_authenticated_proof_and_trace<S: TraceSigner, V: TraceVerifier>(
    source: &str,
    artifact: &str,
    evidence: &[ProofEvidence],
    trust: &TrustRegistry,
    replay: &dyn ReplayStore,
    runtime: &mut TraceRuntimeSet,
    event: TraceEventInput,
    signer: &S,
    verifier: &V,
    limits: ExecutionLimits,
) -> Result<NativeValue, SourceConstructExecutableError> {
    if event.subject_sha256 != sha256_hex(source.as_bytes()) {
        return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
            "Trace subject digest does not match the exact source bytes".into(),
        ));
    }

    let compiled = source_construct_executable::compile_source_with_constructs(source)?;
    if compiled.artifact != artifact {
        return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
            "source does not match the supplied executable artifact".into(),
        ));
    }

    let prepared = crate::source_construct_pipeline::prepare_source_constructs(source)
        .map_err(|error| SourceConstructExecutableError::Preparation(format!("{error:?}")))?;
    let trace_operations: Vec<_> = prepared.operation_bindings.iter().filter_map(|binding| {
        if let crate::construct_operations::ConstructOperationKind::TraceEmit { trace, event } = &binding.operation.kind {
            Some((trace.as_str(), event.as_str()))
        } else { None }
    }).collect();
    if trace_operations.len() != 1 {
        return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
            "authenticated Proof/Trace execution requires exactly one TraceEmit operation".into(),
        ));
    }
    if event.event_kind != trace_operations[0].1 {
        return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
            "runtime event kind does not match the source TraceEmit label".into(),
        ));
    }
    if !prepared.operation_bindings.iter().any(|binding|
        matches!(&binding.operation.kind, crate::construct_operations::ConstructOperationKind::ProofEvaluate { .. })
    ) {
        return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
            "authenticated Proof/Trace execution requires at least one ProofEvaluate operation".into(),
        ));
    }

    let module = crate::parse(source).map_err(SourceConstructExecutableError::Parse)?;
    let typed_constructs = crate::typed_ir::lower_constructs(&module)
        .map_err(SourceConstructExecutableError::TypedIr)?;
    let mut required = BTreeMap::<String, BTreeSet<String>>::new();
    for construct in typed_constructs.iter().filter(|item| item.kind == ConstructKind::Proof) {
        let instruction = native::proof_check_instruction(construct)
            .map_err(|error| SourceConstructExecutableError::Native(format!("{error:?}")))?;
        let native::NativeInstr::ProofCheck { spec } = instruction else {
            return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
                "Proof declaration did not lower to a ProofCheck instruction".into(),
            ));
        };
        if required.insert(
            spec.name.clone(),
            spec.required.iter().map(|item| item.as_str().to_owned()).collect(),
        ).is_some() {
            return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(
                "duplicate Proof gate declaration".into(),
            ));
        }
    }

    let outcomes = proof_evidence::admit(
        &required,
        evidence,
        &sha256_hex(artifact.as_bytes()),
        trust,
        replay,
    ).map_err(|error| SourceConstructExecutableError::UnsupportedRuntimeBoundary(
        format!("authenticated Proof evidence admission failed: {error:?}"),
    ))?;

    source_construct_executable::run_combined_artifact_with_proof_and_trace(
        source, artifact, &outcomes, runtime, event, signer, verifier, limits,
    )
}


/// Execute authenticated Proof + Trace using the currently committed, root-signed
/// producer trust configuration. The supplied store must enforce its own durable
/// rollback boundary; use an anchored production store rather than an in-memory
/// or ordinary snapshot-backed implementation.
pub fn run_authenticated_proof_and_trace_with_trust_config<
    S: TraceSigner,
    V: TraceVerifier,
>(
    source: &str,
    artifact: &str,
    evidence: &[ProofEvidence],
    trust_config_store: &dyn TrustConfigStore,
    root_verifier: &dyn TrustConfigSignatureVerifier,
    producer_verifiers: &dyn ProducerVerifierResolver,
    replay: &dyn ReplayStore,
    runtime: &mut TraceRuntimeSet,
    event: TraceEventInput,
    signer: &S,
    verifier: &V,
    limits: ExecutionLimits,
) -> Result<NativeValue, SourceConstructExecutableError> {
    let trust = proof_trust_config::load_authenticated_registry(
        trust_config_store,
        root_verifier,
        producer_verifiers,
    ).map_err(|error| SourceConstructExecutableError::UnsupportedRuntimeBoundary(
        format!("authenticated Proof trust configuration could not be loaded: {error:?}"),
    ))?;

    run_authenticated_proof_and_trace(
        source, artifact, evidence, &trust, replay, runtime, event, signer, verifier, limits,
    )
}
