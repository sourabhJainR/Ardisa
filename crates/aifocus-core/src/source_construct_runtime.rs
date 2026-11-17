//! Unified host-runtime facade for Ardisa's source-backed AI constructs.
//!
//! This composes the focused host-side Phase, Cell, Vault, Proof, and Trace
//! adapters for one source module. It is separate from the bounded native
//! executable path in `pipeline::compile_source`; it does not add native
//! instructions or provide atomicity across external host effects.
use std::collections::BTreeMap;

use crate::{
    ai_mode::{ProofRequirement, ProofStatus, TraceEventInput, TraceSigner, TraceVerifier, Capability},
    cell_runtime::{CellRuntime, CellRuntimeError},
    construct_runtime::{ConstructRuntime, ConstructRuntimeError},
    evidence::{EvidenceGraph, EvidenceSourceVerifier},
    proof_policy_runtime::{ProofRuntimeError, ProofRuntimeSet},
    trace_policy_runtime::{TraceRuntimeError, TraceRuntimeSet},
    vault_runtime::{VaultRuntimeError, VaultRuntimeSet},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceRuntimeError {
    Phase(ConstructRuntimeError),
    Cell(CellRuntimeError),
    Vault(VaultRuntimeError),
    Proof(ProofRuntimeError),
    Trace(TraceRuntimeError),
}

/// A single source module's host-side enforcement adapters.
///
/// Public adapter fields make each policy boundary explicit. This type does not
/// claim atomic transactions across unrelated host effects; each operation is
/// individually checked by its owning adapter.
pub struct SourceConstructRuntime {
    pub phases: ConstructRuntime,
    pub cells: CellRuntime,
    pub vaults: VaultRuntimeSet,
    pub proofs: ProofRuntimeSet,
    pub traces: TraceRuntimeSet,
}

impl SourceConstructRuntime {
    /// Build all available runtime adapters from one source module.
    /// Initial states and values are explicit; no source declaration silently
    /// receives a default state.
    pub fn from_source(
        source: &str,
        phase_initial_states: impl IntoIterator<Item = (String, String)>,
        cell_initial_values: impl IntoIterator<Item = (String, i64)>,
    ) -> Result<Self, SourceRuntimeError> {
        let phases = ConstructRuntime::from_source(source, phase_initial_states)
            .map_err(SourceRuntimeError::Phase)?;
        let cells = CellRuntime::from_source(source, cell_initial_values)
            .map_err(SourceRuntimeError::Cell)?;
        let vaults = VaultRuntimeSet::from_source(source).map_err(SourceRuntimeError::Vault)?;
        let proofs = ProofRuntimeSet::from_source(source).map_err(SourceRuntimeError::Proof)?;
        let traces = TraceRuntimeSet::from_source(source).map_err(SourceRuntimeError::Trace)?;
        Ok(Self { phases, cells, vaults, proofs, traces })
    }

    pub fn validate_proofs(
        &self,
        proof: &str,
        results: &BTreeMap<ProofRequirement, ProofStatus>,
    ) -> Result<(), SourceRuntimeError> {
        self.proofs.validate_results(proof, results).map_err(SourceRuntimeError::Proof)
    }

    /// Run Cell and Phase state changes as one in-memory transaction.
    ///
    /// The operation receives staged clones. Neither state machine is replaced
    /// unless the complete closure succeeds, so a late Cell invariant failure
    /// cannot leave an earlier Phase transition or history entry committed.
    /// This transaction does not roll back external host effects, Vault closures,
    /// Trace receipts, or other resources outside these two state stores.
    pub fn transact_cell_phase<T>(
        &mut self,
        operation: impl FnOnce(
            &mut crate::construct_runtime::ConstructRuntime,
            &mut CellRuntime,
        ) -> Result<T, SourceRuntimeError>,
    ) -> Result<T, SourceRuntimeError> {
        let mut staged_phases = self.phases.clone();
        let mut staged_cells = self.cells.clone();
        let result = operation(&mut staged_phases, &mut staged_cells)?;
        self.phases = staged_phases;
        self.cells = staged_cells;
        Ok(result)
    }

    /// Require all declared Proof outcomes before committing a Cell/Phase
    /// transaction. Missing, failed, or unknown outcomes reject before staging.
    pub fn transact_proof_gated_cell_phase<T>(
        &mut self,
        proof: &str,
        results: &BTreeMap<ProofRequirement, ProofStatus>,
        operation: impl FnOnce(
            &mut crate::construct_runtime::ConstructRuntime,
            &mut CellRuntime,
        ) -> Result<T, SourceRuntimeError>,
    ) -> Result<T, SourceRuntimeError> {
        self.validate_proofs(proof, results)?;
        self.transact_cell_phase(operation)
    }

    pub fn execute_vault<T>(
        &self,
        vault: &str,
        required: impl IntoIterator<Item = Capability>,
        operation: impl FnOnce() -> T,
    ) -> Result<T, SourceRuntimeError> {
        self.vaults.execute(vault, required, operation).map_err(SourceRuntimeError::Vault)
    }

    pub fn record_trace<S: TraceSigner>(
        &mut self,
        trace: &str,
        event: TraceEventInput,
        signer: Option<&S>,
    ) -> Result<&crate::ai_mode::TraceReceipt, SourceRuntimeError> {
        self.traces.record(trace, event, signer).map_err(SourceRuntimeError::Trace)
    }

    pub fn verify_trace<V: TraceVerifier>(
        &self,
        trace: &str,
        verifier: &V,
    ) -> Result<(), SourceRuntimeError> {
        self.traces.verify(trace, verifier).map_err(SourceRuntimeError::Trace)
    }

    pub fn verified_trace_evidence<V: TraceVerifier, E: EvidenceSourceVerifier>(
        &self,
        trace: &str,
        source_uri: &str,
        verifier: &V,
        evidence_verifier: &E,
    ) -> Result<EvidenceGraph, SourceRuntimeError> {
        self.traces.verified_evidence_graph(trace, source_uri, verifier, evidence_verifier)
            .map_err(SourceRuntimeError::Trace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_mode::{ProofRequirement, ProofStatus, TraceError};
    use std::cell::Cell as FlagCell;

    const SOURCE: &str = "module integrated\nphase Lifecycle\n  Ready -> Done\ncell Counter\n  value: Int\n  invariant: value >= 0\nvault Local\n  capabilities: filesystem_read clock\nproof ReleaseGate\n  requires: type_safety bounds\n  ensures: true\n  on_unknown: reject\n  required: true\ntrace Build\n  signature: required\n";

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

    #[test]
    fn source_module_composes_all_runtime_enforcement_adapters() {
        let mut runtime = SourceConstructRuntime::from_source(
            SOURCE,
            [("Lifecycle".to_string(), "Ready".to_string())],
            [("Counter".to_string(), 1)],
        ).unwrap();

        assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Ready");
        runtime.phases.transition_phase("Lifecycle", "Done").unwrap();
        assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Done");
        runtime.cells.update("Counter", 2).unwrap();
        assert_eq!(runtime.cells.get("Counter").unwrap(), 2);

        let called = FlagCell::new(false);
        runtime.execute_vault("Local", [Capability::Network], || called.set(true)).unwrap_err();
        assert!(!called.get());

        let mut outcomes = BTreeMap::new();
        outcomes.insert(ProofRequirement::TypeSafety, ProofStatus::Proved);
        outcomes.insert(ProofRequirement::Bounds, ProofStatus::RuntimeChecked);
        runtime.validate_proofs("ReleaseGate", &outcomes).unwrap();

        assert!(matches!(
            runtime.record_trace("Build", TraceEventInput {
                event_kind: "build".into(), actor: "compiler".into(),
                subject_sha256: "a".repeat(64), outcome: "passed".into(),
                provider_id: None, session_sha256: None,
            }, None::<&TestSigner>),
            Err(SourceRuntimeError::Trace(TraceRuntimeError::Trace(TraceError::SigningFailed(_))))
        ));
        runtime.record_trace("Build", TraceEventInput {
            event_kind: "build".into(), actor: "compiler".into(),
            subject_sha256: "a".repeat(64), outcome: "passed".into(),
            provider_id: None, session_sha256: None,
        }, Some(&TestSigner)).unwrap();
        runtime.verify_trace("Build", &TestSigner).unwrap();
    }

    #[test]
    fn declarations_without_executable_operations_fail_closed_in_standard_compilation() {
        assert!(matches!(
            crate::pipeline::compile_source(SOURCE),
            Err(crate::pipeline::PipelineError::CompilerContract(diagnostics))
                if diagnostics.iter().any(|diagnostic| diagnostic.contains("construct declarations without executable source operations"))
        ));
    }
    #[test]
    fn late_cell_failure_rolls_back_phase_state_and_history() {
        let mut runtime = SourceConstructRuntime::from_source(
            SOURCE,
            [("Lifecycle".to_string(), "Ready".to_string())],
            [("Counter".to_string(), 1)],
        ).unwrap();

        let result = runtime.transact_cell_phase(|phases, cells| {
            phases.transition_phase("Lifecycle", "Done").map_err(SourceRuntimeError::Phase)?;
            cells.update("Counter", -1).map_err(SourceRuntimeError::Cell)?;
            Ok(())
        });

        assert!(matches!(result, Err(SourceRuntimeError::Cell(CellRuntimeError::InvariantViolation(_)))));
        assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Ready");
        assert!(runtime.phases.phase_history("Lifecycle").unwrap().is_empty());
        assert_eq!(runtime.cells.get("Counter").unwrap(), 1);
    }

    #[test]
    fn failed_proof_rejects_before_cell_phase_transaction_starts() {
        let mut runtime = SourceConstructRuntime::from_source(
            SOURCE,
            [("Lifecycle".to_string(), "Ready".to_string())],
            [("Counter".to_string(), 1)],
        ).unwrap();
        let mut outcomes = BTreeMap::new();
        outcomes.insert(ProofRequirement::TypeSafety, ProofStatus::Failed);
        outcomes.insert(ProofRequirement::Bounds, ProofStatus::Proved);
        let invoked = FlagCell::new(false);

        let result = runtime.transact_proof_gated_cell_phase(
            "ReleaseGate",
            &outcomes,
            |phases, cells| {
                invoked.set(true);
                phases.transition_phase("Lifecycle", "Done").map_err(SourceRuntimeError::Phase)?;
                cells.update("Counter", 2).map_err(SourceRuntimeError::Cell)?;
                Ok(())
            },
        );

        assert!(matches!(result, Err(SourceRuntimeError::Proof(_))));
        assert!(!invoked.get(), "failed proof must reject before staged state changes");
        assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Ready");
        assert!(runtime.phases.phase_history("Lifecycle").unwrap().is_empty());
        assert_eq!(runtime.cells.get("Counter").unwrap(), 1);
    }

    #[test]
    fn successful_proof_commits_cell_and_phase_changes_together() {
        let mut runtime = SourceConstructRuntime::from_source(
            SOURCE,
            [("Lifecycle".to_string(), "Ready".to_string())],
            [("Counter".to_string(), 1)],
        ).unwrap();
        let mut outcomes = BTreeMap::new();
        outcomes.insert(ProofRequirement::TypeSafety, ProofStatus::Proved);
        outcomes.insert(ProofRequirement::Bounds, ProofStatus::RuntimeChecked);

        runtime.transact_proof_gated_cell_phase(
            "ReleaseGate",
            &outcomes,
            |phases, cells| {
                phases.transition_phase("Lifecycle", "Done").map_err(SourceRuntimeError::Phase)?;
                cells.update("Counter", 2).map_err(SourceRuntimeError::Cell)?;
                Ok(())
            },
        ).unwrap();

        assert_eq!(runtime.phases.phase_current("Lifecycle").unwrap(), "Done");
        assert_eq!(runtime.phases.phase_history("Lifecycle").unwrap().len(), 1);
        assert_eq!(runtime.cells.get("Counter").unwrap(), 2);
    }

}
