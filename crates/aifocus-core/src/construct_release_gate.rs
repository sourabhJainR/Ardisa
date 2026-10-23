//! Cross-construct release acceptance matrix.
//!
//! These tests exercise each currently supported source-to-native adapter and
//! retain the normal compiler's AIF610 fail-closed boundary for unintegrated syntax.
#[cfg(test)]
mod tests {
    use crate::{
        cell_native::{compile_cell_source, execute_cell_artifact},
        native::NativeError,
        phase_native::{compile_phase_source, execute_phase_artifact},
        pipeline::{compile_source, PipelineError},
        proof_native::{compile_proof_source, execute_proof_artifact},
        trace_native::{compile_trace_source, execute_trace_event},
        trace_policy_runtime::TraceRuntimeSet,
        vault_native::{compile_vault_source, execute_vault_authorization, execute_vault_operation},
        ai_mode::{TraceEventInput, TraceSigner, TraceVerifier},
    };

    #[test]
    fn phase_release_gate_executes_declared_edges_and_rejects_undeclared_edges() {
        let source = "module release\nphase Job\n  todo -> running\n  running -> done\n";
        let artifact = compile_phase_source(source, "Job", "todo", &["running".into(), "done".into()]).unwrap();
        assert_eq!(execute_phase_artifact(&artifact).unwrap(), "done");
        let invalid = compile_phase_source(source, "Job", "todo", &["unknown".into()]).unwrap();
        assert!(execute_phase_artifact(&invalid).is_err());
    }

    #[test]
    fn cell_release_gate_enforces_invariants_on_initial_state_and_updates() {
        let source = "module release\ncell Counter\n  value: Int\n  invariant: value >= 0\n";
        let valid = compile_cell_source(source, "Counter", 1, &[2, 3]).unwrap();
        assert_eq!(execute_cell_artifact(&valid).unwrap(), 3);
        let invalid = compile_cell_source(source, "Counter", 1, &[2, -1]).unwrap();
        assert!(matches!(execute_cell_artifact(&invalid), Err(crate::cell_native::NativeCellCompileError::Native(NativeError::Type(_)))));
        assert!(compile_cell_source(source, "Counter", -1, &[]).is_err());
    }

    #[test]
    fn vault_release_gate_checks_capability_before_host_effect() {
        let source = "module release\nvault ReadOnly\n  capabilities: filesystem_read clock\n  denies: network\n";
        let allowed = compile_vault_source(source, "ReadOnly", "filesystem_read").unwrap();
        execute_vault_authorization(&allowed).unwrap();
        let mut invoked = false;
        assert!(execute_vault_operation(source, "ReadOnly", "network", || {
            invoked = true;
        }).is_err());
        assert!(!invoked);
    }

    #[test]
    fn proof_release_gate_rejects_missing_failed_and_unknown_outcomes() {
        let source = "module release\nproof ReleaseGate\n  requires: type_safety, bounds\n  ensures: true\n  on_unknown: reject\n  required: true\n";
        let artifact = compile_proof_source(source, "ReleaseGate").unwrap();
        execute_proof_artifact(&artifact, &["type_safety=proved".into(), "bounds=runtime_checked".into()]).unwrap();
        for outcomes in [
            vec!["type_safety=proved".into()],
            vec!["type_safety=failed".into(), "bounds=proved".into()],
            vec!["type_safety=unknown".into(), "bounds=proved".into()],
        ] {
            assert!(execute_proof_artifact(&artifact, &outcomes).is_err());
        }
    }

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
    fn trace_release_gate_requires_signed_receipts_and_commits_only_verified_events() {
        let source = "module release\ntrace Build\n  signature: required\n";
        let artifact = compile_trace_source(source, "Build").unwrap();
        let mut runtime = TraceRuntimeSet::from_source(source).unwrap();
        let event = TraceEventInput {
            event_kind: "build".into(),
            actor: "release-gate".into(),
            subject_sha256: "b".repeat(64),
            outcome: "passed".into(),
            provider_id: None,
            session_sha256: None,
        };
        execute_trace_event(&artifact, &mut runtime, event, &TestSigner, &TestSigner).unwrap();
        runtime.verify("Build", &TestSigner).unwrap();
        assert_eq!(runtime.receipts("Build").unwrap().len(), 1);
    }

    #[test]
    fn general_compiler_stays_fail_closed_until_constructs_are_integrated_into_its_pipeline() {
        let source = "module release\ntrace Build\n  signature: required\nfn main() -> Int\n  1\n";
        let result = compile_source(source);
        assert!(matches!(result, Err(PipelineError::Semantic(errors)) if errors.iter().any(|error| error.code == "AIF610")));
    }
}
