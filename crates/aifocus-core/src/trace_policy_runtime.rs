//! Source-to-runtime bridge for required, signed Trace receipts.
//!
//! Only `signature: required` is supported. Signing and verification are
//! delegated to caller-provided trusted implementations; evidence promotion also requires
//! HTTPS source backing and verifier approval. This does not enable
//! native Trace instructions or claim that a caller-supplied signer is trusted.
use std::collections::BTreeMap;

use crate::{
    ai_mode::{TraceError, TraceEventInput, TraceLedger, TracePolicy, TraceSigner, TraceVerifier},
    ast::ConstructKind,
    evidence::{EvidenceEnvelope, EvidenceGraph, EvidenceSourceVerifier},
    typed_ir::{self, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceRuntimeError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    UnknownTrace(String),
    UnsupportedConfiguration { trace: String, detail: String },
    Trace(TraceError),
    EvidenceVerificationFailed(String),
}

#[derive(Clone)]
struct SourceTrace {
    policy: TracePolicy,
    ledger: TraceLedger,
}

#[derive(Clone, Default)]
pub struct TraceRuntimeSet {
    traces: BTreeMap<String, SourceTrace>,
}

impl TraceRuntimeSet {
    /// Lower source Trace declarations. Unknown signature policies fail closed.
    pub fn from_source(source: &str) -> Result<Self, TraceRuntimeError> {
        let module = crate::parse(source).map_err(TraceRuntimeError::Parse)?;
        let constructs = typed_ir::lower_constructs(&module).map_err(TraceRuntimeError::TypedIr)?;
        let mut traces = BTreeMap::new();
        for construct in constructs.iter().filter(|c| c.kind == ConstructKind::Trace) {
            let signature = construct.members.iter().find_map(|member| match member {
                TypedIrConstructMember::Clause { name, value, .. } if name == "signature" => Some(value.trim()),
                _ => None,
            });
            if signature != Some("required") {
                return Err(TraceRuntimeError::UnsupportedConfiguration {
                    trace: construct.name.clone(),
                    detail: "only signature: required is supported".into(),
                });
            }
            let policy = TracePolicy {
                required: true,
                include_ai_metadata: false,
                require_signature: true,
            };
            if traces.insert(construct.name.clone(), SourceTrace { policy, ledger: TraceLedger::new() }).is_some() {
                return Err(TraceRuntimeError::UnsupportedConfiguration {
                    trace: construct.name.clone(),
                    detail: "duplicate Trace name".into(),
                });
            }
        }
        Ok(Self { traces })
    }

    pub fn receipts(&self, trace: &str) -> Result<&[crate::ai_mode::TraceReceipt], TraceRuntimeError> {
        self.traces.get(trace).map(|entry| entry.ledger.receipts())
            .ok_or_else(|| TraceRuntimeError::UnknownTrace(trace.to_string()))
    }

    /// Record a bounded, canonical receipt. When source requires signatures,
    /// missing signers and signing failures prevent the append.
    pub fn record<S: TraceSigner>(
        &mut self,
        trace: &str,
        event: TraceEventInput,
        signer: Option<&S>,
    ) -> Result<&crate::ai_mode::TraceReceipt, TraceRuntimeError> {
        let entry = self.traces.get_mut(trace)
            .ok_or_else(|| TraceRuntimeError::UnknownTrace(trace.to_string()))?;
        entry.ledger.record(&entry.policy, event, signer).map_err(TraceRuntimeError::Trace)
    }

    /// Verify every receipt and predecessor link against a trusted verifier.
    pub fn verify<V: TraceVerifier>(&self, trace: &str, verifier: &V) -> Result<(), TraceRuntimeError> {
        let entry = self.traces.get(trace)
            .ok_or_else(|| TraceRuntimeError::UnknownTrace(trace.to_string()))?;
        entry.ledger.verify_chain(&entry.policy, verifier).map_err(TraceRuntimeError::Trace)
    }

    /// Append and verify on a candidate copy, committing only if trusted verification passes.
    /// A broken signer/verifier pair therefore cannot poison the live receipt chain.
    pub fn record_verified<S: TraceSigner, V: TraceVerifier>(
        &mut self,
        trace: &str,
        event: TraceEventInput,
        signer: &S,
        verifier: &V,
    ) -> Result<(), TraceRuntimeError> {
        let mut candidate = self.clone();
        candidate.record(trace, event, Some(signer))?;
        candidate.verify(trace, verifier)?;
        *self = candidate;
        Ok(())
    }

    /// Promote verified receipts into an evidence graph only after both signature
    /// verification and the configured source-backing verifier accept each envelope.
    pub fn verified_evidence_graph<V: TraceVerifier, E: EvidenceSourceVerifier>(
        &self,
        trace: &str,
        source_uri: &str,
        verifier: &V,
        evidence_verifier: &E,
    ) -> Result<EvidenceGraph, TraceRuntimeError> {
        if !source_uri.starts_with("https://") || source_uri.len() <= "https://".len() {
            return Err(TraceRuntimeError::EvidenceVerificationFailed("source URI must be HTTPS".into()));
        }
        self.verify(trace, verifier)?;
        let entry = self.traces.get(trace)
            .ok_or_else(|| TraceRuntimeError::UnknownTrace(trace.to_string()))?;
        let mut graph = EvidenceGraph::default();
        for receipt in entry.ledger.receipts() {
            let signature_hex = receipt.signature.iter().map(|b| format!("{b:02x}")).collect::<String>();
            let previous_hex = receipt.previous_signature.iter().map(|b| format!("{b:02x}")).collect::<String>();
            let envelope = EvidenceEnvelope {
                episode_id: format!("{trace}:{}", receipt.sequence),
                capability: receipt.event_kind.clone(),
                task_kind: receipt.actor.clone(),
                outcome: receipt.outcome.clone(),
                verification: "trace_signature_verified".into(),
                evidence: vec![
                    format!("source_uri={source_uri}"),
                    format!("source_sha256={}", receipt.subject_sha256),
                    format!("verification_receipt={signature_hex};previous={previous_hex}"),
                ],
            };
            evidence_verifier.verify_source_backing(&envelope)
                .map_err(TraceRuntimeError::EvidenceVerificationFailed)?;
            graph.add(envelope);
        }
        graph.validate_integrity().map_err(TraceRuntimeError::EvidenceVerificationFailed)?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    const SOURCE: &str = "module traces\ntrace Build\n  signature: required\n";

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
    fn source_policy_requires_a_signer_and_verifies_receipt_chain() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        assert!(matches!(
            runtime.record("Build", event(), None::<&TestSigner>),
            Err(TraceRuntimeError::Trace(TraceError::SigningFailed(_)))
        ));
        assert!(runtime.receipts("Build").unwrap().is_empty());
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();
        runtime.verify("Build", &TestSigner).unwrap();
        assert_eq!(runtime.receipts("Build").unwrap().len(), 1);
    }

    #[test]
    fn invalid_or_unsupported_trace_policy_fails_closed() {
        assert!(matches!(
            TraceRuntimeSet::from_source("module t\ntrace Build\n  signature: sha256\n"),
            Err(TraceRuntimeError::UnsupportedConfiguration { .. })
        ));
        let runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        assert!(matches!(
            runtime.verify("Missing", &TestSigner),
            Err(TraceRuntimeError::UnknownTrace(_))
        ));
    }

    struct BadSigner;
    impl TraceSigner for BadSigner {
        fn sign(&self, _payload: &[u8]) -> Result<Vec<u8>, String> {
            Ok(b"not-a-valid-signature".to_vec())
        }
    }

    struct EvidenceVerifier;
    impl crate::evidence::EvidenceSourceVerifier for EvidenceVerifier {
        fn verify_source_backing(&self, envelope: &crate::evidence::EvidenceEnvelope) -> Result<(), String> {
            if envelope.verification == "trace_signature_verified"
                && envelope.evidence.iter().any(|item| item.starts_with("source_uri=https://"))
                && envelope.evidence.iter().any(|item| item.starts_with("source_sha256=") && item.len() == "source_sha256=".len() + 64)
                && envelope.evidence.iter().any(|item| item.starts_with("verification_receipt=") && item.len() > "verification_receipt=".len()) {
                Ok(())
            } else {
                Err("trace receipt lacks verified source backing".into())
            }
        }
    }

    #[test]
    fn verified_receipts_enter_evidence_graph_only_after_both_verifiers_pass() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();
        let graph = runtime.verified_evidence_graph("Build", "https://example.invalid/source", &TestSigner, &EvidenceVerifier).unwrap();
        assert_eq!(graph.envelopes.len(), 1);
        assert_eq!(graph.envelopes[0].episode_id, "Build:0");
        assert_eq!(graph.envelopes[0].verification, "trace_signature_verified");
        assert!(graph.validate_integrity().is_ok());
    }

    struct RejectingEvidenceVerifier;
    impl crate::evidence::EvidenceSourceVerifier for RejectingEvidenceVerifier {
        fn verify_source_backing(&self, _envelope: &crate::evidence::EvidenceEnvelope) -> Result<(), String> {
            Err("source registry unavailable".into())
        }
    }

    #[test]
    fn evidence_verifier_failure_prevents_evidence_graph_promotion() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();
        assert!(matches!(
            runtime.verified_evidence_graph("Build", "https://example.invalid/source", &TestSigner, &RejectingEvidenceVerifier),
            Err(TraceRuntimeError::EvidenceVerificationFailed(message)) if message == "source registry unavailable"
        ));
    }

    #[test]
    fn evidence_promotion_rejects_non_https_source_uris() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();
        assert!(matches!(
            runtime.verified_evidence_graph("Build", "file:///tmp/source", &TestSigner, &EvidenceVerifier),
            Err(TraceRuntimeError::EvidenceVerificationFailed(message)) if message == "source URI must be HTTPS"
        ));
    }

    #[test]
    fn required_trace_rejects_an_empty_ledger() {
        let runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        assert!(matches!(
            runtime.verify("Build", &TestSigner),
            Err(TraceRuntimeError::Trace(TraceError::MissingRequiredTrace))
        ));
    }

    #[test]
    fn invalid_signature_fails_verification() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&BadSigner)).unwrap();
        assert!(matches!(
            runtime.verify("Build", &TestSigner),
            Err(TraceRuntimeError::Trace(TraceError::InvalidSignature))
        ));
    }

    #[test]
    fn trace_policy_rejects_provider_and_session_metadata_without_appending() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();

        let mut provider_metadata = event();
        provider_metadata.provider_id = Some("provider-x".into());
        assert!(matches!(
            runtime.record("Build", provider_metadata, Some(&TestSigner)),
            Err(TraceRuntimeError::Trace(TraceError::MetadataNotAllowed))
        ));
        assert!(runtime.receipts("Build").unwrap().is_empty());

        let mut session_metadata = event();
        session_metadata.session_sha256 = Some("b".repeat(64));
        assert!(matches!(
            runtime.record("Build", session_metadata, Some(&TestSigner)),
            Err(TraceRuntimeError::Trace(TraceError::MetadataNotAllowed))
        ));
        assert!(runtime.receipts("Build").unwrap().is_empty());
    }

    #[test]
    fn accepted_trace_receipt_contains_only_explicitly_allowed_fields() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();

        let receipt = &runtime.receipts("Build").unwrap()[0];
        assert_eq!(receipt.event_kind, "build");
        assert_eq!(receipt.actor, "compiler");
        assert_eq!(receipt.outcome, "passed");
        assert_eq!(receipt.subject_sha256, "a".repeat(64));
        assert_eq!(receipt.provider_id, None);
        assert_eq!(receipt.session_sha256, None);
        assert!(!format!("{receipt:?}").to_ascii_lowercase().contains("prompt"));
    }

    #[test]
    fn evidence_graph_exports_only_redacted_trace_provenance_fields() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();

        let graph = runtime.verified_evidence_graph(
            "Build",
            "https://example.invalid/source",
            &TestSigner,
            &EvidenceVerifier,
        ).unwrap();

        let evidence = &graph.envelopes[0].evidence;
        assert_eq!(evidence.len(), 3);
        assert!(evidence.iter().any(|item| item == "source_uri=https://example.invalid/source"));
        assert!(evidence.iter().any(|item| item == &format!("source_sha256={}", "a".repeat(64))));
        assert!(evidence.iter().any(|item| item.starts_with("verification_receipt=")));
        assert!(evidence.iter().all(|item| {
            item.starts_with("source_uri=")
                || item.starts_with("source_sha256=")
                || item.starts_with("verification_receipt=")
        }));
    }

    #[test]
    fn malformed_trace_events_are_rejected_without_receipt_mutation() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();

        let mut invalid_token = event();
        invalid_token.actor = "user supplied raw prompt".into();
        assert!(matches!(
            runtime.record("Build", invalid_token, Some(&TestSigner)),
            Err(TraceRuntimeError::Trace(TraceError::InvalidToken("actor")))
        ));
        assert!(runtime.receipts("Build").unwrap().is_empty());

        let mut invalid_digest = event();
        invalid_digest.subject_sha256 = "not-a-digest".into();
        assert!(matches!(
            runtime.record("Build", invalid_digest, Some(&TestSigner)),
            Err(TraceRuntimeError::Trace(TraceError::InvalidDigest("subject_sha256")))
        ));
        assert!(runtime.receipts("Build").unwrap().is_empty());
    }
}
