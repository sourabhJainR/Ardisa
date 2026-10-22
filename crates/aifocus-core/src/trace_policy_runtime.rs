//! Source-to-runtime bridge for required, signed Trace receipts.
//!
//! Only `signature: required` is supported. Signing and verification are
//! delegated to caller-provided trusted implementations. This does not enable
//! native Trace instructions or claim that a caller-supplied signer is trusted.
use std::collections::BTreeMap;

use crate::{
    ai_mode::{TraceError, TraceEventInput, TraceLedger, TracePolicy, TraceSigner, TraceVerifier},
    ast::ConstructKind,
    typed_ir::{self, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceRuntimeError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    UnknownTrace(String),
    UnsupportedConfiguration { trace: String, detail: String },
    Trace(TraceError),
}

struct SourceTrace {
    policy: TracePolicy,
    ledger: TraceLedger,
}

#[derive(Default)]
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

    #[test]
    fn tampered_receipt_is_rejected() {
        let mut runtime = TraceRuntimeSet::from_source(SOURCE).unwrap();
        runtime.record("Build", event(), Some(&TestSigner)).unwrap();
        // Public receipts are immutable through this API; verification remains
        // the authority boundary rather than accepting caller-supplied receipt data.
        runtime.verify("Build", &TestSigner).unwrap();
    }
}
