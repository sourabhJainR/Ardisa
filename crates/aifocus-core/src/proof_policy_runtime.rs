//! Source-to-runtime bridge for explicit Proof requirements.
//!
//! This bridge accepts only named proof requirements and the conservative
//! `on_unknown: reject` policy. It validates caller-supplied proof outcomes;
//! it does not prove claims or enable native Proof instructions.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ai_mode::{PolicyViolation, ProofRequirement, ProofStatus},
    ast::ConstructKind,
    typed_ir::{self, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofRuntimeError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    UnknownProof(String),
    UnsupportedConfiguration { proof: String, detail: String },
    Rejected(Vec<PolicyViolation>),
}

fn source_requirement(name: &str) -> Option<ProofRequirement> {
    match name {
        "type_safety" => Some(ProofRequirement::TypeSafety),
        "bounds" => Some(ProofRequirement::Bounds),
        "effect_policy" => Some(ProofRequirement::EffectPolicy),
        "contract_postconditions" => Some(ProofRequirement::ContractPostconditions),
        "constant_time" => Some(ProofRequirement::ConstantTime),
        _ => None,
    }
}

fn parse_requirements(proof: &str, value: &str) -> Result<BTreeSet<ProofRequirement>, ProofRuntimeError> {
    let mut requirements = BTreeSet::new();
    for token in value.split(|ch: char| ch.is_whitespace() || ch == ',').filter(|s| !s.is_empty()) {
        let requirement = source_requirement(token).ok_or_else(|| ProofRuntimeError::UnsupportedConfiguration {
            proof: proof.to_string(),
            detail: format!("unknown required proof '{token}'"),
        })?;
        if !requirements.insert(requirement) {
            return Err(ProofRuntimeError::UnsupportedConfiguration {
                proof: proof.to_string(),
                detail: format!("duplicate required proof '{token}'"),
            });
        }
    }
    if requirements.is_empty() {
        return Err(ProofRuntimeError::UnsupportedConfiguration {
            proof: proof.to_string(),
            detail: "requires must name at least one supported proof requirement".into(),
        });
    }
    Ok(requirements)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceProofPolicy {
    required: BTreeSet<ProofRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProofRuntimeSet {
    policies: BTreeMap<String, SourceProofPolicy>,
}

impl ProofRuntimeSet {
    /// Parse Proof declarations. Optional proof policies and non-trivial
    /// postconditions fail closed until their runtime semantics are specified.
    pub fn from_source(source: &str) -> Result<Self, ProofRuntimeError> {
        let module = crate::parse(source).map_err(ProofRuntimeError::Parse)?;
        let constructs = typed_ir::lower_constructs(&module).map_err(ProofRuntimeError::TypedIr)?;
        let mut policies = BTreeMap::new();
        for construct in constructs.iter().filter(|c| c.kind == ConstructKind::Proof) {
            let mut requires = None;
            let mut ensures = None;
            let mut on_unknown = None;
            let mut required = None;
            for member in &construct.members {
                let TypedIrConstructMember::Clause { name, value, .. } = member else { continue };
                match name.as_str() {
                    "requires" => requires = Some(value.as_str()),
                    "ensures" => ensures = Some(value.trim()),
                    "on_unknown" => on_unknown = Some(value.trim()),
                    "required" => required = Some(value.trim()),
                    _ => {}
                }
            }
            if required != Some("true") {
                return Err(ProofRuntimeError::UnsupportedConfiguration {
                    proof: construct.name.clone(),
                    detail: "required must be true for this runtime bridge".into(),
                });
            }
            if on_unknown != Some("reject") {
                return Err(ProofRuntimeError::UnsupportedConfiguration {
                    proof: construct.name.clone(),
                    detail: "only on_unknown: reject is supported".into(),
                });
            }
            if ensures != Some("true") {
                return Err(ProofRuntimeError::UnsupportedConfiguration {
                    proof: construct.name.clone(),
                    detail: "only ensures: true is supported; postcondition evaluation is not implemented".into(),
                });
            }
            let required = parse_requirements(
                &construct.name,
                requires.ok_or_else(|| ProofRuntimeError::UnsupportedConfiguration {
                    proof: construct.name.clone(),
                    detail: "missing requires clause".into(),
                })?,
            )?;
            if policies.insert(construct.name.clone(), SourceProofPolicy { required }).is_some() {
                return Err(ProofRuntimeError::UnsupportedConfiguration {
                    proof: construct.name.clone(),
                    detail: "duplicate Proof name".into(),
                });
            }
        }
        Ok(Self { policies })
    }

    pub fn requirements(&self, proof: &str) -> Result<&BTreeSet<ProofRequirement>, ProofRuntimeError> {
        self.policies.get(proof).map(|policy| &policy.required)
            .ok_or_else(|| ProofRuntimeError::UnknownProof(proof.to_string()))
    }

    /// Accept only complete required outcomes. Missing and failed results always
    /// reject; unknown results reject under the only supported source policy.
    pub fn validate_results(
        &self,
        proof: &str,
        results: &BTreeMap<ProofRequirement, ProofStatus>,
    ) -> Result<(), ProofRuntimeError> {
        let required = self.requirements(proof)?;
        let mut violations = Vec::new();
        for requirement in required {
            match results.get(requirement) {
                None => violations.push(PolicyViolation::MissingProofResult(*requirement)),
                Some(ProofStatus::Failed) => violations.push(PolicyViolation::FailedProof(*requirement)),
                Some(ProofStatus::Unknown) => violations.push(PolicyViolation::UnknownProof(*requirement)),
                Some(ProofStatus::Proved | ProofStatus::RuntimeChecked) => {}
            }
        }
        if violations.is_empty() { Ok(()) } else { Err(ProofRuntimeError::Rejected(violations)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module proofs\nproof ReleaseGate\n  requires: type_safety, bounds\n  ensures: true\n  on_unknown: reject\n  required: true\n";

    #[test]
    fn source_policy_requires_all_named_outcomes() {
        let runtime = ProofRuntimeSet::from_source(SOURCE).unwrap();
        let mut results = BTreeMap::new();
        results.insert(ProofRequirement::TypeSafety, ProofStatus::Proved);
        results.insert(ProofRequirement::Bounds, ProofStatus::RuntimeChecked);
        assert_eq!(runtime.validate_results("ReleaseGate", &results), Ok(()));
    }

    #[test]
    fn missing_failed_and_unknown_results_reject() {
        let runtime = ProofRuntimeSet::from_source(SOURCE).unwrap();
        assert!(matches!(
            runtime.validate_results("ReleaseGate", &BTreeMap::new()),
            Err(ProofRuntimeError::Rejected(v)) if v.len() == 2
        ));
        let mut failed = BTreeMap::new();
        failed.insert(ProofRequirement::TypeSafety, ProofStatus::Failed);
        failed.insert(ProofRequirement::Bounds, ProofStatus::Proved);
        assert!(matches!(
            runtime.validate_results("ReleaseGate", &failed),
            Err(ProofRuntimeError::Rejected(v)) if v == vec![PolicyViolation::FailedProof(ProofRequirement::TypeSafety)]
        ));
        failed.insert(ProofRequirement::TypeSafety, ProofStatus::Unknown);
        assert!(matches!(
            runtime.validate_results("ReleaseGate", &failed),
            Err(ProofRuntimeError::Rejected(v)) if v == vec![PolicyViolation::UnknownProof(ProofRequirement::TypeSafety)]
        ));
    }

    #[test]
    fn unknown_requirement_optional_policy_and_unimplemented_postconditions_fail_closed() {
        assert!(matches!(
            ProofRuntimeSet::from_source("module p\nproof Gate\n  requires: valid\n  ensures: true\n  on_unknown: reject\n  required: true\n"),
            Err(ProofRuntimeError::UnsupportedConfiguration { .. })
        ));
        assert!(matches!(
            ProofRuntimeSet::from_source("module p\nproof Gate\n  requires: bounds\n  ensures: true\n  on_unknown: reject\n  required: false\n"),
            Err(ProofRuntimeError::UnsupportedConfiguration { .. })
        ));
        assert!(matches!(
            ProofRuntimeSet::from_source("module p\nproof Gate\n  requires: bounds\n  ensures: result > 0\n  on_unknown: reject\n  required: true\n"),
            Err(ProofRuntimeError::UnsupportedConfiguration { .. })
        ));
    }
}
