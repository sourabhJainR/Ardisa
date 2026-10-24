//! Typed semantic contract for operations that bind functions to AI-native constructs.
//!
//! This is an intermediate validation layer, not parser integration or executable
//! semantics. Source syntax must lower into these operations before the compiler can
//! safely enable construct-bearing modules. The regular compiler remains fail-closed.

use std::collections::BTreeSet;

use crate::{ast::{ConstructKind, TypeKind}, typed_ir::{TypedIrConstruct, TypedIrConstructMember}};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructOperation {
    pub function: String,
    pub kind: ConstructOperationKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructOperationKind {
    /// Initialize a Cell field from an explicit typed expression in the function.
    CellInitialize { cell: String, field: String, expression: u64, value_type: TypeKind },
    /// Update a Cell field. Runtime integration must validate atomically before commit.
    CellUpdate { cell: String, field: String, expression: u64, value_type: TypeKind },
    /// Request a declared Phase edge. The runtime must check current state.
    PhaseTransition { phase: String, from: String, to: String },
    /// Request a capability before any protected effect is attempted.
    VaultAuthorize { vault: String, capability: String },
    /// Evaluate a named Proof policy. A result must be represented by runtime evidence.
    ProofEvaluate { proof: String },
    /// Prepare a Trace event. This does not itself establish trusted signing.
    TraceEmit { trace: String, event: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedConstructOperations {
    pub operations: Vec<ConstructOperation>,
}

/// Validate the explicit declaration references and required operation context.
/// Expression IDs are supplied by the typed expression lowering; no initial values,
/// capabilities, transitions, proof results, or trace signatures are inferred here.
pub fn validate_operations(
    constructs: &[TypedIrConstruct],
    operations: &[ConstructOperation],
) -> Result<ValidatedConstructOperations, Vec<String>> {
    let mut errors = Vec::new();
    if let Err(metadata_errors) = crate::typed_ir::validate_construct_metadata(constructs) {
        errors.extend(metadata_errors);
    }
    let names: std::collections::BTreeMap<&str, &TypedIrConstruct> =
        constructs.iter().map(|construct| (construct.name.as_str(), construct)).collect();

    for (index, operation) in operations.iter().enumerate() {
        if operation.function.trim().is_empty() {
            errors.push(format!("operation {index}: function name must not be empty"));
        }
        match &operation.kind {
            ConstructOperationKind::CellInitialize { cell, field, expression, value_type }
            | ConstructOperationKind::CellUpdate { cell, field, expression, value_type } => {
                if *expression == u64::MAX {
                    errors.push(format!("operation {index}: invalid expression reference"));
                }
                match names.get(cell.as_str()) {
                    None => errors.push(format!("operation {index}: unresolved Cell '{cell}'")),
                    Some(declaration) if declaration.kind != ConstructKind::Cell =>
                        errors.push(format!("operation {index}: '{cell}' is not a Cell")),
                    Some(declaration) => {
                        match declaration.members.iter().find(|member| matches!(member,
                            TypedIrConstructMember::Field { name, .. } if name == field)) {
                            None => errors.push(format!("operation {index}: unresolved Cell field '{cell}.{field}'")),
                            Some(TypedIrConstructMember::Field { ty, .. }) if ty != value_type =>
                                errors.push(format!("operation {index}: type mismatch for '{cell}.{field}'")),
                            _ => {}
                        }
                    }
                }
            }
            ConstructOperationKind::PhaseTransition { phase, from, to } => {
                match names.get(phase.as_str()) {
                    None => errors.push(format!("operation {index}: unresolved Phase '{phase}'")),
                    Some(declaration) if declaration.kind != ConstructKind::Phase =>
                        errors.push(format!("operation {index}: '{phase}' is not a Phase")),
                    Some(declaration) if !declaration.members.iter().any(|member| matches!(member,
                        TypedIrConstructMember::Transition { from: edge_from, to: edge_to, .. }
                        if edge_from == from && edge_to == to)) =>
                        errors.push(format!("operation {index}: undeclared Phase edge '{phase}: {from} -> {to}'")),
                    _ => {}
                }
            }
            ConstructOperationKind::VaultAuthorize { vault, capability } => {
                match names.get(vault.as_str()) {
                    None => errors.push(format!("operation {index}: unresolved Vault '{vault}'")),
                    Some(declaration) if declaration.kind != ConstructKind::Vault =>
                        errors.push(format!("operation {index}: '{vault}' is not a Vault")),
                    Some(declaration) => {
                        let declared = declaration.members.iter().find_map(|member| match member {
                            TypedIrConstructMember::Clause { name, value, .. } if name == "capabilities" =>
                                Some(value.split(',').map(str::trim).collect::<BTreeSet<_>>()),
                            _ => None,
                        });
                        if capability.trim().is_empty() || !declared.is_some_and(|set| set.contains(capability.trim())) {
                            errors.push(format!("operation {index}: capability '{capability}' is not explicitly declared by Vault '{vault}'"));
                        }
                    }
                }
            }
            ConstructOperationKind::ProofEvaluate { proof } => match names.get(proof.as_str()) {
                None => errors.push(format!("operation {index}: unresolved Proof '{proof}'")),
                Some(declaration) if declaration.kind != ConstructKind::Proof =>
                    errors.push(format!("operation {index}: '{proof}' is not a Proof")),
                _ => {}
            },
            ConstructOperationKind::TraceEmit { trace, event } => {
                match names.get(trace.as_str()) {
                    None => errors.push(format!("operation {index}: unresolved Trace '{trace}'")),
                    Some(declaration) if declaration.kind != ConstructKind::Trace =>
                        errors.push(format!("operation {index}: '{trace}' is not a Trace")),
                    _ => {}
                }
                if event.trim().is_empty() {
                    errors.push(format!("operation {index}: Trace event must not be empty"));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(ValidatedConstructOperations { operations: operations.to_vec() })
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn declarations(source: &str) -> Vec<TypedIrConstruct> {
        let module = parse(source).unwrap();
        crate::typed_ir::lower_constructs(&module).unwrap()
    }

    #[test]
    fn validates_explicit_cell_field_binding_and_rejects_type_mismatch() {
        let constructs = declarations("module cell_ops\ncell State\n  value: Int\n  invariant: value >= 0\n");
        let valid = ConstructOperation { function: "main".into(), kind: ConstructOperationKind::CellInitialize {
            cell: "State".into(), field: "value".into(), expression: 7, value_type: TypeKind::Int,
        }};
        assert!(validate_operations(&constructs, &[valid.clone()]).is_ok());
        let invalid = ConstructOperation { kind: ConstructOperationKind::CellUpdate {
            cell: "State".into(), field: "value".into(), expression: 8, value_type: TypeKind::Bool,
        }, ..valid };
        assert!(validate_operations(&constructs, &[invalid]).unwrap_err().iter().any(|e| e.contains("type mismatch")));
    }

    #[test]
    fn validates_phase_edges_and_vault_capabilities_without_inference() {
        let constructs = declarations("module phase_vault_ops\nphase Lifecycle\n  initial: Ready\n  Ready -> Running\nvault Files\n  capabilities: filesystem_read, filesystem_write\n");
        let good = vec![
            ConstructOperation { function: "main".into(), kind: ConstructOperationKind::PhaseTransition {
                phase: "Lifecycle".into(), from: "Ready".into(), to: "Running".into(),
            }},
            ConstructOperation { function: "main".into(), kind: ConstructOperationKind::VaultAuthorize {
                vault: "Files".into(), capability: "filesystem_read".into(),
            }},
        ];
        assert!(validate_operations(&constructs, &good).is_ok());
        let bad = ConstructOperation { function: "main".into(), kind: ConstructOperationKind::VaultAuthorize {
            vault: "Files".into(), capability: "admin".into(),
        }};
        assert!(validate_operations(&constructs, &[bad]).unwrap_err().iter().any(|e| e.contains("not explicitly declared")));
    }

    #[test]
    fn unresolved_and_wrong_kind_bindings_fail_closed() {
        let constructs = declarations("module trace_ops\ntrace Audit\n  signature: ed25519\n");
        let operation = ConstructOperation { function: "main".into(), kind: ConstructOperationKind::CellInitialize {
            cell: "Missing".into(), field: "value".into(), expression: 1, value_type: TypeKind::Int,
        }};
        assert!(validate_operations(&constructs, &[operation]).unwrap_err().iter().any(|e| e.contains("unresolved Cell")));
    }

    #[test]
    fn proof_and_trace_operations_require_resolved_declarations_and_event_names() {
        let constructs = declarations("module proof_trace_ops\nproof Safety\n  requires: bounds\n  ensures: true\n  on_unknown: reject\n  required: true\ntrace Audit\n  signature: ed25519\n");
        let operations = vec![
            ConstructOperation { function: "main".into(), kind: ConstructOperationKind::ProofEvaluate { proof: "Safety".into() }},
            ConstructOperation { function: "main".into(), kind: ConstructOperationKind::TraceEmit { trace: "Audit".into(), event: "cell.updated".into() }},
        ];
        assert!(validate_operations(&constructs, &operations).is_ok());
        let invalid = ConstructOperation { function: "main".into(), kind: ConstructOperationKind::TraceEmit { trace: "Audit".into(), event: " ".into() }};
        assert!(validate_operations(&constructs, &[invalid]).unwrap_err().iter().any(|e| e.contains("event must not be empty")));
    }
}
