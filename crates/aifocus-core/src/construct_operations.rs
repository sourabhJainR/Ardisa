//! Typed semantic contract for operations that bind functions to AI-native constructs.
//!
//! This is an intermediate validation layer, not parser integration or executable
//! semantics. Source syntax must lower into these operations before the compiler can
//! safely enable construct-bearing modules. The regular compiler remains fail-closed.

use std::collections::BTreeSet;

use crate::{ast::{ConstructKind, Item, Module, TypeKind}, typed_ir::{TypedIrConstruct, TypedIrConstructMember}};

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
                // Expression IDs are opaque typed-IR references. Their validity is
                // established by the expression-lowering stage, not by a sentinel value.
                let _ = expression;
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
                                Some(parse_declared_capabilities(value)),
                            _ => None,
                        });
                        let capability_is_declared = declared
                            .and_then(|result| result.ok())
                            .is_some_and(|set| set.contains(capability.trim()));
                        if capability.trim().is_empty() || !capability_is_declared {
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


/// Parse declared capabilities without guessing intent. Commas and whitespace
/// separate names, but empty comma-delimited entries fail closed.
fn parse_declared_capabilities(value: &str) -> Result<BTreeSet<String>, String> {
    let comma_parts: Vec<&str> = value.split(',').collect();
    if comma_parts.iter().any(|part| part.trim().is_empty()) {
        return Err("Vault capabilities contain an empty list entry".into());
    }
    let capabilities: BTreeSet<String> = comma_parts
        .iter()
        .flat_map(|part| part.split_whitespace())
        .map(str::to_owned)
        .collect();
    if capabilities.is_empty() {
        return Err("Vault capabilities must not be empty".into());
    }
    Ok(capabilities)
}

/// Validate operations against a complete parsed module.
///
/// Unlike `validate_operations`, this entry point also proves that every operation
/// names a function declared in the same module. It remains a semantic contract:
/// it does not execute operations or enable construct-bearing compilation.
pub fn validate_module_operations(
    module: &Module,
    operations: &[ConstructOperation],
) -> Result<ValidatedConstructOperations, Vec<String>> {
    let constructs = crate::typed_ir::lower_constructs(module)?;
    let validated = validate_operations(&constructs, operations)?;
    let function_map: std::collections::BTreeMap<&str, &crate::ast::Function> = module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Function(function) => Some((function.name.as_str(), function)),
        })
        .collect();
    let mut errors = Vec::new();

    // Analyze the function bodies without construct declarations: ordinary semantic
    // analysis intentionally emits AIF610 for construct-bearing modules. This still
    // gives us authoritative inferred expression types without weakening that gate.
    let mut function_only_module = Module {
        constructs: Vec::new(),
        ..module.clone()
    };
    // The source-operation extractor owns these reserved call forms. They are not
    // ordinary function calls, so erase their dispatch nodes only in this private
    // semantic-analysis clone. Keep Cell value expressions in the tree so inferred
    // types remain available; ordinary compile_source still sees the untouched AST
    // and retains AIF610.
    for item in &mut function_only_module.items {
        let Item::Function(function) = item;
        sanitize_operation_calls_in_block(&mut function.body);
    }
    let semantic = crate::sema::analyze(&function_only_module).map_err(|errors| {
        errors.into_iter()
            .map(|error| format!("function semantic validation {}: {}", error.code, error.message))
            .collect::<Vec<_>>()
    })?;

    for (index, operation) in validated.operations.iter().enumerate() {
        let Some(function) = function_map.get(operation.function.as_str()) else {
            errors.push(format!(
                "operation {index}: unresolved function '{}'",
                operation.function
            ));
            continue;
        };

        if let ConstructOperationKind::CellInitialize { expression, value_type, .. }
            | ConstructOperationKind::CellUpdate { expression, value_type, .. } = &operation.kind
        {
            let mut expression_ids = BTreeSet::new();
            collect_block_expression_ids(&function.body, &mut expression_ids);
            if !expression_ids.contains(expression) {
                errors.push(format!(
                    "operation {index}: expression reference {expression} does not belong to function '{}'",
                    operation.function
                ));
                continue;
            }
            let Some(inferred) = semantic.inferred_types.get(&crate::ast::NodeId(*expression)) else {
                errors.push(format!(
                    "operation {index}: expression reference {expression} has no inferred type"
                ));
                continue;
            };
            if &inferred.kind != value_type {
                errors.push(format!(
                    "operation {index}: expression reference {expression} has type {:?}, expected {:?}",
                    inferred.kind, value_type
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(validated)
    } else {
        Err(errors)
    }
}


/// Replace reserved source-operation calls only in the validator's cloned AST.
/// Cell operation values remain as child expressions so semantic type inference can
/// validate the value against the declared Cell field. All other operation calls
/// become inert integer placeholders; this function never executes an operation.
fn sanitize_operation_calls_in_block(block: &mut crate::ast::Block) {
    use crate::ast::{Expr, ExprKind, StmtKind};

    fn visit_expr(expr: &mut Expr) {
        let operation = match &expr.kind {
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Name(name)
                    if matches!(
                        name.as_str(),
                        "cell_initialize" | "cell_update" | "phase_transition"
                            | "vault_authorize" | "proof_evaluate" | "trace_emit"
                    ) => Some((name.clone(), args.len(), args.get(2).cloned())),
                _ => None,
            },
            _ => None,
        };
        if let Some((name, arity, value)) = operation {
            if matches!(name.as_str(), "cell_initialize" | "cell_update") && arity == 3 {
                let mut value = value.expect("third Cell operation argument exists at arity three");
                visit_expr(&mut value);
                expr.kind = ExprKind::Group(Box::new(value));
            } else {
                expr.kind = ExprKind::Int(0);
            }
            return;
        }

        match &mut expr.kind {
            ExprKind::Binary { left, right, .. } => {
                visit_expr(left);
                visit_expr(right);
            }
            ExprKind::Call { callee, args } => {
                visit_expr(callee);
                for arg in args { visit_expr(arg); }
            }
            ExprKind::Group(inner) => visit_expr(inner),
            ExprKind::List(items) => for item in items { visit_expr(item); },
            ExprKind::Index { collection, index } => {
                visit_expr(collection);
                visit_expr(index);
            }
            ExprKind::Slice { collection, start, end } => {
                visit_expr(collection);
                visit_expr(start);
                visit_expr(end);
            }
            ExprKind::If { condition, then_branch, else_branch } => {
                visit_expr(condition);
                sanitize_operation_calls_in_block(then_branch);
                if let Some(branch) = else_branch {
                    sanitize_operation_calls_in_block(branch);
                }
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)
            | ExprKind::String(_) | ExprKind::Name(_) => {}
        }
    }

    for stmt in &mut block.stmts {
        match &mut stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Set { value, .. }
            | StmtKind::Return(Some(value)) | StmtKind::Expr(value) => visit_expr(value),
            StmtKind::SetIndex { collection, index, value } => {
                visit_expr(collection);
                visit_expr(index);
                visit_expr(value);
            }
            StmtKind::Scope { body } => sanitize_operation_calls_in_block(body),
            StmtKind::Spawn { call, .. } => visit_expr(call),
            StmtKind::While { condition, body } => {
                visit_expr(condition);
                sanitize_operation_calls_in_block(body);
            }
            StmtKind::For { iterable, body, .. } => {
                visit_expr(iterable);
                sanitize_operation_calls_in_block(body);
            }
            StmtKind::Return(None) | StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
        }
    }
}

/// Collect expression node IDs from a function body. These IDs are opaque outside
/// the AST, so a module-bound operation must prove that its reference belongs to
/// the function it names before later typed lowering can resolve its type/value.
fn collect_block_expression_ids(
    block: &crate::ast::Block,
    ids: &mut BTreeSet<u64>,
) {
    use crate::ast::{Expr, ExprKind, StmtKind};

    fn visit_expr(expr: &Expr, ids: &mut BTreeSet<u64>) {
        ids.insert(expr.id.0);
        match &expr.kind {
            ExprKind::Binary { left, right, .. } => {
                visit_expr(left, ids);
                visit_expr(right, ids);
            }
            ExprKind::Call { callee, args } => {
                visit_expr(callee, ids);
                for arg in args {
                    visit_expr(arg, ids);
                }
            }
            ExprKind::Group(inner) => visit_expr(inner, ids),
            ExprKind::List(elements) => {
                for element in elements {
                    visit_expr(element, ids);
                }
            }
            ExprKind::Index { collection, index } => {
                visit_expr(collection, ids);
                visit_expr(index, ids);
            }
            ExprKind::Slice { collection, start, end } => {
                visit_expr(collection, ids);
                visit_expr(start, ids);
                visit_expr(end, ids);
            }
            ExprKind::If { condition, then_branch, else_branch } => {
                visit_expr(condition, ids);
                collect_block_expression_ids(then_branch, ids);
                if let Some(branch) = else_branch {
                    collect_block_expression_ids(branch, ids);
                }
            }
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)
            | ExprKind::String(_) | ExprKind::Name(_) => {}
        }
    }

    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Set { value, .. }
            | StmtKind::Expr(value) => visit_expr(value, ids),
            StmtKind::SetIndex { collection, index, value } => {
                visit_expr(collection, ids);
                visit_expr(index, ids);
                visit_expr(value, ids);
            }
            StmtKind::Return(value) => {
                if let Some(value) = value {
                    visit_expr(value, ids);
                }
            }
            StmtKind::Scope { body } => collect_block_expression_ids(body, ids),
            StmtKind::Spawn { call, .. } => visit_expr(call, ids),
            StmtKind::While { condition, body } => {
                visit_expr(condition, ids);
                collect_block_expression_ids(body, ids);
            }
            StmtKind::For { iterable, body, .. } => {
                visit_expr(iterable, ids);
                collect_block_expression_ids(body, ids);
            }
            StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
        }
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
    fn module_validation_requires_each_operation_to_bind_a_declared_function() {
        let source = "module operation_binding\nfn main() -> Int\n  return 1\ncell State\n  value: Int\n  invariant: value >= 0\n";
        let module = parse(source).unwrap();
        let operation = |function: &str, expression| ConstructOperation {
            function: function.into(),
            kind: ConstructOperationKind::CellInitialize {
                cell: "State".into(),
                field: "value".into(),
                expression,
                value_type: TypeKind::Int,
            },
        };

        let expression_id = match &module.items[0] {
            Item::Function(function) => match &function.body.stmts[0].kind {
                crate::ast::StmtKind::Return(Some(expression)) => expression.id.0,
                other => panic!("expected return expression, got {other:?}"),
            },
        };
        assert!(validate_module_operations(&module, &[operation("main", expression_id)]).is_ok());

        let errors = validate_module_operations(&module, &[operation("main", u64::MAX)]).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("does not belong to function 'main'")));

        let errors = validate_module_operations(&module, &[operation("missing", expression_id)]).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("unresolved function 'missing'")));
    }

    #[test]
    fn module_validation_rejects_expression_type_mismatch_even_when_field_type_matches() {
        let source = "module operation_type_binding\ncell State\n  value: Int\n  invariant: value >= 0\nfn main() -> Int\n  let flag = true\n  return 1\n";
        let module = parse(source).unwrap();
        let expression_id = match &module.items[0] {
            Item::Function(function) => match &function.body.stmts[0].kind {
                crate::ast::StmtKind::Let { value, .. } => value.id.0,
                other => panic!("expected let expression, got {other:?}"),
            },
        };
        let operation = ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::CellInitialize {
                cell: "State".into(),
                field: "value".into(),
                expression: expression_id,
                value_type: TypeKind::Int,
            },
        };
        let errors = validate_module_operations(&module, &[operation]).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("has type Bool, expected Int")));
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
    fn vault_capabilities_accept_comma_or_whitespace_separators_and_reject_empty_entries() {
        let whitespace = declarations("module vault_spaces\nvault Files\n  capabilities: filesystem_read clock\n");
        let comma = declarations("module vault_commas\nvault Files\n  capabilities: filesystem_read, clock\n");
        let operation = |capability: &str| ConstructOperation {
            function: "main".into(),
            kind: ConstructOperationKind::VaultAuthorize {
                vault: "Files".into(),
                capability: capability.into(),
            },
        };
        assert!(validate_operations(&whitespace, &[operation("filesystem_read"), operation("clock")]).is_ok());
        assert!(validate_operations(&comma, &[operation("filesystem_read"), operation("clock")]).is_ok());

        let malformed = declarations("module vault_bad_list\nvault Files\n  capabilities: filesystem_read,,clock\n");
        assert!(validate_operations(&malformed, &[operation("filesystem_read")]).is_err());
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
