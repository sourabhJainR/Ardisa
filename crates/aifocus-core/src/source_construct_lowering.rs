//! Lower source-level construct operation calls into typed operation bindings.
//!
//! This is the source-to-operation boundary for the compiler integration. It does
//! not enable native compilation: the standard pipeline retains AIF610 until these
//! bindings are also lowered into executable native instructions and artifact data.

use crate::ast::{Block, ConstructKind, ConstructMember, Expr, ExprKind, Function, Item, Module, Stmt, StmtKind, TypeKind};
use crate::construct_operations::{ConstructOperation, ConstructOperationKind};

/// Extract source operations in deterministic source order.
///
/// Reserved call forms:
/// - `cell_initialize("Cell", "field", value)`
/// - `cell_update("Cell", "field", value)`
/// - `phase_transition("Phase", "from", "to")`
/// - `vault_authorize("Vault", "capability")`
/// - `proof_evaluate("Proof")`
/// - `trace_emit("Trace", "event")`
///
/// Construct names, fields, and policy labels must be string literals so binding is
/// explicit and cannot be redirected by runtime data. Cell value types are taken
/// from the declared field; the operation validator later checks the expression's
/// inferred type against that declaration.
pub fn lower_source_operations(module: &Module) -> Result<Vec<ConstructOperation>, Vec<String>> {
    let mut operations = Vec::new();
    let mut errors = Vec::new();
    for item in &module.items {
        let Item::Function(function) = item;
        collect_block(module, function, &function.body, &mut operations, &mut errors);
    }
    if errors.is_empty() { Ok(operations) } else { Err(errors) }
}

fn collect_block(
    module: &Module,
    function: &Function,
    block: &Block,
    operations: &mut Vec<ConstructOperation>,
    errors: &mut Vec<String>,
) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Set { value, .. } |
            StmtKind::Return(Some(value)) | StmtKind::Expr(value) => {
                collect_expr(module, function, value, operations, errors);
            }
            StmtKind::Return(None) | StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
            StmtKind::SetIndex { collection, index, value } => {
                collect_expr(module, function, collection, operations, errors);
                collect_expr(module, function, index, operations, errors);
                collect_expr(module, function, value, operations, errors);
            }
            StmtKind::Scope { body } => collect_block(module, function, body, operations, errors),
            StmtKind::Spawn { call, .. } => collect_expr(module, function, call, operations, errors),
            StmtKind::While { condition, body } => {
                collect_expr(module, function, condition, operations, errors);
                collect_block(module, function, body, operations, errors);
            }
            StmtKind::For { iterable, body, .. } => {
                collect_expr(module, function, iterable, operations, errors);
                collect_block(module, function, body, operations, errors);
            }
        }
    }
}

fn literal_string<'a>(expr: &'a Expr, what: &str, errors: &mut Vec<String>) -> Option<&'a str> {
    match &expr.kind {
        ExprKind::String(value) if !value.trim().is_empty() => Some(value),
        _ => {
            errors.push(format!("{}: {what} must be a non-empty string literal", expr.span.start));
            None
        }
    }
}

fn cell_field_type(module: &Module, cell: &str, field: &str) -> Option<TypeKind> {
    module.constructs.iter().find(|declaration| {
        declaration.kind == ConstructKind::Cell && declaration.name == cell
    })?.members.iter().find_map(|member| match member {
        ConstructMember::Field { name, ty, .. } if name == field => Some(ty.kind.clone()),
        _ => None,
    })
}

fn collect_expr(
    module: &Module,
    function: &Function,
    expr: &Expr,
    operations: &mut Vec<ConstructOperation>,
    errors: &mut Vec<String>,
) {
    if let ExprKind::Call { callee, args } = &expr.kind {
        if let ExprKind::Name(name) = &callee.kind {
            let expected_arity = match name.as_str() {
                "cell_initialize" | "cell_update" => Some(3),
                "phase_transition" => Some(3),
                "vault_authorize" => Some(2),
                "proof_evaluate" => Some(1),
                "trace_emit" => Some(2),
                _ => None,
            };
            if let Some(expected) = expected_arity {
                if args.len() != expected {
                    errors.push(format!(
                        "function '{}': operation '{}' expects {expected} arguments, got {}",
                        function.name, name, args.len()
                    ));
                } else {
                    let before = errors.len();
                    let operation = match name.as_str() {
                        "cell_initialize" | "cell_update" => {
                            let cell = literal_string(&args[0], "Cell name", errors);
                            let field = literal_string(&args[1], "Cell field", errors);
                            match (cell, field) {
                                (Some(cell), Some(field)) => match cell_field_type(module, cell, field) {
                                    Some(value_type) => {
                                        let kind = if name == "cell_initialize" {
                                            ConstructOperationKind::CellInitialize {
                                                cell: cell.into(), field: field.into(),
                                                expression: args[2].id.0, value_type,
                                            }
                                        } else {
                                            ConstructOperationKind::CellUpdate {
                                                cell: cell.into(), field: field.into(),
                                                expression: args[2].id.0, value_type,
                                            }
                                        };
                                        Some(kind)
                                    }
                                    None => {
                                        errors.push(format!("function '{}': unresolved Cell field '{}.{}'", function.name, cell, field));
                                        None
                                    }
                                },
                                _ => None,
                            }
                        }
                        "phase_transition" => {
                            let phase = literal_string(&args[0], "Phase name", errors);
                            let from = literal_string(&args[1], "Phase source state", errors);
                            let to = literal_string(&args[2], "Phase target state", errors);
                            match (phase, from, to) {
                                (Some(phase), Some(from), Some(to)) => Some(ConstructOperationKind::PhaseTransition {
                                    phase: phase.into(), from: from.into(), to: to.into(),
                                }),
                                _ => None,
                            }
                        }
                        "vault_authorize" => {
                            let vault = literal_string(&args[0], "Vault name", errors);
                            let capability = literal_string(&args[1], "Vault capability", errors);
                            match (vault, capability) {
                                (Some(vault), Some(capability)) => Some(ConstructOperationKind::VaultAuthorize {
                                    vault: vault.into(), capability: capability.into(),
                                }),
                                _ => None,
                            }
                        }
                        "proof_evaluate" => literal_string(&args[0], "Proof name", errors)
                            .map(|proof| ConstructOperationKind::ProofEvaluate { proof: proof.into() }),
                        "trace_emit" => {
                            let trace = literal_string(&args[0], "Trace name", errors);
                            let event = literal_string(&args[1], "Trace event", errors);
                            match (trace, event) {
                                (Some(trace), Some(event)) => Some(ConstructOperationKind::TraceEmit {
                                    trace: trace.into(), event: event.into(),
                                }),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                    if errors.len() == before {
                        if let Some(kind) = operation {
                            operations.push(ConstructOperation { function: function.name.clone(), kind });
                        }
                    }
                }
                // Traverse the value expression for nested calls, but do not reinterpret
                // literal policy arguments as independent operations.
                for arg in args.iter().skip(if name == "cell_initialize" || name == "cell_update" { 2 } else { args.len() }) {
                    collect_expr(module, function, arg, operations, errors);
                }
                return;
            }
        }
    }
    match &expr.kind {
        ExprKind::Binary { left, right, .. } => {
            collect_expr(module, function, left, operations, errors);
            collect_expr(module, function, right, operations, errors);
        }
        ExprKind::Call { callee, args } => {
            collect_expr(module, function, callee, operations, errors);
            for arg in args { collect_expr(module, function, arg, operations, errors); }
        }
        ExprKind::Group(inner) => collect_expr(module, function, inner, operations, errors),
        ExprKind::List(items) => for item in items { collect_expr(module, function, item, operations, errors); },
        ExprKind::Index { collection, index } => {
            collect_expr(module, function, collection, operations, errors);
            collect_expr(module, function, index, operations, errors);
        }
        ExprKind::Slice { collection, start, end } => {
            collect_expr(module, function, collection, operations, errors);
            collect_expr(module, function, start, operations, errors);
            collect_expr(module, function, end, operations, errors);
        }
        ExprKind::If { condition, then_branch, else_branch } => {
            collect_expr(module, function, condition, operations, errors);
            collect_block(module, function, then_branch, operations, errors);
            if let Some(branch) = else_branch {
                collect_block(module, function, branch, operations, errors);
            }
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) |
        ExprKind::String(_) | ExprKind::Name(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn lowers_mixed_source_construct_calls_in_source_order() {
        let source = r#"module source_ops
phase Job
  initial: todo
  todo -> running
  running -> done
cell Counter
  value: Int
  invariant: value >= 0
vault Local
  capabilities: filesystem_read, clock
proof ReleaseGate
  requires: type_safety
  ensures: true
  on_unknown: reject
  required: true
trace Build
  signature: required
fn main() -> Int
  cell_initialize("Counter", "value", 1)
  phase_transition("Job", "todo", "running")
  vault_authorize("Local", "filesystem_read")
  proof_evaluate("ReleaseGate")
  trace_emit("Build", "compile")
  cell_update("Counter", "value", 2)
  return 0
"#;
        let module = parse(source).unwrap();
        let operations = lower_source_operations(&module).unwrap();
        assert_eq!(operations.len(), 6);
        assert_eq!(operations[0].function, "main");
        assert!(matches!(operations[0].kind, ConstructOperationKind::CellInitialize {
            ref cell, ref field, expression, value_type: TypeKind::Int
        } if cell == "Counter" && field == "value" && expression != 0));
        assert!(matches!(operations[1].kind, ConstructOperationKind::PhaseTransition { ref phase, ref from, ref to }
            if phase == "Job" && from == "todo" && to == "running"));
        assert!(matches!(operations[2].kind, ConstructOperationKind::VaultAuthorize { ref vault, ref capability }
            if vault == "Local" && capability == "filesystem_read"));
        assert!(matches!(operations[3].kind, ConstructOperationKind::ProofEvaluate { ref proof } if proof == "ReleaseGate"));
        assert!(matches!(operations[4].kind, ConstructOperationKind::TraceEmit { ref trace, ref event }
            if trace == "Build" && event == "compile"));
        assert!(matches!(operations[5].kind, ConstructOperationKind::CellUpdate { .. }));
    }

    #[test]
    fn malformed_source_operation_arguments_fail_closed() {
        let source = "module bad_ops\nfn main() -> Int\n  vault_authorize(dynamic_vault, \"network\")\n  return 0\n";
        let module = parse(source).unwrap();
        let errors = lower_source_operations(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("Vault name must be a non-empty string literal")));
    }

    #[test]
    fn ordinary_source_without_operations_produces_empty_binding_list() {
        let module = parse("module ordinary\nfn main() -> Int\n  return 0\n").unwrap();
        assert!(lower_source_operations(&module).unwrap().is_empty());
    }
}
