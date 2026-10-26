//! Stable source locations for validated AI-native operation bindings.
//!
//! This layer records where each reserved operation call appeared in the AST. It
//! does not lower operations to executable instructions or authorize effects. The
//! normal compiler continues to reject construct-bearing source with AIF610.

use crate::{
    ast::{Block, Expr, ExprKind, Item, Module, NodeId, StmtKind},
    source::Span,
    construct_operations::{self, ConstructOperation},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceConstructOperationBinding {
    pub ordinal: usize,
    pub operation: ConstructOperation,
    pub call_node_id: NodeId,
    pub source_span: Span,
}

/// Lower reserved source calls and bind each validated operation to its exact call
/// node. Ordering must match source traversal; any mismatch is rejected rather than
/// guessed, because future native instruction placement depends on this identity.
pub fn bind_source_operations(
    module: &Module,
) -> Result<Vec<SourceConstructOperationBinding>, Vec<String>> {
    let operations = construct_operations::lower_source_operations(module)?;
    let mut sites = Vec::new();
    for item in &module.items {
        let Item::Function(function) = item;
        collect_block_sites(&function.body, &mut sites);
    }
    if sites.len() != operations.len() {
        return Err(vec![format!(
            "source operation binding count mismatch: {} validated operations, {} source call sites",
            operations.len(),
            sites.len()
        )]);
    }
    Ok(operations
        .into_iter()
        .zip(sites)
        .enumerate()
        .map(|(ordinal, (operation, (call_node_id, source_span)))| {
            SourceConstructOperationBinding {
                ordinal,
                operation,
                call_node_id,
                source_span,
            }
        })
        .collect())
}

fn is_reserved_operation(name: &str) -> bool {
    matches!(
        name,
        "cell_initialize"
            | "cell_update"
            | "phase_transition"
            | "vault_authorize"
            | "proof_evaluate"
            | "trace_emit"
    )
}

fn collect_block_sites(block: &Block, sites: &mut Vec<(NodeId, Span)>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { value, .. }
            | StmtKind::Set { value, .. }
            | StmtKind::Return(Some(value))
            | StmtKind::Expr(value) => collect_expr_sites(value, sites),
            StmtKind::SetIndex { collection, index, value } => {
                collect_expr_sites(collection, sites);
                collect_expr_sites(index, sites);
                collect_expr_sites(value, sites);
            }
            StmtKind::Scope { body } => collect_block_sites(body, sites),
            StmtKind::Spawn { call, .. } => collect_expr_sites(call, sites),
            StmtKind::While { condition, body } => {
                collect_expr_sites(condition, sites);
                collect_block_sites(body, sites);
            }
            StmtKind::For { iterable, body, .. } => {
                collect_expr_sites(iterable, sites);
                collect_block_sites(body, sites);
            }
            StmtKind::Return(None) | StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
        }
    }
}

fn collect_expr_sites(expr: &Expr, sites: &mut Vec<(NodeId, Span)>) {
    if let ExprKind::Call { callee, args } = &expr.kind {
        if let ExprKind::Name(name) = &callee.kind {
            if is_reserved_operation(name) {
                sites.push((expr.id, expr.span));
                // Match source_construct_lowering's traversal: only Cell values
                // are recursively visited because they are executable expressions.
                if matches!(name.as_str(), "cell_initialize" | "cell_update") {
                    for arg in args.iter().skip(2) {
                        collect_expr_sites(arg, sites);
                    }
                }
                return;
            }
        }
    }

    match &expr.kind {
        ExprKind::Binary { left, right, .. } => {
            collect_expr_sites(left, sites);
            collect_expr_sites(right, sites);
        }
        ExprKind::Call { callee, args } => {
            collect_expr_sites(callee, sites);
            for arg in args {
                collect_expr_sites(arg, sites);
            }
        }
        ExprKind::Group(inner) => collect_expr_sites(inner, sites),
        ExprKind::List(items) => {
            for item in items {
                collect_expr_sites(item, sites);
            }
        }
        ExprKind::Index { collection, index } => {
            collect_expr_sites(collection, sites);
            collect_expr_sites(index, sites);
        }
        ExprKind::Slice { collection, start, end } => {
            collect_expr_sites(collection, sites);
            collect_expr_sites(start, sites);
            collect_expr_sites(end, sites);
        }
        ExprKind::If { condition, then_branch, else_branch } => {
            collect_expr_sites(condition, sites);
            collect_block_sites(then_branch, sites);
            if let Some(branch) = else_branch {
                collect_block_sites(branch, sites);
            }
        }
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::String(_)
        | ExprKind::Name(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    const SOURCE: &str = r#"module site_bindings
cell Counter
  value: Int
  invariant: value >= 0
phase Job
  initial: todo
  todo -> running
vault Local
  capabilities: filesystem_read
proof ReleaseGate
  requires: bounds
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

    #[test]
    fn binds_each_operation_to_a_deterministic_source_node_and_span() {
        let module = parse(SOURCE).unwrap();
        let first = bind_source_operations(&module).unwrap();
        let second = bind_source_operations(&module).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 6);
        assert!(first.iter().enumerate().all(|(index, binding)| binding.ordinal == index));
        assert!(first.windows(2).all(|pair| pair[0].source_span.start < pair[1].source_span.start));
        assert!(first.iter().all(|binding| binding.source_span.end > binding.source_span.start));
        assert!(first.iter().all(|binding| binding.call_node_id.0 != 0));
    }

    #[test]
    fn malformed_operation_source_does_not_produce_partial_bindings() {
        let source = SOURCE.replace(
            "vault_authorize(\"Local\", \"filesystem_read\")",
            "vault_authorize(dynamic, \"filesystem_read\")",
        );
        let module = parse(&source).unwrap();
        assert!(bind_source_operations(&module).is_err());
    }
}
