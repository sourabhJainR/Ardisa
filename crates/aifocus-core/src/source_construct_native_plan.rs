//! Ordered, fail-closed lowering plan from validated source operation sites to native instructions.
//!
//! This plan is deliberately not a complete executable module. It keeps operation
//! identity and source order attached to existing native lowering results, and it
//! refuses the unsupported Trace path rather than dropping event payload semantics.
//! `pipeline::compile_source` continues to enforce AIF610 for construct-bearing modules.

use crate::{
    ast::Module,
    native::NativeInstr,
    source::Span,
    source_construct_bindings::{self, SourceConstructOperationBinding},
    construct_operations::ConstructOperationKind,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeConstructOperationPlan {
    pub ordinal: usize,
    pub call_node_id: crate::ast::NodeId,
    pub source_span: Span,
    pub operation: crate::construct_operations::ConstructOperation,
    pub instructions: Vec<NativeInstr>,
}

/// Lower all source operations as one all-or-nothing planning pass.
///
/// Every operation must have a stable AST call-site binding and a supported native
/// lowering. If any operation cannot be represented, no partial plan is returned.
pub fn lower_source_construct_operations(
    module: &Module,
) -> Result<Vec<NativeConstructOperationPlan>, Vec<String>> {
    let bindings = source_construct_bindings::bind_source_operations(module)?;
    let operations: Vec<_> = bindings.iter().map(|binding| binding.operation.clone()).collect();
    let validated = crate::construct_operations::validate_module_operations(module, &operations)
        .map_err(|errors| errors.into_iter().map(|error| format!("operation validation: {error}")).collect::<Vec<_>>())?;

    let mut plans = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let lowered = match &binding.operation.kind {
            ConstructOperationKind::CellInitialize { .. }
            | ConstructOperationKind::CellUpdate { .. } => {
                crate::native::lower_cell_construct_operation(module, &binding.operation)
            }
            _ => crate::native::lower_construct_operation(module, &binding.operation),
        };
        let instructions = lowered.map_err(|error| {
            vec![format!(
                "operation {} at {}..{} cannot be lowered: {:?}",
                binding.ordinal, binding.source_span.start, binding.source_span.end, error
            )]
        })?;
        plans.push(NativeConstructOperationPlan {
            ordinal: binding.ordinal,
            call_node_id: binding.call_node_id,
            source_span: binding.source_span,
            operation: binding.operation,
            instructions,
        });
    }

    // Keep the whole-batch validation result live as a structural guard: the
    // operation order passed to the validator must equal the order in the plan.
    if plans.len() != validated.operations.len()
        || plans.iter().zip(&validated.operations).any(|(plan, operation)| &plan.operation != operation)
    {
        return Err(vec!["native construct plan differs from validated source operation order".into()]);
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::NativeInstr;

    const SOURCE: &str = r#"module native_plan
cell Counter
  value: Int
  invariant: value >= 0
phase Job
  initial: todo
  todo -> running
vault Local
  capabilities: filesystem_read
proof ReleaseGate
  requires: type_safety, bounds
  ensures: true
  on_unknown: reject
  required: true
fn main() -> Int
  cell_initialize("Counter", "value", 1)
  phase_transition("Job", "todo", "running")
  vault_authorize("Local", "filesystem_read")
  proof_evaluate("ReleaseGate")
  return 0
"#;

    #[test]
    fn lowers_supported_operations_in_source_order_with_stable_sites() {
        let module = crate::parse(SOURCE).unwrap();
        let plan = lower_source_construct_operations(&module).unwrap();
        assert_eq!(plan.len(), 4);
        assert!(plan.iter().enumerate().all(|(index, item)| item.ordinal == index));
        assert!(plan.iter().all(|item| item.call_node_id.0 != 0 && item.source_span.end > item.source_span.start));
        assert!(matches!(plan[0].instructions.as_slice(), [NativeInstr::CellStart { .. }]));
        assert!(matches!(plan[1].instructions.as_slice(), [
            NativeInstr::PhaseStart { .. }, NativeInstr::PushString(target), NativeInstr::PhaseTransition
        ] if target == "running"));
        assert!(matches!(plan[2].instructions.as_slice(), [NativeInstr::VaultAuthorize { .. }]));
        assert!(matches!(plan[3].instructions.as_slice(), [NativeInstr::ProofCheck { .. }]));
    }

    #[test]
    fn unsupported_trace_payload_rejects_the_entire_plan() {
        let source = SOURCE.replace(
            "  return 0",
            "  trace_emit(\"Build\", \"compile\")\n  return 0",
        ).replace(
            "fn main() -> Int",
            "trace Build\n  signature: required\nfn main() -> Int",
        );
        let module = crate::parse(&source).unwrap();
        let error = lower_source_construct_operations(&module).unwrap_err();
        assert!(error.iter().any(|message| message.contains("Trace operation")));
    }

    #[test]
    fn ordinary_module_has_an_empty_native_construct_plan() {
        let module = crate::parse("module plain\nfn main() -> Int\n  return 0\n").unwrap();
        assert!(lower_source_construct_operations(&module).unwrap().is_empty());
    }
}
