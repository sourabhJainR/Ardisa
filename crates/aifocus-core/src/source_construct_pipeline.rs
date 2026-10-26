//! Integrated source-to-typed-operation preparation for AI-native constructs.
//!
//! This gate wires parsing, source operation extraction, declaration/reference
//! validation, expression type checking, and typed CFG lowering into one deterministic
//! API. It intentionally does not produce ARDISA-EXEC-V1; pipeline::compile_source
//! continues to reject construct-bearing modules with AIF610 until native operation
//! placement and artifact semantics are implemented end-to-end.

use crate::{
    ast::Module,
    construct_operations::ValidatedConstructOperations,
    source_construct_lowering,
    typed_ir::{self, TypedIrModule},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceConstructPipelineError {
    Parse(Vec<crate::source::Diagnostic>),
    SourceLowering(Vec<String>),
    OperationValidation(Vec<String>),
    TypedLowering(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSourceConstructs {
    pub module: Module,
    pub typed_module: TypedIrModule,
    pub operations: ValidatedConstructOperations,
}

/// Prepare one source module through every available pre-execution compiler gate.
///
/// Operations are extracted in source order and validated as a complete batch before
/// the typed module is returned. No provider, filesystem, network, signer, or other
/// host effect is invoked by this function.
pub fn prepare_source_constructs(
    source: &str,
) -> Result<PreparedSourceConstructs, SourceConstructPipelineError> {
    let module = crate::parse(source).map_err(SourceConstructPipelineError::Parse)?;
    let operations = source_construct_lowering::lower_source_operations(&module)
        .map_err(SourceConstructPipelineError::SourceLowering)?;
    let validated = crate::construct_operations::validate_module_operations(&module, &operations)
        .map_err(SourceConstructPipelineError::OperationValidation)?;
    let typed_module = typed_ir::lower_cfg(&module)
        .map_err(SourceConstructPipelineError::TypedLowering)?;

    Ok(PreparedSourceConstructs {
        module,
        typed_module,
        operations: validated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"module integrated_source
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
  requires: type_safety, bounds
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
    fn integrates_source_extraction_declaration_validation_and_typed_lowering() {
        let prepared = prepare_source_constructs(SOURCE).expect("source construct gate");
        assert_eq!(prepared.module.name, "integrated_source");
        assert_eq!(prepared.typed_module.name, "integrated_source");
        assert_eq!(prepared.typed_module.functions.len(), 1);
        assert_eq!(prepared.typed_module.constructs.len(), 5);
        assert_eq!(prepared.operations.operations.len(), 6);
        assert!(matches!(
            &prepared.operations.operations[0].kind,
            crate::construct_operations::ConstructOperationKind::CellInitialize { .. }
        ));
        assert!(matches!(
            &prepared.operations.operations[5].kind,
            crate::construct_operations::ConstructOperationKind::CellUpdate { .. }
        ));
    }

    #[test]
    fn malformed_or_unresolved_source_operation_fails_before_prepared_output() {
        let malformed = SOURCE.replace(
            "vault_authorize(\"Local\", \"filesystem_read\")",
            "vault_authorize(dynamic_vault, \"filesystem_read\")",
        );
        assert!(matches!(
            prepare_source_constructs(&malformed),
            Err(SourceConstructPipelineError::SourceLowering(_))
        ));

        let unresolved = SOURCE.replace("\"Local\", \"filesystem_read\"", "\"Missing\", \"filesystem_read\"");
        assert!(matches!(
            prepare_source_constructs(&unresolved),
            Err(SourceConstructPipelineError::OperationValidation(_))
        ));
    }

    #[test]
    fn ordinary_compile_gate_keeps_aif610_until_native_artifact_integration() {
        let error = crate::pipeline::compile_source(SOURCE)
            .expect_err("construct-bearing source must remain blocked at native compile");
        assert!(matches!(
            error,
            crate::pipeline::PipelineError::Semantic(ref diagnostics)
                if diagnostics.iter().filter(|diagnostic| diagnostic.code == "AIF610").count() == 5
        ));
    }

    #[test]
    fn operation_order_is_deterministic_across_preparation_runs() {
        let first = prepare_source_constructs(SOURCE).unwrap();
        let second = prepare_source_constructs(SOURCE).unwrap();
        assert_eq!(first.operations, second.operations);
        assert_eq!(first.typed_module, second.typed_module);
    }
}
