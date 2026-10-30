//! Source-bound lowering from validated AI-native operations to native VM instructions.
//!
//! This is an explicit compiler integration seam: it preserves the operation's AST
//! identity and source span beside the exact native instruction sequence. It does
//! not splice instructions into ARDISA-EXEC-V1 function bodies yet, so compile_source
//! continues to enforce AIF610 for construct-bearing modules.

use crate::{
    ast::Module,
    native::{self, NativeInstr},
    source_construct_bindings::{self, SourceConstructOperationBinding},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredSourceConstructOperation {
    pub binding: SourceConstructOperationBinding,
    pub instructions: Vec<NativeInstr>,
}

/// Lower all source operation bindings in source order.
///
/// The entire source batch is bound and validated before native lowering starts.
/// Unsupported semantics (including Trace events that do not yet carry their event
/// payload in the executable instruction) fail closed; no partial plan is returned.
pub fn lower_source_construct_operations(
    module: &Module,
) -> Result<Vec<LoweredSourceConstructOperation>, Vec<String>> {
    let bindings = source_construct_bindings::bind_source_operations(module)?;
    let mut lowered = Vec::with_capacity(bindings.len());

    for binding in bindings {
        let result = match &binding.operation.kind {
            crate::construct_operations::ConstructOperationKind::CellInitialize { .. }
            | crate::construct_operations::ConstructOperationKind::CellUpdate { .. } => {
                native::lower_cell_construct_operation(module, &binding.operation)
            }
            _ => native::lower_construct_operation(module, &binding.operation),
        };
        let instructions = result.map_err(|error| {
            vec![format!(
                "source operation {} at {}..{} cannot be lowered to native instructions: {:?}",
                binding.ordinal,
                binding.source_span.start,
                binding.source_span.end,
                error
            )]
        })?;
        lowered.push(LoweredSourceConstructOperation { binding, instructions });
    }

    Ok(lowered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{construct_operations::ConstructOperationKind, native::NativeInstr};

    const SOURCE: &str = r#"module native_ops
phase Job
  initial: todo
  todo -> running
  running -> done
cell Counter
  value: Int
  invariant: value >= 0
vault Local
  capabilities: filesystem_read, clock
fn main() -> Int
  cell_initialize("Counter", "value", 1 + 2)
  phase_transition("Job", "todo", "running")
  vault_authorize("Local", "filesystem_read")
  cell_update("Counter", "value", 4)
  return 0
"#;

    #[test]
    fn binds_native_instruction_sequences_to_source_sites_in_stable_order() {
        let module = crate::parse(SOURCE).unwrap();
        let first = lower_source_construct_operations(&module).unwrap();
        let second = lower_source_construct_operations(&module).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 4);
        assert!(first.iter().enumerate().all(|(index, item)| {
            item.binding.ordinal == index && item.binding.source_span.end > item.binding.source_span.start
        }));
        assert!(matches!(&first[0].binding.operation.kind, ConstructOperationKind::CellInitialize { .. }));
        assert!(matches!(first[0].instructions.as_slice(), [NativeInstr::CellStart { spec }] if spec.initial == 3));
        assert!(matches!(first[1].instructions.as_slice(), [
            NativeInstr::PhaseStart { .. },
            NativeInstr::PushString(target),
            NativeInstr::PhaseTransition
        ] if target == "running"));
        assert!(matches!(first[2].instructions.as_slice(), [NativeInstr::VaultAuthorize { capability, .. }]
            if *capability == crate::native::NativeVaultCapability::FilesystemRead));
        assert!(matches!(first[3].instructions.as_slice(), [NativeInstr::PushInt(4), NativeInstr::CellUpdate]));
    }

    #[test]
    fn unsupported_trace_lowering_rejects_the_whole_source_batch() {
        let source = SOURCE.replace(
            "fn main() -> Int",
            "trace Build\n  signature: required\nfn main() -> Int",
        ).replace(
            "  return 0",
            "  trace_emit(\"Build\", \"compile\")\n  return 0",
        );
        let module = crate::parse(&source).unwrap();
        let error = lower_source_construct_operations(&module).unwrap_err();
        assert!(error.iter().any(|message| message.contains("Trace operation")));
    }

    #[test]
    fn ordinary_compilation_integrates_supported_construct_source() {
        let compiled = crate::pipeline::compile_source(SOURCE)
            .expect("the standard compiler integrates supported Cell, Phase, and Vault constructs");
        let program = crate::native::decode_program(&compiled.artifact)
            .expect("combined construct artifact decodes");
        assert!(program.functions.contains_key("main"));
    }
}
