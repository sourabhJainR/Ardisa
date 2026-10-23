//! Source-to-native artifact path for invariant-checked integer Cells.
//!
//! This narrow subset supports one `value: Int` field and one supported integer
//! invariant. Updates use immutable native state tokens: a rejected update returns
//! an error and cannot mutate a previously held token. General compiler support
//! remains fail-closed for constructs that do not meet this contract.
use std::collections::BTreeMap;

use crate::{
    ast::ConstructKind,
    native::{self, NativeFunction, NativeInstr, NativeProgram, NativeValue},
    typed_ir,
};

pub const MAX_CELL_UPDATES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCellCompileError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    ExpectedSingleCell,
    UnknownCell(String),
    TooManyUpdates,
    Native(native::NativeError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCellArtifact {
    pub artifact: String,
    pub cell_name: String,
    pub initial_value: i64,
    pub update_count: usize,
}

/// Compile one source Cell and a fixed update sequence to a deterministic artifact.
pub fn compile_cell_source(
    source: &str,
    cell_name: &str,
    initial_value: i64,
    updates: &[i64],
) -> Result<NativeCellArtifact, NativeCellCompileError> {
    if updates.len() > MAX_CELL_UPDATES {
        return Err(NativeCellCompileError::TooManyUpdates);
    }
    let module = crate::parse(source).map_err(NativeCellCompileError::Parse)?;
    let constructs = typed_ir::lower_constructs(&module).map_err(NativeCellCompileError::TypedIr)?;
    if constructs.len() != 1 || constructs[0].kind != ConstructKind::Cell {
        return Err(NativeCellCompileError::ExpectedSingleCell);
    }
    let construct = constructs.iter().find(|item| item.name == cell_name)
        .ok_or_else(|| NativeCellCompileError::UnknownCell(cell_name.to_string()))?;
    let start = native::cell_start_instruction(construct, initial_value)
        .map_err(NativeCellCompileError::Native)?;
    let mut code = vec![start];
    for update in updates {
        code.push(NativeInstr::PushInt(*update));
        code.push(NativeInstr::CellUpdate);
    }
    code.push(NativeInstr::CellCurrent);
    code.push(NativeInstr::Return);
    let mut functions = BTreeMap::new();
    functions.insert("main".to_string(), NativeFunction { params: Vec::new(), code });
    let artifact = native::encode_program(&NativeProgram { functions });
    let decoded = native::decode_program(&artifact).map_err(NativeCellCompileError::Native)?;
    native::ValidatedNativeProgram::try_new(decoded).map_err(NativeCellCompileError::Native)?;
    Ok(NativeCellArtifact {
        artifact,
        cell_name: cell_name.to_string(),
        initial_value,
        update_count: updates.len(),
    })
}

/// Execute the exact serialized native artifact and return the final Cell value.
pub fn execute_cell_artifact(artifact: &NativeCellArtifact) -> Result<i64, NativeCellCompileError> {
    let program = native::decode_program(&artifact.artifact).map_err(NativeCellCompileError::Native)?;
    match native::run_program(&program, "main", &[]) {
        Ok(NativeValue::Int(value)) => Ok(value),
        Ok(_) => Err(NativeCellCompileError::Native(native::NativeError::InvalidProgram(
            "Cell entry point did not return an integer value".into(),
        ))),
        Err(error) => Err(NativeCellCompileError::Native(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module cells\ncell Counter\n  value: Int\n  invariant: value >= 0\n";

    #[test]
    fn source_cell_executes_valid_updates_from_serialized_artifact() {
        let artifact = compile_cell_source(SOURCE, "Counter", 2, &[4, 9]).unwrap();
        assert_eq!(execute_cell_artifact(&artifact).unwrap(), 9);
        assert_eq!(artifact.update_count, 2);
    }

    #[test]
    fn source_cell_rejects_invalid_initial_value_and_update() {
        assert!(compile_cell_source(SOURCE, "Counter", -1, &[]).is_err());
        let artifact = compile_cell_source(SOURCE, "Counter", 3, &[5, -1]).unwrap();
        assert!(matches!(
            execute_cell_artifact(&artifact),
            Err(NativeCellCompileError::Native(native::NativeError::Type(_)))
        ));
        // A new execution starts from the original token; the failed invocation
        // cannot mutate this independently held artifact's initial state.
        let unchanged = compile_cell_source(SOURCE, "Counter", 3, &[]).unwrap();
        assert_eq!(execute_cell_artifact(&unchanged).unwrap(), 3);
    }

    #[test]
    fn source_cell_artifact_is_deterministic_and_rejects_unsupported_shapes() {
        let first = compile_cell_source(SOURCE, "Counter", 3, &[8]).unwrap();
        let second = compile_cell_source(SOURCE, "Counter", 3, &[8]).unwrap();
        assert_eq!(first.artifact, second.artifact);
        assert!(compile_cell_source(
            "module c\ncell C\n  value: Int\n  invariant: value > 0 && value < 10\n",
            "C", 1, &[]
        ).is_err());
        assert!(matches!(
            compile_cell_source(SOURCE, "Counter", 1, &vec![1; MAX_CELL_UPDATES + 1]),
            Err(NativeCellCompileError::TooManyUpdates)
        ));
    }

    #[test]
    fn source_cell_rejects_mixed_constructs() {
        let mixed = format!("{SOURCE}phase Work\n  initial: todo\n  transition: todo -> done\n");
        assert!(compile_cell_source(&mixed, "Counter", 1, &[]).is_err());
    }
}
