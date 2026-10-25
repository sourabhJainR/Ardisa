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


/// Compile a source-backed sequence of typed Cell operations into one native artifact.
///
/// The first operation must initialize exactly one integer Cell. Subsequent operations
/// must update that same Cell. Each expression is resolved against its named function
/// and lowered by the shared, fail-closed literal subset; no value is inferred from
/// operation metadata. The artifact is validated before it is returned.
pub fn compile_cell_construct_operations(
    source: &str,
    operations: &[crate::construct_operations::ConstructOperation],
) -> Result<NativeCellArtifact, NativeCellCompileError> {
    use crate::construct_operations::ConstructOperationKind;

    if operations.is_empty() {
        return Err(NativeCellCompileError::ExpectedSingleCell);
    }
    if operations.len() > MAX_CELL_UPDATES + 1 {
        return Err(NativeCellCompileError::TooManyUpdates);
    }

    let module = crate::parse(source).map_err(NativeCellCompileError::Parse)?;
    crate::construct_operations::validate_module_operations(&module, operations)
        .map_err(NativeCellCompileError::TypedIr)?;

    let (cell_name, initial_instructions) = match &operations[0].kind {
        ConstructOperationKind::CellInitialize { cell, .. } => {
            (cell.clone(), crate::native::lower_cell_construct_operation(&module, &operations[0])
                .map_err(NativeCellCompileError::Native)?)
        }
        _ => return Err(NativeCellCompileError::TypedIr(vec![
            "the first Cell operation must initialize the Cell".into(),
        ])),
    };
    let initial_value = match initial_instructions.first() {
        Some(NativeInstr::CellStart { spec }) => spec.initial,
        _ => return Err(NativeCellCompileError::Native(native::NativeError::InvalidProgram(
            "Cell initializer did not lower to CellStart".into(),
        ))),
    };
    let mut code = initial_instructions;

    for operation in operations.iter().skip(1) {
        match &operation.kind {
            ConstructOperationKind::CellUpdate { cell, .. } if *cell == cell_name => {
                code.extend(crate::native::lower_cell_construct_operation(&module, operation)
                    .map_err(NativeCellCompileError::Native)?);
            }
            ConstructOperationKind::CellUpdate { cell, .. } => {
                return Err(NativeCellCompileError::TypedIr(vec![format!(
                    "Cell update targets '{cell}', but this artifact initialized '{cell_name}'"
                )]));
            }
            ConstructOperationKind::CellInitialize { .. } => {
                return Err(NativeCellCompileError::TypedIr(vec![
                    "a Cell may be initialized only once per native operation sequence".into(),
                ]));
            }
            _ => return Err(NativeCellCompileError::TypedIr(vec![
                "Cell operation sequence contains a non-Cell operation".into(),
            ])),
        }
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
        cell_name,
        initial_value,
        update_count: operations.len() - 1,
    })
}


#[cfg(test)]
mod construct_operation_tests {
    use super::*;
    use crate::ast::StmtKind;
    use crate::construct_operations::{ConstructOperation, ConstructOperationKind};

    const SOURCE: &str = "module cells\ncell Counter\n  value: Int\n  invariant: value >= 0\nfn main() -> Int\n  let initial = 2\n  let next = 5\n  return next\n";

    fn operations() -> Vec<ConstructOperation> {
        let module = crate::parse(SOURCE).unwrap();
        let function = module.items.iter().find_map(|item| match item {
            crate::ast::Item::Function(function) if function.name == "main" => Some(function),
            _ => None,
        }).unwrap();
        let expression_id = |name: &str| function.body.stmts.iter().find_map(|stmt| match &stmt.kind {
            StmtKind::Let { name: binding, value } if binding == name => Some(value.id.0),
            _ => None,
        }).unwrap();
        vec![
            ConstructOperation {
                function: "main".into(),
                kind: ConstructOperationKind::CellInitialize {
                    cell: "Counter".into(), field: "value".into(),
                    expression: expression_id("initial"), value_type: crate::ast::TypeKind::Int,
                },
            },
            ConstructOperation {
                function: "main".into(),
                kind: ConstructOperationKind::CellUpdate {
                    cell: "Counter".into(), field: "value".into(),
                    expression: expression_id("next"), value_type: crate::ast::TypeKind::Int,
                },
            },
        ]
    }

    #[test]
    fn ordered_source_cell_operations_execute_through_serialized_native_artifact() {
        let artifact = compile_cell_construct_operations(SOURCE, &operations()).unwrap();
        assert_eq!(artifact.initial_value, 2);
        assert_eq!(artifact.update_count, 1);
        assert_eq!(execute_cell_artifact(&artifact).unwrap(), 5);
        let repeat = compile_cell_construct_operations(SOURCE, &operations()).unwrap();
        assert_eq!(artifact.artifact, repeat.artifact);
    }

    #[test]
    fn source_cell_operation_sequence_rejects_missing_initializer_and_reinitialization() {
        let mut ops = operations();
        assert!(compile_cell_construct_operations(SOURCE, &ops[1..]).is_err());
        let initializer = ops[0].clone();
        ops.push(initializer);
        assert!(compile_cell_construct_operations(SOURCE, &ops).is_err());
    }

    #[test]
    fn source_cell_operation_sequence_rejects_cross_cell_updates() {
        let source = SOURCE.replace("cell Counter", "cell Counter").replace(
            "invariant: value >= 0\nfn main()",
            "invariant: value >= 0\ncell Other\n  value: Int\n  invariant: value >= 0\nfn main()"
        );
        let mut ops = operations();
        if let ConstructOperationKind::CellUpdate { cell, .. } = &mut ops[1].kind {
            *cell = "Other".into();
        }
        assert!(compile_cell_construct_operations(&source, &ops).is_err());
    }
}
