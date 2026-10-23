//! Source-to-runtime bridge for the deliberately small, fail-closed Cell subset.
//!
//! Supported invariant syntax is a single integer comparison over `value` and
//! an integer literal. This is not native ARDISA-EXEC-V1 execution; the compiler
//! continues to reject Cell declarations until native lowering is implemented.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ast::ConstructKind,
    typed_ir::{self, TypedIrConstruct, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellRuntimeError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    DuplicateInitialValue(String),
    MissingInitialValue(String),
    UnknownInitialValue(String),
    UnsupportedCellShape(String),
    UnsupportedInvariant { cell: String, invariant: String },
    InitialValueRejected(String),
    InvariantViolation(String),
    UnknownCell(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Comparison {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IntegerInvariant {
    comparison: Comparison,
    bound: i64,
}

impl IntegerInvariant {
    fn parse(source: &str) -> Option<Self> {
        let source = source.trim();
        let tail = source.strip_prefix("value")?.trim_start();
        for (token, comparison) in [
            (">=", Comparison::Ge),
            ("<=", Comparison::Le),
            ("==", Comparison::Eq),
            ("!=", Comparison::Ne),
            (">", Comparison::Gt),
            ("<", Comparison::Lt),
        ] {
            if let Some(rhs) = tail.strip_prefix(token) {
                let bound = rhs.trim().parse::<i64>().ok()?;
                return Some(Self { comparison, bound });
            }
        }
        None
    }

    fn accepts(self, value: i64) -> bool {
        match self.comparison {
            Comparison::Eq => value == self.bound,
            Comparison::Ne => value != self.bound,
            Comparison::Lt => value < self.bound,
            Comparison::Le => value <= self.bound,
            Comparison::Gt => value > self.bound,
            Comparison::Ge => value >= self.bound,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CellState {
    value: i64,
    invariant: IntegerInvariant,
}

/// Runtime state for validated integer Cell declarations.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CellRuntime {
    cells: BTreeMap<String, CellState>,
}

impl CellRuntime {
    /// Build cells from source and explicit initial values. No value is inferred.
    /// Only one `value: Int` field and one supported invariant are accepted.
    pub fn from_source(
        source: &str,
        initial_values: impl IntoIterator<Item = (String, i64)>,
    ) -> Result<Self, CellRuntimeError> {
        let module = crate::parse(source).map_err(CellRuntimeError::Parse)?;
        let constructs = typed_ir::lower_constructs(&module).map_err(CellRuntimeError::TypedIr)?;
        let mut supplied = BTreeMap::new();
        for (name, value) in initial_values {
            if supplied.insert(name.clone(), value).is_some() {
                return Err(CellRuntimeError::DuplicateInitialValue(name));
            }
        }

        let cell_names: BTreeSet<String> = constructs
            .iter()
            .filter(|construct| construct.kind == ConstructKind::Cell)
            .map(|construct| construct.name.clone())
            .collect();
        if let Some(name) = supplied.keys().find(|name| !cell_names.contains(*name)) {
            return Err(CellRuntimeError::UnknownInitialValue(name.clone()));
        }

        let mut cells = BTreeMap::new();
        for construct in constructs.iter().filter(|c| c.kind == ConstructKind::Cell) {
            let value = supplied.remove(&construct.name).ok_or_else(|| {
                CellRuntimeError::MissingInitialValue(construct.name.clone())
            })?;
            let invariant = Self::invariant_for(construct)?;
            if !invariant.accepts(value) {
                return Err(CellRuntimeError::InitialValueRejected(construct.name.clone()));
            }
            cells.insert(construct.name.clone(), CellState { value, invariant });
        }
        Ok(Self { cells })
    }

    fn invariant_for(construct: &TypedIrConstruct) -> Result<IntegerInvariant, CellRuntimeError> {
        let mut fields = Vec::new();
        let mut invariant = None;
        for member in &construct.members {
            match member {
                TypedIrConstructMember::Field { name, ty, .. } => {
                    fields.push((name.as_str(), ty));
                }
                TypedIrConstructMember::Clause { name, value, .. } if name == "invariant" => {
                    if invariant.replace(value.as_str()).is_some() {
                        return Err(CellRuntimeError::UnsupportedInvariant {
                            cell: construct.name.clone(),
                            invariant: "duplicate invariant clause".into(),
                        });
                    }
                }
                TypedIrConstructMember::Clause { name, .. } => {
                    return Err(CellRuntimeError::UnsupportedInvariant {
                        cell: construct.name.clone(),
                        invariant: format!("unsupported clause '{name}'"),
                    });
                }
                TypedIrConstructMember::Transition { from, to, .. } => {
                    return Err(CellRuntimeError::UnsupportedInvariant {
                        cell: construct.name.clone(),
                        invariant: format!("unsupported transition '{from} -> {to}'"),
                    });
                }
            }
        }
        if fields.len() != 1
            || fields[0].0 != "value"
            || fields[0].1 != &crate::ast::TypeKind::Int
        {
            return Err(CellRuntimeError::UnsupportedCellShape(construct.name.clone()));
        }
        let source = invariant.ok_or_else(|| {
            CellRuntimeError::UnsupportedInvariant {
                cell: construct.name.clone(),
                invariant: String::new(),
            }
        })?;
        IntegerInvariant::parse(source).ok_or_else(|| CellRuntimeError::UnsupportedInvariant {
            cell: construct.name.clone(),
            invariant: source.to_string(),
        })
    }

    pub fn get(&self, name: &str) -> Result<i64, CellRuntimeError> {
        self.cells
            .get(name)
            .map(|cell| cell.value)
            .ok_or_else(|| CellRuntimeError::UnknownCell(name.to_string()))
    }

    /// Validate before commit. A rejected update leaves the previous value intact.
    pub fn update(&mut self, name: &str, next: i64) -> Result<(), CellRuntimeError> {
        let cell = self.cells.get_mut(name)
            .ok_or_else(|| CellRuntimeError::UnknownCell(name.to_string()))?;
        if !cell.invariant.accepts(next) {
            return Err(CellRuntimeError::InvariantViolation(name.to_string()));
        }
        cell.value = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module cells\ncell Counter\n  value: Int\n  invariant: value >= 0\n";

    #[test]
    fn source_cell_checks_initial_value_and_commits_valid_updates() {
        let mut runtime = CellRuntime::from_source(
            SOURCE,
            [("Counter".to_string(), 1)],
        ).unwrap();
        assert_eq!(runtime.get("Counter"), Ok(1));
        runtime.update("Counter", 4).unwrap();
        assert_eq!(runtime.get("Counter"), Ok(4));
    }

    #[test]
    fn invalid_update_is_rejected_without_mutating_state() {
        let mut runtime = CellRuntime::from_source(
            SOURCE,
            [("Counter".to_string(), 3)],
        ).unwrap();
        assert_eq!(
            runtime.update("Counter", -1),
            Err(CellRuntimeError::InvariantViolation("Counter".into()))
        );
        assert_eq!(runtime.get("Counter"), Ok(3));
    }

    #[test]
    fn invalid_initial_value_and_missing_or_duplicate_values_fail_closed() {
        assert_eq!(
            CellRuntime::from_source(SOURCE, [("Counter".into(), -1)]),
            Err(CellRuntimeError::InitialValueRejected("Counter".into()))
        );
        assert_eq!(
            CellRuntime::from_source(SOURCE, std::iter::empty()),
            Err(CellRuntimeError::MissingInitialValue("Counter".into()))
        );
        assert_eq!(
            CellRuntime::from_source(
                SOURCE,
                [("Counter".into(), 1), ("Counter".into(), 2)]
            ),
            Err(CellRuntimeError::DuplicateInitialValue("Counter".into()))
        );
    }

    #[test]
    fn unsupported_cell_members_are_rejected_instead_of_ignored() {
        let unknown_clause = "module cells\ncell Counter\n  value: Int\n  invariant: value >= 0\n  reset: allowed\n";
        assert!(CellRuntime::from_source(unknown_clause, [("Counter".into(), 1)]).is_err());

        let duplicate_invariant = "module cells\ncell Counter\n  value: Int\n  invariant: value >= 0\n  invariant: value <= 100\n";
        assert!(CellRuntime::from_source(duplicate_invariant, [("Counter".into(), 1)]).is_err());
    }

    #[test]
    fn every_integer_comparison_is_enforced_at_initialization_and_update() {
        let cases = [
            ("value == 3", 3, 4),
            ("value != 3", 4, 3),
            ("value < 3", 2, 3),
            ("value <= 3", 3, 4),
            ("value > 3", 4, 3),
            ("value >= 3", 3, 2),
        ];
        for (invariant, valid, invalid) in cases {
            let source = format!(
                "module cells\ncell Counter\n  value: Int\n  invariant: {invariant}\n"
            );
            let mut runtime = CellRuntime::from_source(&source, [("Counter".into(), valid)])
                .unwrap_or_else(|error| panic!("valid initial value rejected for {invariant}: {error:?}"));
            assert_eq!(
                runtime.update("Counter", invalid),
                Err(CellRuntimeError::InvariantViolation("Counter".into())),
                "invalid update accepted for {invariant}"
            );
            assert_eq!(runtime.get("Counter"), Ok(valid), "state changed for {invariant}");
        }
    }

    #[test]
    fn malformed_or_trailing_invariant_tokens_fail_closed() {
        for invariant in [
            "value >= 0 trailing",
            "value >=",
            "value <> 0",
            "value >= 0 || value < 10",
            "other >= 0",
        ] {
            let source = format!(
                "module cells\ncell Counter\n  value: Int\n  invariant: {invariant}\n"
            );
            assert!(
                matches!(
                    CellRuntime::from_source(&source, [("Counter".into(), 1)]),
                    Err(CellRuntimeError::UnsupportedInvariant { .. })
                ),
                "unexpectedly accepted invariant: {invariant}"
            );
        }
    }

    #[test]
    fn unsupported_invariant_expressions_are_not_guessed() {
        let source = "module cells\ncell Counter\n  value: Int\n  invariant: value >= limit\n";
        assert!(matches!(
            CellRuntime::from_source(source, [("Counter".into(), 1)]),
            Err(CellRuntimeError::UnsupportedInvariant { .. })
        ));
    }
}
