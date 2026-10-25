//! Combined construct dispatch boundary for modules containing multiple AI-native
//! declarations.
//!
//! This module validates the complete source/operation batch before invoking a
//! runtime. It intentionally does not enable `pipeline::compile_source`: source
//! constructs still need integrated typed lowering and a native executable artifact.
//! The supplied executor must stage the entire batch and commit atomically; this
//! module guarantees validation-before-dispatch, not rollback of a host executor.

use crate::construct_operations::{
    self, ConstructOperation, ValidatedConstructOperations,
};
use crate::ast::Module;

/// Errors raised before a combined operation batch is dispatched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinedConstructError {
    Parse(Vec<crate::source::Diagnostic>),
    Validation(Vec<String>),
    Runtime(String),
}

/// A validated, ordered operation batch bound to one parsed source module.
///
/// Construct declarations and function references are validated together, so a
/// runtime never receives a partially validated list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinedConstructPlan {
    module_name: String,
    operations: ValidatedConstructOperations,
}

impl CombinedConstructPlan {
    /// Parse the source and validate every construct declaration and operation
    /// before producing an executable plan. No runtime callback is invoked here.
    pub fn validate(
        source: &str,
        operations: &[ConstructOperation],
    ) -> Result<Self, CombinedConstructError> {
        let module: Module = crate::parse(source).map_err(CombinedConstructError::Parse)?;
        let validated = construct_operations::validate_module_operations(&module, operations)
            .map_err(CombinedConstructError::Validation)?;
        Ok(Self {
            module_name: module.name,
            operations: validated,
        })
    }

    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn operations(&self) -> &[ConstructOperation] {
        &self.operations.operations
    }

    /// Dispatch the complete ordered batch exactly once, after validation.
    /// Atomicity is provided by the executor, not by this dispatch wrapper.
    ///
    /// The executor is the transaction boundary: it must stage effects and state
    /// changes and commit only when every operation succeeds. On any executor
    /// error it must leave externally visible state unchanged. The closure is
    /// deliberately passed the full batch rather than called once per operation,
    /// avoiding accidental partial execution by this dispatcher.
    pub fn dispatch_validated_batch<T, F>(&self, executor: F) -> Result<T, CombinedConstructError>
    where
        F: FnOnce(&[ConstructOperation]) -> Result<T, String>,
    {
        executor(self.operations())
            .map_err(CombinedConstructError::Runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construct_operations::{ConstructOperation, ConstructOperationKind};

    const SOURCE: &str = "module combined
trace Build
  signature: required
cell Counter
  value: Int
  invariant: value >= 0
vault ReadOnly
  capabilities: filesystem_read, clock
  denies: network
proof ReleaseGate
  requires: type_safety, bounds
  ensures: true
  on_unknown: reject
  required: true
phase Job
  initial: todo
  todo -> running
  running -> done
fn main() -> Int
  1
";

    fn operation(kind: ConstructOperationKind) -> ConstructOperation {
        ConstructOperation { function: "main".into(), kind }
    }

    #[test]
    fn validates_mixed_construct_batch_before_single_dispatch() {
        let operations = vec![
            operation(ConstructOperationKind::PhaseTransition {
                phase: "Job".into(), from: "todo".into(), to: "running".into(),
            }),
            operation(ConstructOperationKind::VaultAuthorize {
                vault: "ReadOnly".into(), capability: "filesystem_read".into(),
            }),
            operation(ConstructOperationKind::ProofEvaluate { proof: "ReleaseGate".into() }),
            operation(ConstructOperationKind::TraceEmit {
                trace: "Build".into(), event: "compile".into(),
            }),
        ];
        let plan = CombinedConstructPlan::validate(SOURCE, &operations).unwrap();
        assert_eq!(plan.module_name(), "combined");
        assert_eq!(plan.operations(), operations.as_slice());

        let mut calls = 0;
        let returned = plan.dispatch_validated_batch(|batch| {
            calls += 1;
            assert_eq!(batch, operations.as_slice());
            Ok(batch.len())
        }).unwrap();
        assert_eq!(returned, 4);
        assert_eq!(calls, 1);
    }

    #[test]
    fn invalid_late_operation_prevents_any_dispatch() {
        let operations = vec![
            operation(ConstructOperationKind::VaultAuthorize {
                vault: "ReadOnly".into(), capability: "filesystem_read".into(),
            }),
            operation(ConstructOperationKind::VaultAuthorize {
                vault: "ReadOnly".into(), capability: "network".into(),
            }),
        ];
        let result = CombinedConstructPlan::validate(SOURCE, &operations);
        assert!(matches!(result, Err(CombinedConstructError::Validation(_))));
    }

    #[test]
    fn unresolved_function_and_wrong_construct_kind_fail_closed() {
        let unresolved = vec![ConstructOperation {
            function: "missing".into(),
            kind: ConstructOperationKind::ProofEvaluate { proof: "ReleaseGate".into() },
        }];
        assert!(matches!(
            CombinedConstructPlan::validate(SOURCE, &unresolved),
            Err(CombinedConstructError::Validation(_))
        ));

        let wrong_kind = vec![operation(ConstructOperationKind::ProofEvaluate {
            proof: "ReadOnly".into(),
        })];
        assert!(matches!(
            CombinedConstructPlan::validate(SOURCE, &wrong_kind),
            Err(CombinedConstructError::Validation(_))
        ));
    }

    #[test]
    fn executor_failure_is_propagated_without_retry_or_follow_on_dispatch() {
        let operations = vec![operation(ConstructOperationKind::VaultAuthorize {
            vault: "ReadOnly".into(), capability: "filesystem_read".into(),
        })];
        let plan = CombinedConstructPlan::validate(SOURCE, &operations).unwrap();
        let mut calls = 0;
        let result = plan.execute_atomically(|_| {
            calls += 1;
            Err::<(), _>("staged operation rejected".into())
        });
        assert_eq!(result, Err(CombinedConstructError::Runtime("staged operation rejected".into())));
        assert_eq!(calls, 1);
    }

    #[test]
    fn malformed_source_rejects_before_dispatch() {
        let source = SOURCE.replace("fn main() -> Int\n", "fn main( -> Int\n");
        assert!(CombinedConstructPlan::validate(&source, &[]).is_err());
    }
}
