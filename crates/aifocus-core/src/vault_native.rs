//! Source-to-native artifact path for Vault capability checks.
//!
//! The emitted instruction checks a declared capability before subsequent native
//! instructions. Host effects must still be routed through VaultRuntimeSet; this
//! API is not an operating-system sandbox and does not enable arbitrary I/O.
use std::collections::BTreeMap;

use crate::{
    ast::ConstructKind,
    native::{self, NativeFunction, NativeInstr, NativeProgram, NativeVaultCapability, NativeValue},
    typed_ir,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeVaultCompileError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    ExpectedSingleVault,
    UnknownVault(String),
    UnsupportedCapability(String),
    Native(native::NativeError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeVaultArtifact {
    pub artifact: String,
    pub vault_name: String,
    pub required_capability: String,
}

/// Compile a single Vault declaration and one required capability to an executable
/// authorization check. Unknown and overlapping grants/denials fail closed.
pub fn compile_vault_source(
    source: &str,
    vault_name: &str,
    required_capability: &str,
) -> Result<NativeVaultArtifact, NativeVaultCompileError> {
    let required = NativeVaultCapability::parse(required_capability)
        .ok_or_else(|| NativeVaultCompileError::UnsupportedCapability(required_capability.to_string()))?;
    let module = crate::parse(source).map_err(NativeVaultCompileError::Parse)?;
    let constructs = typed_ir::lower_constructs(&module).map_err(NativeVaultCompileError::TypedIr)?;
    if constructs.len() != 1 || constructs[0].kind != ConstructKind::Vault {
        return Err(NativeVaultCompileError::ExpectedSingleVault);
    }
    let construct = constructs.iter().find(|item| item.name == vault_name)
        .ok_or_else(|| NativeVaultCompileError::UnknownVault(vault_name.to_string()))?;
    let authorize = native::vault_authorize_instruction(construct, required)
        .map_err(NativeVaultCompileError::Native)?;
    let mut functions = BTreeMap::new();
    functions.insert("main".to_string(), NativeFunction {
        params: Vec::new(),
        code: vec![authorize, NativeInstr::Return],
    });
    let artifact = native::encode_program(&NativeProgram { functions });
    let decoded = native::decode_program(&artifact).map_err(NativeVaultCompileError::Native)?;
    native::ValidatedNativeProgram::try_new(decoded).map_err(NativeVaultCompileError::Native)?;
    Ok(NativeVaultArtifact {
        artifact,
        vault_name: vault_name.to_string(),
        required_capability: required.as_str().to_string(),
    })
}

/// Execute the serialized artifact. Success means the required capability was
/// present in the source policy; it does not itself execute a host-side effect.
pub fn execute_vault_authorization(artifact: &NativeVaultArtifact) -> Result<(), NativeVaultCompileError> {
    let program = native::decode_program(&artifact.artifact).map_err(NativeVaultCompileError::Native)?;
    match native::run_program(&program, "main", &[]) {
        Ok(NativeValue::Unit) => Ok(()),
        Ok(_) => Err(NativeVaultCompileError::Native(native::NativeError::InvalidProgram(
            "Vault authorization did not return Unit".into(),
        ))),
        Err(error) => Err(NativeVaultCompileError::Native(error)),
    }
}

/// Authorize through the compiled native artifact before invoking a host operation.
/// The closure is never called when compilation or capability validation fails.
pub fn execute_vault_operation<T>(
    source: &str,
    vault_name: &str,
    required_capability: &str,
    operation: impl FnOnce() -> T,
) -> Result<T, NativeVaultCompileError> {
    let artifact = compile_vault_source(source, vault_name, required_capability)?;
    execute_vault_authorization(&artifact)?;
    Ok(operation())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module vaults\nvault ReadOnly\n  capabilities: filesystem_read clock\n  denies: network\n";

    #[test]
    fn source_vault_authorization_round_trips_and_is_deterministic() {
        let first = compile_vault_source(SOURCE, "ReadOnly", "filesystem_read").unwrap();
        let second = compile_vault_source(SOURCE, "ReadOnly", "filesystem_read").unwrap();
        assert_eq!(first.artifact, second.artifact);
        assert!(first.artifact.starts_with(native::ARTIFACT_MAGIC));
        execute_vault_authorization(&first).unwrap();
    }

    #[test]
    fn denied_native_vault_check_never_invokes_host_operation() {
        let called = std::cell::Cell::new(false);
        let result = execute_vault_operation(SOURCE, "ReadOnly", "network", || called.set(true));
        assert!(result.is_err());
        assert!(!called.get());
    }

    #[test]
    fn native_vault_instruction_denies_ungranted_capabilities() {
        let artifact = compile_vault_source(SOURCE, "ReadOnly", "network").unwrap();
        assert!(matches!(
            execute_vault_authorization(&artifact),
            Err(NativeVaultCompileError::Native(native::NativeError::Type(message)))
                if message.contains("denied capability 'network'")
        ));
    }

    #[test]
    fn unknown_capability_missing_grants_and_overlapping_denials_fail_closed() {
        assert!(matches!(
            compile_vault_source(SOURCE, "ReadOnly", "hyperdrive"),
            Err(NativeVaultCompileError::UnsupportedCapability(_))
        ));
        assert!(matches!(
            compile_vault_source("module v\nvault Empty\n  denies: network\n", "Empty", "network"),
            Err(NativeVaultCompileError::Native(native::NativeError::InvalidProgram(_)))
        ));
        assert!(matches!(
            compile_vault_source("module v\nvault Broken\n  capabilities: network\n  denies: network\n", "Broken", "network"),
            Err(NativeVaultCompileError::Native(native::NativeError::InvalidProgram(_)))
        ));
    }

    #[test]
    fn mixed_constructs_and_unknown_vault_names_are_rejected() {
        assert!(matches!(
            compile_vault_source(SOURCE, "Missing", "clock"),
            Err(NativeVaultCompileError::UnknownVault(_))
        ));
        let mixed = format!("{SOURCE}cell Counter\n  value: Int\n  invariant: value >= 0\n");
        assert_eq!(
            compile_vault_source(&mixed, "ReadOnly", "clock"),
            Err(NativeVaultCompileError::ExpectedSingleVault)
        );
    }
}
