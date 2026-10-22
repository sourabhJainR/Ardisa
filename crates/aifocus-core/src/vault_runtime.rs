//! Source-to-runtime bridge for Vault capability policies.
//!
//! This routes host operations through explicit source-declared capability
//! grants. It is not an OS sandbox, and it does not enable native bytecode
//! execution; normal compilation remains fail-closed for Vault declarations.
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ai_mode::{Capability, PolicyViolation, VaultPolicy, VaultRuntime},
    ast::ConstructKind,
    typed_ir::{self, TypedIrConstructMember},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaultRuntimeError {
    Parse(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    DuplicateVault(String),
    UnknownVault(String),
    UnsupportedCapability { vault: String, capability: String },
    Policy(PolicyViolation),
}

fn source_capability(name: &str) -> Option<Capability> {
    match name {
        "clock" => Some(Capability::Clock),
        "random" => Some(Capability::Random),
        "filesystem_read" => Some(Capability::FilesystemRead),
        "filesystem_write" => Some(Capability::FilesystemWrite),
        "network" => Some(Capability::Network),
        "process" => Some(Capability::Process),
        "secrets" => Some(Capability::Secrets),
        "unsafe_interop" => Some(Capability::UnsafeInterop),
        _ => None,
    }
}

fn parse_capabilities(vault: &str, value: &str) -> Result<BTreeSet<Capability>, VaultRuntimeError> {
    let mut capabilities = BTreeSet::new();
    for token in value.split(|ch: char| ch.is_whitespace() || ch == ',').filter(|s| !s.is_empty()) {
        let capability = source_capability(token).ok_or_else(|| {
            VaultRuntimeError::UnsupportedCapability { vault: vault.to_string(), capability: token.to_string() }
        })?;
        if !capabilities.insert(capability) {
            return Err(VaultRuntimeError::UnsupportedCapability { vault: vault.to_string(), capability: token.to_string() });
        }
    }
    Ok(capabilities)
}

/// A collection of Vaults lowered from source and enforced at the host-operation boundary.
pub struct VaultRuntimeSet {
    vaults: BTreeMap<String, VaultRuntime>,
}

impl VaultRuntimeSet {
    pub fn from_source(source: &str) -> Result<Self, VaultRuntimeError> {
        let module = crate::parse(source).map_err(VaultRuntimeError::Parse)?;
        let constructs = typed_ir::lower_constructs(&module).map_err(VaultRuntimeError::TypedIr)?;
        let mut vaults = BTreeMap::new();
        for construct in constructs.iter().filter(|c| c.kind == ConstructKind::Vault) {
            let mut grants = BTreeSet::new();
            let mut saw_capabilities = false;
            for member in &construct.members {
                let TypedIrConstructMember::Clause { name, value, .. } = member else { continue };
                match name.as_str() {
                    "capabilities" => {
                        saw_capabilities = true;
                        grants = parse_capabilities(&construct.name, value)?;
                    }
                    // Denials are never converted to grants. Semantic analysis also
                    // rejects any capability present in both lists.
                    "denies" => {
                        let denied = parse_capabilities(&construct.name, value)?;
                        if !grants.is_disjoint(&denied) {
                            return Err(VaultRuntimeError::UnsupportedCapability {
                                vault: construct.name.clone(),
                                capability: "capability both granted and denied".into(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            if !saw_capabilities {
                return Err(VaultRuntimeError::TypedIr(vec![
                    format!("Vault '{}' is missing its capabilities clause", construct.name)
                ]));
            }
            if vaults.insert(
                construct.name.clone(),
                VaultRuntime::new(VaultPolicy { allowed_capabilities: grants }),
            ).is_some() {
                return Err(VaultRuntimeError::DuplicateVault(construct.name.clone()));
            }
        }
        Ok(Self { vaults })
    }

    pub fn policy(&self, name: &str) -> Result<&VaultPolicy, VaultRuntimeError> {
        self.vaults.get(name)
            .map(VaultRuntime::policy)
            .ok_or_else(|| VaultRuntimeError::UnknownVault(name.to_string()))
    }

    /// Check every required capability before invoking the host operation.
    /// If any grant is missing, the closure is not called.
    pub fn execute<T>(
        &self,
        vault: &str,
        required: impl IntoIterator<Item = Capability>,
        operation: impl FnOnce() -> T,
    ) -> Result<T, VaultRuntimeError> {
        self.vaults.get(vault)
            .ok_or_else(|| VaultRuntimeError::UnknownVault(vault.to_string()))?
            .execute(required, operation)
            .map_err(VaultRuntimeError::Policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    const SOURCE: &str = "module vaults\nvault ReadOnly\n  capabilities: filesystem_read clock\n  denies: network\nvault NoAccess\n  capabilities: \n";

    #[test]
    fn source_vault_grants_only_declared_capabilities() {
        let runtime = VaultRuntimeSet::from_source(
            "module vaults\nvault ReadOnly\n  capabilities: filesystem_read clock\n"
        ).unwrap();
        assert!(runtime.policy("ReadOnly").unwrap().permits(Capability::FilesystemRead));
        assert!(runtime.policy("ReadOnly").unwrap().permits(Capability::Clock));
        assert!(!runtime.policy("ReadOnly").unwrap().permits(Capability::Network));
    }

    #[test]
    fn denied_capability_prevents_operation_from_running() {
        let runtime = VaultRuntimeSet::from_source(
            "module vaults\nvault ReadOnly\n  capabilities: filesystem_read\n  denies: network\n"
        ).unwrap();
        let called = Cell::new(false);
        let result = runtime.execute("ReadOnly", [Capability::Network], || called.set(true));
        assert!(matches!(result, Err(VaultRuntimeError::Policy(PolicyViolation::UndeclaredCapabilities(_)))));
        assert!(!called.get());
    }

    #[test]
    fn allowed_capability_runs_operation_after_full_set_is_checked() {
        let runtime = VaultRuntimeSet::from_source(
            "module vaults\nvault ReadOnly\n  capabilities: filesystem_read clock\n"
        ).unwrap();
        let called = Cell::new(false);
        runtime.execute("ReadOnly", [Capability::FilesystemRead, Capability::Clock], || called.set(true)).unwrap();
        assert!(called.get());
    }

    #[test]
    fn unknown_vault_and_unknown_capabilities_fail_closed() {
        let runtime = VaultRuntimeSet::from_source(
            "module vaults\nvault ReadOnly\n  capabilities: filesystem_read\n"
        ).unwrap();
        assert!(matches!(
            runtime.execute("Missing", [Capability::Network], || ()),
            Err(VaultRuntimeError::UnknownVault(_))
        ));
        assert!(matches!(
            VaultRuntimeSet::from_source("module vaults\nvault Broken\n  capabilities: hyperdrive\n"),
            Err(VaultRuntimeError::TypedIr(_))
        ));
    }
}
