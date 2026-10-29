//! Opt-in executable source-to-native integration for the supported AI-native construct subset.
//!
//! This entry point does not alter pipeline::compile_source or its AIF610 gate.
//! Source operation calls are replaced with unique markers, compiled, then replaced
//! by native instructions at the same bytecode positions. Jump targets are remapped.

use std::collections::{BTreeMap, BTreeSet};
use crate::{
    ast::{Block, Expr, ExprKind, Item, Module, StmtKind},
    construct_operations::ConstructOperationKind,
    native::{self, NativeInstr, NativeProgram, NativeValue, ValidatedNativeProgram},
    pipeline::{CompiledArtifact, PipelineEvidence},
    source_construct_bindings::SourceConstructOperationBinding,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceConstructExecutableError {
    Parse(Vec<crate::source::Diagnostic>),
    Preparation(String),
    Semantic(Vec<crate::source::Diagnostic>),
    Ownership(Vec<crate::source::Diagnostic>),
    Concurrency(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    Native(String),
    InstructionPlacement(String),
    UnsupportedRuntimeBoundary(String),
    CompilerContract(Vec<String>),
}
fn native_error(error: impl std::fmt::Display) -> SourceConstructExecutableError {
    SourceConstructExecutableError::Native(error.to_string())
}

/// Compile Cell Int constants, validated Phase transitions, and Vault checks.
/// Proof and Trace are rejected until their runtime evidence and signing inputs are
/// explicit in the executable contract. The standard compile_source path is unchanged.
pub fn compile_source_with_constructs(source: &str) -> Result<CompiledArtifact, SourceConstructExecutableError> {
    let tokens = crate::token::lex(source).map_err(SourceConstructExecutableError::Parse)?;
    let original = crate::parse(source).map_err(SourceConstructExecutableError::Parse)?;
    let prepared = crate::source_construct_pipeline::prepare_source_constructs(source)
        .map_err(|error| SourceConstructExecutableError::Preparation(format!("{error:?}")))?;
    for binding in &prepared.operation_bindings {
        match &binding.operation.kind {
            ConstructOperationKind::ProofEvaluate { proof } =>
                return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(format!(
                    "Proof '{proof}' requires per-invocation evidence outcomes; refusing to invent or bake evidence"))),
            ConstructOperationKind::TraceEmit { trace, event } =>
                return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(format!(
                    "Trace '{trace}' event '{event}' requires complete event fields and a trusted signing boundary"))),
            _ => {}
        }
    }
    ensure_direct_call_sites(&original, &prepared.operation_bindings)?;

    let markers = prepared.operation_bindings.iter()
        .map(|binding| (binding.call_node_id, i64::MIN + binding.ordinal as i64 + 1))
        .collect::<BTreeMap<_, _>>();
    let mut marked = original.clone();
    marked.constructs.clear();
    let mut seen = BTreeSet::new();
    for item in &mut marked.items {
        let Item::Function(function) = item;
        mark_block(&mut function.body, &markers, &mut seen);
    }
    if seen.len() != prepared.operation_bindings.len() {
        return Err(SourceConstructExecutableError::InstructionPlacement(format!(
            "operation marker mismatch: expected {}, marked {}", prepared.operation_bindings.len(), seen.len())));
    }

    let semantic = crate::sema::analyze(&marked).map_err(SourceConstructExecutableError::Semantic)?;
    let ownership = crate::ownership::analyze(&marked).map_err(SourceConstructExecutableError::Ownership)?;
    crate::concurrency::analyze_with_diagnostics(&marked).map_err(SourceConstructExecutableError::Concurrency)?;
    let effects = crate::effects::analyze(&marked);
    let typed = crate::typed_ir::optimize(
        crate::typed_ir::lower_cfg(&marked).map_err(SourceConstructExecutableError::TypedIr)?);
    let ir_module = crate::optimizer::optimize(
        crate::typed_ir::to_legacy_ir(&typed).map_err(SourceConstructExecutableError::TypedIr)?);
    let mut program = native::compile_program(&ir_module).map_err(native_error)?;
    let injections = build_instruction_injections(&original, &prepared.operation_bindings, &markers)?;
    inject_at_markers(&mut program, &injections, &markers)?;

    let verified = ValidatedNativeProgram::try_new(program).map_err(native_error)?;
    let report = crate::compiler_engine::BackendReport::verify(verified.as_program())
        .map_err(SourceConstructExecutableError::CompilerContract)?;
    let artifact = native::encode_program(verified.as_program());
    debug_assert_eq!(report.artifact_hash,
        crate::compiler_engine::BackendReport::verify(verified.as_program())
            .map(|item| item.artifact_hash).unwrap_or_default());

    Ok(CompiledArtifact {
        artifact: artifact.clone(),
        evidence: PipelineEvidence {
            lexer_tokens: tokens.len(),
            ast_functions: original.items.len(),
            semantic_nodes: semantic.inferred_types.len(),
            ownership_accesses: ownership.accesses.len(),
            effect_functions: effects.functions.len(),
            typed_functions: typed.functions.len(),
            ir_functions: ir_module.functions.len(),
            native_functions: verified.as_program().functions.len(),
            artifact_bytes: artifact.len(),
            phase_order: vec!["lexer", "parser_ast", "construct_binding", "semantic_type",
                "ownership", "structured_concurrency", "effects", "typed_ir", "optimizer",
                "instruction_selection", "native_construct_placement", "ardisa_exec_v1"],
        },
    })
}

/// Decode, validate, and execute the same deterministic combined artifact.
pub fn run_combined_artifact(
    artifact: &str, entry: &str, args: &[NativeValue], limits: native::ExecutionLimits,
) -> Result<NativeValue, SourceConstructExecutableError> {
    let decoded = native::decode_program(artifact).map_err(native_error)?;
    ValidatedNativeProgram::try_new(decoded).map_err(native_error)?
        .run_with_limits(entry, args, limits).map_err(native_error)
}

fn ensure_direct_call_sites(module: &Module, bindings: &[SourceConstructOperationBinding])
    -> Result<(), SourceConstructExecutableError> {
    let mut direct = BTreeSet::new();
    for item in &module.items { let Item::Function(function) = item; collect_direct_sites(&function.body, &mut direct); }
    for binding in bindings {
        if !direct.contains(&binding.call_node_id) {
            return Err(SourceConstructExecutableError::InstructionPlacement(format!(
                "operation at {}..{} must be a direct statement/let/return value; nested operation expressions are unsupported",
                binding.source_span.start, binding.source_span.end)));
        }
    }
    Ok(())
}
fn is_reserved(expr: &Expr) -> bool {
    matches!(&expr.kind, ExprKind::Call { callee, .. } if matches!(&callee.kind,
        ExprKind::Name(name) if matches!(name.as_str(), "cell_initialize" | "cell_update" |
        "phase_transition" | "vault_authorize" | "proof_evaluate" | "trace_emit")))
}
fn collect_direct_sites(block: &Block, sites: &mut BTreeSet<crate::ast::NodeId>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Expr(expr) => {
                if is_reserved(expr) { sites.insert(expr.id); }
                collect_nested_blocks(expr, sites);
            }
            StmtKind::Let { value, .. } | StmtKind::Return(Some(value)) | StmtKind::Set { value, .. } => collect_nested_blocks(value, sites),
            StmtKind::SetIndex { collection, index, value } => {
                collect_nested_blocks(collection, sites); collect_nested_blocks(index, sites); collect_nested_blocks(value, sites);
            }
            StmtKind::Scope { body } => collect_direct_sites(body, sites),
            StmtKind::While { condition, body } => { collect_nested_blocks(condition, sites); collect_direct_sites(body, sites); }
            StmtKind::For { iterable, body, .. } => { collect_nested_blocks(iterable, sites); collect_direct_sites(body, sites); }
            StmtKind::Spawn { call, .. } => collect_nested_blocks(call, sites),
            StmtKind::Return(None) | StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
        }
    }
}
fn collect_nested_blocks(expr: &Expr, sites: &mut BTreeSet<crate::ast::NodeId>) {
    match &expr.kind {
        ExprKind::Binary { left, right, .. } => { collect_nested_blocks(left, sites); collect_nested_blocks(right, sites); }
        ExprKind::Call { callee, args } => { collect_nested_blocks(callee, sites); for arg in args { collect_nested_blocks(arg, sites); } }
        ExprKind::Group(inner) => collect_nested_blocks(inner, sites),
        ExprKind::List(items) => for item in items { collect_nested_blocks(item, sites); },
        ExprKind::Index { collection, index } => { collect_nested_blocks(collection, sites); collect_nested_blocks(index, sites); }
        ExprKind::Slice { collection, start, end } => { collect_nested_blocks(collection, sites); collect_nested_blocks(start, sites); collect_nested_blocks(end, sites); }
        ExprKind::If { condition, then_branch, else_branch } => {
            collect_nested_blocks(condition, sites); collect_direct_sites(then_branch, sites);
            if let Some(branch) = else_branch { collect_direct_sites(branch, sites); }
        }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::String(_) | ExprKind::Name(_) => {}
    }
}
fn mark_block(block: &mut Block, markers: &BTreeMap<crate::ast::NodeId, i64>, seen: &mut BTreeSet<crate::ast::NodeId>) {
    for stmt in &mut block.stmts {
        if let StmtKind::Expr(expr) = &stmt.kind {
            if let Some(marker) = markers.get(&expr.id) {
                let mut value = expr.clone();
                value.kind = ExprKind::Int(*marker);
                seen.insert(value.id);
                stmt.kind = StmtKind::Let { name: format!("__ardisa_operation_marker_{}", marker), value };
                continue;
            }
        }
        match &mut stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Set { value, .. } | StmtKind::Expr(value) | StmtKind::Return(Some(value)) => mark_expr(value, markers, seen),
            StmtKind::SetIndex { collection, index, value } => { mark_expr(collection, markers, seen); mark_expr(index, markers, seen); mark_expr(value, markers, seen); }
            StmtKind::Scope { body } => mark_block(body, markers, seen),
            StmtKind::Spawn { call, .. } => mark_expr(call, markers, seen),
            StmtKind::While { condition, body } => { mark_expr(condition, markers, seen); mark_block(body, markers, seen); }
            StmtKind::For { iterable, body, .. } => { mark_expr(iterable, markers, seen); mark_block(body, markers, seen); }
            StmtKind::Return(None) | StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
        }
    }
}
fn mark_expr(expr: &mut Expr, markers: &BTreeMap<crate::ast::NodeId, i64>, seen: &mut BTreeSet<crate::ast::NodeId>) {
    if let Some(marker) = markers.get(&expr.id) { expr.kind = ExprKind::Int(*marker); seen.insert(expr.id); return; }
    match &mut expr.kind {
        ExprKind::Binary { left, right, .. } => { mark_expr(left, markers, seen); mark_expr(right, markers, seen); }
        ExprKind::Call { callee, args } => { mark_expr(callee, markers, seen); for arg in args { mark_expr(arg, markers, seen); } }
        ExprKind::Group(inner) => mark_expr(inner, markers, seen),
        ExprKind::List(items) => for item in items { mark_expr(item, markers, seen); },
        ExprKind::Index { collection, index } => { mark_expr(collection, markers, seen); mark_expr(index, markers, seen); }
        ExprKind::Slice { collection, start, end } => { mark_expr(collection, markers, seen); mark_expr(start, markers, seen); mark_expr(end, markers, seen); }
        ExprKind::If { condition, then_branch, else_branch } => { mark_expr(condition, markers, seen); mark_block(then_branch, markers, seen); if let Some(branch) = else_branch { mark_block(branch, markers, seen); } }
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::String(_) | ExprKind::Name(_) => {}
    }
}

fn build_instruction_injections(
    module: &Module, bindings: &[SourceConstructOperationBinding], markers: &BTreeMap<crate::ast::NodeId, i64>,
) -> Result<BTreeMap<String, BTreeMap<i64, Vec<NativeInstr>>>, SourceConstructExecutableError> {
    let constructs = crate::typed_ir::lower_constructs(module).map_err(SourceConstructExecutableError::TypedIr)?;
    let mut result: BTreeMap<String, BTreeMap<i64, Vec<NativeInstr>>> = BTreeMap::new();
    let mut cells = BTreeSet::new();
    let mut phases: BTreeMap<(String, String), String> = BTreeMap::new();
    for binding in bindings {
        let marker = *markers.get(&binding.call_node_id).ok_or_else(|| SourceConstructExecutableError::InstructionPlacement("missing source marker".into()))?;
        let operation = &binding.operation;
        let instructions = match &operation.kind {
            ConstructOperationKind::CellInitialize { cell, .. } => {
                if !cells.insert((operation.function.clone(), cell.clone())) {
                    return Err(SourceConstructExecutableError::InstructionPlacement(format!("Cell '{cell}' initialized more than once in '{}'", operation.function)));
                }
                let mut lowered = native::lower_cell_construct_operation(module, operation).map_err(native_error)?;
                lowered.push(NativeInstr::Store(cell_local(cell)));
                lowered.push(NativeInstr::PushUnit);
                lowered
            }
            ConstructOperationKind::CellUpdate { cell, .. } => {
                if !cells.contains(&(operation.function.clone(), cell.clone())) {
                    return Err(SourceConstructExecutableError::InstructionPlacement(format!("Cell '{cell}' must be initialized earlier in '{}'", operation.function)));
                }
                let mut lowered = vec![NativeInstr::Load(cell_local(cell))];
                lowered.extend(native::lower_cell_construct_operation(module, operation).map_err(native_error)?);
                lowered.push(NativeInstr::Store(cell_local(cell)));
                lowered.push(NativeInstr::PushUnit);
                lowered
            }
            ConstructOperationKind::PhaseTransition { phase, from, to } => {
                let declaration = constructs.iter().find(|item| item.name == *phase)
                    .ok_or_else(|| SourceConstructExecutableError::InstructionPlacement(format!("unresolved Phase '{phase}'")))?;
                let key = (operation.function.clone(), phase.clone());
                let mut lowered = if let Some(current) = phases.get(&key) {
                    if current != from { return Err(SourceConstructExecutableError::InstructionPlacement(format!("Phase '{phase}' expected '{from}', preceding transition leaves '{current}'"))); }
                    vec![NativeInstr::Load(phase_local(phase)), NativeInstr::PushString(to.clone()), NativeInstr::PhaseTransition]
                } else {
                    let start = native::phase_start_instruction(declaration).map_err(native_error)?;
                    let initial = match &start { NativeInstr::PhaseStart { graph } => graph.initial.as_str(), _ => return Err(SourceConstructExecutableError::InstructionPlacement("invalid PhaseStart lowering".into())) };
                    if initial != from { return Err(SourceConstructExecutableError::InstructionPlacement(format!("Phase '{phase}' starts at '{initial}', not '{from}'"))); }
                    vec![start, NativeInstr::PushString(to.clone()), NativeInstr::PhaseTransition]
                };
                phases.insert(key, to.clone());
                lowered.push(NativeInstr::Store(phase_local(phase)));
                lowered.push(NativeInstr::PushUnit);
                lowered
            }
            ConstructOperationKind::VaultAuthorize { .. } => native::lower_construct_operation(module, operation).map_err(native_error)?,
            ConstructOperationKind::ProofEvaluate { proof } => return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(format!("Proof '{proof}' requires runtime evidence outcomes"))),
            ConstructOperationKind::TraceEmit { trace, event } => return Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(format!("Trace '{trace}' event '{event}' requires complete signed event fields"))),
        };
        result.entry(operation.function.clone()).or_default().insert(marker, instructions);
    }
    Ok(result)
}
fn cell_local(name: &str) -> String { format!("__ardisa_construct_cell_{name}") }
fn phase_local(name: &str) -> String { format!("__ardisa_construct_phase_{name}") }

fn inject_at_markers(
    program: &mut NativeProgram,
    injections: &BTreeMap<String, BTreeMap<i64, Vec<NativeInstr>>>,
    markers: &BTreeMap<crate::ast::NodeId, i64>,
) -> Result<(), SourceConstructExecutableError> {
    let expected = markers.values().copied().collect::<BTreeSet<_>>();
    for (function_name, function) in &mut program.functions {
        let replacements = injections.get(function_name);
        let mut marker_counts = BTreeMap::<i64, usize>::new();
        for instruction in &function.code {
            if let NativeInstr::PushInt(marker) = instruction {
                if expected.contains(marker) { *marker_counts.entry(*marker).or_default() += 1; }
            }
        }
        for marker in expected.iter().filter(|marker| replacements.and_then(|map| map.get(marker)).is_some()) {
            if marker_counts.get(marker).copied().unwrap_or(0) != 1 {
                return Err(SourceConstructExecutableError::InstructionPlacement(format!(
                    "source marker {marker} must compile to exactly one native instruction in '{function_name}'"
                )));
            }
        }
        let mut offsets = vec![0usize; function.code.len() + 1];
        for (index, instruction) in function.code.iter().enumerate() {
            let length = match instruction {
                NativeInstr::PushInt(marker) if expected.contains(marker) => replacements.and_then(|map| map.get(marker))
                    .ok_or_else(|| SourceConstructExecutableError::InstructionPlacement(format!("marker {marker} in unexpected function '{function_name}'")))?.len(),
                _ => 1,
            };
            offsets[index + 1] = offsets[index].checked_add(length)
                .ok_or_else(|| SourceConstructExecutableError::InstructionPlacement("instruction offset overflow".into()))?;
        }
        let mut rewritten = Vec::with_capacity(*offsets.last().unwrap_or(&0));
        for instruction in &function.code {
            match instruction {
                NativeInstr::PushInt(marker) if expected.contains(marker) => rewritten.extend(
                    replacements.and_then(|map| map.get(marker)).ok_or_else(|| SourceConstructExecutableError::InstructionPlacement(format!("missing replacement for marker {marker}")))?.iter().cloned()),
                NativeInstr::Jump(target) => rewritten.push(NativeInstr::Jump(*offsets.get(*target).ok_or_else(|| SourceConstructExecutableError::InstructionPlacement(format!("invalid jump target {target}")))?)),
                NativeInstr::JumpIfFalse(target) => rewritten.push(NativeInstr::JumpIfFalse(*offsets.get(*target).ok_or_else(|| SourceConstructExecutableError::InstructionPlacement(format!("invalid conditional jump target {target}")))?)),
                _ => rewritten.push(instruction.clone()),
            }
        }
        function.code = rewritten;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = r#"module executable_constructs
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
  cell_initialize("Counter", "value", 1)
  phase_transition("Job", "todo", "running")
  vault_authorize("Local", "filesystem_read")
  cell_update("Counter", "value", 2)
  return 7
"#;
    #[test]
    fn emits_instructions_at_source_sites_and_round_trips_deterministically() {
        let first = compile_source_with_constructs(SOURCE).expect("supported source compiles");
        let second = compile_source_with_constructs(SOURCE).expect("repeat compile");
        assert_eq!(first.artifact, second.artifact);
        assert!(first.artifact.starts_with(native::ARTIFACT_MAGIC));
        let decoded = native::decode_program(&first.artifact).expect("artifact decodes");
        assert_eq!(native::encode_program(&decoded), first.artifact);
        assert_eq!(run_combined_artifact(&first.artifact, "main", &[], native::ExecutionLimits::default()).unwrap(), NativeValue::Int(7));
    }
    #[test]
    fn failed_cell_invariant_stops_execution_before_following_instructions() {
        let source = SOURCE.replace("cell_update(\"Counter\", \"value\", 2)", "cell_update(\"Counter\", \"value\", -1)");
        let compiled = compile_source_with_constructs(&source).expect("constant expression compiles");
        assert!(run_combined_artifact(&compiled.artifact, "main", &[], native::ExecutionLimits::default()).is_err());
    }
    #[test]
    fn proof_and_trace_fail_closed_without_runtime_evidence_inputs() {
        let proof = SOURCE.replace("fn main() -> Int", "proof ReleaseGate\n  requires: bounds\n  ensures: true\n  on_unknown: reject\n  required: true\nfn main() -> Int")
            .replace("cell_initialize(\"Counter\", \"value\", 1)", "proof_evaluate(\"ReleaseGate\")");
        assert!(matches!(compile_source_with_constructs(&proof), Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(_))));
        let trace = SOURCE.replace("fn main() -> Int", "trace Build\n  signature: required\nfn main() -> Int")
            .replace("cell_initialize(\"Counter\", \"value\", 1)", "trace_emit(\"Build\", \"compile\")");
        assert!(matches!(compile_source_with_constructs(&trace), Err(SourceConstructExecutableError::UnsupportedRuntimeBoundary(_))));
    }
    #[test]
    fn standard_compiler_still_rejects_construct_source_with_aif610() {
        assert!(matches!(crate::pipeline::compile_source(SOURCE),
            Err(crate::pipeline::PipelineError::Semantic(diagnostics)) if diagnostics.iter().any(|d| d.code == "AIF610")));
    }
}
