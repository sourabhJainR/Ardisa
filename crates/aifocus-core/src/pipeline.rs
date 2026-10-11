use std::time::Instant;

use crate::{effects, ir, native, optimizer, ownership, sema, typed_ir, concurrency};

/// Deterministic, auditable compiler pipeline used by the native CLI and bootstrap
/// infrastructure. Each phase consumes structured data from the preceding phase;
/// no textual AST/IR serialization is used between phases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineEvidence {
    pub lexer_tokens: usize,
    pub ast_functions: usize,
    pub semantic_nodes: usize,
    pub ownership_accesses: usize,
    pub effect_functions: usize,
    pub typed_functions: usize,
    pub ir_functions: usize,
    pub native_functions: usize,
    pub artifact_bytes: usize,
    pub phase_order: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledArtifact {
    pub artifact: String,
    pub evidence: PipelineEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    Parse(Vec<crate::source::Diagnostic>),
    Semantic(Vec<crate::source::Diagnostic>),
    Ownership(Vec<crate::source::Diagnostic>),
    Concurrency(Vec<crate::source::Diagnostic>),
    TypedIr(Vec<String>),
    Native(native::NativeError),
    CompilerContract(Vec<String>),
}

pub fn compile_source(source: &str) -> Result<CompiledArtifact, PipelineError> {
    let mut phase_order = Vec::new();

    let contract = crate::compiler_engine::validate_source(source).map_err(PipelineError::CompilerContract)?;
    let tokens = crate::token::lex(source).map_err(PipelineError::Parse)?;
    phase_order.push("lexer");

    let module = crate::parse(source).map_err(PipelineError::Parse)?;
    phase_order.push("parser_ast");

    let semantic = sema::analyze(&module).map_err(PipelineError::Semantic)?;
    phase_order.push("semantic_type");

    let ownership_model = ownership::analyze(&module).map_err(PipelineError::Ownership)?;
    phase_order.push("ownership");

    concurrency::analyze_with_diagnostics(&module).map_err(PipelineError::Concurrency)?;
    phase_order.push("structured_concurrency");

    let effect_model = effects::analyze(&module);
    phase_order.push("effects");

    let typed = typed_ir::optimize(typed_ir::lower_cfg(&module).map_err(PipelineError::TypedIr)?);
    phase_order.push("typed_ir");

    let ir_module = optimizer::optimize(typed_ir::to_legacy_ir(&typed));
    phase_order.push("optimizer");

    let native_program = native::compile_program(&ir_module).map_err(PipelineError::Native)?;
    phase_order.push("instruction_selection");

    let artifact = native::encode_program(&native_program);
    phase_order.push("ardisa_exec_v1");
    debug_assert_eq!(contract.artifact_hash, crate::compiler_engine::BackendReport::verify(&native_program).map(|x| x.artifact_hash).unwrap_or_default());

    Ok(CompiledArtifact {
        artifact: artifact.clone(),
        evidence: PipelineEvidence {
            lexer_tokens: tokens.len(),
            ast_functions: module.items.len(),
            semantic_nodes: semantic.inferred_types.len(),
            ownership_accesses: ownership_model.accesses.len(),
            effect_functions: effect_model.functions.len(),
            typed_functions: typed.functions.len(),
            ir_functions: ir_module.functions.len(),
            native_functions: native_program.functions.len(),
            artifact_bytes: artifact.len(),
            phase_order,
        },
    })
}

/// A small timing probe kept separate from correctness evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineTiming {
    pub parse_ns: u64,
    pub semantic_ns: u64,
    pub ownership_ns: u64,
    pub effects_ns: u64,
    pub ir_ns: u64,
    pub native_ns: u64,
}

pub fn time_pipeline(source: &str) -> Result<PipelineTiming, PipelineError> {
    let start = Instant::now();
    let module = crate::parse(source).map_err(PipelineError::Parse)?;
    let parse_ns = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    sema::check(&module).map_err(PipelineError::Semantic)?;
    let semantic_ns = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    ownership::infer(&module).map_err(PipelineError::Ownership)?;
    let ownership_ns = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    effects::analyze(&module);
    let effects_ns = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    let ir = optimizer::optimize(ir::lower(&module));
    let ir_ns = start.elapsed().as_nanos() as u64;

    let start = Instant::now();
    native::compile_program(&ir).map_err(PipelineError::Native)?;
    let native_ns = start.elapsed().as_nanos() as u64;

    Ok(PipelineTiming { parse_ns, semantic_ns, ownership_ns, effects_ns, ir_ns, native_ns })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"module pipeline
fn add(a: Int, b: Int) -> Int
  a + b
fn main() -> Int
  add(20, 22)
"#;

    #[test]
    fn runs_every_structured_phase_in_order() {
        let compiled = compile_source(SOURCE).expect("pipeline should compile");
        assert_eq!(
            compiled.evidence.phase_order,
            vec![
                "lexer",
                "parser_ast",
                "semantic_type",
                "ownership",
                "structured_concurrency",
                "effects",
                "typed_ir",
                "optimizer",
                "instruction_selection",
                "ardisa_exec_v1",
            ]
        );
        assert_eq!(compiled.evidence.ast_functions, 2);
        assert_eq!(compiled.evidence.native_functions, 2);
        assert!(compiled.artifact.starts_with(native::ARTIFACT_MAGIC));
    }

    #[test]
    fn artifact_is_deterministic() {
        let first = compile_source(SOURCE).expect("first compilation");
        let second = compile_source(SOURCE).expect("second compilation");
        assert_eq!(first.artifact, second.artifact);
        assert_eq!(first.evidence, second.evidence);
    }

    #[test]
    fn source_ai_tensor_constructors_and_ops_compile_and_execute() {
        let source = r#"module ai
fn main() -> Tensor<F32>
  let left = tensor([2], [1, 2])
  let right = tensor([2], [3, 4])
  tensor_add(left, right)
"#;
        let artifact = compile_source(source).expect("AI tensor source should compile through the full pipeline");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        let value = native::run_program(&program, "main", &[]).expect("tensor addition should execute");
        assert_eq!(value, native::NativeValue::Tensor { shape: vec![2], values: [4.0f32, 6.0].iter().map(|v| v.to_bits()).collect() });
    }

    #[test]
    fn source_ai_embedding_construction_and_dot_execute_with_domain_guard() {
        let source = r#"module ai
fn main() -> F32
  let left = embedding("text", [1, 2])
  let right = embedding("text", [3, 4])
  embedding_dot(left, right)
"#;
        let artifact = compile_source(source).expect("AI embedding source should compile through the full pipeline");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert_eq!(native::run_program(&program, "main", &[]).unwrap(), native::NativeValue::Float(11.0f32.to_bits()));
    }

    #[test]
    fn source_ai_tensor_and_embedding_invalid_inputs_fail_closed() {
        let bad_shape = r#"module ai
fn main() -> Tensor<F32>
  tensor([0], [1])
"#;
        let artifact = compile_source(bad_shape).expect("shape is runtime-validated");
        let program = native::decode_program(&artifact.artifact).unwrap();
        assert!(native::run_program(&program, "main", &[]).is_err());

        let mismatch = r#"module ai
fn main() -> F32
  let left = embedding("text", [1, 2])
  let right = embedding("image", [3, 4])
  embedding_dot(left, right)
"#;
        let artifact = compile_source(mismatch).expect("domain comparison is runtime-validated");
        let program = native::decode_program(&artifact.artifact).unwrap();
        assert!(matches!(native::run_program(&program, "main", &[]), Err(native::NativeError::Type(_))));
    }
    #[test]
    fn pipeline_artifact_executes() {
        let compiled = compile_source(SOURCE).expect("compile");
        let program = native::decode_program(&compiled.artifact).expect("decode");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("run"),
            native::NativeValue::Int(42)
        );
    }
}
