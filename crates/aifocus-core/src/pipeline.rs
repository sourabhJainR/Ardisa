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
  return tensor_add(left, right)
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
  return embedding_dot(left, right)
"#;
        let artifact = compile_source(source).expect("AI embedding source should compile through the full pipeline");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert_eq!(native::run_program(&program, "main", &[]).unwrap(), native::NativeValue::Float(11.0f32.to_bits()));
    }

    #[test]
    fn source_ai_tensor_and_embedding_invalid_inputs_fail_closed() {
        let bad_shape = r#"module ai
fn main() -> Tensor<F32>
  return tensor([0], [1])
"#;
        let artifact = compile_source(bad_shape).expect("shape is runtime-validated");
        let program = native::decode_program(&artifact.artifact).unwrap();
        assert!(native::run_program(&program, "main", &[]).is_err());

        let mismatch = r#"module ai
fn main() -> F32
  let left = embedding("text", [1, 2])
  let right = embedding("image", [3, 4])
  return embedding_dot(left, right)
"#;
        let artifact = compile_source(mismatch).expect("domain comparison is runtime-validated");
        let program = native::decode_program(&artifact.artifact).unwrap();
        assert!(matches!(native::run_program(&program, "main", &[]), Err(native::NativeError::Type(_))));
    }
    #[test]
    fn source_float_literals_flow_through_tensor_construction_and_native_execution() {
        let source = r#"module ai_float
fn main() -> Tensor<F32>
  let left = tensor([2], [1.5, 2.25])
  let right = tensor([2], [2.5, 3.75])
  return tensor_add(left, right)
"#;
        let artifact = compile_source(source).expect("float tensor source should compile");
        let program = native::decode_program(&artifact.artifact).expect("float artifact should decode");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("float tensor should execute"),
            native::NativeValue::Tensor {
                shape: vec![2],
                values: [4.0f32, 6.0].iter().map(|value| value.to_bits()).collect(),
            }
        );
    }

    #[test]
    fn source_float_literals_flow_through_embedding_dot() {
        let source = r#"module ai_float_embedding
fn main() -> F32
  let left = embedding("text", [0.5, 2.0])
  let right = embedding("text", [4.0, 1.5])
  return embedding_dot(left, right)
"#;
        let artifact = compile_source(source).expect("float embedding source should compile");
        let program = native::decode_program(&artifact.artifact).expect("float artifact should decode");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("embedding dot should execute"),
            native::NativeValue::Float(5.0f32.to_bits())
        );
    }

    #[test]
    fn non_finite_float_artifact_payloads_are_rejected() {
        let program = native::NativeProgram {
            functions: [(
                "main".to_string(),
                native::NativeFunction {
                    params: vec![],
                    code: vec![native::NativeInstr::PushFloat(f32::INFINITY.to_bits()), native::NativeInstr::Return],
                },
            )].into_iter().collect(),
        };
        let encoded = native::encode_program(&program);
        assert!(native::decode_program(&encoded).is_err());
    }

    #[test]
    fn source_quantized_tensor_dequantizes_through_native_execution() {
        let source = r#"module quantized_ai
fn main() -> Tensor<F32>
  let packed = quantized_tensor([2], 0.5, 0, [2, 4])
  return dequantize(packed)
"#;
        let artifact = compile_source(source).expect("quantized tensor source should compile");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("dequantization should execute"),
            native::NativeValue::Tensor {
                shape: vec![2],
                values: [1.0f32, 2.0].iter().map(|v| v.to_bits()).collect(),
            }
        );
        let repeated = compile_source(source).expect("repeat compilation should succeed");
        assert_eq!(artifact.artifact, repeated.artifact, "quantized tensor artifact must be deterministic");
    }

    #[test]
    fn source_quantized_tensor_invalid_metadata_and_payload_fail_closed() {
        let invalid_sources = [
            r#"module bad_scale
fn main() -> Tensor<F32>
  return dequantize(quantized_tensor([1], 0.0, 0, [1]))
"#,
            r#"module bad_zero_point
fn main() -> Tensor<F32>
  return dequantize(quantized_tensor([1], 1.0, 128, [1]))
"#,
            r#"module bad_value
fn main() -> Tensor<F32>
  return dequantize(quantized_tensor([1], 1.0, 0, [128]))
"#,
            r#"module bad_shape
fn main() -> Tensor<F32>
  return dequantize(quantized_tensor([2], 1.0, 0, [1]))
"#,
        ];
        for source in invalid_sources {
            let artifact = compile_source(source).expect("metadata and payload are runtime-validated");
            let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
            assert!(native::run_program(&program, "main", &[]).is_err(), "invalid quantized input must fail closed");
        }
    }

    #[test]
    fn source_tensor_matmul_executes_with_rank_two_shape_contract() {
        let source = r#"module ai_matmul
fn main() -> Tensor<F32>
  let left = tensor([2, 2], [1, 2, 3, 4])
  let right = tensor([2, 2], [5, 6, 7, 8])
  return tensor_matmul(left, right)
"#;
        let artifact = compile_source(source).expect("matrix multiplication source should compile");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("matrix multiplication should execute"),
            native::NativeValue::Tensor {
                shape: vec![2, 2],
                values: [19.0f32, 22.0, 43.0, 50.0].iter().map(|v| v.to_bits()).collect(),
            }
        );
    }

    #[test]
    fn source_tensor_matmul_rejects_incompatible_shapes() {
        let source = r#"module ai_bad_matmul
fn main() -> Tensor<F32>
  let left = tensor([2, 3], [1, 2, 3, 4, 5, 6])
  let right = tensor([2, 2], [1, 2, 3, 4])
  return tensor_matmul(left, right)
"#;
        let artifact = compile_source(source).expect("shape compatibility is runtime-validated");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(native::run_program(&program, "main", &[]).is_err(), "incompatible matrix dimensions must fail closed");
    }

    #[test]
    fn source_embedding_dot_rejects_dimension_mismatch() {
        let source = r#"module ai_bad_embedding
fn main() -> F32
  let left = embedding("text", [1.0, 2.0])
  let right = embedding("text", [3.0])
  return embedding_dot(left, right)
"#;
        let artifact = compile_source(source).expect("embedding dimensions are runtime-validated");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(native::run_program(&program, "main", &[]).is_err(), "embedding dimensions must match");
    }

    #[test]
    fn source_tensor_constructor_enforces_element_limit() {
        let source = r#"module tensor_limit
fn main() -> Tensor<F32>
  return tensor([16385], [1])
"#;
        let artifact = compile_source(source).expect("element limits are enforced at runtime");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(
            matches!(
                native::run_program(&program, "main", &[]),
                Err(native::NativeError::ResourceLimit(_))
            ),
            "oversized tensor shapes must fail closed"
        );
    }

    #[test]
    fn source_quantized_tensor_constructor_enforces_element_limit() {
        let source = r#"module quantized_tensor_limit
fn main() -> Tensor<F32>
  return dequantize(quantized_tensor([16385], 1.0, 0, [1]))
"#;
        let artifact = compile_source(source).expect("element limits are enforced at runtime");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(
            matches!(
                native::run_program(&program, "main", &[]),
                Err(native::NativeError::ResourceLimit(_))
            ),
            "oversized quantized tensor shapes must fail closed"
        );
    }

    #[test]
    fn source_tensor_matmul_rejects_non_finite_results() {
        let source = r#"module tensor_matmul_overflow
fn main() -> Tensor<F32>
  let left = tensor([1, 1], [340000000000000000000000000000000000000.0])
  let right = tensor([1, 1], [2.0])
  return tensor_matmul(left, right)
"#;
        let artifact = compile_source(source).expect("finite inputs should compile");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(
            matches!(
                native::run_program(&program, "main", &[]),
                Err(native::NativeError::Type(_))
            ),
            "matmul overflow must not produce a non-finite tensor"
        );
    }

    #[test]
    fn source_embedding_dot_rejects_non_finite_results() {
        let source = r#"module embedding_dot_overflow
fn main() -> F32
  let left = embedding("text", [340000000000000000000000000000000000000.0])
  let right = embedding("text", [2.0])
  return embedding_dot(left, right)
"#;
        let artifact = compile_source(source).expect("finite embeddings should compile");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(
            matches!(
                native::run_program(&program, "main", &[]),
                Err(native::NativeError::Type(_))
            ),
            "embedding dot overflow must fail closed"
        );
    }

    #[test]
    fn source_dequantize_rejects_non_finite_results() {
        let source = r#"module dequantize_overflow
fn main() -> Tensor<F32>
  return dequantize(quantized_tensor([1], 340282300000000000000000000000000000000.0, 0, [127]))
"#;
        let artifact = compile_source(source).expect("finite quantization scale should compile");
        let program = native::decode_program(&artifact.artifact).expect("artifact should decode");
        assert!(
            native::run_program(&program, "main", &[]).is_err(),
            "dequantization overflow must not yield a non-finite tensor"
        );
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

    #[test]
    fn string_indexing_is_explicitly_utf8_byte_based() {
        let source = r#"module unicode_byte_index
fn main() -> Int
  "é"[0]
"#;
        let compiled = compile_source(source).expect("Unicode string index fixture compiles");
        let program = native::decode_program(&compiled.artifact).expect("artifact decodes");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("byte index executes"),
            native::NativeValue::Int(195),
            "string indexing currently returns the first UTF-8 byte, not a Unicode scalar"
        );
    }

    #[test]
    fn list_indexing_returns_values_and_rejects_out_of_bounds() {
        let valid = r#"module list_index_valid
fn main() -> Int
  [10, 20, 30][1]
"#;
        let compiled = compile_source(valid).expect("list index fixture compiles");
        let program = native::decode_program(&compiled.artifact).expect("artifact decodes");
        assert_eq!(
            native::run_program(&program, "main", &[]).expect("valid list index executes"),
            native::NativeValue::Int(20)
        );

        let invalid = r#"module list_index_oob
fn main() -> Int
  [10, 20, 30][3]
"#;
        let compiled = compile_source(invalid).expect("out-of-bounds fixture compiles to runtime check");
        let program = native::decode_program(&compiled.artifact).expect("artifact decodes");
        assert!(
            matches!(
                native::run_program(&program, "main", &[]),
                Err(native::NativeError::Type(message)) if message.contains("out of bounds")
            ),
            "out-of-bounds list access must fail explicitly"
        );
    }

    #[test]
    fn list_push_preserves_source_to_runtime_values_and_supports_indexing() {
        let source = r#"module list_mutation_contract
fn main() -> Int
  let values = [10, 20, 30]
  push(values, 40)
  values[1] + values[3]
"#;
        let compiled = compile_source(source).expect("list mutation fixture compiles");
        let program = native::decode_program(&compiled.artifact).expect("artifact decodes");
        assert_eq!(
            native::run_program(&program, "main", &[]).unwrap_or_else(|error| panic!("list mutation executes: {error:?}; native code: {:?}", program.functions["main"].code)),
            native::NativeValue::Int(60),
            "native code: {:?}", program.functions["main"].code
        );
    }

    #[test]
    fn list_push_updates_length_in_source_to_runtime_execution() {
        let source = r#"module list_push_length_contract
fn main() -> Int
  let values = [10, 20]
  push(values, 30)
  len(values)
"#;
        let compiled = compile_source(source).expect("list push length fixture compiles");
        let program = native::decode_program(&compiled.artifact).expect("artifact decodes");
        assert_eq!(
            native::run_program(&program, "main", &[]).unwrap_or_else(|error| panic!("list push length executes: {error:?}; native code: {:?}", program.functions["main"].code)),
            native::NativeValue::Int(3),
            "native code: {:?}", program.functions["main"].code
        );
    }

}
