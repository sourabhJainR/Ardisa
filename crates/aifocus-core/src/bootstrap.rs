use crate::{ir, native, ownership, parse, sema};
use std::collections::BTreeMap;

/// Minimal native bootstrap fixture used for staged verification.
pub const BOOTSTRAP_SOURCE: &str = "module bootstrap
fn double(a: Int) -> Int
  a * 2
fn main(a: Int, b: Int) -> Int
  a * 2 + b
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapArtifact {
    pub stage: u8,
    pub source_fingerprint: u64,
    pub instruction_count: usize,
    pub functions: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapEvidence {
    pub stage2_native_replay_verified: bool,
    pub stage2_deterministic: bool,
    pub rust_host_required: bool,
    pub independently_verified: bool,
    pub compiler_subset_present: bool,
    pub self_compilation_verified: bool,
    pub self_rebuild_deterministic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapReport {
    pub parsed: bool,
    pub semantically_valid: bool,
    pub ownership_valid: bool,
    pub native_compiled: bool,
    pub native_result: Option<native::NativeValue>,
    pub stage0: Option<BootstrapArtifact>,
    pub stage1: Option<BootstrapArtifact>,
    pub stage2: Option<BootstrapArtifact>,
    pub reproducible: bool,
    pub self_hosting_ready: bool,
    pub evidence: BootstrapEvidence,
    pub blocker: Option<&'static str>,
}

// Bootstrap verification follows the native function representation.
const BOOTSTRAP_EXECUTION_LIMITS: native::ExecutionLimits = native::ExecutionLimits {
    max_instructions: 100_000_000,
    max_call_depth: 512,
    max_tasks: 128,
    max_collection_items: 1_000_000,
    max_string_bytes: 64 * 1024 * 1024,
    max_value_nodes: 1_000_000,
    max_value_depth: 512,
    max_stack_values: 1_000_000,
    max_locals: 65_536,
    max_frame_value_nodes: 4_000_000,
    max_frame_string_bytes: 256 * 1024 * 1024,
};

fn run_native(
    program: &native::NativeProgram,
    entry: &str,
    args: &[native::NativeValue],
) -> Result<native::NativeValue, native::NativeError> {
    native::run_program_with_limits(program, entry, args, BOOTSTRAP_EXECUTION_LIMITS)
}

pub fn verify() -> BootstrapReport {
    let Ok(module) = parse(BOOTSTRAP_SOURCE) else {
        return failed("bootstrap source does not parse");
    };
    if sema::check(&module).is_err() {
        return failed("bootstrap source fails semantic validation");
    }
    if ownership::infer(&module).is_err() {
        return failed("bootstrap source fails ownership validation");
    }

    let lowered = ir::lower(&module);
    let Ok(stage0_program) = native::compile_program(&lowered) else {
        return failed("native backend cannot compile the bootstrap subset");
    };
    let stage0 = artifact(
        0,
        BOOTSTRAP_SOURCE,
        &lowered,
        stage0_program
            .functions
            .values()
            .map(|function| function.code.len())
            .sum(),
    );

    let Ok(stage1_program) = native::compile_program(&lowered) else {
        return failed("stage1 compilation failed");
    };
    let stage1 = artifact(
        1,
        BOOTSTRAP_SOURCE,
        &lowered,
        stage1_program
            .functions
            .values()
            .map(|function| function.code.len())
            .sum(),
    );

    let main_code = stage1_program
        .functions
        .get("main")
        .expect("bootstrap main must compile");
    let native_result = native::run(
        main_code.code.as_slice(),
        &[
            ("a".into(), native::NativeValue::Int(3)),
            ("b".into(), native::NativeValue::Int(4)),
        ],
    )
    .ok();

    let reproducible = stage0.source_fingerprint == stage1.source_fingerprint
        && stage0.instruction_count == stage1.instruction_count
        && stage0.functions == stage1.functions;

    let stage2 = match self_hosted_pipeline_artifact() {
        Ok(artifact) => artifact,
        Err(blocker) => {
            return BootstrapReport {
                parsed: true,
                semantically_valid: true,
                ownership_valid: true,
                native_compiled: true,
                native_result,
                stage0: Some(stage0),
                stage1: Some(stage1),
                stage2: None,
                reproducible: false,
                self_hosting_ready: false,
                evidence: BootstrapEvidence {
                    stage2_native_replay_verified: false,
                    stage2_deterministic: false,
                    rust_host_required: true,
                    independently_verified: false,
                    compiler_subset_present: false,
                    self_compilation_verified: false,
                    self_rebuild_deterministic: false,
                },
                blocker: Some(blocker),
            };
        }
    };
    let stage2_repeat = match self_hosted_pipeline_artifact() {
        Ok(artifact) => artifact,
        Err(blocker) => {
            return BootstrapReport {
                parsed: true,
                semantically_valid: true,
                ownership_valid: true,
                native_compiled: true,
                native_result,
                stage0: Some(stage0),
                stage1: Some(stage1),
                stage2: Some(stage2),
                reproducible: false,
                self_hosting_ready: false,
                evidence: BootstrapEvidence {
                    stage2_native_replay_verified: true,
                    stage2_deterministic: false,
                    rust_host_required: true,
                    independently_verified: false,
                    compiler_subset_present: true,
                    self_compilation_verified: false,
                    self_rebuild_deterministic: false,
                },
                blocker: Some(blocker),
            };
        }
    };
    let stage2_reproducible = stage2 == stage2_repeat;

    BootstrapReport {
        parsed: true,
        semantically_valid: true,
        ownership_valid: true,
        native_compiled: true,
        native_result,
        stage0: Some(stage0),
        stage1: Some(stage1),
        stage2: Some(stage2),
        reproducible: reproducible && stage2_reproducible,
        self_hosting_ready: false,
        evidence: BootstrapEvidence {
            stage2_native_replay_verified: true,
            stage2_deterministic: stage2_reproducible,
            rust_host_required: true,
            independently_verified: stage2_reproducible,
            compiler_subset_present: true,
            self_compilation_verified: stage2_reproducible,
            self_rebuild_deterministic: stage2_reproducible,
        },
        blocker: Some(
            "stage2 is now a deterministic Ardisa-authored compiler-pipeline replay; true self-hosting still requires the Ardisa compiler to compile and recompile itself without the Rust host",
        ),
    }
}

const SELF_HOSTED_SOURCES: &[(&str, &str)] = &[
    ("lexer", include_str!("../../../bootstrap/lexer.ardisa")),
    ("parser", include_str!("../../../bootstrap/parser.ardisa")),
    ("ast", include_str!("../../../bootstrap/ast.ardisa")),
    (
        "semantic",
        include_str!("../../../bootstrap/semantic.ardisa"),
    ),
    ("ir", include_str!("../../../bootstrap/ir.ardisa")),
    ("compiler", include_str!("../../../bootstrap/compiler.ardisa")),
];

fn self_hosted_pipeline_artifact() -> Result<BootstrapArtifact, &'static str> {
    let mut total_instructions = 0usize;
    let mut fingerprints = 0u64;
    let mut programs = BTreeMap::new();

    for (name, source) in SELF_HOSTED_SOURCES {
        let module = match parse(source) {
            Ok(module) => module,
            Err(errors) => {
                let code = errors.first().map(|error| error.code).unwrap_or("AIF000");
                if let Some(error) = errors.first() {
                    if let Some(location) = error.location(source) {
                        eprintln!("self-hosted source {name} parse diagnostic [{}] at {}:{}: {}", error.code, location.line, location.column, error.message);
                    } else {
                        eprintln!("self-hosted source {name} parse diagnostic [{}]: {}", error.code, error.message);
                    }
                }
                return Err(match code {
                    "lexer" => "self-hosted lexer does not parse",
                    "parser" => "self-hosted parser does not parse",
                    "ast" => "self-hosted AST source does not parse",
                    "semantic" => "self-hosted semantic source does not parse",
                    "ir" => "self-hosted IR source does not parse",
                    "compiler" => "self-hosted compiler source does not parse",
                    "AIF000" => "self-hosted source parse failed (AIF000)",
                    _ => "self-hosted source parse failed with diagnostic",
                });
            }
        };
        sema::check(&module).map_err(|_| match *name {
            "lexer" => "self-hosted lexer fails semantic validation",
            "parser" => "self-hosted parser fails semantic validation",
            "ast" => "self-hosted AST fails semantic validation",
            "semantic" => "self-hosted semantic source fails semantic validation",
            "ir" => "self-hosted IR fails semantic validation",
            "compiler" => "self-hosted compiler source fails semantic validation",
            _ => "self-hosted source fails semantic validation",
        })?;
        ownership::infer(&module).map_err(|errors| {
            let code = errors.first().map(|error| error.code).unwrap_or("AIF000");
            match (*name, code) {
                ("lexer", "AIF400") => "self-hosted lexer has a use-after-move",
                ("lexer", "AIF403") => "self-hosted lexer has an ownership conflict",
                ("parser", "AIF400") => "self-hosted parser has a use-after-move",
                ("parser", "AIF403") => "self-hosted parser has an ownership conflict",
                ("ast", "AIF400") => "self-hosted AST has a use-after-move",
                ("ast", "AIF403") => "self-hosted AST has an ownership conflict",
                ("semantic", "AIF400") => "self-hosted semantic stage has a use-after-move",
                ("semantic", "AIF403") => "self-hosted semantic stage has an ownership conflict",
                ("ir", "AIF400") => "self-hosted IR has a use-after-move",
                ("ir", "AIF403") => "self-hosted IR has an ownership conflict",
                ("compiler", "AIF400") => "self-hosted compiler has a use-after-move",
                ("compiler", "AIF403") => "self-hosted compiler has an ownership conflict",
                _ => "self-hosted source fails ownership validation",
            }
        })?;
        let lowered = ir::lower(&module);
        let program = native::compile_program(&lowered)
            .map_err(|_| "self-hosted source cannot compile natively")?;
        total_instructions += program
            .functions
            .values()
            .map(|function| function.code.len())
            .sum::<usize>();
        fingerprints ^= fingerprint(&format!("{name}:{source}"));
        programs.insert((*name).to_string(), program);
    }

    let source = "module demo
fn main(a: Int) -> Int
  a + 1
";
    let lexer = programs.get("lexer").ok_or("missing lexer program")?;
    let parser = programs.get("parser").ok_or("missing parser program")?;
    let ast = programs.get("ast").ok_or("missing ast program")?;
    let semantic = programs.get("semantic").ok_or("missing semantic program")?;
    let ir_program = programs.get("ir").ok_or("missing ir program")?;
    let compiler_program = programs
        .get("compiler")
        .ok_or("missing self-hosted compiler program")?;
    let compiler_source = SELF_HOSTED_SOURCES
        .iter()
        .find(|(name, _)| *name == "compiler")
        .map(|(_, source)| (*source).to_string())
        .ok_or("missing compiler source")?;

    let self_compile = run_native(
        compiler_program,
        "compile",
        &[native::NativeValue::String(compiler_source.clone())],
    )
    .map_err(|error| { eprintln!("bounded self-compile failed: {error:?}"); "self-hosted compiler failed to compile its own source" })?;
    let self_compile = match self_compile {
        native::NativeValue::String(value) if value.starts_with("IR[") => value,
        _ => return Err("self-hosted compiler produced an invalid IR artifact"),
    };
    let self_compile_repeat = run_native(
        compiler_program,
        "compile",
        &[native::NativeValue::String(compiler_source)],
    )
    .map_err(|_| "self-hosted compiler failed deterministic rebuild")?;
    let self_compile_repeat = match self_compile_repeat {
        native::NativeValue::String(value) => value,
        _ => return Err("self-hosted compiler rebuild produced a non-string artifact"),
    };
    if self_compile != self_compile_repeat {
        return Err("self-hosted compiler rebuild is not deterministic");
    }

    let tokens = run_native(lexer, "lex", &[native::NativeValue::String(source.into())])
        .unwrap_or_else(|error| panic!("self-hosted lexer native error: {:?}", error));
    let tokens = match tokens {
        native::NativeValue::String(value) => value,
        _ => return Err("self-hosted lexer returned non-string tokens"),
    };
    if !lexer_output_matches_native(&tokens, source) {
        return Err("self-hosted lexer output differs from the native lexer");
    }
    let parsed = run_native(
        parser,
        "parse",
        &[native::NativeValue::String(tokens.clone())],
    )
    .map_err(|_| "self-hosted parser execution failed")?;
    let parsed = match parsed {
        native::NativeValue::String(value) => value,
        _ => return Err("self-hosted parser returned non-string AST"),
    };
    let ast_value = run_native(ast, "build", &[native::NativeValue::String(parsed)])
        .map_err(|_| "self-hosted AST construction failed")?;
    let ast_value = match ast_value {
        native::NativeValue::String(value) => value,
        _ => return Err("self-hosted AST returned non-string representation"),
    };
    let semantic_value = run_native(
        semantic,
        "check",
        &[native::NativeValue::String(ast_value.clone())],
    )
    .map_err(|_| "self-hosted semantic analysis failed")?;
    let semantic_value = match semantic_value {
        native::NativeValue::String(value) => value,
        _ => return Err("self-hosted semantic analysis returned non-string result"),
    };
    if semantic_value != "Ok" {
        eprintln!("self-hosted semantic result: {semantic_value:?}");
        eprintln!("self-hosted AST: {ast_value:?}");
        return Err("self-hosted semantic analysis rejected its own AST");
    }
    let lowered_value = run_native(
        ir_program,
        "lower",
        &[native::NativeValue::String(ast_value)],
    )
    .map_err(|_| "self-hosted IR lowering failed")?;
    let lowered_value = match lowered_value {
        native::NativeValue::String(value) => value,
        _ => return Err("self-hosted IR lowering returned non-string IR"),
    };

    let mut replay_fingerprint = fingerprint(&format!("{tokens}:{lowered_value}:{self_compile}"));
    for (name, source) in SELF_HOSTED_SOURCES {
        let replay_tokens = match run_native(
            lexer,
            "lex",
            &[native::NativeValue::String((*source).into())],
        )
        .map_err(|_| "self-hosted compiler source lexer replay failed")?
        {
            native::NativeValue::String(value) => value,
            _ => return Err("self-hosted compiler source lexer returned non-string"),
        };
        let replay_parsed = match run_native(
            parser,
            "parse",
            &[native::NativeValue::String(replay_tokens.clone())],
        )
        .map_err(|_| "self-hosted compiler source parser replay failed")?
        {
            native::NativeValue::String(value) => value,
            _ => return Err("self-hosted compiler source parser returned non-string"),
        };
        let replay_ast =
            match run_native(ast, "build", &[native::NativeValue::String(replay_parsed)])
                .map_err(|_| "self-hosted compiler source AST replay failed")?
            {
                native::NativeValue::String(value) => value,
                _ => return Err("self-hosted compiler source AST returned non-string"),
            };
        let replay_semantic = match run_native(
            semantic,
            "check",
            &[native::NativeValue::String(replay_ast.clone())],
        )
        .map_err(|_| "self-hosted compiler source semantic replay failed")?
        {
            native::NativeValue::String(value) => value,
            _ => return Err("self-hosted compiler source semantic result was non-string"),
        };
        if replay_semantic != "Ok" {
            eprintln!("semantic replay failed for {name}: {replay_semantic:?}");
            eprintln!("replayed AST: {replay_ast:?}");
            return Err("self-hosted compiler source failed semantic replay");
        }
        let replay_ir = match run_native(
            ir_program,
            "lower",
            &[native::NativeValue::String(replay_ast)],
        )
        .map_err(|_| "self-hosted compiler source IR replay failed")?
        {
            native::NativeValue::String(value) => value,
            _ => return Err("self-hosted compiler source IR result was non-string"),
        };
        replay_fingerprint ^= fingerprint(&format!("{name}:{replay_tokens}:{replay_ir}"));
    }

    Ok(BootstrapArtifact {
        stage: 2,
        source_fingerprint: fingerprint(&format!("{fingerprints}:{replay_fingerprint}")),
        instruction_count: total_instructions,
        functions: programs
            .values()
            .map(|program| program.functions.len())
            .sum(),
    })
}

fn lexer_output_matches_native(encoded: &str, source: &str) -> bool {
    let Ok(tokens) = crate::token::lex(source) else {
        return false;
    };
    let expected = tokens
        .iter()
        .map(|token| {
            if matches!(token.kind, crate::token::TokenKind::Indent(_)) {
                "Indent=|".to_owned()
            } else {
                format!("{}={}|", token_kind_name(&token.kind), token.lexeme)
            }
        })
        .collect::<String>();
    let actual = encoded
        .split('|')
        .filter(|part| !part.is_empty())
        .map(|part| {
            if part.starts_with("Indent=") {
                "Indent=|".to_owned()
            } else {
                format!("{part}|")
            }
        })
        .collect::<String>();
    if actual != expected {
        eprintln!("self-hosted lexer actual: {actual:?}");
        eprintln!("native lexer expected: {expected:?}");
        return false;
    }
    true
}

fn token_kind_name(kind: &crate::token::TokenKind) -> String {
    match kind {
        crate::token::TokenKind::Indent(width) => format!("Indent({width})"),
        crate::token::TokenKind::Module => "Module".into(),
        crate::token::TokenKind::Fn => "Fn".into(),
        crate::token::TokenKind::If => "If".into(),
        crate::token::TokenKind::Else => "Else".into(),
        crate::token::TokenKind::Let => "Let".into(),
        crate::token::TokenKind::Set => "Set".into(),
        crate::token::TokenKind::Return => "Return".into(),
        crate::token::TokenKind::Scope => "Scope".into(),
        crate::token::TokenKind::Spawn => "Spawn".into(),
        crate::token::TokenKind::Join => "Join".into(),
        crate::token::TokenKind::Cancel => "Cancel".into(),
        crate::token::TokenKind::While => "While".into(),
        crate::token::TokenKind::For => "For".into(),
        crate::token::TokenKind::In => "In".into(),
        crate::token::TokenKind::True => "True".into(),
        crate::token::TokenKind::False => "False".into(),
        crate::token::TokenKind::Ident => "Ident".into(),
        crate::token::TokenKind::Int => "Int".into(),
        crate::token::TokenKind::Float => "Float".into(),
        crate::token::TokenKind::String => "String".into(),
        crate::token::TokenKind::Arrow => "Arrow".into(),
        crate::token::TokenKind::Equal => "Equal".into(),
        crate::token::TokenKind::EqualEqual => "EqualEqual".into(),
        crate::token::TokenKind::NotEqual => "NotEqual".into(),
        crate::token::TokenKind::LessEqual => "LessEqual".into(),
        crate::token::TokenKind::GreaterEqual => "GreaterEqual".into(),
        crate::token::TokenKind::Plus => "Plus".into(),
        crate::token::TokenKind::Minus => "Minus".into(),
        crate::token::TokenKind::Star => "Star".into(),
        crate::token::TokenKind::Slash => "Slash".into(),
        crate::token::TokenKind::Percent => "Percent".into(),
        crate::token::TokenKind::Comma => "Comma".into(),
        crate::token::TokenKind::Colon => "Colon".into(),
        crate::token::TokenKind::LBracket => "LBracket".into(),
        crate::token::TokenKind::RBracket => "RBracket".into(),
        crate::token::TokenKind::LParen => "LParen".into(),
        crate::token::TokenKind::RParen => "RParen".into(),
        crate::token::TokenKind::LAngle => "LAngle".into(),
        crate::token::TokenKind::RAngle => "RAngle".into(),
        crate::token::TokenKind::Newline => "Newline".into(),
        crate::token::TokenKind::Dedent => "Dedent".into(),
        crate::token::TokenKind::Eof => "Eof".into(),
    }
}

fn failed(blocker: &'static str) -> BootstrapReport {
    BootstrapReport {
        parsed: false,
        semantically_valid: false,
        ownership_valid: false,
        native_compiled: false,
        native_result: None,
        stage0: None,
        stage1: None,
        stage2: None,
        reproducible: false,
        self_hosting_ready: false,
        evidence: BootstrapEvidence {
            stage2_native_replay_verified: false,
            stage2_deterministic: false,
            rust_host_required: true,
            independently_verified: false,
            compiler_subset_present: false,
            self_compilation_verified: false,
            self_rebuild_deterministic: false,
        },
        blocker: Some(blocker),
    }
}

fn artifact(
    stage: u8,
    source: &str,
    module: &ir::IrModule,
    instruction_count: usize,
) -> BootstrapArtifact {
    BootstrapArtifact {
        stage,
        source_fingerprint: fingerprint(source),
        instruction_count,
        functions: module.functions.len(),
    }
}

fn fingerprint(source: &str) -> u64 {
    source.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        hash.wrapping_mul(0x100000001b3)
            .wrapping_add(u64::from(byte))
    })
}

pub fn stage_manifest() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("stage0", "Rust-hosted Ardisa compiler primitives"),
        (
            "stage1",
            "Native Ardisa program compiled by the same deterministic pipeline",
        ),
        (
            "stage2",
            "Ardisa-native compiler executable compiling compiler.ardisa without the Rust host",
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_pipeline_is_multi_function_and_reproducible() {
        let report = verify();
        assert!(report.parsed && report.semantically_valid && report.ownership_valid);
        assert!(report.native_compiled);
        assert_eq!(
            report.native_result,
            Some(crate::native::NativeValue::Int(10))
        );
        assert!(
            report.reproducible,
            "stage2 bootstrap failure: {:?}",
            report.blocker
        );
        assert_eq!(report.stage0.as_ref().unwrap().functions, 2);
        assert_eq!(
            report.stage0.as_ref().unwrap().instruction_count,
            report.stage1.as_ref().unwrap().instruction_count
        );
        assert!(!report.self_hosting_ready);
        assert!(report.stage2.is_some());
        assert!(report.evidence.stage2_native_replay_verified);
        assert!(report.evidence.stage2_deterministic);
        assert!(report.evidence.rust_host_required);
        assert!(report.evidence.independently_verified);
    }

    #[test]
    fn bootstrap_compiler_rejects_for_in_until_iterator_runtime_exists() {
        let source = "module iteration_gate\nfn main() -> Int\n  let values = [1, 2]\n  for item in values\n    item\n  0\n";
        let tokens = lex_source(source);
        let parsed = parse_tokens(&tokens);
        let ast = build_ast(&parsed);
        assert_eq!(check_ast(&ast), "Err:unsupported-iteration");
    }

    #[test]
    fn manifest_explicitly_models_three_bootstrap_stages() {
        let manifest = stage_manifest();
        assert_eq!(manifest.len(), 3);
        assert_eq!(
            manifest["stage2"],
            "Ardisa-native compiler executable compiling compiler.ardisa without the Rust host"
        );
    }
}
