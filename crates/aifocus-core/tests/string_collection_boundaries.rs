use ardisa_core::{compile_source, native, parse, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn empty_string_index_fails_at_runtime() {
    let source = r#"module empty_string_index
fn main() -> Int
  ""[0]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")),
        "indexing an empty string must fail explicitly"
    );
}

#[test]
fn string_index_at_byte_length_fails_at_runtime() {
    let source = r#"module string_index_at_length
fn main() -> Int
  "abc"[3]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")),
        "index equal to UTF-8 byte length must fail explicitly"
    );
}

#[test]
fn negative_string_index_fails_at_runtime() {
    let source = r#"module negative_string_index
fn main() -> Int
  let index = 0 - 1
  "abc"[index]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(_))),
        "negative string indexes must not wrap or clamp"
    );
}

#[test]
fn list_index_at_length_fails_at_runtime() {
    let source = r#"module list_index_at_length
fn main() -> Int
  let values = [7]
  values[1]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")),
        "index equal to list length must fail explicitly"
    );
}

#[test]
fn list_slice_syntax_parses_as_a_distinct_expression() {
    use ardisa_core::{ast::ExprKind, sema, Item, StmtKind};
    let source = r#"module parsed_slice
fn main() -> Int
  let values = [1, 2, 3]
  len(values[0:2])
"#;
    let module = parse(source).expect("list slice syntax should parse");
    let Item::Function(function) = &module.items[0];
    let StmtKind::Expr(call) = &function.body.stmts[1].kind else { panic!("expected call expression"); };
    let ExprKind::Call { args, .. } = &call.kind else { panic!("expected len call"); };
    assert!(matches!(args[0].kind, ExprKind::Slice { .. }));
    assert!(sema::analyze(&module).is_ok());
}


#[test]
fn string_payload_fails_closed_at_configured_utf8_byte_limit() {
    let source = r#"module string_byte_limit
fn main() -> Int
  len("éé")
"#;
    let compiled = compile_source(source).expect("fixture should compile before runtime limits are applied");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_string_bytes: 3, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("string")),
        "two two-byte UTF-8 scalars must exceed a three-byte string budget"
    );
}
