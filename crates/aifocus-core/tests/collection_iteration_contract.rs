use ardisa_core::{compile_source, native, parse, sema, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn while_loop_supports_explicit_index_based_list_iteration() {
    let source = r#"module while_list_iteration
fn main() -> Int
  let values = [2, 3, 4]
  let i = 0
  let total = 0
  while i < len(values)
    set total = total + values[i]
    set i = i + 1
  total
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(9));
}

#[test]
fn for_in_iterator_syntax_is_rejected_until_implemented() {
    let source = r#"module unsupported_for_in
fn main() -> Int
  let values = [1, 2, 3]
  for value in values
    value
  0
"#;
    let module = parse(source).expect("for-in syntax should parse to a targeted diagnostic");
    let errors = sema::analyze(&module).expect_err("for-in must remain unsupported until iterator semantics exist");
    assert!(
        format!("{errors:?}").contains("AIF320"),
        "for-in should be rejected by semantic analysis with AIF320, got {errors:?}"
    );
    assert!(compile_source(source).is_err(), "unsupported for-in must never reach native execution");
}
