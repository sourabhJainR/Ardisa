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
fn slice_syntax_is_rejected_until_slice_semantics_are_implemented() {
    let source = r#"module unsupported_slice
fn main() -> Int
  let values = [1, 2, 3]
  values[0:2]
"#;
    let errors = parse(source).expect_err("slice syntax must not be silently interpreted as indexing");
    assert!(
        errors.iter().any(|error| error.code.starts_with("AIF")),
        "unsupported slice syntax should produce a language diagnostic"
    );
}
