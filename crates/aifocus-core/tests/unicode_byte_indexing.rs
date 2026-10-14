use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn multibyte_string_length_counts_utf8_bytes() {
    let source = r#"module multibyte_length
fn main() -> Int
  len("é")
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(2));
}

#[test]
fn string_index_returns_first_utf8_byte_not_unicode_scalar() {
    let source = r#"module utf8_first_byte
fn main() -> Int
  "é"[0]
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(195));
}

#[test]
fn string_index_can_address_utf8_continuation_byte() {
    let source = r#"module utf8_continuation_byte
fn main() -> Int
  "é"[1]
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(169));
}

#[test]
fn supplementary_plane_string_uses_four_utf8_bytes() {
    let source = r#"module emoji_utf8_length
fn main() -> Int
  len("😀")
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(4));
}

#[test]
fn indexing_past_multibyte_utf8_length_fails_closed() {
    let source = r#"module utf8_past_length
fn main() -> Int
  "é"[2]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")),
        "index equal to UTF-8 byte length must fail explicitly"
    );
}
