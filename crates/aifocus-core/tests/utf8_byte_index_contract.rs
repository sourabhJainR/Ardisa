use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn string_len_counts_utf8_bytes_not_unicode_scalars() {
    let source = r#"module utf8_byte_length
fn main() -> Int
  len("é")
"#;
    assert_eq!(
        execute(source).expect("UTF-8 byte length should execute"),
        NativeValue::Int(2)
    );
}

#[test]
fn string_index_returns_each_utf8_byte_as_an_int() {
    let first_byte = r#"module utf8_first_byte
fn main() -> Int
  "é"[0]
"#;
    let second_byte = r#"module utf8_second_byte
fn main() -> Int
  "é"[1]
"#;

    assert_eq!(
        execute(first_byte).expect("first byte of a UTF-8 string should be indexable"),
        NativeValue::Int(0xC3)
    );
    assert_eq!(
        execute(second_byte).expect("second byte of a UTF-8 string should be indexable"),
        NativeValue::Int(0xA9)
    );
}

#[test]
fn string_index_inside_a_multibyte_scalar_is_not_scalar_indexing() {
    let source = r#"module utf8_byte_not_scalar
fn main() -> Int
  "é"[1]
"#;
    assert_eq!(
        execute(source).expect("byte indexing is the declared behavior"),
        NativeValue::Int(169)
    );
}
