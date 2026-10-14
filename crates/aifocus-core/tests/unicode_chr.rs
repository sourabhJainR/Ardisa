use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn chr_constructs_non_ascii_unicode_scalar() {
    let source = r#"module chr_non_ascii
fn main() -> String
  chr(233)
"#;
    assert_eq!(
        execute(source).expect("U+00E9 is a valid Unicode scalar"),
        NativeValue::String("é".into())
    );
}

#[test]
fn chr_constructs_supplementary_plane_scalar() {
    let source = r#"module chr_supplementary
fn main() -> String
  chr(128512)
"#;
    assert_eq!(
        execute(source).expect("U+1F600 is a valid Unicode scalar"),
        NativeValue::String("😀".into())
    );
}

#[test]
fn chr_rejects_surrogate_code_points() {
    let source = r#"module chr_surrogate
fn main() -> String
  chr(55296)
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("Unicode scalar")),
        "surrogate code points are not Unicode scalar values"
    );
}

#[test]
fn chr_rejects_values_above_unicode_range() {
    let source = r#"module chr_above_range
fn main() -> String
  chr(1114112)
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("Unicode scalar")),
        "values above U+10FFFF must fail closed"
    );
}

#[test]
fn chr_rejects_negative_values() {
    let source = r#"module chr_negative
fn main() -> String
  let value = 0 - 1
  chr(value)
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("Unicode scalar")),
        "negative values must not wrap to an unsigned scalar"
    );
}
