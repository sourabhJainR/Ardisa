use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn dividing_i64_min_by_negative_one_fails_closed() {
    let source = r#"module int_min_div_negative_one
fn main() -> Int
  let min = 0 - 9223372036854775807 - 1
  let negative_one = 0 - 1
  min / negative_one
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("overflow")),
        "i64::MIN / -1 must report checked overflow rather than panic or wrap"
    );
}

#[test]
fn remainder_i64_min_by_negative_one_fails_closed() {
    let source = r#"module int_min_rem_negative_one
fn main() -> Int
  let min = 0 - 9223372036854775807 - 1
  let negative_one = 0 - 1
  min % negative_one
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("overflow")),
        "i64::MIN % -1 must report checked overflow consistently"
    );
}

#[test]
fn largest_positive_int_is_preserved_through_native_artifact() {
    let source = r#"module int_max_round_trip
fn main() -> Int
  9223372036854775807
"#;
    assert_eq!(
        execute(source).expect("largest positive signed Int should be representable"),
        NativeValue::Int(i64::MAX)
    );
}
