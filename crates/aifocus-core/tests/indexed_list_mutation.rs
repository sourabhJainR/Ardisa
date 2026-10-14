use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn indexed_list_assignment_updates_the_selected_element() {
    let source = r#"module list_indexed_assignment
fn main() -> Int
  let mut values = [10, 20, 30]
  set values[1] = 99
  values[1]
"#;
    assert_eq!(
        execute(source).expect("valid indexed assignment should execute"),
        NativeValue::Int(99)
    );
}

#[test]
fn indexed_list_assignment_out_of_bounds_fails_closed() {
    let source = r#"module list_indexed_assignment_oob
fn main() -> Int
  let mut values = [10, 20, 30]
  set values[3] = 99
  values[0]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")),
        "out-of-bounds mutation must fail explicitly"
    );
}
