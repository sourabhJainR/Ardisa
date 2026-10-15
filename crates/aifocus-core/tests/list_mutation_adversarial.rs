use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("compiler artifact should decode");
    native::run_program(&program, "main", &[])
}

#[test]
fn negative_list_index_fails_closed_without_wrapping() {
    let source = r#"module negative_list_index
fn main() -> Int
  let values = [10, 20, 30]
  let index = 0 - 1
  values[index]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("negative index")),
        "negative list indexes must fail explicitly rather than wrap or read from the end"
    );
}

#[test]
fn negative_list_mutation_index_fails_closed_without_mutating_other_elements() {
    let source = r#"module negative_list_mutation_index
fn main() -> Int
  let values = [10, 20, 30]
  let index = 0 - 1
  set values[index] = 99
  values[0]
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("negative list index")),
        "negative indexed assignment must fail explicitly before mutation"
    );
}

#[test]
fn indexed_mutation_changes_only_the_selected_element() {
    let source = r#"module selected_list_mutation
fn main() -> Int
  let values = [10, 20, 30]
  set values[1] = 99
  values[0] + values[1] + values[2]
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(139));
}
