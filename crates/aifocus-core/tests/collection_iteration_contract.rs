use ardisa_core::{compile_source, native, ExecutionLimits, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = ExecutionLimits { max_instructions: 10_000, ..ExecutionLimits::default() };
    native::run_program_with_limits(&program, "main", &[], limits)
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
fn for_in_iterator_executes_through_the_native_backend() {
    let source = r#"module supported_for_in
fn main() -> Int
  let values = [1, 2, 3]
  let total = 0
  for value in values
    set total = total + value
  total
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(6));
}
