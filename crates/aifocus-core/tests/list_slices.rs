use ardisa_core::{compile_source, native, NativeError, NativeValue};

fn execute(source: &str) -> Result<NativeValue, NativeError> {
    let compiled = compile_source(source).expect("fixture should compile to a checked native artifact");
    let program = native::decode_program(&compiled.artifact).expect("slice instruction must round-trip in artifact encoding");
    native::run_program(&program, "main", &[])
}

#[test]
fn list_slice_returns_half_open_owned_copy() {
    let source = r#"module list_slice_copy
fn main() -> Int
  let values = [10, 20, 30, 40]
  let part = values[1:3]
  set part[0] = 99
  values[1] + values[2] + part[0] + len(part)
"#;
    // Original [20,30] remains independent of slice [99,30].
    assert_eq!(execute(source).unwrap(), NativeValue::Int(20 + 30 + 99 + 2));
}

#[test]
fn list_slice_allows_empty_range_at_end() {
    let source = r#"module list_slice_empty
fn main() -> Int
  let values = [1, 2]
  len(values[2:2])
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(0));
}

#[test]
fn list_slice_allows_full_range() {
    let source = r#"module list_slice_full
fn main() -> Int
  let values = [1, 2, 3]
  len(values[0:3])
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(3));
}

#[test]
fn list_slice_rejects_negative_start() {
    let source = r#"module list_slice_negative_start
fn main() -> Int
  let values = [1, 2, 3]
  let start = 0 - 1
  len(values[start:2])
"#;
    assert!(matches!(execute(source), Err(NativeError::Type(message)) if message.contains("negative slice start")));
}

#[test]
fn list_slice_rejects_negative_end() {
    let source = r#"module list_slice_negative_end
fn main() -> Int
  let values = [1, 2, 3]
  let end = 0 - 1
  len(values[0:end])
"#;
    assert!(matches!(execute(source), Err(NativeError::Type(message)) if message.contains("negative slice end")));
}

#[test]
fn list_slice_rejects_reversed_range() {
    let source = r#"module list_slice_reversed
fn main() -> Int
  let values = [1, 2, 3]
  len(values[2:1])
"#;
    assert!(matches!(execute(source), Err(NativeError::Type(message)) if message.contains("must not exceed")));
}

#[test]
fn list_slice_rejects_end_past_length() {
    let source = r#"module list_slice_oob
fn main() -> Int
  let values = [1, 2, 3]
  len(values[1:4])
"#;
    assert!(matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")));
}

#[test]
fn string_slice_uses_utf8_byte_offsets_and_requires_scalar_boundaries() {
    let valid = r#"module string_slice_utf8_boundaries
fn main() -> Int
  len("aé😀"[1:3])
"#;
    assert_eq!(execute(valid).unwrap(), NativeValue::Int(2), "slice byte offsets 1..3 select the complete two-byte scalar");

    let invalid = r#"module string_slice_splits_scalar
fn main() -> Int
  len("é"[0:1])
"#;
    assert!(
        matches!(execute(invalid), Err(NativeError::Type(message)) if message.contains("UTF-8 scalar boundaries")),
        "byte slices must not create invalid UTF-8 strings"
    );
}

#[test]
fn moving_a_list_binding_rejects_later_use_of_the_source_name() {
    let source = r#"module list_binding_move
fn main() -> Int
  let values = [10, 20]
  let other = values
  set other[0] = 99
  values[0]
"#;
    let error = compile_source(source).expect_err("moving a List binding must not permit later use of the source name");
    assert!(
        format!("{error:?}").contains("use of moved value 'values'"),
        "the ownership gate should diagnose the moved source binding, got: {error:?}"
    );
}

#[test]
fn list_mutation_obeys_runtime_collection_item_limit() {
    let source = r#"module list_collection_limit
fn main() -> Int
  let values = [1]
  push(values, 2)
  len(values)
"#;
    let compiled = compile_source(source).expect("fixture should compile");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_collection_items: 1, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("collection")),
        "append must fail closed when the configured item budget is reached"
    );
}


#[test]
fn slice_instruction_encoding_is_deterministic_across_decode_encode() {
    let source = r#"module deterministic_slice_artifact
fn main() -> Int
  let values = [5, 8, 13, 21]
  len(values[1:3])
"#;
    let compiled = compile_source(source).expect("slice fixture should compile");
    let first = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let encoded_again = native::encode_program(&first);
    assert_eq!(encoded_again, compiled.artifact, "slice instruction encoding must round-trip byte-for-byte");
    let second = native::decode_program(&encoded_again).expect("re-encoded artifact should decode");
    assert_eq!(native::run_program(&first, "main", &[]).unwrap(), NativeValue::Int(2));
    assert_eq!(native::run_program(&second, "main", &[]).unwrap(), NativeValue::Int(2));
}

#[test]
fn iteration_syntax_is_rejected_until_iterator_semantics_exist() {
    let source = r#"module unsupported_for_iteration
fn main() -> Int
  let values = [1, 2, 3]
  for item in values
    item
  0
"#;
    assert!(
        compile_source(source).is_err(),
        "for/in iteration must not be accepted until iterator typing, ownership, and runtime semantics exist"
    );
}

#[test]
fn list_construction_fails_closed_at_configured_collection_limit() {
    let source = r#"module list_item_limit
fn main() -> Int
  let values = [1, 2, 3]
  len(values)
"#;
    let compiled = compile_source(source).expect("fixture should compile before runtime limits are applied");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_collection_items: 2, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("collection")),
        "constructing a list above the item budget must fail with ResourceLimit"
    );
}


#[test]
fn nested_list_slice_is_deeply_independent_after_mutation() {
    let source = r#"module nested_list_slice_copy
fn main() -> Int
  let values = [[1, 2], [3, 4]]
  let part = values[0:1]
  set part[0] = [99, 100]
  values[0][0] + part[0][0] + len(part)
"#;
    assert_eq!(execute(source).unwrap(), NativeValue::Int(1 + 99 + 1));
}

#[test]
fn string_slice_rejects_negative_and_reversed_bounds() {
    let negative = r#"module string_slice_negative
fn main() -> Int
  let start = 0 - 1
  len("abc"[start:2])
"#;
    assert!(
        matches!(execute(negative), Err(NativeError::Type(_))),
        "negative string slice bounds must fail explicitly"
    );

    let reversed = r#"module string_slice_reversed
fn main() -> Int
  len("abc"[2:1])
"#;
    assert!(
        matches!(execute(reversed), Err(NativeError::Type(message)) if message.contains("must not exceed")),
        "reversed string slice bounds must fail explicitly"
    );
}

#[test]
fn string_slice_rejects_end_past_utf8_byte_length() {
    let source = r#"module string_slice_oob
fn main() -> Int
  len("é"[0:3])
"#;
    assert!(
        matches!(execute(source), Err(NativeError::Type(message)) if message.contains("out of bounds")),
        "string slice end is a UTF-8 byte offset and must be within byte length"
    );
}
