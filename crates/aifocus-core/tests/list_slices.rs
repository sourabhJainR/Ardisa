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
fn for_in_parses_but_is_rejected_by_semantic_analysis() {
    use ardisa_core::{parse, sema};
    let source = r#"module unsupported_for_iteration
fn main() -> Int
  let values = [1, 2, 3]
  for item in values
    item
  0
"#;
    let module = parse(source).expect("for/in syntax should parse to a targeted diagnostic");
    let errors = sema::analyze(&module).expect_err("for/in must be rejected until typed iteration and runtime paths exist");
    assert!(
        format!("{errors:?}").contains("AIF320"),
        "unsupported iteration should fail at semantic analysis with AIF320, got: {errors:?}"
    );
    assert!(
        compile_source(source).is_err(),
        "unsupported for/in syntax must never reach native code generation"
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

#[test]
fn unicode_scalar_apis_are_distinct_from_byte_indexing() {
    let count = r#"module scalar_count
fn main() -> Int
  scalar_len("Aé😀")
"#;
    assert_eq!(execute(count).unwrap(), NativeValue::Int(3));

    let at = r#"module scalar_index
fn main() -> Int
  len(scalar_at("Aé😀", 2))
"#;
    assert_eq!(execute(at).unwrap(), NativeValue::Int(4));

    let negative = r#"module scalar_index_negative
fn main() -> Int
  let index = 0 - 1
  len(scalar_at("Aé", index))
"#;
    assert!(matches!(execute(negative), Err(NativeError::Type(message)) if message.contains("negative Unicode scalar index")));
}

#[test]
fn grapheme_apis_keep_combining_marks_flags_and_zwj_emoji_together() {
    let count = r#"module grapheme_count
fn main() -> Int
  grapheme_len("é🇮🇳👨‍👩‍👧‍👦")
"#;
    assert_eq!(execute(count).unwrap(), NativeValue::Int(3));

    let combining = r#"module grapheme_combining
fn main() -> Int
  len(grapheme_at("éX", 0))
"#;
    assert_eq!(execute(combining).unwrap(), NativeValue::Int(3), "e plus combining acute is one grapheme but three UTF-8 bytes");

    let flag = r#"module grapheme_flag
fn main() -> Int
  len(grapheme_at("A🇮🇳B", 1))
"#;
    assert_eq!(execute(flag).unwrap(), NativeValue::Int(8), "regional-indicator flag is one grapheme");

    let family = r#"module grapheme_family
fn main() -> Int
  len(grapheme_at("👨‍👩‍👧‍👦", 0))
"#;
    assert_eq!(execute(family).unwrap(), NativeValue::Int(25), "ZWJ family emoji is one grapheme");

    let oob = r#"module grapheme_oob
fn main() -> Int
  grapheme_len(grapheme_at("x", 1))
"#;
    assert!(matches!(execute(oob), Err(NativeError::Type(message)) if message.contains("grapheme index out of bounds")));
}


#[test]
fn scalar_and_grapheme_apis_reject_negative_and_out_of_range_indices() {
    let scalar_negative = r#"module scalar_negative_bound
fn main() -> Int
  let index = 0 - 1
  scalar_len(scalar_at("é", index))
"#;
    assert!(
        matches!(execute(scalar_negative), Err(NativeError::Type(message)) if message.contains("negative Unicode scalar index")),
        "negative scalar indexes must fail explicitly"
    );

    let scalar_oob = r#"module scalar_oob_bound
fn main() -> Int
  scalar_len(scalar_at("é", 1))
"#;
    assert!(
        matches!(execute(scalar_oob), Err(NativeError::Type(message)) if message.contains("scalar index out of bounds")),
        "scalar indexes at the scalar count must fail explicitly"
    );

    let grapheme_negative = r#"module grapheme_negative_bound
fn main() -> Int
  let index = 0 - 1
  grapheme_len(grapheme_at("🇮🇳", index))
"#;
    assert!(
        matches!(execute(grapheme_negative), Err(NativeError::Type(message)) if message.contains("negative Unicode grapheme index")),
        "negative grapheme indexes must fail explicitly"
    );

    let grapheme_oob = r#"module grapheme_oob_bound
fn main() -> Int
  grapheme_len(grapheme_at("🇮🇳", 1))
"#;
    assert!(
        matches!(execute(grapheme_oob), Err(NativeError::Type(message)) if message.contains("grapheme index out of bounds")),
        "grapheme indexes at the grapheme count must fail explicitly"
    );
}


#[test]
fn nested_list_slice_is_an_independent_deep_copy() {
    let source = r#"module nested_slice_copy
fn main() -> Int
  let values = [[1, 2], [3, 4]]
  let part = values[0:1]
  set part[0] = [9, 9]
  values[0][0] + part[0][0] + len(values[0]) + len(part[0])
"#;
    assert_eq!(
        execute(source).unwrap(),
        NativeValue::Int(1 + 9 + 2 + 2),
        "mutating a nested slot in the slice must not mutate the source list"
    );
}

#[test]
fn nested_values_are_checked_against_runtime_collection_limits() {
    let source = r#"module nested_collection_limit
fn main() -> Int
  let values = [[1, 2, 3], [4, 5, 6]]
  len(values[0:1])
"#;
    let compiled = compile_source(source).expect("fixture should compile");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_collection_items: 2, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("collection")),
        "nested lists must not bypass configured per-collection item limits"
    );
}

#[test]
fn unicode_instruction_artifact_round_trip_is_byte_deterministic() {
    let source = r#"module unicode_round_trip
fn main() -> Int
  len(scalar_at("Aé😀", 2)) + len(grapheme_at("A🇮🇳B", 1)) + scalar_len("Aé😀") + grapheme_len("é🇮🇳👨‍👩‍👧‍👦")
"#;
    let compiled = compile_source(source).expect("Unicode fixture should compile");
    let first = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let encoded_again = native::encode_program(&first);
    assert_eq!(encoded_again, compiled.artifact, "Unicode instruction encoding must round-trip byte-for-byte");
    let second = native::decode_program(&encoded_again).expect("re-encoded artifact should decode");
    let first_result = native::run_program(&first, "main", &[]).unwrap();
    let second_result = native::run_program(&second, "main", &[]).unwrap();
    assert_eq!(first_result, NativeValue::Int(4 + 8 + 3 + 3));
    assert_eq!(second_result, first_result);
}


#[test]
fn nested_slice_respects_value_depth_budget() {
    let source = r#"module nested_slice_depth
fn main() -> Int
  let values = [[1]]
  len(values[0:1])
"#;
    let compiled = compile_source(source).expect("fixture should compile");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_value_depth: 1, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("depth")),
        "nested list/slice values must be checked against the depth budget"
    );
}

#[test]
fn slice_output_obeys_collection_item_limit() {
    let source = r#"module slice_item_limit
fn main() -> Int
  let values = [1, 2, 3]
  len(values[0:3])
"#;
    let compiled = compile_source(source).expect("fixture should compile");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_collection_items: 2, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("collection")),
        "slice result must be rejected when it exceeds the configured item budget"
    );
}

#[test]
fn artifact_encoding_is_deterministic_after_decode_and_reencode() {
    let source = r#"module deterministic_slice_artifact
fn main() -> Int
  let values = [[1, 2], [3, 4]]
  len(values[0:2])
"#;
    let compiled = compile_source(source).expect("fixture should compile");
    let decoded = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let reencoded = native::encode_program(&decoded);
    assert_eq!(reencoded, compiled.artifact, "native artifacts must round-trip byte-for-byte");
    let decoded_again = native::decode_program(&reencoded).expect("reencoded artifact should decode");
    assert_eq!(
        native::run_program(&decoded, "main", &[]).unwrap(),
        native::run_program(&decoded_again, "main", &[]).unwrap()
    );
}

#[test]
fn utf8_string_slice_requires_scalar_boundaries_and_returns_copy() {
    let valid = r#"module utf8_string_slice
fn main() -> Int
  len("Aé😀"[1:3])
"#;
    assert_eq!(execute(valid).unwrap(), NativeValue::Int(2));

    let split_scalar = r#"module utf8_string_slice_split
fn main() -> Int
  len("Aé😀"[2:3])
"#;
    assert!(
        matches!(execute(split_scalar), Err(NativeError::Type(message)) if message.contains("UTF-8 scalar boundaries")),
        "byte slicing that splits a scalar must fail explicitly"
    );
}


#[test]
fn collection_binding_cannot_create_a_second_mutable_alias() {
    let source = r#"module no_list_alias
fn main() -> Int
  let values = [1, 2]
  let alias = values
  set alias[0] = 99
  values[0]
"#;
    assert!(
        compile_source(source).is_err(),
        "binding an owned list consumes the source name; source code must not gain two mutable aliases"
    );
}

#[test]
fn append_fails_closed_at_the_collection_item_limit() {
    let source = r#"module push_item_limit
fn main() -> Int
  let values = [1, 2]
  push(values, 3)
  len(values)
"#;
    let compiled = compile_source(source).expect("fixture should compile before applying runtime limits");
    let program = native::decode_program(&compiled.artifact).expect("artifact should decode");
    let limits = native::ExecutionLimits { max_collection_items: 2, ..native::ExecutionLimits::default() };
    assert!(
        matches!(native::run_program_with_limits(&program, "main", &[], limits), Err(NativeError::ResourceLimit(message)) if message.contains("collection")),
        "push must not grow a list beyond its configured item budget"
    );
}
