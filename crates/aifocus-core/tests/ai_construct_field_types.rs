use ardisa_core::{parse, sema};

#[test]
fn ai_construct_fields_reject_unsupported_reference_types() {
    let source = "module invalid_field_type\ncell Secrets\n  payload: SharedRef<String>\n  invariant: present\n";
    let module = parse(source).expect("construct syntax should parse before semantic validation");
    let errors = sema::analyze_constructs(&module)
        .expect_err("unsupported reference types must fail closed inside construct fields");
    assert!(
        errors.iter().any(|error| error.code == "AIF320"),
        "expected AIF320 for unsupported reference field type, got {errors:?}"
    );
}

#[test]
fn ai_construct_fields_validate_numeric_ai_type_parameters() {
    let source = "module invalid_tensor_field\ncell State\n  payload: Tensor<String>\n  invariant: present\n";
    let module = parse(source).expect("construct syntax should parse before semantic validation");
    let errors = sema::analyze_constructs(&module)
        .expect_err("Tensor<String> is not a supported numeric tensor field");
    assert!(
        errors.iter().any(|error| error.code == "AIF320"),
        "expected AIF320 for invalid tensor field type, got {errors:?}"
    );
}
