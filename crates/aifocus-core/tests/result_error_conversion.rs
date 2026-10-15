use ardisa_core::{parse, sema};

#[test]
fn incompatible_result_error_type_has_explicit_diagnostic() {
    let source = "module result_error_mismatch\nfn value() -> Result<Int, String>\n  err(123)\n";
    let module = parse(source).expect("valid source syntax");
    let errors = sema::check(&module).expect_err("Int error payload must not convert implicitly to String");
    assert!(
        errors.iter().any(|error| error.code == "AIF333"),
        "expected AIF333 for incompatible Result error type, got {errors:?}"
    );
    assert!(
        errors.iter().any(|error| error.message.contains("implicit error conversion is unsupported")),
        "diagnostic should state the explicit conversion policy"
    );
}

#[test]
fn result_unwrap_requires_a_typed_result() {
    let valid = parse(
        "module result_unwrap\nfn value(input: Result<Int, String>) -> Int\n  unwrap(input)\n",
    )
    .expect("valid source syntax");
    assert!(sema::check(&valid).is_ok());

    let invalid = parse("module invalid_unwrap\nfn value(input: Int) -> Int\n  unwrap(input)\n")
        .expect("valid source syntax");
    let errors = sema::check(&invalid).expect_err("unwrap must reject non-Result values");
    assert!(
        errors.iter().any(|error| error.code == "AIF329"),
        "expected AIF329, got {errors:?}"
    );
}
