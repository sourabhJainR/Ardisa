use aifocus_core::{parse, sema};

#[test]
fn native_reference_and_raw_pointer_types_fail_closed() {
    for ty in ["&String", "*String", "&mut String"] {
        let source = format!(
            "module rejected_reference\nfn identity(value: {ty}) -> {ty}\n  value\n"
        );
        assert!(
            parse(&source).is_err(),
            "unsupported reference/pointer type {ty:?} must not enter semantic or native compilation"
        );
    }
}

#[test]
fn reference_returning_functions_fail_closed() {
    let source = "module rejected_return\nfn identity(value: &String) -> &String\n  value\n";
    assert!(
        parse(source).is_err(),
        "a reference-returning function must not be accepted before lifetime escape checking exists"
    );
}

#[test]
fn nested_scope_shadowing_is_rejected_until_bindings_have_stable_identity() {
    let source = "module rejected_shadow\nfn f(value: String) -> String\n  scope\n    let value = \"shadow\"\n  value\n";
    let module = parse(source).expect("scope syntax should parse");
    let errors = sema::check(&module).expect_err("shadowing must fail closed");
    assert!(
        errors.iter().any(|error| error.code == "AIF303"),
        "expected the stable shadowing diagnostic AIF303, got {errors:?}"
    );
}
