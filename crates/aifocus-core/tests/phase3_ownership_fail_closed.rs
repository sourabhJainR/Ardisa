use ardisa_core::{ownership, parse, sema};

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
fn nested_scope_shadowing_preserves_outer_binding_identity() {
    let source = "module nested_shadow\nfn f(value: String) -> String\n  scope\n    let value = \"shadow\"\n  value\n";
    let module = parse(source).expect("scope syntax should parse");
    assert!(sema::check(&module).is_ok(), "shadowing in a nested scope is legal");
    assert!(
        ownership::infer(&module).is_ok(),
        "the nested binding must not overwrite or move the outer binding"
    );
}

#[test]
fn loop_binding_shadowing_preserves_outer_binding_identity() {
    let source = "module loop_shadow\nfn f(values: List<String>, item: String) -> String\n  for item in values\n    len(item)\n  item\n";
    let module = parse(source).expect("source syntax should parse");
    assert!(sema::check(&module).is_ok(), "loop bindings may shadow outer locals");
    assert!(
        ownership::infer(&module).is_ok(),
        "the outer item remains available after the loop-local binding leaves scope"
    );
}

#[test]
fn loop_binding_shadowing_fails_in_both_semantic_and_ownership_passes() {
    let source = "module rejected_loop_shadow\nfn f(values: List<String>, item: String) -> String\n  for item in values\n    len(item)\n  item\n";
    let module = parse(source).expect("source syntax should parse");
    let semantic_errors = sema::check(&module).expect_err("semantic analysis must reject shadowing");
    assert!(semantic_errors.iter().any(|error| error.code == "AIF303"));

    let ownership_errors =
        ownership::infer(&module).expect_err("ownership analysis must independently reject shadowing");
    assert!(ownership_errors.iter().any(|error| error.code == "AIF303"));
}
