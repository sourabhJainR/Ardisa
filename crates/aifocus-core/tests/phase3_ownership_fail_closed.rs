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
fn pin_nonnull_and_reference_wrapper_types_fail_closed_semantically() {
    for ty in [
        "Pin<String>",
        "NonNull<String>",
        "RawPtr<String>",
        "SharedRef<String>",
        "MutableRef<String>",
        "Pin",
        "NonNull",
    ] {
        let source = format!(
            "module rejected_pointer_constructor\nfn identity(value: {ty}) -> Int\n  0\n"
        );
        let module = parse(&source)
            .unwrap_or_else(|error| panic!("type spelling {ty:?} should parse for a diagnostic: {error:?}"));
        let errors = sema::check(&module)
            .expect_err("unsupported reference/pointer/pinning constructors must fail semantic validation");
        assert!(
            errors.iter().any(|error| error.code == "AIF320"),
            "type spelling {ty:?} must receive the unsupported native pointer diagnostic: {errors:?}"
        );
    }
}


#[test]
fn nested_reference_like_types_fail_closed_before_native_compilation() {
    for ty in [
        "Result<List<SharedRef<String>>, String>",
        "List<Result<Int, MutableRef<String>>>",
        "Option<List<Pin<String>>>",
    ] {
        let source = format!(
            "module nested_reference_rejected\nfn identity(value: {ty}) -> Int\n  0\n"
        );
        let module = parse(&source)
            .unwrap_or_else(|error| panic!("nested type {ty:?} should parse for semantic rejection: {error:?}"));
        let errors = sema::check(&module)
            .expect_err("reference-like types must be rejected even when nested in supported wrappers");
        assert!(
            errors.iter().any(|error| error.code == "AIF320"),
            "nested reference-like type {ty:?} must report AIF320: {errors:?}"
        );
        assert!(
            ardisa_core::compile_source(&source).is_err(),
            "a rejected nested reference-like type must never produce a native artifact"
        );
    }
}

#[test]
fn partial_move_of_static_index_preserves_disjoint_sibling_access() {
    let source = "module disjoint_partial_move\nfn f(values: List<String>) -> String\n  let first = values[0]\n  values[1]\n";
    let module = parse(source).expect("static index move source should parse");
    assert!(
        ownership::infer(&module).is_ok(),
        "moving one statically known element must not invalidate a disjoint sibling element"
    );
}

#[test]
fn partial_move_rejects_whole_aggregate_reuse() {
    let source = "module aggregate_after_partial_move\nfn f(values: List<String>) -> List<String>\n  let first = values[0]\n  values\n";
    let module = parse(source).expect("partial move source should parse");
    assert!(
        ownership::infer(&module).is_err(),
        "using the aggregate after a non-Copy element was moved must be rejected"
    );
}

#[test]
fn dynamic_index_move_conservatively_blocks_potential_aliases() {
    let source = "module dynamic_partial_move\nfn f(values: List<String>, index: Int) -> String\n  let first = values[index]\n  values[0]\n";
    let module = parse(source).expect("dynamic index move source should parse");
    assert!(
        ownership::infer(&module).is_err(),
        "a dynamic-index move must conservatively invalidate potentially aliased indexed places"
    );
}


#[test]
fn loop_carried_move_rejects_value_after_loop() {
    let source = "module loop_carried_move\nfn f(value: String, condition: Bool) -> String\n  while condition\n    let consumed = value\n  value\n";
    let module = parse(source).expect("loop-carried move source should parse");
    assert!(
        ownership::infer(&module).is_err(),
        "a move on a loop path must not leave the value unconditionally available after the loop"
    );
}

#[test]
fn move_in_nested_scope_does_not_restore_outer_binding() {
    let source = "module scope_move_state\nfn f(value: String) -> String\n  scope\n    let consumed = value\n  value\n";
    let module = parse(source).expect("nested-scope move source should parse");
    assert!(
        ownership::infer(&module).is_err(),
        "leaving a lexical scope must not resurrect an outer binding moved inside it"
    );
}
