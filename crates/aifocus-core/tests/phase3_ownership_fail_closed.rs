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