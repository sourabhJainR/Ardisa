use ardisa_core::{
    compile_source, run_program_with_limits, ExecutionLimits, NativeValue,
};
use ardisa_core::native::decode_program;

fn execute(source: &str, args: &[NativeValue]) -> NativeValue {
    let compiled = compile_source(source).expect("source should compile through all compiler phases");
    let program = decode_program(&compiled.artifact).expect("native artifact should round-trip");
    run_program_with_limits(&program, "main", args, ExecutionLimits::default())
        .expect("native execution should succeed")
}

#[test]
fn for_in_sums_owned_list_in_order_and_runs_native_artifact() {
    let source = "module iterator_sum\nfn main(values: List<Int>) -> Int\n  let total = 0\n  for value in values\n    set total = total + value\n  total\n";
    let result = execute(
        source,
        &[NativeValue::List(vec![
            NativeValue::Int(3),
            NativeValue::Int(1),
            NativeValue::Int(4),
            NativeValue::Int(1),
            NativeValue::Int(5),
        ])],
    );
    assert_eq!(result, NativeValue::Int(14));
}

#[test]
fn for_in_empty_list_executes_zero_times() {
    let source = "module iterator_empty\nfn main(values: List<Int>) -> Int\n  let total = 7\n  for value in values\n    set total = total + value\n  total\n";
    assert_eq!(
        execute(source, &[NativeValue::List(vec![])]),
        NativeValue::Int(7)
    );
}

#[test]
fn nested_for_in_loops_execute_with_independent_iteration_state() {
    let source = "module iterator_nested\nfn main(rows: List<List<Int>>) -> Int\n  let total = 0\n  for row in rows\n    for value in row\n      set total = total + value\n  total\n";
    let rows = NativeValue::List(vec![
        NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)]),
        NativeValue::List(vec![NativeValue::Int(3)]),
    ]);
    assert_eq!(execute(source, &[rows]), NativeValue::Int(6));
}

#[test]
fn return_from_inside_loop_exits_function() {
    let source = "module iterator_return\nfn main(values: List<Int>) -> Int\n  for value in values\n    return value\n  99\n";
    assert_eq!(
        execute(source, &[NativeValue::List(vec![NativeValue::Int(8), NativeValue::Int(9)])]),
        NativeValue::Int(8)
    );
    assert_eq!(
        execute(source, &[NativeValue::List(vec![])]),
        NativeValue::Int(99)
    );
}

#[test]
fn for_in_rejects_non_list_iterables() {
    let source = "module iterator_bad\nfn main(value: Int) -> Int\n  for item in value\n    item\n  0\n";
    let error = compile_source(source).expect_err("only List<T> is iterable in Phase 2");
    assert!(format!("{error:?}").contains("AIF321"), "unexpected error: {error:?}");
}

#[test]
fn for_in_consumes_the_iterable_binding() {
    let source = "module iterator_move\nfn main(values: List<Int>) -> Int\n  for value in values\n    value\n  len(values)\n";
    let error = compile_source(source).expect_err("iterating consumes the owned list binding");
    assert!(format!("{error:?}").contains("AIF400"), "unexpected error: {error:?}");
}
