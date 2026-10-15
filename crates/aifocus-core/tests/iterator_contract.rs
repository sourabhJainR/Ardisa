use ardisa_core::{
    compile_source, run_program_with_limits, ExecutionLimits, NativeError, NativeValue,
};
use ardisa_core::native::decode_program;

fn execute(source: &str, args: &[NativeValue]) -> NativeValue {
    let compiled = compile_source(source).expect("source should compile through all compiler phases");
    let program = decode_program(&compiled.artifact).expect("native artifact should round-trip");
    assert_eq!(ardisa_core::native::encode_program(&program), compiled.artifact);
    let limits = ExecutionLimits { max_instructions: 10_000, ..ExecutionLimits::default() };
    run_program_with_limits(&program, "main", args, limits)
        .expect("native execution should succeed within the regression-test budget")
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

#[test]
fn iterable_expression_is_emitted_once_before_the_loop() {
    let source = "module iterator_once\nfn source() -> List<Int>\n  [2, 4, 6]\nfn main() -> Int\n  let total = 0\n  for value in source()\n    set total = total + value\n  total\n";
    let compiled = compile_source(source).expect("call expression should compile");
    let program = decode_program(&compiled.artifact).expect("artifact should decode");
    let main = program.functions.get("main").expect("main function should exist");
    let calls = main.code.iter().filter(|instr| {
        matches!(instr, ardisa_core::NativeInstr::Call { callee, .. } if callee == "source")
    }).count();
    assert_eq!(calls, 1, "the iterable expression must be evaluated once");
    let limits = ExecutionLimits { max_instructions: 10_000, ..ExecutionLimits::default() };
    assert_eq!(run_program_with_limits(&program, "main", &[], limits).unwrap(), NativeValue::Int(12));
}

#[test]
fn mutation_of_the_consumed_iterable_is_rejected() {
    let source = "module iterator_alias\nfn main() -> Int\n  let values = [1, 2]\n  let alias = values\n  for item in alias\n    set alias[0] = item\n  0\n";
    let error = compile_source(source).expect_err("a consumed iterable must not be mutable in the loop body");
    assert!(format!("{error:?}").contains("AIF400"), "unexpected error: {error:?}");
}

#[test]
fn iteration_respects_the_vm_instruction_budget() {
    let source = "module iterator_budget\nfn main(values: List<Int>) -> Int\n  let total = 0\n  for value in values\n    set total = total + value\n  total\n";
    let compiled = compile_source(source).expect("bounded loop should compile");
    let program = decode_program(&compiled.artifact).expect("artifact should decode");
    let values = NativeValue::List((0..32).map(NativeValue::Int).collect());
    let limits = ExecutionLimits { max_instructions: 12, ..ExecutionLimits::default() };
    let result = run_program_with_limits(&program, "main", &[values], limits);
    assert!(matches!(result, Err(NativeError::ResourceLimit(_))), "expected instruction-budget failure, got {result:?}");
}
