use ardisa_core::{native, pipeline::compile_source, ResourceBudget, ResourceLimits, StructuredScope};

#[test]
fn source_runtime_errors_remain_explicit_native_failures() {
    let source = "module explicit_failure\nfn main() -> Int\n  return 7 / 0\n";
    let compiled = compile_source(source).expect("division-by-zero is a runtime checked failure");
    let program = native::decode_program(&compiled.artifact).expect("artifact should validate");
    assert!(
        matches!(native::run_program(&program, "main", &[]), Err(native::NativeError::Type(_))),
        "runtime failure must remain an explicit NativeError, not a wrapped value"
    );
}

#[test]
fn malformed_native_artifact_is_rejected_before_execution() {
    let source = "module malformed_artifact\nfn main() -> Int\n  return 42\n";
    let compiled = compile_source(source).expect("source should compile");
    let mut artifact = compiled.artifact;
    assert!(!artifact.is_empty());
    artifact.replace_range(0..1, "X");
    assert!(
        native::decode_program(&artifact).is_err(),
        "invalid ARDISA-EXEC-V1 framing must fail before VM execution"
    );
}

#[test]
fn task_panic_is_reported_and_task_budget_is_released() {
    let budget = ResourceBudget::new(ResourceLimits {
        max_tasks: 1,
        max_reserved_bytes: 0,
        max_operations: 0,
    });
    let mut scope = StructuredScope::with_budget(budget.clone());
    scope
        .spawn("broken", |_token| panic!("intentional worker panic for boundary test"))
        .expect("task should start within budget");

    let error = scope
        .finish()
        .expect_err("worker panic must become an explicit task failure");
    assert!(
        error.starts_with("AIF505:"),
        "host panic must not escape as a language-level panic: {error}"
    );
    assert_eq!(
        budget.snapshot().active_tasks,
        0,
        "failed task must release its resource lease"
    );
}

#[test]
fn joined_task_panic_is_reported_and_task_budget_is_released() {
    let budget = ResourceBudget::new(ResourceLimits {
        max_tasks: 1,
        max_reserved_bytes: 0,
        max_operations: 0,
    });
    let mut scope = StructuredScope::with_budget(budget.clone());
    scope
        .spawn("broken", |_token| panic!("intentional joined-worker panic for boundary test"))
        .expect("task should start within budget");

    let error = scope
        .join("broken")
        .expect_err("joining a panicked worker must return an explicit failure");
    assert!(
        error.starts_with("AIF505:"),
        "worker failure should use the stable task-failure diagnostic: {error}"
    );
    assert_eq!(
        budget.snapshot().active_tasks,
        0,
        "joined failed task must release its resource lease"
    );
}

#[test]
fn finish_panic_cancels_siblings_and_releases_all_task_budget() {
    let budget = ResourceBudget::new(ResourceLimits {
        max_tasks: 2,
        max_reserved_bytes: 0,
        max_operations: 0,
    });
    let mut scope = StructuredScope::with_budget(budget.clone());
    scope
        .spawn("a-broken", |_token| panic!("intentional finish-path worker panic"))
        .expect("failing task should start within budget");
    scope
        .spawn("b-cooperative", |token| {
            while !token.is_cancelled() {
                std::thread::yield_now();
            }
        })
        .expect("cooperative sibling should start within budget");

    let error = scope
        .finish()
        .expect_err("finish must report the failing task");
    assert!(
        error.starts_with("AIF505:"),
        "finish must return the stable task-failure diagnostic: {error}"
    );
    assert_eq!(
        budget.snapshot().active_tasks,
        0,
        "finish failure must cancel and join siblings before releasing every task lease"
    );
}


#[test]
fn finish_discovers_later_named_panic_before_joining_cooperative_sibling() {
    let budget = ResourceBudget::new(ResourceLimits {
        max_tasks: 2,
        max_reserved_bytes: 0,
        max_operations: 0,
    });
    let mut scope = StructuredScope::with_budget(budget.clone());
    scope
        .spawn("a-cooperative", |token| {
            while !token.is_cancelled() {
                std::thread::yield_now();
            }
        })
        .expect("cooperative sibling should start within budget");
    scope
        .spawn("z-broken", |_token| panic!("intentional later-named worker panic"))
        .expect("failing task should start within budget");

    let error = scope.finish().expect_err("finish must report the worker panic");
    assert!(error.starts_with("AIF505:"), "unexpected task failure: {error}");
    assert_eq!(budget.snapshot().active_tasks, 0);
}
