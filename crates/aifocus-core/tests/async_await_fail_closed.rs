use ardisa_core::pipeline::compile_source;

#[test]
fn async_function_syntax_fails_closed() {
    let source = "module unsupported_async\nasync fn main() -> Int\n  return 1\n";
    assert!(
        compile_source(source).is_err(),
        "async functions must not compile before a Future/executor contract exists"
    );
}

#[test]
fn await_expression_fails_closed_before_native_execution() {
    let source = "module unsupported_await\nfn main() -> Int\n  return await work()\n";
    assert!(
        compile_source(source).is_err(),
        "await must not be accepted without typed IR, runtime suspension, and cancellation semantics"
    );
}
