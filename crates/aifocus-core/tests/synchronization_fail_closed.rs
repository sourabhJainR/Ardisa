use ardisa_core::pipeline::compile_source;

#[test]
fn atomic_operations_fail_closed_until_native_memory_ordering_exists() {
    let source = "module unsupported_atomic\nfn main() -> Int\n  return atomic_load(1)\n";
    assert!(
        compile_source(source).is_err(),
        "atomic operations must not compile without a source-level memory-ordering contract"
    );
}

#[test]
fn lock_operations_fail_closed_until_lock_lifetimes_are_defined() {
    let source = "module unsupported_lock\nfn main() -> Int\n  return mutex_lock(1)\n";
    assert!(
        compile_source(source).is_err(),
        "lock operations must not compile before native lock and cleanup semantics exist"
    );
}

#[test]
fn memory_ordering_names_are_not_silently_accepted_as_operations() {
    let source = "module unsupported_ordering\nfn main() -> Int\n  return atomic_load(1, SeqCst)\n";
    assert!(
        compile_source(source).is_err(),
        "memory-ordering requests must fail closed while no atomic API is supported"
    );
}
