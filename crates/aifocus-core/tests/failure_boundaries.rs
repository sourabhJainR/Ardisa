use ardisa_core::{native, pipeline::compile_source};

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
    artifact[0] ^= 0xff;
    assert!(
        native::decode_program(&artifact).is_err(),
        "invalid ARDISA-EXEC-V1 framing must fail before VM execution"
    );
}
