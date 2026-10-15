use ardisa_core::{compile_source, PipelineError};

#[test]
fn source_iteration_stays_fail_closed_until_full_native_pipeline_is_integrated() {
    let source = "module iterator_gate\nfn sum(values: List<Int>) -> Int\n  for value in values\n    value\n  0\n";
    let error = compile_source(source).expect_err(
        "a parser-level for/in node must not be promoted before typed IR and native execution exist",
    );

    match error {
        PipelineError::CompilerContract(errors) => assert!(
            errors.iter().any(|message| message.contains("for/in iteration cannot enter typed CFG")),
            "expected stable fail-closed iterator diagnostic, got {errors:?}",
        ),
        other => panic!("expected semantic fail-closed error, got {other:?}"),
    }
}
