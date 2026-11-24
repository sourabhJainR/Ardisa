use ardisa_core::application::{AppLimits, AppMethod, AppRequest, NativeProgramHandler};
use ardisa_core::native::{decode_program, ExecutionLimits};
use ardisa_core::compile_application_source;

const SOURCE: &str = "module parity\napp parity\n  capability http_listen\n  route POST \"/echo\" echo\nfn echo(method: String, path: String, query: String, body: String) -> String\n  body\n";

#[test]
fn shared_source_returns_same_result_through_native_application_host() {
    let compiled = compile_application_source(SOURCE).expect("shared source compiles");
    let runtime = compiled.definition.build_runtime(AppLimits::default()).unwrap();
    let program = decode_program(&compiled.program.artifact).expect("canonical artifact decodes");
    let handler = NativeProgramHandler::new(program, ExecutionLimits::default()).unwrap();
    let response = runtime.dispatch(
        AppRequest::new(AppMethod::Post, "/echo").with_body(b"same-result".to_vec()),
        &handler,
    ).expect("native application dispatch");
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"same-result");
}
