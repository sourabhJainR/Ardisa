//! Acceptance coverage for source-to-runtime and native HTTP host parity.
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::thread;

use ardisa_core::application::{
    AppHandler, AppLimits, AppMethod, AppPolicy, AppRequest, NativeHttpServer,
    NativeProgramHandler, HttpServerConfig, AppCapability,
};
use ardisa_core::native::{decode_program, ExecutionLimits};
use ardisa_core::compile_application_source;

const SOURCE: &str = r#"module parity
app parity
  capability http_listen
  route GET "/" home
  route GET "/query" echo_query
  route POST "/body" echo_body
fn main() -> String
  "shared-result"
fn home(method: String, path: String, query: String, body: String) -> String
  "shared-result"
fn echo_query(method: String, path: String, query: String, body: String) -> String
  query
fn echo_body(method: String, path: String, query: String, body: String) -> String
  body
"#;

fn request(address: SocketAddr, bytes: &[u8]) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(address).expect("connect to loopback test server");
    stream.write_all(bytes).expect("write request");
    let mut response = Vec::new();
    stream.read_to_end(&mut response).expect("read closed response");
    let split = response.windows(4).position(|window| window == b"\r\n\r\n")
        .expect("response header terminator");
    let header = std::str::from_utf8(&response[..split]).expect("HTTP headers are UTF-8");
    let status = header.lines().next().expect("status line")
        .split_whitespace().nth(1).expect("status code")
        .parse::<u16>().expect("numeric status");
    (status, response[split + 4..].to_vec())
}

#[test]
fn compiled_source_dispatch_matches_loopback_http_and_rejects_host_drift() {
    let compiled = compile_application_source(SOURCE).expect("synthetic application compiles");
    let runtime = compiled.definition.build_runtime(AppLimits::default()).expect("runtime");
    let program = decode_program(&compiled.program.artifact).expect("artifact decodes");
    let handler = NativeProgramHandler::new(program, ExecutionLimits::default()).expect("validated handler");

    let direct_root = runtime.dispatch(AppRequest::new(AppMethod::Get, "/"), &handler).expect("root dispatch");
    let direct_query = runtime.dispatch(
        AppRequest::new(AppMethod::Get, "/query").with_query("sample=synthetic"),
        &handler,
    ).expect("query dispatch");
    let direct_body = runtime.dispatch(
        AppRequest::new(AppMethod::Post, "/body").with_body(b"synthetic-body".to_vec()),
        &handler,
    ).expect("body dispatch");
    assert_eq!(direct_root.body, b"shared-result");
    assert_eq!(direct_query.body, b"sample=synthetic");
    assert_eq!(direct_body.body, b"synthetic-body");

    let policy = AppPolicy::allow([AppCapability::HttpListen]);
    let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).expect("loopback bind");
    let address = server.local_addr().expect("server address");
    let app = runtime;
    let host_handler = handler;
    let worker = thread::spawn(move || server.serve_n(&app, &host_handler, 4));

    let host = format!("127.0.0.1:{}", address.port());
    let (status, body) = request(address, format!(
        "GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    ).as_bytes());
    assert_eq!(status, 200);
    assert_eq!(body, direct_root.body);

    let (status, body) = request(address, format!(
        "GET /query?sample=synthetic HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    ).as_bytes());
    assert_eq!(status, 200);
    assert_eq!(body, direct_query.body);

    let (status, body) = request(address, format!(
        "POST /body HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{host}\r\nContent-Length: 14\r\nConnection: close\r\n\r\nsynthetic-body"
    ).as_bytes());
    assert_eq!(status, 200);
    assert_eq!(body, direct_body.body);

    let (status, _) = request(address, b"GET / HTTP/1.1\r\nHost: attacker.example\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 400, "Host drift must be rejected");

    worker.join().expect("server thread").expect("bounded server run");
}
