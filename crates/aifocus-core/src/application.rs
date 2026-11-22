//! Host-neutral application contracts for Ardisa programs.
//!
//! This module separates deterministic application logic from operating-system
//! and browser APIs. HTTP transports and desktop renderers implement host traits.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};
use crate::native::{ExecutionLimits, NativeProgram, NativeValue, ValidatedNativeProgram};

pub const APP_PROTOCOL_VERSION: u16 = 1;
pub const DEFAULT_MAX_APP_ROUTES: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AppMethod { Get, Post, Put, Patch, Delete, Head, Options }

impl AppMethod {
    pub fn as_str(self) -> &'static str {
        match self { Self::Get => "GET", Self::Post => "POST", Self::Put => "PUT",
            Self::Patch => "PATCH", Self::Delete => "DELETE", Self::Head => "HEAD",
            Self::Options => "OPTIONS" }
    }
    pub fn parse(value: &str) -> Result<Self, AppError> {
        match value { "GET" => Ok(Self::Get), "POST" => Ok(Self::Post), "PUT" => Ok(Self::Put),
            "PATCH" => Ok(Self::Patch), "DELETE" => Ok(Self::Delete), "HEAD" => Ok(Self::Head),
            "OPTIONS" => Ok(Self::Options),
            _ => Err(AppError::InvalidRequest("unsupported HTTP method".into())) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppRequest {
    pub method: AppMethod,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl AppRequest {
    pub fn new(method: AppMethod, path: impl Into<String>) -> Self {
        Self { method, path: path.into(), query: String::new(), headers: Vec::new(), body: Vec::new() }
    }
    pub fn with_query(mut self, query: impl Into<String>) -> Self { self.query = query.into(); self }
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self { self.body = body.into(); self }
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into())); self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl AppResponse {
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self { status, headers: Vec::new(), body: body.into() }
    }
    pub fn text(status: u16, body: impl AsRef<str>) -> Self {
        Self::new(status, body.as_ref().as_bytes().to_vec())
            .with_header("content-type", "text/plain; charset=utf-8")
    }
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into())); self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppLimits {
    pub max_path_bytes: usize,
    pub max_headers: usize,
    pub max_header_bytes: usize,
    pub max_request_body_bytes: usize,
    pub max_response_body_bytes: usize,
    pub max_routes: usize,
}
impl Default for AppLimits {
    fn default() -> Self {
        Self { max_path_bytes: 2_048, max_headers: 64, max_header_bytes: 16_384,
            max_request_body_bytes: 1_048_576, max_response_body_bytes: 1_048_576,
            max_routes: DEFAULT_MAX_APP_ROUTES }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppError {
    InvalidRequest(String), InvalidResponse(String), RequestLimit(String), ResponseLimit(String),
    DuplicateRoute { method: AppMethod, path: String }, RouteLimit, NotFound,
    Handler(String), CapabilityDenied(AppCapability), InvalidUiTree(String), InvalidUiEvent(String),
}
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(m) | Self::InvalidResponse(m) | Self::RequestLimit(m)
            | Self::ResponseLimit(m) | Self::Handler(m) | Self::InvalidUiTree(m)
            | Self::InvalidUiEvent(m) => f.write_str(m),
            Self::DuplicateRoute { method, path } => write!(f, "duplicate route {} {}", method.as_str(), path),
            Self::RouteLimit => f.write_str("application route limit exceeded"),
            Self::NotFound => f.write_str("application route not found"),
            Self::CapabilityDenied(c) => write!(f, "application capability denied: {c:?}"),
        }
    }
}
impl std::error::Error for AppError {}

fn validate_path(path: &str, max_bytes: usize) -> Result<(), AppError> {
    if path.is_empty() || path.len() > max_bytes || !path.starts_with('/') || path.starts_with("//")
        || path.chars().any(|c| matches!(c, '?' | '#' | '\\' | '%'))
        || path.bytes().any(|b| b <= 0x20 || b == 0x7f)
        || path.split('/').any(|part| part == "." || part == "..") {
        return Err(AppError::InvalidRequest(
            "path must be a bounded absolute path without query, fragment, escapes, or traversal segments".into()));
    }
    Ok(())
}
fn valid_header_name(name: &str) -> bool {
    !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}
fn valid_header_value(value: &str) -> bool {
    !value.bytes().any(|b| b == b'\r' || b == b'\n' || (b < 0x20 && b != b'\t') || b == 0x7f)
}
fn validate_headers(headers: &[(String, String)], max_count: usize, max_bytes: usize) -> Result<(), AppError> {
    if headers.len() > max_count { return Err(AppError::RequestLimit("header count limit exceeded".into())); }
    let mut total = 0usize;
    for (name, value) in headers {
        if !valid_header_name(name) || !valid_header_value(value) {
            return Err(AppError::InvalidRequest("header contains an invalid name or unsafe value".into()));
        }
        total = total.checked_add(name.len()).and_then(|n| n.checked_add(value.len()))
            .ok_or_else(|| AppError::RequestLimit("header byte accounting overflow".into()))?;
        if total > max_bytes { return Err(AppError::RequestLimit("header byte limit exceeded".into())); }
    }
    Ok(())
}

pub trait AppHandler {
    fn invoke(&self, entry: &str, request: &AppRequest) -> Result<AppResponse, AppError>;
}

#[derive(Debug, Clone)]
pub struct AppRuntime { routes: BTreeMap<(AppMethod, String), String>, limits: AppLimits }
impl AppRuntime {
    pub fn new(limits: AppLimits) -> Self { Self { routes: BTreeMap::new(), limits } }
    pub fn limits(&self) -> AppLimits { self.limits }
    pub fn route(&mut self, method: AppMethod, path: impl Into<String>, entry: impl Into<String>) -> Result<(), AppError> {
        let path = path.into();
        validate_path(&path, self.limits.max_path_bytes)?;
        let entry = entry.into();
        if entry.is_empty() || !entry.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(AppError::InvalidRequest("route handler must be a valid Ardisa function name".into()));
        }
        if self.routes.contains_key(&(method, path.clone())) {
            return Err(AppError::DuplicateRoute { method, path });
        }
        if self.routes.len() >= self.limits.max_routes { return Err(AppError::RouteLimit); }
        self.routes.insert((method, path), entry);
        Ok(())
    }
    pub fn route_count(&self) -> usize { self.routes.len() }
    pub fn dispatch(&self, request: AppRequest, handler: &dyn AppHandler) -> Result<AppResponse, AppError> {
        self.validate_request(&request)?;
        let entry = self.routes.get(&(request.method, request.path.clone())).ok_or(AppError::NotFound)?;
        let response = handler.invoke(entry, &request)?;
        self.validate_response(&response)?;
        Ok(response)
    }
    fn validate_request(&self, request: &AppRequest) -> Result<(), AppError> {
        validate_path(&request.path, self.limits.max_path_bytes)?;
        if request.query.len() > self.limits.max_path_bytes || request.query.bytes().any(|b| b <= 0x20 || b == b'\r' || b == b'\n' || b == b'#' || b == 0x7f) {
            return Err(AppError::InvalidRequest("query string is invalid or exceeds byte limit".into()));
        }
        validate_headers(&request.headers, self.limits.max_headers, self.limits.max_header_bytes)?;
        if request.body.len() > self.limits.max_request_body_bytes {
            return Err(AppError::RequestLimit("request body byte limit exceeded".into()));
        }
        Ok(())
    }
    fn validate_response(&self, response: &AppResponse) -> Result<(), AppError> {
        if !(200..=599).contains(&response.status) {
            return Err(AppError::InvalidResponse("response status must be between 200 and 599".into()));
        }
        if matches!(response.status, 204 | 205 | 304) && !response.body.is_empty() {
            return Err(AppError::InvalidResponse("204, 205 and 304 responses cannot include a body".into()));
        }
        validate_headers(&response.headers, self.limits.max_headers, self.limits.max_header_bytes)
            .map_err(|e| AppError::InvalidResponse(e.to_string()))?;
        if response.headers.iter().any(|(name, _)| matches!(name.to_ascii_lowercase().as_str(), "content-length" | "transfer-encoding" | "connection")) {
            return Err(AppError::InvalidResponse("response cannot override transport framing headers".into()));
        }
        if response.body.len() > self.limits.max_response_body_bytes {
            return Err(AppError::ResponseLimit("response body byte limit exceeded".into()));
        }
        Ok(())
    }
}

/// Bridge to the current native VM. Handler functions take three String
/// parameters (method, path, query, body) and return a String response body.
pub struct NativeProgramHandler { program: ValidatedNativeProgram, execution_limits: ExecutionLimits }
impl NativeProgramHandler {
    pub fn new(program: NativeProgram, execution_limits: ExecutionLimits) -> Result<Self, AppError> {
        let program = ValidatedNativeProgram::try_new(program)
            .map_err(|error| AppError::Handler(format!("{error:?}")))?;
        Ok(Self { program, execution_limits })
    }
}
impl AppHandler for NativeProgramHandler {
    fn invoke(&self, entry: &str, request: &AppRequest) -> Result<AppResponse, AppError> {
        let body = std::str::from_utf8(&request.body)
            .map_err(|_| AppError::InvalidRequest("request body must be UTF-8 for application ABI v1".into()))?;
        let args = [
            NativeValue::String(request.method.as_str().to_owned()),
            NativeValue::String(request.path.clone()),
            NativeValue::String(request.query.clone()),
            NativeValue::String(body.to_owned()),
        ];
        match self.program.run_with_limits(entry, &args, self.execution_limits)
            .map_err(|error| AppError::Handler(format!("{error:?}")))? {
            NativeValue::String(body) => Ok(AppResponse::text(200, body)),
            _ => Err(AppError::Handler("Ardisa HTTP handler must return String in application ABI v1".into())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AppCapability { HttpListen, DesktopWindow, BrowserDom, FilesystemRead, FilesystemWrite, NetworkClient, Clock, Process }
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AppPolicy { allowed: BTreeSet<AppCapability> }
impl AppPolicy {
    pub fn allow(capabilities: impl IntoIterator<Item = AppCapability>) -> Self {
        Self { allowed: capabilities.into_iter().collect() }
    }
    pub fn require(&self, capability: AppCapability) -> Result<(), AppError> {
        if self.allowed.contains(&capability) { Ok(()) } else { Err(AppError::CapabilityDenied(capability)) }
    }
    pub fn allows(&self, capability: AppCapability) -> bool { self.allowed.contains(&capability) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiNodeKind { Window, Container, Label, Button, TextInput }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiNode {
    pub id: u32, pub parent: Option<u32>, pub kind: UiNodeKind,
    pub accessible_name: String, pub text: String, pub enabled: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiTree { root: u32, nodes: BTreeMap<u32, UiNode>, max_nodes: usize }
impl UiTree {
    pub fn new(root: UiNode, max_nodes: usize) -> Result<Self, AppError> {
        if root.kind != UiNodeKind::Window || root.parent.is_some() || max_nodes == 0 {
            return Err(AppError::InvalidUiTree("root must be a Window without a parent and a non-zero node limit".into()));
        }
        validate_ui_text(&root)?;
        let root_id = root.id;
        let mut nodes = BTreeMap::new();
        nodes.insert(root_id, root);
        Ok(Self { root: root_id, nodes, max_nodes })
    }
    pub fn root_id(&self) -> u32 { self.root }
    pub fn nodes(&self) -> &BTreeMap<u32, UiNode> { &self.nodes }
    pub fn add(&mut self, node: UiNode) -> Result<(), AppError> {
        validate_ui_text(&node)?;
        if node.kind == UiNodeKind::Window { return Err(AppError::InvalidUiTree("only one Window root is allowed".into())); }
        let parent = node.parent.ok_or_else(|| AppError::InvalidUiTree("non-root nodes require a parent".into()))?;
        let parent_node = self.nodes.get(&parent).ok_or_else(|| AppError::InvalidUiTree("UI parent does not exist".into()))?;
        if !matches!(parent_node.kind, UiNodeKind::Window | UiNodeKind::Container) {
            return Err(AppError::InvalidUiTree("UI parent must be a Window or Container".into()));
        }
        if self.nodes.contains_key(&node.id) { return Err(AppError::InvalidUiTree("duplicate UI node id".into())); }
        if self.nodes.len() >= self.max_nodes { return Err(AppError::InvalidUiTree("UI node limit exceeded".into())); }
        self.nodes.insert(node.id, node);
        Ok(())
    }
    pub fn validate_event(&self, event: &UiEvent) -> Result<(), AppError> {
        match (event.node_id, &event.kind) {
            (None, UiEventKind::WindowClosed) => Ok(()),
            (Some(_), UiEventKind::WindowClosed) | (None, _) =>
                Err(AppError::InvalidUiEvent("window-close events must not target a control".into())),
            (Some(id), kind) => {
                let node = self.nodes.get(&id).ok_or_else(|| AppError::InvalidUiEvent("event targets unknown UI node".into()))?;
                if !node.enabled { return Err(AppError::InvalidUiEvent("event targets disabled UI node".into())); }
                match (node.kind, kind) {
                    (UiNodeKind::Button, UiEventKind::Clicked) => Ok(()),
                    (UiNodeKind::TextInput, UiEventKind::TextChanged(value)) if value.len() <= 4_096 => Ok(()),
                    (UiNodeKind::TextInput, UiEventKind::TextChanged(_)) =>
                        Err(AppError::InvalidUiEvent("text input event exceeds byte limit".into())),
                    _ => Err(AppError::InvalidUiEvent("event kind does not match its UI control".into())),
                }
            }
        }
    }
    pub fn apply_action(&mut self, action: &UiAction) -> Result<bool, AppError> {
        match action {
            UiAction::CloseWindow => Ok(true),
            UiAction::SetText { node_id, text } => {
                if text.len() > 4_096 {
                    return Err(AppError::InvalidUiTree("UI action text exceeds byte limit".into()));
                }
                let node = self.nodes.get_mut(node_id)
                    .ok_or_else(|| AppError::InvalidUiTree("UI action targets unknown node".into()))?;
                if !matches!(node.kind, UiNodeKind::Label | UiNodeKind::Button | UiNodeKind::TextInput) {
                    return Err(AppError::InvalidUiTree("text action requires a text-capable control".into()));
                }
                node.text = text.clone();
                Ok(false)
            }
            UiAction::SetEnabled { node_id, enabled } => {
                let node = self.nodes.get_mut(node_id)
                    .ok_or_else(|| AppError::InvalidUiTree("UI action targets unknown node".into()))?;
                if node.kind == UiNodeKind::Window {
                    return Err(AppError::InvalidUiTree("window enabled state is controlled by the host".into()));
                }
                node.enabled = *enabled;
                Ok(false)
            }
        }
    }
}
fn validate_ui_text(node: &UiNode) -> Result<(), AppError> {
    if node.accessible_name.trim().is_empty() || node.accessible_name.len() > 512 || node.text.len() > 4_096 {
        return Err(AppError::InvalidUiTree("accessible name and UI text must be bounded".into()));
    }
    Ok(())
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiEventKind { WindowClosed, Clicked, TextChanged(String) }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiEvent { pub node_id: Option<u32>, pub kind: UiEventKind }
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiAction { SetText { node_id: u32, text: String }, SetEnabled { node_id: u32, enabled: bool }, CloseWindow }

/// OS-specific native renderers implement this interface. The UI tree and event
/// model remain shared across Windows, macOS and Linux implementations.
pub trait DesktopHost {
    fn present(&mut self, tree: &UiTree) -> Result<(), AppError>;
    fn next_event(&mut self) -> Result<Option<UiEvent>, AppError>;
    fn apply(&mut self, action: &UiAction) -> Result<(), AppError>;
}


#[derive(Debug, Clone, Copy)]
pub struct HttpServerConfig {
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub max_request_duration: Duration,
    pub max_header_bytes: usize,
    pub max_body_bytes: usize,
}
impl Default for HttpServerConfig {
    fn default() -> Self {
        Self { read_timeout: Duration::from_secs(5), write_timeout: Duration::from_secs(5),
            max_request_duration: Duration::from_secs(15), max_header_bytes: 16_384, max_body_bytes: 1_048_576 }
    }
}

/// Minimal bounded HTTP/1.x host for local Ardisa services. It serves one
/// connection at a time, closes each connection, and rejects chunked framing.
pub struct NativeHttpServer { listener: TcpListener, config: HttpServerConfig }
impl NativeHttpServer {
    pub fn bind(address: &str, policy: &AppPolicy, config: HttpServerConfig) -> Result<Self, AppError> {
        policy.require(AppCapability::HttpListen)?;
        if config.max_header_bytes == 0 || config.max_body_bytes == 0
            || config.read_timeout.is_zero() || config.write_timeout.is_zero()
            || config.max_request_duration.is_zero() {
            return Err(AppError::InvalidRequest("HTTP server limits and timeouts must be non-zero".into()));
        }
        let socket_address = address.parse::<SocketAddr>().map_err(|_| {
            AppError::InvalidRequest(
                "native HTTP server requires a numeric loopback socket address".into(),
            )
        })?;
        if !socket_address.ip().is_loopback() {
            return Err(AppError::InvalidRequest(
                "native HTTP server accepts loopback addresses only until authenticated TLS is supported".into(),
            ));
        }
        let listener = TcpListener::bind(socket_address)
            .map_err(|error| AppError::Handler(format!("HTTP bind failed: {error}")))?;
        Ok(Self { listener, config })
    }
    pub fn local_addr(&self) -> Result<std::net::SocketAddr, AppError> {
        self.listener.local_addr().map_err(|error| AppError::Handler(format!("HTTP local address failed: {error}")))
    }
    pub fn serve_once(&self, app: &AppRuntime, handler: &dyn AppHandler) -> Result<(), AppError> {
        let (stream, _) = self.listener.accept()
            .map_err(|error| AppError::Handler(format!("HTTP accept failed: {error}")))?;
        self.serve_stream(stream, app, handler)
    }
    pub fn serve_forever(&self, app: &AppRuntime, handler: &dyn AppHandler) -> Result<(), AppError> {
        self.serve_connections(app, handler, None)
    }
    /// Serve a bounded number of connections. Per-client I/O failures are
    /// isolated so a disconnected client cannot stop the listener.
    pub fn serve_n(&self, app: &AppRuntime, handler: &dyn AppHandler, count: usize) -> Result<(), AppError> {
        self.serve_connections(app, handler, Some(count))
    }
    fn serve_connections(&self, app: &AppRuntime, handler: &dyn AppHandler, limit: Option<usize>) -> Result<(), AppError> {
        let mut served = 0usize;
        while limit.is_none_or(|max| served < max) {
            let (stream, _) = self.listener.accept()
                .map_err(|error| AppError::Handler(format!("HTTP accept failed: {error}")))?;
            served += 1;
            let _ = self.serve_stream(stream, app, handler);
        }
        Ok(())
    }
    fn serve_stream(&self, mut stream: TcpStream, app: &AppRuntime, handler: &dyn AppHandler) -> Result<(), AppError> {
        stream.set_read_timeout(Some(self.config.read_timeout.min(self.config.max_request_duration)))
            .map_err(|error| AppError::Handler(format!("HTTP read timeout setup failed: {error}")))?;
        stream.set_write_timeout(Some(self.config.write_timeout))
            .map_err(|error| AppError::Handler(format!("HTTP write timeout setup failed: {error}")))?;
        let local_addr = self.local_addr()?;
        let mut head_only = false;
        let response = match read_http_request(&mut stream, self.config, local_addr) {
            Ok(request) => {
                head_only = request.method == AppMethod::Head;
                match app.dispatch(request, handler) { Ok(response) => response, Err(error) => error_response(&error) }
            },
            Err(error) => error_response(&error),
        };
        write_http_response(&mut stream, &response, head_only)
    }
}

fn read_http_request(stream: &mut TcpStream, config: HttpServerConfig, listener_address: SocketAddr) -> Result<AppRequest, AppError> {
    let deadline = Instant::now().checked_add(config.max_request_duration)
        .ok_or_else(|| AppError::InvalidRequest("HTTP request deadline is out of range".into()))?;
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        if Instant::now() >= deadline {
            return Err(AppError::RequestLimit("HTTP request read deadline exceeded".into()));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(AppError::RequestLimit("HTTP request read deadline exceeded".into()));
        }
        stream.set_read_timeout(Some(config.read_timeout.min(remaining)))
            .map_err(|error| AppError::Handler(format!("HTTP read deadline setup failed: {error}")))?;
        stream.read_exact(&mut byte).map_err(|error| AppError::InvalidRequest(format!("incomplete HTTP headers: {error}")))?;
        if Instant::now() >= deadline {
            return Err(AppError::RequestLimit("HTTP request read deadline exceeded".into()));
        }
        header.push(byte[0]);
        if header.len() > config.max_header_bytes {
            return Err(AppError::RequestLimit("HTTP header byte limit exceeded".into()));
        }
        if header.ends_with(b"\r\n\r\n") { break; }
    }
    let text = std::str::from_utf8(&header[..header.len() - 4])
        .map_err(|_| AppError::InvalidRequest("HTTP headers must be ASCII/UTF-8".into()))?;
    if !text.is_ascii() { return Err(AppError::InvalidRequest("HTTP headers must be ASCII".into())); }
    let mut lines = text.split("\r\n");
    let request_line = lines.next().ok_or_else(|| AppError::InvalidRequest("missing HTTP request line".into()))?;
    let parts = request_line.split(' ').collect::<Vec<_>>();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err(AppError::InvalidRequest("malformed HTTP request line".into()));
    }
    if parts[2] != "HTTP/1.1" && parts[2] != "HTTP/1.0" {
        return Err(AppError::InvalidRequest("only HTTP/1.0 and HTTP/1.1 are supported".into()));
    }
    let method = AppMethod::parse(parts[0])?;
    let (path, query) = match parts[1].split_once('?') {
        Some((path, query)) => (path, query),
        None => (parts[1], ""),
    };
    if parts[1].contains('#') {
        return Err(AppError::InvalidRequest("fragments are not valid in HTTP request targets".into()));
    }
    let mut headers = Vec::new();
    let mut content_length = None;
    let mut host_count = 0usize;
    let mut origin_count = 0usize;
    let mut fetch_site_count = 0usize;
    let mut transfer_encoding = false;
    for line in lines {
        let (name, value) = line.split_once(':')
            .ok_or_else(|| AppError::InvalidRequest("malformed HTTP header".into()))?;
        if name.trim() != name {
            return Err(AppError::InvalidRequest("whitespace before HTTP header colon is not allowed".into()));
        }
        let value = value.trim();
        if !valid_header_name(name) || !valid_header_value(value) {
            return Err(AppError::InvalidRequest("invalid HTTP header name or value".into()));
        }
        match name.to_ascii_lowercase().as_str() {
            "content-length" => {
                if content_length.is_some() { return Err(AppError::InvalidRequest("duplicate Content-Length is not allowed".into())); }
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(AppError::InvalidRequest("invalid Content-Length".into()));
                }
                let length = value.parse::<usize>().map_err(|_| AppError::InvalidRequest("invalid Content-Length".into()))?;
                if length > config.max_body_bytes { return Err(AppError::RequestLimit("HTTP request body byte limit exceeded".into())); }
                content_length = Some(length);
            }
            "transfer-encoding" => transfer_encoding = true,
            "host" => host_count += 1,
            "origin" => origin_count += 1,
            "sec-fetch-site" => fetch_site_count += 1,
            "expect" => return Err(AppError::InvalidRequest("Expect request framing is not supported".into())),
            _ => {}
        }
        headers.push((name.to_ascii_lowercase(), value.to_owned()));
    }
    if transfer_encoding { return Err(AppError::InvalidRequest("Transfer-Encoding is not supported; use bounded Content-Length".into())); }
    if host_count > 1 {
        return Err(AppError::InvalidRequest("duplicate Host headers are not allowed".into()));
    }
    if origin_count > 1 {
        return Err(AppError::InvalidRequest("duplicate Origin headers are not allowed".into()));
    }
    if fetch_site_count > 1 {
        return Err(AppError::InvalidRequest("duplicate Sec-Fetch-Site headers are not allowed".into()));
    }
    if parts[2] == "HTTP/1.1" && host_count != 1 {
        return Err(AppError::InvalidRequest("HTTP/1.1 requires exactly one Host header".into()));
    }
    if let Some((_, host)) = headers.iter().find(|(name, _)| name == "host") {
        validate_listener_host(host, listener_address)?;
    }
    if let Some((_, origin)) = headers.iter().find(|(name, _)| name == "origin") {
        if !is_listener_origin(origin, listener_address) {
            return Err(AppError::InvalidRequest(
                "Origin must match the exact HTTP listener origin".into(),
            ));
        }
    }
    if let Some((_, site)) = headers.iter().find(|(name, _)| name == "sec-fetch-site") {
        // A different localhost port is same-site but not same-origin. It must
        // not be trusted to submit forms to state-changing application routes.
        if !matches!(site.to_ascii_lowercase().as_str(), "same-origin" | "none") {
            return Err(AppError::InvalidRequest(
                "only same-origin or user-initiated Fetch Metadata requests are allowed".into(),
            ));
        }
    }
    let length = content_length.unwrap_or(0);
    let mut body = vec![0u8; length];
    let mut received = 0usize;
    while received < length {
        if Instant::now() >= deadline {
            return Err(AppError::RequestLimit("HTTP request read deadline exceeded".into()));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(AppError::RequestLimit("HTTP request read deadline exceeded".into()));
        }
        stream.set_read_timeout(Some(config.read_timeout.min(remaining)))
            .map_err(|error| AppError::Handler(format!("HTTP body deadline setup failed: {error}")))?;
        let count = stream.read(&mut body[received..])
            .map_err(|error| AppError::InvalidRequest(format!("incomplete HTTP request body: {error}")))?;
        if count == 0 { return Err(AppError::InvalidRequest("incomplete HTTP request body".into())); }
        if Instant::now() >= deadline {
            return Err(AppError::RequestLimit("HTTP request read deadline exceeded".into()));
        }
        received += count;
    }
    let request = AppRequest {
        method, path: path.to_owned(), query: query.to_owned(), headers, body,
    };
    validate_path(&request.path, 2_048)?;
    if request.query.len() > 2_048 || request.query.bytes().any(|b| b <= 0x20 || b == b'\r' || b == b'\n' || b == b'#' || b == 0x7f) {
        return Err(AppError::InvalidRequest("query string is invalid or exceeds byte limit".into()));
    }
    Ok(request)
}

fn validate_listener_host(host: &str, listener_address: SocketAddr) -> Result<(), AppError> {
    let host = host.to_ascii_lowercase();
    let mut allowed = vec![listener_address.to_string().to_ascii_lowercase()];
    if listener_address.ip().is_loopback() {
        allowed.push(format!("localhost:{}", listener_address.port()));
    }
    if listener_address.port() == 80 {
        let bare_address = match listener_address.ip() {
            std::net::IpAddr::V4(ip) => ip.to_string(),
            std::net::IpAddr::V6(ip) => format!("[{ip}]"),
        };
        allowed.push(bare_address);
        if listener_address.ip().is_loopback() {
            allowed.push("localhost".to_owned());
        }
    }
    if allowed.iter().any(|candidate| candidate == &host) {
        Ok(())
    } else {
        Err(AppError::InvalidRequest(
            "Host header must match the loopback listener address".into(),
        ))
    }
}

fn is_listener_origin(origin: &str, listener_address: SocketAddr) -> bool {
    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    // This server only speaks plain HTTP. HTTPS origins and opaque "null"
    // origins cannot be same-origin with this listener.
    if !scheme.eq_ignore_ascii_case("http") || authority.is_empty() {
        return false;
    }
    if authority.chars().any(|character| {
        character.is_ascii_whitespace()
            || matches!(character, '/' | '?' | '#' | '@' | '\\')
    }) {
        return false;
    }

    let authority = authority.to_ascii_lowercase();
    let exact = listener_address.to_string().to_ascii_lowercase();
    let localhost_alias = format!("localhost:{}", listener_address.port());
    authority == exact
        || (listener_address.ip().is_loopback() && authority == localhost_alias)
}

fn is_loopback_origin(origin: &str) -> bool {
    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return false;
    }
    if authority.is_empty()
        || authority.chars().any(|character| {
            character.is_ascii_whitespace()
                || matches!(character, '/' | '?' | '#' | '@' | '\\')
        })
    {
        return false;
    }

    let valid_port = |port: &str| port.parse::<u16>().map(|value| value > 0).unwrap_or(false);
    if let Some(rest) = authority.strip_prefix('[') {
        let Some(end) = rest.find(']') else {
            return false;
        };
        let host = &rest[..end];
        let suffix = &rest[end + 1..];
        if !suffix.is_empty() {
            let Some(port) = suffix.strip_prefix(':') else {
                return false;
            };
            if !valid_port(port) {
                return false;
            }
        }
        return host.parse::<std::net::IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false);
    }

    let mut parts = authority.split(':');
    let host = parts.next().unwrap_or_default();
    if host.is_empty() {
        return false;
    }
    if let Some(port) = parts.next() {
        if parts.next().is_some() || !valid_port(port) {
            return false;
        }
    }
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<std::net::IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
}

fn error_response(error: &AppError) -> AppResponse {
    let (status, body) = match error {
        AppError::NotFound => (404, "Not Found"),
        AppError::InvalidRequest(_) => (400, "Bad Request"),
        AppError::RequestLimit(_) => (413, "Payload Too Large"),
        _ => (500, "Internal Server Error"),
    };
    AppResponse::text(status, body)
}

fn write_http_response(stream: &mut TcpStream, response: &AppResponse, head_only: bool) -> Result<(), AppError> {
    let reason = match response.status {
        200 => "OK", 201 => "Created", 202 => "Accepted", 204 => "No Content",
        205 => "Reset Content", 301 => "Moved Permanently", 302 => "Found", 304 => "Not Modified",
        400 => "Bad Request", 401 => "Unauthorized", 403 => "Forbidden", 404 => "Not Found",
        405 => "Method Not Allowed", 409 => "Conflict", 413 => "Payload Too Large",
        415 => "Unsupported Media Type", 429 => "Too Many Requests",
        500 => "Internal Server Error", 501 => "Not Implemented", 503 => "Service Unavailable",
        _ => "Application Response",
    };
    let mut output = format!("HTTP/1.1 {} {}\r\n", response.status, reason);
    for (name, value) in &response.headers {
        output.push_str(name);
        output.push_str(": ");
        output.push_str(value);
        output.push_str("\r\n");
    }
    if !matches!(response.status, 204 | 304) {
        output.push_str(&format!("content-length: {}\r\n", response.body.len()));
    }
    output.push_str("connection: close\r\n\r\n");
    stream.write_all(output.as_bytes())
        .and_then(|_| if head_only { Ok(()) } else { stream.write_all(&response.body) })
        .and_then(|_| stream.flush())
        .map_err(|error| AppError::Handler(format!("HTTP response write failed: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    struct EchoHandler;
    impl AppHandler for EchoHandler {
        fn invoke(&self, entry: &str, request: &AppRequest) -> Result<AppResponse, AppError> {
            Ok(AppResponse::text(200, format!("{entry}:{}:{}", request.method.as_str(), request.path)))
        }
    }
    #[test]
    fn routes_are_exact_and_duplicate_safe() {
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();
        assert!(matches!(app.route(AppMethod::Get, "/", "other"), Err(AppError::DuplicateRoute { .. })));
        let response = app.dispatch(AppRequest::new(AppMethod::Get, "/"), &EchoHandler).unwrap();
        assert_eq!(response.body, b"home:GET:/");
        assert!(matches!(app.dispatch(AppRequest::new(AppMethod::Post, "/"), &EchoHandler), Err(AppError::NotFound)));
    }
    #[test]
    fn paths_reject_traversal_and_ambiguous_normalization() {
        for path in ["/../secret", "/a/./b", "/%2e%2e/secret", "/a%2fb", "/a?x=1", "/a#frag", "//host/path"] {
            let mut app = AppRuntime::new(AppLimits::default());
            assert!(app.route(AppMethod::Get, path, "home").is_err(), "{path}");
        }
    }
    #[test]
    fn request_limits_and_header_injection_fail_closed() {
        let limits = AppLimits { max_request_body_bytes: 2, ..AppLimits::default() };
        let mut app = AppRuntime::new(limits);
        app.route(AppMethod::Post, "/submit", "submit").unwrap();
        assert!(matches!(app.dispatch(AppRequest::new(AppMethod::Post, "/submit").with_body(b"abc".to_vec()), &EchoHandler), Err(AppError::RequestLimit(_))));
        let request = AppRequest::new(AppMethod::Post, "/submit").with_header("x-test", "safe\r\ninjected: yes");
        assert!(matches!(app.dispatch(request, &EchoHandler), Err(AppError::InvalidRequest(_))));
    }
    #[test]
    fn response_status_and_size_are_validated() {
        struct BadHandler;
        impl AppHandler for BadHandler {
            fn invoke(&self, _: &str, _: &AppRequest) -> Result<AppResponse, AppError> {
                Ok(AppResponse::new(101, b"bad".to_vec()))
            }
        }
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();
        assert!(matches!(app.dispatch(AppRequest::new(AppMethod::Get, "/"), &BadHandler), Err(AppError::InvalidResponse(_))));

        struct ResetWithBody;
        impl AppHandler for ResetWithBody {
            fn invoke(&self, _: &str, _: &AppRequest) -> Result<AppResponse, AppError> {
                Ok(AppResponse::new(205, b"not allowed".to_vec()))
            }
        }
        assert!(matches!(
            app.dispatch(AppRequest::new(AppMethod::Get, "/"), &ResetWithBody),
            Err(AppError::InvalidResponse(_))
        ));
    }

    #[test]
    fn native_http_204_response_omits_content_length() {
        use std::io::Read;
        use std::net::{TcpListener, TcpStream};
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let writer = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            write_http_response(&mut stream, &AppResponse::new(204, Vec::new()), false).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        let mut response = Vec::new();
        client.read_to_end(&mut response).unwrap();
        writer.join().unwrap();
        let response = String::from_utf8(response).unwrap().to_ascii_lowercase();
        assert!(response.starts_with("http/1.1 204 no content"));
        assert!(!response.contains("content-length:"));
        assert!(response.contains("connection: close"));
    }

    #[test]
    fn host_capabilities_are_explicit() {
        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        assert!(policy.require(AppCapability::HttpListen).is_ok());
        assert!(matches!(policy.require(AppCapability::DesktopWindow), Err(AppError::CapabilityDenied(AppCapability::DesktopWindow))));
    }
    fn root() -> UiNode {
        UiNode { id: 1, parent: None, kind: UiNodeKind::Window, accessible_name: "Main window".into(), text: String::new(), enabled: true }
    }
    #[test]
    fn ui_tree_requires_valid_parents_and_accessible_names() {
        let mut tree = UiTree::new(root(), 3).unwrap();
        assert!(tree.add(UiNode { id: 2, parent: Some(99), kind: UiNodeKind::Button, accessible_name: "Save".into(), text: "Save".into(), enabled: true }).is_err());
        tree.add(UiNode { id: 2, parent: Some(1), kind: UiNodeKind::Button, accessible_name: "Save".into(), text: "Save".into(), enabled: true }).unwrap();
        assert!(tree.add(UiNode { id: 3, parent: Some(1), kind: UiNodeKind::Button, accessible_name: String::new(), text: "Cancel".into(), enabled: true }).is_err());
    }
    #[test]
    fn ui_events_must_match_enabled_controls() {
        let mut tree = UiTree::new(root(), 4).unwrap();
        tree.add(UiNode { id: 2, parent: Some(1), kind: UiNodeKind::Button, accessible_name: "Save".into(), text: "Save".into(), enabled: true }).unwrap();
        assert!(tree.validate_event(&UiEvent { node_id: Some(2), kind: UiEventKind::Clicked }).is_ok());
        assert!(tree.validate_event(&UiEvent { node_id: Some(2), kind: UiEventKind::TextChanged("x".into()) }).is_err());
        assert!(tree.validate_event(&UiEvent { node_id: None, kind: UiEventKind::WindowClosed }).is_ok());
    }
    #[test]
    fn ui_actions_validate_targets_and_text_budgets() {
        let mut tree = UiTree::new(root(), 4).unwrap();
        tree.add(UiNode { id: 2, parent: Some(1), kind: UiNodeKind::Label,
            accessible_name: "Status".into(), text: "Ready".into(), enabled: true }).unwrap();
        tree.add(UiNode { id: 3, parent: Some(1), kind: UiNodeKind::TextInput,
            accessible_name: "Search".into(), text: String::new(), enabled: true }).unwrap();
        assert!(tree.apply_action(&UiAction::SetText { node_id: 2, text: "Updated".into() }).is_ok());
        assert_eq!(tree.nodes().get(&2).unwrap().text, "Updated");
        assert!(tree.apply_action(&UiAction::SetText { node_id: 2, text: "x".repeat(4_097) }).is_err());
        assert!(tree.apply_action(&UiAction::SetEnabled { node_id: 99, enabled: false }).is_err());
        assert!(tree.apply_action(&UiAction::CloseWindow).unwrap());
        assert!(tree.validate_event(&UiEvent { node_id: Some(3),
            kind: UiEventKind::TextChanged("x".repeat(4_097)) }).is_err());
    }

    #[test]
    fn native_http_server_rejects_non_loopback_bind_addresses() {
        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        for address in ["0.0.0.0:0", "[::]:0", "192.0.2.1:0", "localhost:0"] {
            let result = NativeHttpServer::bind(
                address,
                &policy,
                HttpServerConfig::default(),
            );
            assert!(
                matches!(result, Err(AppError::InvalidRequest(message)) if message.contains("loopback")),
                "address {address} must be rejected before binding"
            );
        }
    }

    #[test]
    fn native_http_server_rejects_dns_rebinding_host_headers() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();

        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!(
            "GET / HTTP/1.1\r\nHost: attacker.example:{}\r\n\r\n",
            address.port()
        ).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    }

    #[test]
    fn native_http_server_rejects_remote_browser_origins() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();

        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!(
            "GET / HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nOrigin: https://attacker.example\r\n\r\n",
            address.port()
        ).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    }

    #[test]
    fn native_http_server_rejects_cross_site_fetch_metadata_without_origin() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();

        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!(
            "GET / HTTP/1.1\r\nHost: localhost:{}\r\nSec-Fetch-Site: cross-site\r\nSec-Fetch-Mode: no-cors\r\n\r\n",
            address.port()
        ).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    }

    #[test]
    fn native_http_server_accepts_same_origin_fetch_metadata_without_origin() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();

        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!(
            "GET / HTTP/1.1\r\nHost: localhost:{}\r\nSec-Fetch-Site: same-origin\r\nSec-Fetch-Mode: cors\r\n\r\n",
            address.port()
        ).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
    }

    #[test]
    fn native_http_server_accepts_loopback_host_and_origin() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();

        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!(
            "GET / HTTP/1.1\r\nHost: localhost:{}\r\nOrigin: http://localhost:3000\r\n\r\n",
            address.port()
        ).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.ends_with("home:GET:/"));
    }

    #[test]
    fn native_http_server_dispatches_to_ardisa_and_preserves_query() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let compiled = crate::compile_source(include_str!("../../../examples/shared_app_logic.ardisa")).unwrap();
        let program = crate::native::decode_program(&compiled.artifact).unwrap();
        let handler = NativeProgramHandler::new(program, ExecutionLimits::default()).unwrap();
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();

        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!("GET /?q=ardisa HTTP/1.1\r\nHost: localhost:{}\r\n\r\n", address.port()).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.ends_with("Hello from shared Ardisa application logic"));
    }

    #[test]
    fn native_ardisa_handler_receives_the_raw_query_string() {
        let source = include_str!("../../../examples/shared_app_logic.ardisa");
        let compiled = crate::compile_source(source).unwrap();
        let program = crate::native::decode_program(&compiled.artifact).unwrap();
        let handler = NativeProgramHandler::new(program, ExecutionLimits::default()).unwrap();
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/echo", "echo_query").unwrap();
        let request = AppRequest::new(AppMethod::Get, "/echo").with_query("q=ardisa&mode=test");
        let response = app.dispatch(request, &handler).unwrap();
        assert_eq!(response.body, b"q=ardisa&mode=test");
    }

    #[test]
    fn native_http_server_enforces_total_request_read_deadline() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;
        use std::time::Duration;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let config = HttpServerConfig {
            read_timeout: Duration::from_secs(2),
            max_request_duration: Duration::from_millis(150),
            ..HttpServerConfig::default()
        };
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, config).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();
        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(b"G").unwrap();
        thread::sleep(Duration::from_millis(250));
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request")
            || response.starts_with("HTTP/1.1 413 Payload Too Large"));
    }

    #[test]
    fn native_http_server_continues_after_a_disconnected_client() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();
        let worker = thread::spawn(move || server.serve_n(&app, &handler, 2).unwrap());

        let mut disconnected = TcpStream::connect(address).unwrap();
        disconnected.write_all(b"G").unwrap();
        drop(disconnected);

        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!("GET / HTTP/1.1\r\nHost: localhost:{}\r\n\r\n", address.port()).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.ends_with("home:GET:/"));
    }

    #[test]
    fn native_http_server_rejects_ambiguous_request_framing() {
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::thread;

        let policy = AppPolicy::allow([AppCapability::HttpListen]);
        let server = NativeHttpServer::bind("127.0.0.1:0", &policy, HttpServerConfig::default()).unwrap();
        let address = server.local_addr().unwrap();
        let handler = EchoHandler;
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Post, "/", "post").unwrap();
        let worker = thread::spawn(move || server.serve_once(&app, &handler).unwrap());
        let mut client = TcpStream::connect(address).unwrap();
        client.write_all(format!("POST / HTTP/1.1\r\nHost: localhost:{}\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n", address.port()).as_bytes()).unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    }

    #[test]
    fn native_ardisa_program_can_handle_a_shared_application_route() {
        let source = include_str!("../../../examples/shared_app_logic.ardisa");
        let compiled = crate::compile_source(source).unwrap();
        let program = crate::native::decode_program(&compiled.artifact).unwrap();
        let handler = NativeProgramHandler::new(program, ExecutionLimits::default()).unwrap();
        let mut app = AppRuntime::new(AppLimits::default());
        app.route(AppMethod::Get, "/", "home").unwrap();
        let response = app.dispatch(AppRequest::new(AppMethod::Get, "/"), &handler).unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"Hello from shared Ardisa application logic");
    }
}
