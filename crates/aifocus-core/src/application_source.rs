//! First-class source declarations for Ardisa applications.
//!
//! This is the first source-level application slice. It deliberately keeps
//! application declarations outside executable function bodies and compiles
//! the remaining module through the normal Ardisa pipeline.

use std::collections::BTreeSet;

use crate::application::{AppCapability, AppLimits, AppMethod, AppRuntime};
use crate::pipeline::{compile_source, CompiledArtifact, PipelineError};
use crate::{parse, Item};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplicationRoute {
    pub method: AppMethod,
    pub path: String,
    pub handler: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplicationDefinition {
    pub name: String,
    pub routes: Vec<ApplicationRoute>,
    pub capabilities: BTreeSet<AppCapability>,
}

impl ApplicationDefinition {
    pub fn build_runtime(&self, limits: AppLimits) -> Result<AppRuntime, crate::application::AppError> {
        let mut runtime = AppRuntime::new(limits);
        for route in &self.routes {
            runtime.route(route.method, route.path.clone(), route.handler.clone())?;
        }
        Ok(runtime)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledApplication {
    pub definition: ApplicationDefinition,
    pub program: CompiledArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplicationSourceError {
    Declaration(String),
    Pipeline(PipelineError),
}

impl From<PipelineError> for ApplicationSourceError {
    fn from(value: PipelineError) -> Self { Self::Pipeline(value) }
}

/// Compile an Ardisa module with one top-level application declaration.
///
/// Syntax:
/// app inventory
///   capability http_listen
///   route GET "/" home
///
/// The application block is metadata for host composition. Handler bodies
/// remain ordinary Ardisa functions and pass through the standard compiler.
pub fn compile_application_source(source: &str) -> Result<CompiledApplication, ApplicationSourceError> {
    let (definition, module_source) = extract_application(source)?;
    let module = parse(&module_source).map_err(|errors| {
        ApplicationSourceError::Declaration(format!("invalid Ardisa module: {errors:?}"))
    })?;
    let functions: BTreeSet<&str> = module.items.iter().filter_map(|item| match item {
        Item::Function(function) => Some(function.name.as_str()),
    }).collect();

    for route in &definition.routes {
        if !functions.contains(route.handler.as_str()) {
            return Err(ApplicationSourceError::Declaration(format!(
                "route handler '{}' is not a function in this module", route.handler
            )));
        }
    }

    let program = compile_source(&module_source)?;
    Ok(CompiledApplication { definition, program })
}

fn extract_application(source: &str) -> Result<(ApplicationDefinition, String), ApplicationSourceError> {
    let mut output = String::with_capacity(source.len());
    let mut name: Option<String> = None;
    let mut routes = Vec::new();
    let mut capabilities = BTreeSet::new();
    let mut in_app = false;
    let mut saw_route = false;

    for (line_number, line) in source.split_inclusive('\n').enumerate() {
        let raw = line.strip_suffix('\n').unwrap_or(line);
        if !in_app && raw.starts_with("app ") {
            if name.is_some() {
                return Err(ApplicationSourceError::Declaration("only one app declaration is allowed per module".into()));
            }
            let candidate = raw[4..].trim();
            if !valid_identifier(candidate) {
                return Err(ApplicationSourceError::Declaration(format!("line {}: invalid application name", line_number + 1)));
            }
            name = Some(candidate.to_owned());
            in_app = true;
            continue;
        }
        if in_app {
            if raw.trim().is_empty() || raw.trim_start().starts_with('#') { continue; }
            if !raw.starts_with("  ") || raw.starts_with("   ") {
                in_app = false;
                output.push_str(line);
                continue;
            }
            let declaration = raw[2..].trim();
            if let Some(rest) = declaration.strip_prefix("route ") {
                let fields = split_route(rest).ok_or_else(|| ApplicationSourceError::Declaration(
                    format!("line {}: expected route METHOD \"/path\" handler", line_number + 1)
                ))?;
                let method = AppMethod::parse(fields.0).map_err(|error| ApplicationSourceError::Declaration(
                    format!("line {}: {error}", line_number + 1)
                ))?;
                let path = fields.1.to_owned();
                if !path.starts_with('/') || path.starts_with("//") || path.contains('%')
                    || path.contains('?') || path.contains('#') || path.contains('\\')
                    || path.split('/').any(|part| part == "." || part == "..")
                {
                    return Err(ApplicationSourceError::Declaration(format!("line {}: route path must be a canonical absolute path", line_number + 1)));
                }
                if !valid_identifier(fields.2) {
                    return Err(ApplicationSourceError::Declaration(format!("line {}: invalid route handler name", line_number + 1)));
                }
                if routes.iter().any(|existing: &ApplicationRoute| existing.method == method && existing.path == path) {
                    return Err(ApplicationSourceError::Declaration(format!("line {}: duplicate route", line_number + 1)));
                }
                routes.push(ApplicationRoute { method, path, handler: fields.2.to_owned() });
                saw_route = true;
            } else if let Some(value) = declaration.strip_prefix("capability ") {
                let capability = match value {
                    "http_listen" => AppCapability::HttpListen,
                    "desktop_window" => AppCapability::DesktopWindow,
                    "browser_dom" => AppCapability::BrowserDom,
                    "filesystem_read" => AppCapability::FilesystemRead,
                    "filesystem_write" => AppCapability::FilesystemWrite,
                    "network_client" => AppCapability::NetworkClient,
                    "clock" => AppCapability::Clock,
                    "process" => AppCapability::Process,
                    _ => return Err(ApplicationSourceError::Declaration(format!("line {}: unknown capability '{value}'", line_number + 1))),
                };
                if !capabilities.insert(capability) {
                    return Err(ApplicationSourceError::Declaration(format!("line {}: duplicate capability", line_number + 1)));
                }
            } else {
                return Err(ApplicationSourceError::Declaration(format!("line {}: unknown application declaration", line_number + 1)));
            }
            continue;
        }
        output.push_str(line);
    }

    let name = name.ok_or_else(|| ApplicationSourceError::Declaration("missing top-level app declaration".into()))?;
    if !saw_route {
        return Err(ApplicationSourceError::Declaration("application must declare at least one route".into()));
    }
    if !capabilities.contains(&AppCapability::HttpListen) {
        return Err(ApplicationSourceError::Declaration("HTTP routes require an explicit http_listen capability".into()));
    }
    Ok((ApplicationDefinition { name, routes, capabilities }, output))
}

fn valid_identifier(value: &str) -> bool {
    let mut bytes = value.bytes();
    matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn split_route(value: &str) -> Option<(&str, &str, &str)> {
    let (method, rest) = value.split_once(' ')?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('"')?;
    let (path, tail) = rest.split_once('"')?;
    let handler = tail.trim();
    if handler.is_empty() || handler.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return None;
    }
    Some((method, path, handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = "module inventory\napp inventory\n  capability http_listen\n  route GET \"/\" home\nfn home(method: String, path: String, query: String, body: String) -> String\n  \"ok\"\n";

    #[test]
    fn compiles_application_declarations_and_builds_runtime() {
        let compiled = compile_application_source(SOURCE).expect("valid app source");
        assert_eq!(compiled.definition.name, "inventory");
        assert_eq!(compiled.definition.routes.len(), 1);
        assert!(compiled.definition.capabilities.contains(&AppCapability::HttpListen));
        let runtime = compiled.definition.build_runtime(AppLimits::default()).unwrap();
        assert_eq!(runtime.route_count(), 1);
    }

    #[test]
    fn rejects_undeclared_capability_and_duplicate_route() {
        let unknown = SOURCE.replace("http_listen", "ambient_network");
        assert!(matches!(compile_application_source(&unknown), Err(ApplicationSourceError::Declaration(_))));
        let duplicate = SOURCE.replace("fn home", "  route GET \"/\" home\nfn home");
        assert!(matches!(compile_application_source(&duplicate), Err(ApplicationSourceError::Declaration(_))));
    }

    #[test]
    fn rejects_missing_handler_and_noncanonical_path() {
        let missing = SOURCE.replace("home\nfn home", "missing\nfn home");
        assert!(matches!(compile_application_source(&missing), Err(ApplicationSourceError::Declaration(_))));
        let traversal = SOURCE.replace("GET \"/\"", "GET \"/../secret\"");
        assert!(matches!(compile_application_source(&traversal), Err(ApplicationSourceError::Declaration(_))));
    }
}
