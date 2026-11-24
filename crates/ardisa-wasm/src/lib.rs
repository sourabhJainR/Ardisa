use ardisa_core::native::{decode_program, ExecutionLimits, NativeValue, ValidatedNativeProgram};
use js_sys::Array;
use wasm_bindgen::prelude::*;

const MAX_ARTIFACT_BYTES: usize = 1_048_576;
const MAX_ARGUMENTS: u32 = 32;
const MAX_STRING_BYTES: usize = 262_144;

fn browser_error(message: impl AsRef<str>) -> JsValue { JsValue::from_str(message.as_ref()) }
fn limits() -> ExecutionLimits {
    ExecutionLimits { max_instructions: 100_000, max_call_depth: 64, max_tasks: 0,
        max_collection_items: 4_096, max_string_bytes: MAX_STRING_BYTES, max_value_nodes: 4_096,
        max_value_depth: 32, max_stack_values: 4_096, max_locals: 1_024,
        max_frame_value_nodes: 8_192, max_frame_string_bytes: 524_288 }
}
fn parse_argument(value: JsValue) -> Result<NativeValue, JsValue> {
    if let Some(value) = value.as_string() {
        if value.len() > MAX_STRING_BYTES { return Err(browser_error("string argument exceeds browser runtime limit")); }
        return Ok(NativeValue::String(value));
    }
    if let Some(value) = value.as_bool() { return Ok(NativeValue::Bool(value)); }
    if let Some(value) = value.as_f64() {
        if !value.is_finite() || value.fract() != 0.0 || value.abs() > 9_007_199_254_740_991.0 {
            return Err(browser_error("numeric arguments must be finite signed 64-bit integers"));
        }
        return Ok(NativeValue::Int(value as i64));
    }
    Err(browser_error("arguments must be strings, booleans or signed 64-bit integers"))
}
fn render_result(value: NativeValue) -> Result<String, JsValue> {
    match value {
        NativeValue::String(value) if value.len() <= MAX_STRING_BYTES => Ok(value),
        NativeValue::Int(value) => Ok(value.to_string()),
        NativeValue::Bool(value) => Ok(value.to_string()),
        NativeValue::Unit => Ok(String::new()),
        NativeValue::Float(bits) => {
            let value = f32::from_bits(bits);
            if value.is_finite() { Ok(value.to_string()) } else { Err(browser_error("non-finite floating-point result rejected")) }
        }
        _ => Err(browser_error("browser ABI v1 only returns bounded scalar values")),
    }
}

/// Version of the stable JavaScript-to-WASM call boundary.
#[wasm_bindgen]
pub fn runtime_abi_version() -> u32 { 1 }

/// Execute a validated ARDISA-EXEC-V1 artifact with conservative browser limits.
#[wasm_bindgen]
pub fn run_artifact(artifact: &str, entry: &str, args: JsValue) -> Result<String, JsValue> {
    if artifact.len() > MAX_ARTIFACT_BYTES { return Err(browser_error("executable artifact exceeds browser runtime limit")); }
    if entry.is_empty() || entry.len() > 128 || !entry.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
        return Err(browser_error("invalid Ardisa entry function name"));
    }
    if !Array::is_array(&args) { return Err(browser_error("arguments must be a JavaScript array")); }
    let args = Array::from(&args);
    if args.length() > MAX_ARGUMENTS { return Err(browser_error("argument count exceeds browser runtime limit")); }
    let values = args.iter().map(parse_argument).collect::<Result<Vec<_>, _>>()?;
    let decoded = decode_program(artifact).map_err(|error| browser_error(format!("invalid Ardisa artifact: {error:?}")))?;
    let program = ValidatedNativeProgram::try_new(decoded)
        .map_err(|error| browser_error(format!("Ardisa validation failed: {error:?}")))?;
    let result = program.run_with_limits(entry, &values, limits())
        .map_err(|error| browser_error(format!("Ardisa execution failed: {error:?}")))?;
    render_result(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn browser_limits_are_conservative() {
        let value = limits();
        assert_eq!(value.max_instructions, 100_000);
        assert_eq!(value.max_tasks, 0);
        assert_eq!(value.max_string_bytes, MAX_STRING_BYTES);
    }
}


#[cfg(all(test, target_arch = "wasm32"))]
mod browser_parity_tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    const SOURCE: &str = "module parity\napp parity\n  capability http_listen\n  route POST \"/echo\" echo\nfn echo(method: String, path: String, query: String, body: String) -> String\n  body\n";

    #[wasm_bindgen_test]
    fn browser_wasm_matches_native_application_handler_result() {
        let compiled = ardisa_core::compile_application_source(SOURCE)
            .expect("shared application source compiles");
        let args = Array::new();
        args.push(&JsValue::from_str("POST"));
        args.push(&JsValue::from_str("/echo"));
        args.push(&JsValue::from_str(""));
        args.push(&JsValue::from_str("same-result"));
        let result = run_artifact(&compiled.program.artifact, "echo", args.into())
            .expect("browser runtime invocation succeeds");
        assert_eq!(result, "same-result");
    }
}
