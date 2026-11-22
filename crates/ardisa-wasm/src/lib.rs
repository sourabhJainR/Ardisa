//! Browser-facing WebAssembly runtime for canonical ARDISA-EXEC-V1 artifacts.
//!
//! The runtime exposes only pure Ardisa execution. It does not grant browser,
//! network, filesystem, process, or DOM capabilities.
use ardisa_core::native::{
    decode_program, encode_program, encode_program_v1, ExecutionLimits, NativeInstr, NativeValue, ValidatedNativeProgram,
};
use wasm_bindgen::prelude::*;
use js_sys::Array;

const MAX_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
const MAX_ARGUMENTS: usize = 64;
const MAX_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_RESULT_BYTES: usize = 1024 * 1024;

#[wasm_bindgen]
pub struct ArdisaBrowserRuntime {
    program: ValidatedNativeProgram,
    limits: ExecutionLimits,
}

#[wasm_bindgen]
impl ArdisaBrowserRuntime {
    /// Load a canonical ARDISA-EXEC-V1 artifact. Invalid or non-canonical
    /// artifacts are rejected before any application function can run.
    #[wasm_bindgen(constructor)]
    pub fn new(artifact: &str) -> Result<ArdisaBrowserRuntime, JsValue> {
        if artifact.len() > MAX_ARTIFACT_BYTES {
            return Err(js_error("executable artifact exceeds the browser runtime size limit"));
        }
        let decoded = decode_program(artifact)
            .map_err(|error| js_error(&format!("invalid ARDISA-EXEC-V1 artifact: {error:?}")))?;
        let canonical = if artifact.starts_with("ARDISA-EXEC-V1\\n") {
            encode_program_v1(&decoded)
        } else {
            encode_program(&decoded)
        };
        if canonical != artifact {
            return Err(js_error("executable artifact is not canonically encoded"));
        }
        if decoded.functions.values().any(|function| {
            function.code.iter().any(|instruction| matches!(instruction, NativeInstr::Spawn { .. }))
        }) {
            return Err(js_error("browser WebAssembly runtime does not support task-spawning instructions"));
        }
        let program = ValidatedNativeProgram::try_new(decoded)
            .map_err(|error| js_error(&format!("executable validation failed: {error:?}")))?;
        let mut limits = ExecutionLimits::default();
        limits.max_instructions = 250_000;
        limits.max_call_depth = 64;
        limits.max_tasks = 0;
        limits.max_string_bytes = MAX_RESULT_BYTES;
        limits.max_collection_items = 4_096;
        Ok(Self { program, limits })
    }

    /// Invoke a named Ardisa function with a bounded array of string arguments.
    /// ABI v1 deliberately returns strings only; unsupported values fail closed.
    pub fn invoke(&self, entry: &str, args: Array) -> Result<String, JsValue> {
        if entry.is_empty() || entry.len() > 128
            || !entry.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
            return Err(js_error("entry must be a valid bounded Ardisa function name"));
        }
        if args.length() as usize > MAX_ARGUMENTS {
            return Err(js_error("argument count exceeds browser runtime limit"));
        }
        let mut values = Vec::with_capacity(args.length() as usize);
        for value in args.iter() {
            let text = value.as_string()
                .ok_or_else(|| js_error("browser runtime ABI accepts string arguments only"))?;
            if text.len() > MAX_ARGUMENT_BYTES {
                return Err(js_error("argument exceeds browser runtime byte limit"));
            }
            values.push(NativeValue::String(text));
        }
        match self.program.run_with_limits(entry, &values, self.limits) {
            Ok(NativeValue::String(value)) if value.len() <= MAX_RESULT_BYTES => Ok(value),
            Ok(NativeValue::String(_)) => Err(js_error("Ardisa result exceeds browser runtime byte limit")),
            Ok(_) => Err(js_error("browser runtime ABI v1 requires a String result")),
            Err(error) => Err(js_error(&format!("Ardisa execution failed: {error:?}"))),
        }
    }
}

fn js_error(message: &str) -> JsValue {
    JsValue::from_str(message)
}

#[cfg(test)]
mod tests {
    #[test]
    fn browser_runtime_limits_are_explicit_and_nonzero() {
        assert!(super::MAX_ARTIFACT_BYTES > 0);
        assert!(super::MAX_ARGUMENTS > 0);
        assert!(super::MAX_RESULT_BYTES > 0);
    }
}
