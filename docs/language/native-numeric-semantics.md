# Native numeric semantics (supported subset)

This document is the current contract for the numeric subset exercised by the Ardisa source-to-native pipeline. It is intentionally narrower than Rust numeric and conversion semantics.

## Integer values

- `Int` is represented as a signed 64-bit integer in the current native execution path.
- Addition, subtraction, multiplication, division, and remainder must not rely on debug-versus-release overflow differences.
- Integer overflow is a checked runtime failure. It must not wrap, saturate, or be folded into a different value by the optimizer.
- Division or remainder by zero is a checked runtime failure.
- Integer division follows the native integer operation for representable operands; the language does not claim Rust-wide numeric compatibility beyond the tested subset.
- Integer literals and arithmetic must pass through the same source → semantic/typed IR → ARDISA-EXEC-V1 path used by ordinary programs.

## Floating-point values

- The current native scalar floating-point representation is IEEE-754 binary32 (`F32`).
- Encoded native artifacts reject non-finite embedded float payloads. Operations that produce non-finite tensor results are rejected by the corresponding native operations.
- Scalar `F32` arithmetic and ordering comparisons are currently rejected by semantic analysis; they must not reach the integer-only native arithmetic instructions. Float literals are accepted where the existing AI/tensor APIs explicitly require them.
- This does not establish a complete policy for scalar float operations, casts, rounding modes, signed zero, NaN propagation, or cross-platform bit-for-bit floating-point equivalence.

## Conversions and unsupported forms

- Do not infer implicit casts between integer and floating-point values.
- Conversion forms without an explicit source-level and runtime contract remain unsupported and must fail closed.
- The subset does not claim parity with Rust's full integer widths, unsigned types, `as` casts, checked/saturating/wrapping APIs, or floating-point library.

## Evidence and limitations

- `crates/aifocus-core/src/pipeline.rs`: source-to-native regressions cover checked integer overflow and division/remainder by zero, deterministic artifacts, and rejection of non-finite encoded float payloads.
- `crates/aifocus-core/src/native.rs`: native instruction validation and execution enforce the runtime checks and artifact validation.
- `docs/RUST_CAPABILITY_COMPATIBILITY_MATRIX.md`: keep primitive scalar arithmetic marked **Partial** until the complete conversion, float, and differential/holdout contracts are independently demonstrated.

Any expansion of the numeric contract must add positive and negative source fixtures and preserve deterministic ARDISA-EXEC-V1 artifacts. This document is not evidence of full Rust numeric compatibility.
