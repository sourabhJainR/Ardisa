# Native numeric contract (implemented subset)

This document describes the numeric behavior currently enforced by Ardisa's native execution path. It is deliberately narrower than Rust's complete numeric model.

## Integer values

- \`Int\` is a signed 64-bit integer (\`i64\`) in the current native value representation.
- Addition, subtraction, multiplication, division, remainder, and integer compound addition are checked. Overflow returns a deterministic native type/runtime error; it must not wrap or panic.
- Division and remainder by zero return errors.
- The \`i64::MIN / -1\` and \`i64::MIN % -1\` edge cases return overflow errors because the mathematical result is not representable in \`Int\`.
- Integer division truncates toward zero, and remainder follows that quotient convention (for example, \`-7 / 3 == -2\` and \`-7 % 3 == -1\`).
- The current contract does not imply implicit narrowing, widening, signed/unsigned conversions, or checked/unchecked cast syntax. Unsupported conversions must remain rejected rather than inheriting host-language behavior.

## Floating-point values and conversions

- Floating literals are currently represented as \`F32\` in the front-end type model.
- Full source-level floating arithmetic, cast semantics, NaN/Infinity policy, and cross-platform bitwise reproducibility are **not** claimed by this contract. These remain explicit implementation gaps until source-to-native conformance tests cover them.
- AI tensor/adapter operations that require finite values validate them at their native operation boundary; that does not establish a universal scalar-float policy.

## Evidence boundary

The native VM's checked integer operations and source-to-native arithmetic fixtures are the evidence for the subset above. Host Rust implementation details are not a language specification. Any change to these rules requires updates to this document, source-level boundary tests, native artifact verification, differential/holdout tests, and the Stage 2/Stage 3 reproducibility gate.
