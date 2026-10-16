# Result and Option semantics: current native contract

This document describes the subset that is currently implemented and tested. It is not a claim that the full Phase 4 gate is complete.

## Implemented Result subset

- `Result<T, E>` is represented in source types and native values as either `ResultOk(T)` or `ResultErr(E)`.
- `ok(value)` and `err(value)` construct these variants. The surrounding declared `Result<T, E>` return type supplies the opposite-side type where it is unambiguous; otherwise the missing side currently defaults to `()`.
- `unwrap(result)` returns the payload of `Ok`. Calling it on `Err` is a runtime type failure with the stable message `unwrap on Err`; passing a non-Result is also a runtime type failure and is rejected by semantic analysis when the source type is known.
- An explicit error-type mismatch between a returned `Result` and the declared return type is rejected. Implicit error conversion is unsupported.
- The source-to-native tests exercise successful unwrap, failing unwrap, artifact decode/re-encode determinism, and ordinary bounded execution.

## Implemented Option subset

- `Option<T>` has native `OptionSome(T)` and `OptionNone` values.
- `some(value)` constructs `Some(value)`; `none()` constructs `None` when the function's declared return type supplies an unambiguous `Option<T>` context.
- `unwrap(OptionSome(value))` returns the payload. `unwrap(OptionNone)` fails at runtime with the stable message `unwrap on None`.
- `propagate(value)` unwraps `Ok(T)`/`Some(T)` to `T`. On `Err(E)` or `None`, it returns that variant directly from the current function. Result propagation requires the same error type as the function's declared Result return; Option propagation requires an Option return. The ownership checker treats unwrap and propagate as consuming the input value.
- Source-to-native tests cover construction, unwrap, propagation short-circuiting, deterministic artifact round-tripping, and None/Err boundaries.

## Explicitly not implemented

- The `?` operator syntax is not implemented; `propagate(value)` is the current limited equivalent.
- Option/Result pattern matching, exhaustive variant analysis, and error conversion traits are not implemented as a unified language feature.
- Cleanup on propagation cannot be claimed because drop/cleanup semantics are not yet implemented. Runtime `unwrap(Err)` and `unwrap(None)` remain execution errors, not automatic propagation.
- This subset does not claim Rust `Option`/`Result` compatibility.

## Promotion gate

Do not mark Phase 4 complete until constructors and payload types are validated end-to-end; invalid and mismatched types have stable compile-time diagnostics; success, error, nested propagation, and early-return cleanup pass source-to-native tests; malformed artifacts and instruction/resource limits are tested; and independent differential/holdout evidence passes on the exact promoted commit. Unsupported syntax must remain rejected rather than silently lowered.
