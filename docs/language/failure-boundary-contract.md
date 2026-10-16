# Ardisa failure-boundary contract (implemented subset)

Ardisa uses explicit compiler diagnostics and native runtime errors for the currently supported language subset. This is intentionally **not** Rust panic/unwind compatibility.

## Source, artifact, and VM boundaries

- Parse, semantic, ownership, concurrency, effect, and typed-IR failures stop compilation; they are not converted into runtime values.
- The ARDISA-EXEC-V1 decoder validates artifact framing and instruction/value constraints before execution. Malformed artifacts must be rejected rather than partially executed.
- Runtime failures are returned as `NativeError` values. Checked integer overflow, division/remainder by zero, invalid AI payloads, and resource-budget exhaustion are explicit failures; the language does not wrap these into values or promise host-Rust panic behavior.
- The caller owns the decision to report or terminate after a returned native error. No implicit conversion between an error result and a panic is defined.

## Structured task boundary

- The supported structured-task subset is documented in `docs/language/concurrency-contract.md`.
- A host worker panic is observed through the thread join result and reported as `AIF505`; this is an explicit task failure, not an Ardisa panic payload or a language-level unwind.
- A joined task failure requests sibling cancellation. Cancellation is cooperative, so non-cooperative work may delay teardown.
- Dropping a `StructuredScope` requests cancellation, joins remaining workers, and releases task resource leases after joining. This is not a guarantee of bounded teardown for a worker that ignores cancellation.

## Cleanup and ABI limits

- The language does not currently expose user-defined `Drop`/RAII glue or claim Rust drop-order, panic-safety, or unwind-safety equivalence. Native values rely on the runtime's host-managed value lifecycle; that is not evidence of a source-level deterministic destructor contract.
- Failure cleanup for user-defined destructors, borrowed references, and lifetime escapes remains unsupported until ownership/lifetime work in #158 is complete.
- This document does not extend panic guarantees across arbitrary foreign ABIs. The narrow C ABI subset, where present, has its own explicit type boundary; unwinding across it is not supported by this contract.

## Evidence

- `crates/aifocus-core/src/native.rs`: artifact validation, explicit runtime errors, checked arithmetic and resource-limit enforcement.
- `crates/aifocus-core/src/concurrency.rs`: worker panic capture at join, sibling cancellation requests, task resource leases, and scope-drop joining.
- `crates/aifocus-core/tests/failure_boundaries.rs`: source-to-runtime error propagation and malformed artifact rejection.
- `docs/RUST_CAPABILITY_COMPATIBILITY_MATRIX.md`: panic/unwind and drop remain unsupported or partial, not Rust-equivalent.
