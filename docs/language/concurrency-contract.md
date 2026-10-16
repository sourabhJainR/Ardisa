# Ardisa concurrency contract (supported subset)

Ardisa currently provides a deliberately smaller structured-task model. It does **not** implement Rust `async fn`, `.await`, `Future`, executor polling, `Pin`, or Rust async cancellation semantics.

## Source-level contract

- Tasks are introduced by `spawn name = named_function(...)` inside an explicit `scope`.
- Every spawned task must reach one terminal operation, `join name` or `cancel name`, before the lexical scope exits. Duplicate task names, unknown task names, and repeated terminal operations are diagnostics.
- `cancel` is a cancellation request, not preemption. The worker must observe its `CancellationToken` and cooperate.
- A task that is cancelled still must be joined before its task resource reservation is released.
- The source checker and native pipeline must reject async/await constructs until they have a complete source-to-runtime representation and ownership contract.

## Runtime and failure boundaries

- The host runtime uses a join handle to observe worker completion. A worker panic is captured by the host thread join boundary and reported as an explicit task failure; it is not an Ardisa panic payload or Rust-compatible unwind contract.
- On a joined task failure, the runtime requests cancellation of running siblings. Cancellation remains cooperative; a non-cooperative worker can delay scope teardown indefinitely.
- Dropping a runtime `StructuredScope` requests cancellation of still-running tasks, joins their workers, and releases resource leases only after joining. This prevents orphaned host threads, but does not promise bounded teardown for non-cooperative work.
- Resource budgets are shared across cloned budget handles. Task reservations are retained until the worker is joined.
- Scope event records preserve the API's deterministic spawn/join/cancel/cleanup ordering. They do not make thread scheduling, task completion timing, or arbitrary concurrent side effects deterministic.
- Native VM/runtime errors remain explicit failures. They are not silently converted to panics, and this subset makes no guarantee of Rust panic/unwind, `Drop`, or unwind-safety equivalence.

## Explicit exclusions

No native Future trait, suspension point, async lifetime inference, pinning contract, cancellation-safe await point, lock/atomic/memory-ordering source API, or foreign-work preemption is implied by `spawn`/`join`/`cancel`. Those require separate design, runtime support, and adversarial conformance evidence.

## Evidence

- `crates/aifocus-core/src/concurrency.rs`: source scope analysis, task state transitions, cancellation token, resource leases, panic capture at join, scope-drop cleanup, and deterministic event tests.
- `crates/aifocus-core/tests/async_await_fail_closed.rs`: source-to-native fail-closed checks for async/await syntax.
- `docs/RUST_CAPABILITY_COMPATIBILITY_MATRIX.md`: keeps Rust async/await unsupported and structured concurrency partial.
