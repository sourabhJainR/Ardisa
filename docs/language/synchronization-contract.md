# Ardisa synchronization contract (supported subset)

Ardisa does not currently expose native source-level threads, atomics, mutexes, read/write locks, channels, or memory-ordering operations. The host runtime may use Rust threads and atomics internally; those implementation details are **not** an Ardisa language guarantee.

## Supported concurrency boundary

- The source-level concurrency subset is the explicit structured task model documented in `docs/language/concurrency-contract.md`: `scope`, `spawn`, `join`, and cooperative `cancel`.
- There is no supported shared-memory synchronization API, no native atomic type, and no source-level `Ordering`/Acquire/Release/SeqCst contract.
- Programs attempting atomic or lock operations must fail compilation until those operations have declared types, typed-IR/native instructions, runtime semantics, and adversarial conformance tests.
- Cancellation is cooperative and scoped task cleanup is not a substitute for lock semantics, race freedom, or a memory model.

## Exclusions and promotion gates

The language makes no claim about data-race freedom for arbitrary host integrations, fairness, deadlock detection, lock poisoning, lock release during unwinding, atomic read-modify-write behavior, or Rust-equivalent memory ordering. No source-level synchronization primitive may be marked supported based only on internal Rust primitives.

A future synchronization API requires source-to-runtime tests for shared state, lock lifetime/release, cancellation interactions, invalid ordering requests, and deterministic adversarial traces before its capability status can change.

## Evidence

- `crates/aifocus-core/tests/synchronization_fail_closed.rs`: atomic/lock APIs are rejected before native execution.
- `crates/aifocus-core/src/concurrency.rs`: the separate structured task model and its cancellation/resource tests.
- `docs/RUST_CAPABILITY_COMPATIBILITY_MATRIX.md`: synchronization remains unsupported at the source-language boundary.
