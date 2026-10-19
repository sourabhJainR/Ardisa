# Ardisa failure-boundary contract

Status: **partial implementation; explicit failures are supported, panic/unwind parity is not**.

This contract distinguishes language-level behavior from host Rust behavior. A Rust panic, destructor, or process abort must never be inferred to be an Ardisa guarantee merely because the implementation currently uses Rust internally.

## Boundary matrix

| Boundary | Current behavior | Guarantee | Explicitly not guaranteed |
|---|---|---|---|
| Function/native VM | Checked runtime failures are returned as `NativeError`; malformed programs are rejected by artifact decoding and native-program validation before dispatch | Stable error category for covered malformed control flow, invalid operations and resource limits | General catchable Ardisa panic, stack unwinding, panic payloads, rollback of arbitrary prior mutations |
| Structured task | Worker panic is translated to `AIF505`; `finish` cancels cooperative siblings, joins workers, releases task leases, and selects the externally reported panic by stable task-name order | Cleanup/join of tasks that cooperate with cancellation; deterministic reported error for tested simultaneous failures | Forced termination of a non-cooperative worker, language-level unwind, destructor guarantees |
| Resource budget | Failed and joined tasks release task leases; instruction budget exhaustion returns an explicit resource-limit error | Bounded operations where the relevant budget is enforced | Universal memory/time bounds for every allocation or host call |
| Native artifact | Decoder rejects invalid magic/records, unterminated functions, malformed control flow and invalid jump targets | Invalid serialized programs are rejected before VM execution | Cryptographic authenticity of artifacts or trust in an external signer |
| FFI / host boundary | No general Ardisa panic ABI is specified | Callers must handle declared errors from the supported ABI | No blanket promise that arbitrary host panics are caught, that panic payloads cross the ABI, or that unwind crosses foreign frames safely |
| Process boundary | No language-defined panic/abort profile is established | Host process termination remains an implementation/runtime concern | Rust `panic=abort` parity, language-defined exit codes for panics, guaranteed process survival |

## Failure categories

- **Compile/validation failure:** source diagnostics or `NativeError::InvalidProgram`; no execution begins for rejected artifacts.
- **Checked runtime failure:** a typed `NativeError` is returned to the caller.
- **Task failure:** a worker panic becomes `AIF505`; the scope drains cooperative siblings and releases their leases before returning.
- **Resource exhaustion:** a specific resource-limit error is returned where a budget is enforced.
- **Host panic outside a guarded boundary:** not currently a language-level recovery feature; do not assume it is converted to an Ardisa error.
- **Abort/process death:** outside the current Ardisa language contract.

## Cleanup and mutation rules

1. A structured scope must cancel running siblings after the first observed worker failure and join every worker before returning.
2. Task-budget leases must remain held until the corresponding worker has been joined, then be released on both success and failure.
3. Failure diagnostics must not depend on thread completion order; simultaneous worker failures are reported by lexicographically stable task name.
4. Native artifacts must be fully decoded and structurally validated before the first instruction executes.
5. An error return is not a transaction. Unless a specific operation documents rollback, mutations completed before a later failure are not promised to be reverted.
6. Cooperative cancellation is not a kill primitive. A worker that ignores its cancellation token can delay scope completion.
7. There is no user-defined destructor contract in the current subset; do not promise drop-on-error, drop-on-panic, or Rust unwind safety.

## Evidence

The regression suite `crates/aifocus-core/tests/failure_boundaries.rs` covers checked runtime errors, invalid artifact framing and jump targets, deterministic instruction-budget failure, repeated task panics, sibling cancellation/joining, lease release, and native artifact mutation. The CI workflow separately runs workspace/all-target tests, generated differential/fuzz/holdout-labelled tests, bootstrap probe reproducibility, and Native Phases.

Those jobs validate the current implemented subset; they do not prove Rust panic/unwind parity, independent sealed holdouts, or native self-hosting. Capability promotion requires specific evidence for each claim and must remain fail-closed when evidence is absent.

## Work required before closing #172

- Add a complete host-panic containment design for each actual FFI entry point; test it without assuming that catching a panic is safe across every foreign ABI.
- Define whether Ardisa will add a language-level `panic` construct, explicit `Result` propagation only, or a narrower abort-only failure model.
- Coordinate destructor and partial-move cleanup semantics with #158 before promising cleanup of language-owned values.
- Add source-to-runtime tests for the selected failure syntax and any ABI boundary, including mutation and partial-state behavior.
- Update the capability evidence manifest only after those behaviors are implemented and independently tested.
