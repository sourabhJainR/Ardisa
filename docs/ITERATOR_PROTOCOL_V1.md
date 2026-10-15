# Ardisa Core Semantics: Iterator Protocol v1

Status: proposal and implementation contract. This document does not mark iteration as implemented.

## Purpose

Define a small, testable iterator subset for Ardisa before enabling `for/in` in the compiler. Parser acceptance alone is not language support. A construct is supported only when semantic analysis, typed IR, native encoding, verification, execution, and evidence tests agree.

## Initial supported domain

The first implementation target is iteration over an owned `List<T>` where `T` is already a supported runtime value type. Do not infer support for arbitrary user-defined iterators, closures, async streams, references, or trait-based dispatch.

The initial form is:

```ardisa
module iterate
fn sum(values: List<Int>) -> Int
  let total = 0
  for value in values
    set total = total + value
  total
```

This is a target fixture, not a claim that the current compiler accepts or executes it.

## Semantics contract

1. **Single evaluation.** Evaluate the iterable expression exactly once before entering the loop.
2. **Ownership.** The initial implementation consumes the owned list binding used as the iterable. After the loop begins, the source binding is unavailable, including when the loop exits early. If implementation cannot enforce this on every control-flow path, reject the construct.
3. **Element binding.** Each iteration binds one element with its declared/inferred element type. Binding must not create a second mutable alias to the collection's storage.
4. **Order.** Visit elements in increasing index order, exactly once each unless control flow exits early.
5. **Mutation.** The initial subset does not permit mutation of the active iterable through another name or a loop-body alias. Reject any case the ownership checker cannot prove safe.
6. **Bounds.** Iteration is bounded by the existing collection-item limit and VM instruction/frame budgets. No unbounded allocation is required for iteration.
7. **Control flow.** `return` from the enclosing function exits the loop and function deterministically. Nested loops must not share iterator state. Until `break` and `continue` are implemented across AST, typed IR and native execution, reject those forms explicitly.
8. **Errors.** Invalid iterable types, unavailable/moved bindings, type mismatches, unsupported iterator implementations, and budget exhaustion must fail deterministically. Never silently skip a loop or lower it to a no-op.
9. **Determinism.** Identical source and compiler version must produce byte-identical ARDISA-EXEC-V1 artifacts. Runtime iteration order must be deterministic.
10. **Compatibility.** Existing programs and artifact decoding remain unchanged. Any instruction-format change requires an explicit versioned format decision rather than silently changing ARDISA-EXEC-V1.

## Required pipeline work

- **AST/parser:** retain source spans and structured `For { name, iterable, body }` nodes; reject malformed headers and indentation with stable diagnostics.
- **Semantic/type checking:** require `List<T>` for the first subset; infer the loop binding as `T`; check body operations against `T`; reject unsupported iterator protocols.
- **Ownership:** model the iterable move and loop binding in the control-flow graph, including empty collections, nested loops, early return, and scope exit.
- **Typed IR:** represent iterator initialization, next-element test, element binding, loop body, and exit as explicit typed operations/terminators. Do not encode control flow in strings.
- **Native lowering and verifier:** add a bounded instruction path or a formally verified desugaring. Validate register/slot references, element type, branch targets, and iterator state before execution.
- **Runtime:** ensure empty lists, singleton lists, nested lists, repeated loops, nested loops, and budget exhaustion behave consistently.
- **Bootstrap:** do not add a host-Rust compiler fallback. Bootstrap parity remains a separate gate; the native compiler must not claim self-hosting solely because the host implementation supports the construct.

## Acceptance matrix

| Case | Required result |
|---|---|
| Empty `List<Int>` | Body executes zero times |
| One and multiple elements | Ordered, exactly-once visits |
| Nested loops | Independent iterator state |
| Nested `List<List<Int>>` | Correct element type or explicit rejection if nested element lowering is not supported |
| Non-list iterable | Stable semantic error |
| Iterable expression with side effects | Evaluated once, or explicitly rejected if side effects are not supported |
| Source binding used after consuming iteration | Stable moved-value diagnostic |
| Mutation through an alias | Rejected unless ownership analysis proves it safe |
| Early return from body | Deterministic return and cleanup |
| Resource budget exceeded | Fail closed; no partial artifact or silent truncation |
| Encode/decode round trip | Equivalent program and byte-stable re-encoding |
| Differential and holdout fixtures | Expected outputs and expected compile failures match |
| Stage 2/3 bootstrap | Existing byte-identical gate remains green |

## Promotion gate

Do not remove the existing fail-closed diagnostic for `for/in` until all applicable rows above are covered by tests, the full workspace and all-target suites pass, native-phase and differential/holdout gates pass on the exact PR head, and Stage 2/3 reproducibility remains byte-identical. Keep the compatibility matrix at partial/not implemented until this evidence exists.

## Non-goals for v1

- Rust trait parity or blanket `IntoIterator` compatibility.
- User-defined iterator implementations.
- Closure captures, async streams, raw pointers, reference iterators, or implicit cloning.
- Claims of complete Rust ownership/lifetime compatibility.
