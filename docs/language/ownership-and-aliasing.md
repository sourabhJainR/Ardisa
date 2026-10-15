# Ardisa ownership and collection safety contract

This document describes the currently evidenced Ardisa subset. It is not a statement of Rust compatibility.

## Owned values and moves

- Int, Bool, Unit, and the currently supported scalar numeric values are copy-class values.
- String and List<T> are owned values. Binding an owned value to a new local consumes the source name under the flow-sensitive move checker. A later use of that name is rejected with AIF400.
- Assignment to an existing local reinitializes that local only after the assigned expression passes analysis. Conditional and loop-body moves are merged conservatively; use after a potentially moved value is rejected. Loop bindings are rejected if they shadow an existing local (AIF303); consuming a loop-local binding must not change the state of unrelated outer locals.
- The VM's internal cloning for loads/serialization is an implementation detail; it does not grant source programs a reference or shared-mutation capability.

## Borrowing

- The analyzer models shared and mutable access modes and rejects overlapping shared/mutable borrow regions where those regions are represented. These checks are partial: the current checker does not yet use binding identities or a complete control-flow graph, so this must not be treated as a full borrow checker.
- Mutation operations such as indexed assignment and push are analyzed as mutable access to the named collection.
- The current language does not expose a complete reference type, reborrow syntax, raw pointers, interior mutability, or a lifetime-parameterized API. Borrow analysis is therefore a partial safety model, not a proof of Rust-equivalent soundness.
- An active borrow transition must be ended before scope exit; move while borrowed and mutable/shared conflicts are rejected by the borrow-transition validator.

## Lifetimes, escape, aliasing, and cleanup

- Since source-level references and reference-returning functions are not implemented, lifetime escape through a returned reference is not a supported operation. Reference-like syntax that is not part of the grammar must be rejected; it must not be silently interpreted as a value.
- List[start:end] and String[start:end] return owned copies. List slice elements are recursively validated and copy-isolated from the source; string slice bounds must be UTF-8 scalar boundaries. Neither operation creates a view.
- Lists are value containers in the supported subset. The source language has no supported operation that creates two mutable references to the same list storage. Mutation is bounded to the named collection and checked at runtime for item count, nested value nodes/depth, frame footprint, and configured string byte limits.
- Native values are reclaimed by the host runtime's ordinary value destruction when they leave frames or are replaced. There is no user-defined Drop trait, deterministic user cleanup hook, or guarantee of Rust drop-order equivalence.
- Unsupported iterator syntax is parsed only where explicitly implemented and is rejected during semantic analysis with AIF320; no implicit iteration or collection mutation semantics are inferred.

## Evidence and limits

The ownership.rs unit tests exercise use-after-move, modeled borrow conflicts, active-borrow scope exit, conservative loop/conditional move tracking, and loop-local move isolation. list_slices.rs exercises owned-copy isolation, bounds, nested values, resource limits, and byte-identical native artifact round trips. These tests are useful contract evidence but do not establish complete path-sensitive borrowing, lifetime regions, drop order, or aliasing soundness. Keep the compatibility matrix at Partial until those gaps have independent source-to-runtime and adversarial evidence.
