# Ardisa Ownership, Borrowing, and Lifetime Contract

Status: **partial implementation; not Rust parity**.

This document is the current language contract for the ownership checker in
`crates/aifocus-core/src/ownership.rs`. A feature is supported only to the
extent exercised by the compiler's diagnostics and native-pipeline tests.

## Implemented subset

- `Int`, `Bool`, and `Unit` are treated as Copy by the current ownership
  analysis. `String`, named/generic/result values, and lists are treated as
  Move unless a specific expression context uses a borrow.
- Moving an owned local invalidates that binding until a valid assignment
  reinitializes it. Use-after-move reports AIF400.
- List element moves are tracked by place path. Distinct static index paths
  may remain usable; repeated moves and paths involving unknown/dynamic indices
  are conservatively rejected when aliasing cannot be ruled out.
- Control-flow joins account for if branches, early returns, loops, and scopes.
  A value moved on only some reachable paths is treated as maybe moved.
- The current implicit shared-borrow convention for String function arguments
  is part of the language contract. It must not be changed to move semantics
  without a deliberate language-design change and compatibility tests.
- `len` is modeled as a shared borrow and `push` as a mutable borrow.
  Borrow regions are checked for overlapping access to the same binding and
  potentially aliasing place paths.
- Slices currently produce owned copies, not borrowed views. Moving a slice
  therefore does not move the source collection.
- Unsupported references and raw-pointer operations must not be inferred from
  these borrow-analysis internals. A Rust wrapper or runtime API is not proof
  that a native Ardisa reference type exists.

## Explicit limitations

- Ardisa does not claim Rust-equivalent lifetime inference, reference types,
  raw pointers, pointer provenance, pinning, unsafe blocks, or aliasing
  guarantees.
- Borrow regions currently model expression spans and place paths. They are
  not a complete first-class lifetime/type system and must not be described as
  proving arbitrary references cannot escape.
- There is no established user-defined destructor contract in this subset.
  In particular, this document does not promise Rust `Drop`, unwind-time
  destruction, or panic-unwind safety. Those require explicit language/runtime
  semantics and source-to-runtime evidence.
- Native runtime values being represented by Rust-owned values does not make
  host-language destruction behavior a guaranteed Ardisa language feature.
- Borrow transition records are diagnostic/model evidence, not a standalone
  proof of soundness. The source-level checker and adversarial tests remain the
  authority for the implemented subset.

## Required validation for future changes

Any extension to ownership, borrowing, or cleanup must include:

1. Positive source-to-runtime cases for accepted patterns.
2. Compile-fail cases for use-after-move, maybe-moved paths, conflicting
   mutable/shared borrows, dynamic aliasing, and scope/escape violations where
   the relevant language construct exists.
3. Reinitialization tests across branches, loops, and nested scopes.
4. Deterministic cleanup tests on normal return and explicit runtime failure
   before claiming a destruction guarantee.
5. Differential, mutation, holdout, workspace/all-target, Native Phases, and
   bootstrap evidence on the exact proposed commit.
6. A capability-matrix update that distinguishes supported, partial,
   runtime-checked, unknown, and unsupported behavior.

Do not add native references or destructor hooks until the syntax, semantic
rules, typed representation, native artifact validation, runtime behavior, and
failure-path cleanup contract agree. Keep unsupported forms fail-closed.
