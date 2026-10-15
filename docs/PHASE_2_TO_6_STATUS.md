# Ardisa Language Implementation Roadmap: Phases 2–6

Status: **Phase 2 merged; promotion evidence remains gated. Phases 3–6 are staged implementation work.**

This roadmap separates source implementation from capability promotion. A phase is not complete because its parser accepts syntax or because unit tests pass in isolation. Each capability must traverse parse, semantic analysis, ownership, typed IR, native encoding, artifact verification, bounded execution, adversarial tests, and bootstrap/differential evidence where applicable.

## Phase 2 — Owned `List<T>` iteration

**Implementation:** merged in PR #215.

The supported subset is native source-level `for/in` over owned `List<T>`, with a typed element binding, single evaluation of the iterable expression, move checking, nested loops, early return, deterministic index-based lowering, and VM instruction limits.

**Promotion remains gated on:** independent differential/holdout fixtures and Stage 2/3 reproducibility evidence on the merged revision. Do not describe this as general iterator-trait support or zero-copy iteration.

## Phase 3 — Flow-sensitive ownership and cleanup soundness

**Goal:** make ownership decisions consistent across branches, loops, scopes, returns, and reinitialization before adding richer callable abstractions.

Implementation work:
- Define the ownership state lattice and control-flow join rules explicitly; distinguish definitely moved, definitely available, and path-dependent states.
- Analyze `if/else`, `while`, `for/in`, nested scopes, early return, and reinitialization using shared transfer/join rules instead of ad hoc state mutation.
- Track local declarations and shadowing by binding identity, not just a string name, so nested scopes cannot accidentally update an outer binding.
- Keep borrows fail-closed until the language has a real reference/value model; do not treat expression-span overlap as a complete lifetime checker.
- Specify deterministic cleanup/resource release at scope exit and on early return. Do not claim user-defined `Drop` or unwind safety.
- Add positive and negative source-to-native tests for branch joins, loop-carried moves, shadowing, reinitialization, nested collections, early returns, and resource limits.

**Acceptance gate:** control-flow ownership tests, compiler workspace/all-target tests, negative diagnostic stability, native execution tests, and an independently generated move/borrow adversarial corpus all pass. No new reference or lifetime syntax is enabled until the runtime representation supports it.

## Phase 4 — Closed, typed error semantics

**Goal:** make `Option<T>` and `Result<T, E>` a coherent language-level contract rather than a collection of built-ins.

Implementation work:
- Specify `Option<T>` / `Result<T, E>` constructors, payload typing, matching, and failure behavior.
- Add typed propagation only after AST, semantic rules, typed IR, native lowering, and cleanup-on-return are defined together.
- Reject invalid propagation types and mismatched error types at compile time.
- Define whether failure is a returned value or a runtime error for each construct; never conflate `unwrap` failure with ordinary `Err`.
- Add success, failure, nested propagation, early-return cleanup, invalid-type, artifact round-trip, and instruction-budget tests.

**Acceptance gate:** complete source-to-native conformance and negative tests for every enabled construct; no host-language fallback and no undocumented panic behavior.

## Phase 5 — Closures, callable types, and general iterator protocol

**Goal:** add callable abstractions only after ownership joins and error/cleanup semantics are stable.

Implementation work:
- Introduce explicit closure AST and typed callable representation, including capture-by-value rules first; reject unsupported capture modes.
- Define callable invocation and effect/ownership behavior in typed IR and native artifacts.
- Define a versioned iterator protocol with `next`/completion semantics and an explicit mutation/aliasing policy. Keep `List<T>` iteration as the compatibility baseline.
- Add `break` and `continue` only with correct nested-loop targets, cleanup behavior, and instruction-budget accounting.
- Keep async streams, trait-based dispatch, arbitrary user-defined iterators, and Rust `Fn`/`FnMut`/`FnOnce` parity unsupported until independently evidenced.

**Acceptance gate:** callable capture and iterator conformance corpus, adversarial aliasing/move tests, deterministic artifact round-trip, nested-loop control-flow tests, and no regression to Phase 2.

## Phase 6 — Independent bootstrap and self-hosting promotion

**Goal:** demonstrate that Ardisa's own compiler pipeline can compile and rebuild itself without a trusted Rust compiler fallback.

Implementation work:
- Ensure `bootstrap/compiler.ardisa` owns the full source-to-native pipeline: source → lexer → AST → semantic/type checks → typed representation → Ardisa IR → native executable IR → ARDISA-EXEC-V1.
- Make Stage 0, Stage 1, Stage 2, and Stage 3 artifacts explicit and reproducible; Stage 2 and Stage 3 must be byte-identical for the same compiler/source inputs.
- Separate host reference execution from the trusted compiler path. A Rust implementation may remain a bootstrap substrate, but it must not silently produce or validate the artifact claimed as self-hosted.
- Generate independent differential fixtures and sealed holdouts; record source, compiler, artifact, expected result, actual result, and verifier evidence.
- Test clean-room rebuilds, malformed artifacts, deterministic diagnostics, resource limits, and rollback to the last known-good compiler revision.

**Acceptance gate:** no host-compiler fallback in the trusted path; Stage 2/3 byte-identical reproducibility; independent differential/holdout success; clean-room rerun; all required CI green. Only then may `self_hosting_ready=true` be reported.

## Cross-phase invariants

- Unsupported syntax and semantics fail closed with stable diagnostics.
- Never claim Rust ownership, borrowing, lifetime, trait, iterator, drop, or standard-library parity without dedicated evidence.
- Preserve ARDISA-EXEC-V1 compatibility unless a deliberate, documented version change is approved.
- Every promotion is tied to an exact commit and exact CI results.
- Changes to compiler behavior must include regression tests and a rollback path. If a promoted gate regresses, disable the feature or revert the enabling commit while preserving the failing fixture.
- Phases 3–6 are sequential gates, not a promise that all capabilities are already implemented.