# Phase 2 End-to-End Completion Gate

Status: **IN PROGRESS — NOT PROMOTED**

This document tracks Phase 2 source-level `for/in` implementation. The branch now wires `List<T>` iteration through semantic typing, ownership/move analysis, typed IR, native lowering, artifact decoding, and runtime execution. It is not yet promoted: exact-head CI and remaining alias/mutation, side-effect, resource-budget, differential/holdout, and Stage 2/3 evidence gates must pass before Phase 2 is called complete.

## Phase 2 — required implementation

- [ ] **AST and parsing:** preserve the `For { name, iterable, body }` node and source spans; reject malformed headers deterministically.
- [ ] **Semantic/type analysis:** initially accept only owned `List<T>` values supported by the native backend; infer the element binding as `T`; reject non-list and unsupported nested element cases with stable diagnostics.
- [ ] **Ownership:** evaluate the iterable once and consume the owned binding; reject use-after-move, mutation through aliases, and any control-flow path for which move state cannot be proven. Check empty loops, nested loops, early return, and scope exit.
- [ ] **Typed CFG/IR:** represent iterator initialization, next-element test, element binding, body, and exit as structured typed operations/terminators. Do not encode control flow in strings or silently omit loop bodies.
- [ ] **Native backend and verifier:** lower the typed loop to a bounded native execution path or verified desugaring; validate branch targets, slots/registers, element types, and iterator state. Preserve ARDISA-EXEC-V1 compatibility; any format change requires an explicit version decision.
- [ ] **Runtime limits:** enforce collection item and instruction/frame budgets; fail deterministically without producing a partial artifact on compile-time validation failure.
- [ ] **End-to-end tests:** native execution for empty, singleton, multiple and nested loops; ordered exactly-once visits; iterable evaluation once; early return; source moved after iteration; alias mutation rejection; invalid iterable; budget exhaustion; malformed source; repeated execution determinism.
- [ ] **Artifact and bootstrap evidence:** deterministic encode/decode/re-encode; differential and holdout fixtures; exact-head workspace/all-target and native-phase CI; Stage 2/3 byte-identical reproducibility remains green.
- [ ] **Status promotion:** update the capability matrix only after all applicable checks pass on the merged revision. Keep issue #160 open until its accepted iterator behavior is demonstrably executable; closure/callable scope must be accounted for separately.

## Explicit exclusions from Phase 2

Do not imply support for arbitrary user-defined iterators, trait-based dispatch, closures, async streams, raw-pointer iteration, or full Rust ownership/lifetime parity. Unsupported constructs must continue to fail closed.

## Phase 3 — FUTURE

Not started. Define its scope and acceptance criteria after Phase 2's source-to-native iteration gate is green. Do not mix Phase 3 work into Phase 2 or weaken Phase 2's promotion requirements.

## Phase 4 — FUTURE

Not started. Requires a separately reviewed scope, implementation, negative tests, and evidence gate after Phase 3 is authorized.

## Phase 5 — FUTURE

Not started. Requires a separately reviewed scope, implementation, negative tests, and evidence gate after Phase 4 is authorized.

## Phase 6 — FUTURE

Not started. Requires a separately reviewed scope, implementation, negative tests, and evidence gate after Phase 5 is authorized.

## Merge and rollback policy

- Do not merge a change that enables `for/in` unless all required Phase 2 gates pass on the exact PR head.
- Keep the fail-closed diagnostic as the rollback-safe behavior until promotion.
- If any post-merge regression affects compiler correctness, native artifact validation, or Stage 2/3 reproducibility, revert the enabling commit or disable the feature gate and restore fail-closed behavior; preserve the failing fixture as a regression test.
- A green compile-only test is insufficient: the source program must execute through the native backend and match its expected result.
