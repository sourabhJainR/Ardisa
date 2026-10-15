# Phase 2 End-to-End Completion Gate

Status: **IN PROGRESS — NOT PROMOTED**

This document tracks Phase 2 source-level `for/in` implementation. The branch now wires `List<T>` iteration through semantic typing, ownership/move analysis, typed IR, native lowering, artifact decoding, and runtime execution. It is not yet promoted: exact-head CI and remaining alias/mutation, side-effect, resource-budget, differential/holdout, and Stage 2/3 evidence gates must pass before Phase 2 is called complete.

## Phase 2 — required implementation

- [x] **AST and parsing:** the parser retains the structured `For { name, iterable, body }` node and source spans.
- [x] **Semantic/type analysis:** accepts the supported `List<T>` subset, infers the loop binding as `T`, and rejects non-list iterables with a stable diagnostic.
- [x] **Ownership:** evaluates the iterable once and marks its source binding moved; loop-body use-after-move and mutation of the consumed binding are rejected.
- [x] **Typed CFG/IR:** represents iteration as a structured typed `ForEach` operation with a typed element binding and nested body.
- [x] **Native backend and verifier:** lowers loops through existing length/index/branch instructions, with validated jump targets and no new ARDISA-EXEC-V1 opcode.
- [x] **Runtime limits:** iteration consumes the VM instruction budget and the existing value/collection/frame limits remain active.
- [x] **End-to-end tests:** covers ordered sums, empty lists, nested lists/loops, early return, iterable call emission once, invalid iterable, use-after-move, alias mutation rejection, artifact round-trip, and instruction-budget exhaustion.
- [ ] **Remaining evidence:** exact-head workspace/all-target and Native Phases results must pass; add/confirm independent differential and holdout fixtures for loop success/failure cases and preserve Stage 2/3 byte-identical reproducibility.
- [ ] **Status promotion:** promote the capability matrix only after all evidence gates pass on the merged revision. Keep issue #160 open because closure/callable scope is still outstanding.

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
