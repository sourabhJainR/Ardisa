# Rust capability compatibility audit

**Purpose:** Track Rust language and runtime capabilities against the implemented Ardisa subset. This is a gap inventory, not a claim of Rust parity.

## Status meanings

- **Implemented (subset):** evidenced in the current Ardisa source/compiler/runtime path for the stated subset.
- **Partial:** a limited form exists, but Rust-equivalent semantics or coverage are not established.
- **Not implemented / not evidenced:** no complete native Ardisa source-to-execution contract was found in this audit.
- **Interop-only:** a narrow boundary exists; this does not imply native language support.

## Language capability matrix

| Rust capability | Ardisa status | Evidence / gap |
|---|---|---|
| Functions, parameters, return types | Partial | Function syntax and basic type checking exist; complete Rust type inference and function-item semantics are not established. |
| Primitive scalar types and arithmetic | Partial | Int, Bool, Unit, Float/native arithmetic exist in parts of the compiler/runtime; numeric conversions, overflow policy, and full Rust numeric semantics are not equivalent. |
| Strings and lists | Partial | UTF-8 strings use byte length/indexing; list indexing/mutation and half-open copy-on-slice `List[start:end]` exist with explicit bounds/resource checks. String slices, iterator syntax, borrowed views, Rust ownership/drop parity, Unicode scalar/grapheme indexing, and full standard-library behavior remain unsupported. |
| Generic types | Partial | Generic type syntax exists for selected types; generic function/type declaration, trait bounds, monomorphization and associated types are not established as a complete system. |
| Structs, tuple structs, enums, union types | Not implemented / not evidenced | AST does not show general user-defined record/enum declarations or Rust pattern semantics. |
| Pattern matching (match, if let, let else) | Not implemented / not evidenced | Current AST has if expressions but no general pattern AST or exhaustiveness checker. |
| Traits, impl, associated types/constants | Not implemented / not evidenced | No general trait resolution, coherence/orphan rules, blanket impls, or associated-item system. |
| Ownership and moves | Partial | Ownership analysis exists; full Rust-compatible move paths, partial moves, drop scopes and all control-flow joins need independent conformance evidence. |
| Borrowing and lifetimes | Partial | Ownership/borrow analysis is present in the repository, but explicit lifetime syntax, region inference, reborrowing, higher-ranked lifetimes and variance are not established as Rust-equivalent. |
| References, raw pointers, pinning | Not implemented / not evidenced | No complete source types or native safety contract for references, raw pointers, Pin, NonNull, aliasing and provenance. |
| Closures and capture modes | Not implemented / not evidenced | No closure AST/type inference, Fn/FnMut/FnOnce traits, or capture analysis contract. |
| Iterators and for loops | Not implemented / not evidenced | No iterator trait protocol or for desugaring; while is present. |
| Modules, visibility, crates and dependency graph | Partial | Module graph/compiler tooling exists; Rust-compatible privacy, crate resolution, Cargo dependency semantics and macro-expanded module behavior are not established. |
| Error handling (Result, Option, ?) | Partial | Result/Ok/Err and unwrap support exist in parts of the compiler/runtime; Option, ? propagation, From conversions, residuals and standard error traits are not established. |
| Async/await, futures, executors | Not implemented / not evidenced | Structured spawn/join/cancel primitives exist, but they are not equivalent to Rust async/await, Future polling, pinning or executor semantics. |
| Structured concurrency | Partial | Scope/spawn/join/cancel and bounded runtime foundations exist; full cancellation, task lifetime and failure semantics still need comprehensive source/runtime tests. |
| Threads, atomics, locks and memory ordering | Partial / not evidenced | Runtime uses thread and atomic primitives internally; Ardisa's native language contract for synchronization primitives and Rust memory-ordering semantics is not established. |
| Unsafe blocks and unsafe contracts | Not implemented as a general language feature | Safe Rust interop emits a narrow isolated unsafe wrapper; arbitrary unsafe source and unsafe-op obligations are not part of the supported Ardisa subset. |
| Rust FFI / C ABI | Interop-only | ardisa-c-abi-v1 currently accepts only scalar Int, Bool, and Unit; aggregate layouts, strings, lists, references, callbacks and unwinding are excluded. |
| Declarative/procedural macros, derives | Not implemented / not evidenced | No Rust-compatible macro token expansion, hygiene, proc-macro execution or derive system. |
| Const evaluation and const generics | Not implemented / not evidenced | No general const evaluator or compile-time value/type computation is established. |
| Attributes, lint levels, conditional compilation | Not implemented / not evidenced | No Rust-compatible attribute system, lint model, cfg evaluation or build-script semantics. |
| Drop, RAII and deterministic destruction | Partial / not evidenced | Scope and resource lease cleanup exist in runtime APIs; user-defined Drop, drop glue, drop order and unwind safety are not established. |
| Panic/unwind semantics | Not implemented / not evidenced | Ardisa errors are explicit in supported paths; Rust panic payload, unwind boundaries, panic safety and abort profiles are not equivalent. |
| Standard library and Cargo ecosystem | Not implemented as a compatibility layer | Ardisa is a standalone language/runtime; there is no claim of Rust std or Cargo ecosystem compatibility. |
| no_std, allocators and custom targets | Not implemented / not evidenced | No supported target matrix, allocator contract, no_std core, or target-specific ABI contract is established. |
| SIMD, intrinsics, GPU/NPU backends | Not implemented / not evidenced | Tensor/quantization runtime paths exist, but hardware-specific support requires real lowering, target checks and reproducible benchmarks. |
| Formatting, diagnostics, compiler tooling | Partial | Parser/formatter, source spans, diagnostics, structural edits and compiler protocol exist; Rustfmt/rust-analyzer/cargo-clippy compatibility is not implied. |
| Deterministic native executable format | Partial | ARDISA-EXEC-V1 and native VM validation exist for a subset; complete language coverage and source-to-runtime parity remain gated. |
| Self-hosting | Not complete | Bootstrap probes and evidence contracts exist; full source compilation and Stage 0→1→2→3 byte-identical reproducibility is still required before claiming self-hosting readiness. |

## Implementation principles

1. Do not promise Rust feature parity by counting runtime helper functions or generated Rust output.
2. Every capability marked supported must have source syntax/AST, semantic and ownership checks, typed IR, deterministic native encoding, verifier/runtime behavior, positive and adversarial tests, and documented limitations.
3. Keep unsupported features rejected explicitly; never silently fall back to the host Rust compiler for trusted compilation.
4. Prefer a coherent subset over superficial syntax coverage. Implement references/borrowing, traits/generics, enums/pattern matching, and error propagation as integrated language features before advertising Rust-class safety.
5. Rust interoperability remains a narrow opt-in boundary and must not become a trusted compiler dependency.

## Recommended priority order

1. **P0 — Language contract and executable path:** publish capability statuses; ensure unsupported syntax fails closed; source-to-native conformance suite.
2. **P1 — Type-system foundation:** user-defined structs/enums, exhaustive pattern matching, generic functions/types and trait constraints.
3. **P1 — Ownership soundness:** flow-sensitive move paths, borrow regions, reborrowing, aliasing and drop semantics with adversarial tests.
4. **P1 — Error semantics:** Option, Result, propagation and conversion rules.
5. **P2 — Closures and iterators:** capture modes, callable traits and iterator/for semantics.
6. **P2 — Concurrency:** async/await and cancellation-safe structured task semantics, or explicitly retain a smaller documented concurrency model.
7. **P2 — Module/package system:** visibility, import resolution, package/dependency and build reproducibility.
8. **P3 — Advanced Rust features:** const generics/evaluation, macros, attributes/cfg, unsafe contracts, FFI aggregates and custom targets.
9. **P3 — Ecosystem/backend capabilities:** std-like library surface, no_std, target-specific SIMD/accelerators, tooling integrations, only when there is a measured need and concrete implementation.

This audit should be revised only with evidence from source-to-runtime tests and explicit language specification decisions.
