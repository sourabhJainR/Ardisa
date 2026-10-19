# Native references, aliasing, and pointer boundary

## Current language contract

Ardisa currently supports flow-sensitive move/availability analysis and a bounded borrow-analysis model over resolved places. The borrow analyzer tracks binding identity, static index projections, conservative dynamic-index aliasing, lexical scope depth, and overlapping shared/mutable regions. This is not a first-class reference type system and does not prove Rust-equivalent soundness.

The semantic checker rejects the native type constructors `SharedRef<T>`, `MutableRef<T>`, `RawPtr<T>`, `NonNull<T>`, and `Pin<T>` with diagnostic `AIF320`. Rejection is intentional: these forms do not have a complete source-to-runtime contract.

## Explicitly unsupported

- First-class references and lifetime/region parameters in source signatures.
- Raw-pointer creation, pointer arithmetic, dereference, nullability, alignment and bounds rules.
- Pointer provenance, aliasing guarantees equivalent to Rust's model, and `NonNull` invariants.
- Pinning and address-stability guarantees.
- FFI pointer validity beyond any explicitly documented external ABI contract.

A Rust wrapper, a parser accepting a spelling, or borrow-region metadata alone does not enable any of these features.

## Required implementation sequence before enabling a form

1. Specify source syntax and stable diagnostics, including invalid/dangling/lifetime-escaping cases.
2. Add AST and semantic typing rules with region and ownership constraints.
3. Carry reference/pointer provenance and mutability through typed IR.
4. Encode a versioned native representation and verify it before execution.
5. Enforce nullability, bounds, alignment, provenance, aliasing and invalidation at the actual runtime boundary where static proof is insufficient.
6. Add positive and adversarial source-to-runtime tests, mutation tests, independent differential/holdout cases, and reproducible bootstrap evidence.
7. Update `docs/rust_capabilities.json` only when every required evidence key exists and the feature is demonstrably supported.

Until all stages pass, unsupported forms remain rejected. This is a deliberately narrower contract, not a claim that all possible pointer operations have been implemented or that the language has Rust-equivalent memory safety.
