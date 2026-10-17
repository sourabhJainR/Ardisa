# Ardisa reference, pointer, aliasing, and pinning contract

No native reference or pointer form is currently supported. This is a deliberate fail-closed boundary, not a claim that the existing ownership checker proves Rust reference soundness.

## Rejected source forms

- Shared and mutable references (`&T`, `&mut T`) and raw pointer syntax (`*T`) are rejected by parsing.
- Pointer/reference wrapper type spellings such as `Pin`, `Pin<T>`, `NonNull`, `NonNull<T>`, `RawPtr<T>`, `SharedRef<T>`, and `MutableRef<T>` are rejected by semantic validation with AIF320.
- Reference-returning functions cannot enter typed IR or native execution until lifetime escape analysis and a runtime representation exist.
- A name that resembles a pointer wrapper does not gain pointer semantics through a generic or nominal type spelling.

## Safety properties not claimed

Ardisa currently has stable ownership binding identities, indexed place paths, conservative dynamic-index aliasing, and flow-sensitive move-state joins. Those checks do **not** implement or prove reference provenance, nullability, alignment, dereference bounds, reborrowing, lifetime variance, pinning guarantees, raw-pointer arithmetic, or full Rust aliasing rules.

No reference or pointer value may be treated as supported based on host Rust types, FFI wrappers, or ownership-only tests. Adding any such form requires an end-to-end source → semantic/ownership → typed IR → deterministic ARDISA-EXEC-V1 verification → runtime design and adversarial compile-fail/runtime corpus.

## Evidence

- `crates/aifocus-core/tests/phase3_ownership_fail_closed.rs`: reference/raw-pointer syntax and pointer-wrapper types are rejected before native execution.
- `crates/aifocus-core/src/sema.rs`: explicit AIF320 rejection for pointer/reference/pinning type spellings.
- `docs/RUST_CAPABILITY_COMPATIBILITY_MATRIX.md`: references, raw pointers, and pinning remain unsupported.
- Full ownership, borrowing, lifetime-region, and drop work remains tracked by #158.
