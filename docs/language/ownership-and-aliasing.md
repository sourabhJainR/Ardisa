# Ardisa ownership and collection safety contract

This document describes the currently evidenced Ardisa subset. It is not a statement of Rust compatibility.

## Stable bindings and owned values

- Every parameter and local declaration has a stable `BindingId` derived from its AST declaration identity. Source names are only the current lexical lookup keys; shadowing introduces a distinct binding and must not overwrite the shadowed declaration's state.
- Int, Bool, Unit, and the currently supported scalar numeric values are copy-class values.
- String and List<T> are owned values. Binding an owned value to a new local consumes the source place under the flow-sensitive move checker. A later use of a moved name is rejected with AIF400.
- Assignment to an existing local reinitializes that local only after the assigned expression passes analysis. Conditional and loop-body moves are merged conservatively; use after a potentially moved value is rejected. Loop-local bindings have distinct identities and must not alter unrelated outer bindings.
- The VM's internal cloning for loads/serialization is an implementation detail; it does not grant source programs a reference or shared-mutation capability.

## Path-aware collection moves

- An owned collection place is tracked by a root binding identity plus a sequence of projections. A non-negative integer index is a static projection; a non-constant or negative index is dynamic and may alias any element at that depth.
- Moving a statically indexed element marks that path moved while leaving provably disjoint static paths available. Reading the same moved path is rejected with AIF400.
- A dynamic-index move is conservative: because its target is unknown, later accesses to potentially aliased elements are rejected. Moving an element partially moves its source collection; using the whole collection is rejected until the language's supported reinitialization rules establish availability.
- Branch joins combine the states of matching paths. A path moved on only one reachable branch becomes maybe-moved; a branch that cannot fall through does not poison the state of reachable paths. Loop joins account for the possibility that a loop executes zero times.
- Reinitializing a statically identified indexed path restores that path and its descendants only; it does not silently restore a moved sibling or an unknown dynamic path.

## Borrowing and alias checks

- The analyzer records shared and mutable accesses against stable binding identities and checks overlapping operation regions. Mutable access to an indexed place is carried through intermediate projections so a nested write cannot be downgraded to a shared access to the root collection.
- Mutation operations such as indexed assignment and push are analyzed as mutable access to the named collection. Overlapping shared/mutable regions and moves while an operation's borrow is active are rejected.
- These checks remain a partial safety model. They do not implement a complete CFG-based borrow checker, native reference values, reborrowing, non-lexical lifetimes, interior mutability, raw pointers, pointer provenance, or a lifetime-parameterized API. No Rust-equivalent soundness claim is made.
- The current language does not expose a complete reference type or reference-returning functions. Reference-like syntax that is not part of the grammar must be rejected; it must not be silently interpreted as a value.

## Slices, lifetimes, and cleanup

- List[start:end] and String[start:end] return owned copies, not views. List slice elements are recursively validated and copy-isolated from the source; string slice bounds must be UTF-8 scalar boundaries.
- Lists are value containers in the supported subset. Source code cannot construct two mutable references to the same list storage because native reference values are not implemented. Mutation is bounded by runtime item-count, nested-value-node/depth, frame-footprint, and configured string-byte limits.
- Native values are reclaimed by ordinary host-runtime value destruction when they leave frames or are replaced. There is no user-defined Drop trait, deterministic user cleanup hook, or guarantee of Rust drop-order equivalence. Cleanup on early returns and propagated errors remains a separate implementation gate.
- Unsupported iterator syntax is rejected during semantic analysis; no implicit iteration or collection mutation semantics are inferred.

## Evidence and limits

The ownership analyzer tests exercise stable declaration identity, shadowing, static and dynamic path moves, branch/loop joins, indexed reinitialization, modeled borrow conflicts, and early-return reachability. Native source-to-runtime tests exercise owned collection behavior. These tests do not establish complete reference/lifetime/drop semantics or independent aliasing soundness. Keep the compatibility matrix at Partial until the missing semantics have independent source-to-runtime, adversarial, differential, holdout, and bootstrap evidence.
