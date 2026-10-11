# Ardisa AI-native type feature matrix

**Target:** First-Class AI Primitives Language Specification, Revision 2.0 (user-provided).
**Status:** Gap inventory and implementation contract. This matrix is not evidence that a feature is implemented.

A feature counts as implemented only when the .ardisa source is parsed into a structured AST, its semantics and ownership/effects are checked, typed IR preserves its meaning, deterministic ARDISA-EXEC-V1 encoding carries it, runtime operations enforce its invariants, and positive plus negative end-to-end tests pass. A Rust runtime struct or accepted syntax alone does not satisfy this gate.

## Required feature matrix

| Primitive / family | Required source-level contract | Static checks | Native/runtime obligations | Status from current repository evidence |
|---|---|---|---|---|
| Tensor | Element type, rank and dimensions; initialization; indexing/slicing; elementwise arithmetic; matrix contraction | Shape/rank compatibility, broadcasting, contraction dimensions, numeric types, view ownership and aliasing | Bounded allocation, shape/stride validation, checked indexing, deterministic arithmetic semantics; hardware acceleration only when implemented and measured | Runtime Tensor API exists; Rev. 2.0 source grammar, typed IR, native encoding and end-to-end proof are not established |
| Embedding | embedding<Domain, N> with nominal domain and dimension | Domain declaration/resolution, dimension equality, domain isolation; explicit projection only | Validated finite coordinates and dimension; similarity operation with defined zero-vector behavior | Runtime embedding API exists; nominal source type and compiler enforcement are not established |
| Distribution | distribution<Kind> and constructors such as Normal(mu: ..., sigma: ...) | Recognized distribution kind and parameter validation where static | Defined RNG algorithm/state, reproducible seed and draw order, support checks and deterministic artifact semantics | Runtime distribution API exists; Rev. 2.0 constructors and sampling operator are not established |
| Probabilistic | Bounded confidence value, explicit thresholded conditions and fuzzy operators | Range/finite checks, operator typing, threshold required for control flow | Reject invalid confidence values; define AND, OR, NOT algebra; avoid implicit truthiness | Probability and Probabilistic<T> runtime APIs exist; native source type/operator semantics are not established |
| Guaranteed / validated | guaranteed<T, Constraint> with explicit construction and validation policy | Constraint well-formedness, proof status, composition compatibility | Validation at dynamic/FFI boundaries; no invalid value admitted through safe operations; no silent coercion or implicit repair | Runtime Guaranteed<T> wrapper exists; source constraints and proof-carrying typed IR are not established |
| GraphTensor | GraphTensor<Schema>, schema declarations, graph open/build, node/edge operations | Schema and domain compatibility; valid references and index ranges; sparse structure invariants | Validate CSR/graph metadata, bounded allocation and indices; explicit graph operations; no claim of GPU/SpMM acceleration without evidence | Runtime graph/tensor metadata API exists; source schema semantics and native graph operations are not established |
| Quantization | f16, bf16, int8, int4, quant4 and explicit conversions | Defined ranges, scale compatibility, overflow mode and conversion rules | Canonical storage/encoding; checked, wrapping or saturating behavior specified per type; no scale mismatch | Some numeric support may exist in runtime APIs; Rev. 2.0 storage/arithmetic and native ABI are not established |
| Stream (proposed) | Stream<T, Bounded(N)> and Stream<T, Unbounded> with push/pop/window/eviction | Capacity and mode validation, ownership, boundedness and effect checks | Explicit overflow/underflow errors, bounded resource use, concurrency and cancellation semantics; deterministic behavior where promised | Proposed only; not to be treated as normative or implemented |
| Trace | Typed, redacted provenance records with source/build digests and optional attestation metadata | Schema, redaction policy, evidence status and provenance linkage | Canonical digest generation; configured signer verification; bounded runtime events; never capture raw prompts/secrets by default | Construct metadata work in PR #135; native encoding/runtime enforcement not established |
| Cell | Typed values with predicates/invariants | Predicate validity, static proof when possible, ownership and mutation checks | Validate construction, mutation, and FFI boundaries; explicit failure policy | Construct metadata work in PR #135; native enforcement not established |
| Vault | Explicit capability/effect declarations | Reject undeclared effects and ambient authority where statically visible | Explicit non-forgeable capability handles and checks at actual effect boundaries; no automatic OS sandbox claim | Construct metadata validation is implemented, but capability runtime enforcement is not established |
| Proof | Preconditions, postconditions, invariants, resource bounds | Distinguish proved, runtime_checked, unknown, failed; fail closed on required unknowns | Emit assertions for dynamic checks; evidence report must not equate tests with formal proof | Construct metadata validation is implemented; proof engine/evidence report enforcement is not established |
| Phase | Named state graph and transitions | Duplicate/unknown state checks, reachability and transition validity | Validate untrusted/serialized states and reject illegal transitions; authorization remains separate | Construct metadata validation is implemented; native transition operations are not established |

## Normative semantics that must be resolved before claiming conformance

1. **Numeric safety:** define NaN/infinity behavior, integer overflow, float precision, casts, packed int4, quant4 scale representation, and checked/wrapping/saturating operations.
2. **Tensor shapes:** define rank rules, unknown dimensions, broadcasting result inference, contraction rules, indexing/slicing bounds, aliasing and view lifetimes.
3. **Domain isolation:** define nominal domain identity and explicit projection; same-sized domains must not compare as interchangeable.
4. **Sampling:** specify the PRNG algorithm and version, seed scope, distribution identity, parallel draw ordering, parameter validation, and cross-platform reproducibility limits. A hardware TRNG cannot promise deterministic replay.
5. **Uncertainty:** specify confidence validation and fuzzy algebra. Do not silently clamp invalid probabilities; return a diagnostic/error or use an explicitly specified clamping operation.
6. **Guarantees:** distinguish static proof from runtime validation and unknown status. Relaxation must not erase a safety obligation without an explicit, auditable escape hatch.
7. **Resources and ownership:** establish allocation budgets, view lifetimes, mutation exclusivity, stream bounds, cancellation, and deadlock/resource-starvation behavior.
8. **Backend claims:** LLVM/MLIR, AMX, Tensor Cores, GPU/VRAM, SpMM fusion, branchless lowering and zero-cost claims require implemented lowering plus target-specific tests/benchmarks. They are not implied by syntax or type declarations.
9. **ABI:** define the actual Ardisa native artifact ABI first. Rust repr(C) examples are design references, not proof of a stable cross-backend ABI.
10. **Diagnostics and evidence:** stable diagnostic codes; negative tests for every invalid case; deterministic artifacts; differential/holdout tests; stage-2/stage-3 bootstrap reproducibility.

## Required implementation sequence

1. Preserve the existing fail-closed construct validation. Typed construct metadata and deterministic encoding are tooling foundations, not executable language semantics.
2. Complete KVCache and ContextWindow source-to-runtime integration: parser/AST, semantic rules, typed IR, native instruction encoding/verifier, runtime behavior, and end-to-end positive/negative tests.
3. Then implement MoERouter and VectorIndex, followed by MCTSTree, AttentionBlock and LoRAAdapter, GradientTape and ActivationCache, and TokenizerState. Each construct must add typed IR and deterministic native encoding; unsupported operations must fail explicitly.
4. Implement bounded native runtime representations and resource checks for the selected subset.
5. Add .ardisa end-to-end executable fixtures for each newly supported construct, with observable assertions and adversarial invalid cases.
6. Add quantization and numeric edge-case tests, domain-isolation tests, sampling reproducibility tests, resource-limit tests, and mutation tests.
7. Expand the supported set only after CI passes for workspace/all targets, native phases, generated differential/holdout corpus, and bootstrap stage-2/stage-3 reproducibility.

## Acceptance rule

Until a row has source syntax, AST, semantic rules, typed IR, native encoding, runtime enforcement, and passing end-to-end tests, label it **partial** or **unsupported**. Do not label the specification as fully incorporated, and do not claim general AI capability or accelerator support based on Rust APIs alone.


## Rust language parity audit

The current Rust capability inventory is maintained separately in [RUST_CAPABILITY_COMPATIBILITY_MATRIX.md](RUST_CAPABILITY_COMPATIBILITY_MATRIX.md). Rust interoperability, generated Rust output, and native Ardisa language support are separate capabilities; none should be used as evidence for the others.
