# AI-native source feature matrix

This matrix separates source-language behavior from helper APIs and from claims that require independent evidence. Status is limited to the implemented subset; it does not imply general autograd, model-tokenizer compatibility, cryptographic trust, OS isolation, accelerator support, or complete self-hosting.

| Feature | Source / AST | Semantic validation | Native artifact/runtime path | Limits and exclusions | Status |
|---|---|---|---|---|---|
| Trace declaration | Parsed into `Module.constructs`; compiler AST retains declaration | Structural clauses and required signature | Typed metadata encoding exists; source compilation rejects the declaration with AIF610 | TraceLedger is a helper API. Production signing, key distribution/rotation, and source effect integration are not established | **Helper/metadata only; source path fail-closed** |
| Cell declaration | Parsed; AST retains declaration | Required invariant clause and duplicate/member checks | Metadata lowering exists; source compilation rejects the declaration with AIF610 | InvariantCell helper does not provide source-level state mutation or invariant enforcement | **Helper/metadata only; source path fail-closed** |
| Vault declaration | Parsed; AST retains declaration | Capability vocabulary and allow/deny consistency checks | Metadata lowering exists; source compilation rejects the declaration with AIF610 | VaultRuntime is not an OS sandbox; arbitrary source effects are not intercepted | **Helper/metadata only; source path fail-closed** |
| Proof declaration | Parsed; AST retains declaration | Required/ensures and reject-on-unknown checks | Metadata lowering exists; source compilation rejects the declaration with AIF610 | A digest or ledger status alone does not prove a claim; source proof enforcement is not implemented | **Helper/metadata only; source path fail-closed** |
| Phase declaration | Parsed transition edges; typed metadata lowering exists | Duplicate/self-transition and graph validation | `PhaseMachine::from_typed_construct` bridges typed metadata to runtime; native source compilation still rejects the declaration with AIF610 | Initial state must be supplied explicitly. No source-level phase instruction or effect-boundary integration yet | **Typed-metadata/runtime bridge; source path fail-closed** |
| Tensor / Embedding / QuantizedTensor | Selected source built-ins | Type, domain, shape, finite-value, and bound checks for implemented operations | Deterministic native instructions and bounded runtime operations exist | Only the implemented subset; no general accelerator or model-runtime guarantee | **Partial source-to-runtime support** |
| MoERouter / VectorIndex | Selected source built-ins | Type/domain/dimension/limit checks for supported operations | Bounded native operations; deterministic routing/search behavior | No hardware acceleration claim; supported metrics and operations are narrow | **Partial source-to-runtime support** |
| MCTSTree | Selected source built-ins | Bounds and operation type checks | Bounded native operations with node/branch/depth checks | Only the implemented indexed tree subset | **Partial source-to-runtime support** |
| AttentionBlock / LoRAAdapter | Selected source built-ins | Dimension, shape, rank, and causal-mode checks | Native instructions and bounded tensor execution | No generic compile-time shape parameters or accelerator claims | **Partial source-to-runtime support** |
| GradientTape | Source built-ins: `gradient_tape`, `gradient_input`, `gradient_add`, `gradient_mul`, `gradient_backward`, `gradient_value` / `gradient_get` | Capacity, tape type, slot ID and value checks | Native instruction encoding/decoding and runtime execution are present | Scalar F32/Int input subset; supported derivatives are add and multiply. No arbitrary autograd; prior-slot references, finite values and tape capacity are checked | **Bounded source-to-runtime subset** |
| ActivationCache | Source built-ins and named mutable cache binding | Capacity, stride, byte-budget, tensor and checkpoint validation | Native instructions support storing and restoring checkpoints and querying cache length | Bounded immutable tensor checkpoints; no claim of general graph recomputation. Callers own the recomputation strategy | **Bounded source-to-runtime subset** |
| TokenizerState | Source built-ins: constructor, `tokenizer_feed`, `tokenizer_flush`, `tokenizer_token_count`, `tokenizer_token` | State type, byte-list, vocabulary, bounds and mutation-target checks | Native instructions implement bounded incremental input and token retrieval | Explicit supplied vocabulary and deterministic matching only; no implied BPE, SentencePiece, normalization, or pretrained tokenizer support | **Bounded source-to-runtime subset** |
| Probabilistic / Guaranteed | Runtime APIs; not general first-class source values | Selected constructor checks | No general source ABI/runtime representation | Do not infer source support from Rust types alone | **Runtime API only** |
| GraphTensor | Runtime API | Bounded node/edge checks | No first-class source operation path | Source typing and native execution remain unsupported | **Runtime API only** |

## Evidence and promotion rules

A feature must not be marked fully supported merely because its parser, Rust helper, or native instruction exists. Promotion requires evidence on the exact commit for:

1. Normal `.ardisa` source compilation and observable execution.
2. Positive, negative, malformed-artifact, resource-limit, and mutation-atomicity tests.
3. Deterministic artifact round trips and verifier rejection of invalid programs.
4. Generated differential/holdout results and bootstrap Stage 2/3 reproducibility.
5. Workspace/all-target and Native Phases checks.
6. Documentation of the precise supported subset and known exclusions.

The previous main commit `85034a10445c0dceab614b6f616431b90eec4563` passed Rust workspace/all-target, generated differential/holdout, bootstrap reproducibility/evidence-contract, and Native Phases workflows. The matrix change in PR #310 also passed its exact-head CI and Native Phases workflows before merge. Post-merge checks must still be evaluated on the resulting main commit. These gates do not establish full language parity or full self-hosting. Keep `self_hosting_ready=false` unless the complete Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 and independent holdout contract is satisfied.

## Remaining phases

1. **AI construct native integration:** carry one of Trace, Cell, Vault, Proof, or Phase beyond typed metadata into verified native execution and actual effect boundaries. Keep declarations fail-closed until each contract is complete.
2. **Conformance strengthening:** ensure every AI-native source built-in has a named end-to-end test covering artifact round trip, invalid inputs, resource ceilings, and failure atomicity. Add mutation tests where coverage is missing.
3. **Independent review and benchmark baseline:** review numerical determinism, bounds, verifier/runtime agreement, and honest capability claims; record reproducible benchmarks without claiming unsupported hardware acceleration.
4. **Self-hosting:** retain the Stage 0 -> 1 -> 2 -> 3 gate, byte-identical Stage 2/3 output, independent holdouts, and forbidden-host-fallback checks.

A source declaration without executable semantics must continue to fail compilation with AIF610 rather than being ignored. A runtime helper does not establish cryptographic trust, OS isolation, formal proof, Rust parity, or self-hosting.
