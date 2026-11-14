# AI-native source feature matrix

This matrix separates source-language behavior from helper APIs and from claims that require independent evidence. Status is limited to the implemented subset; it does not imply general autograd, model-tokenizer compatibility, cryptographic trust, OS isolation, accelerator support, or complete self-hosting.

| Feature | Source / AST | Semantic validation | Native artifact/runtime path | Limits and exclusions | Status |
|---|---|---|---|---|---|
| Trace declaration | Parsed into `Module.constructs`; focused Trace source artifact path exists | Required signature policy and structured event shape are checked | Focused native Trace validation artifact plus trusted host receipt adapter; standard `compile_source` still rejects Trace operations | Signer, verifier, key trust/rotation, source-byte authentication, and complete event fields must be explicit; no inferred signing authority | **Focused runtime path only; standard compiler fail-closed** |
| Cell declaration | Parsed; AST retains declaration | Narrow `value: Int` and supported integer-comparison invariant subset; explicit initialization and type-checked updates | Standard compiler emits deterministic native Cell initialization/update instructions; invariant checks run before commit | General field types and arbitrary invariant expressions remain unsupported | **Supported bounded source-to-native subset via PR #371** |
| Vault declaration | Parsed; AST retains declaration | Requested capability must be explicitly declared; unsupported clauses/members reject | Standard compiler emits a native authorization instruction for the supported operation path | Not an OS sandbox; arbitrary host effects are not automatically intercepted; deny/effect policies remain incomplete | **Supported bounded authorization subset via PR #371** |
| Proof declaration | Parsed; AST retains declaration | Focused Proof policy validation and runtime evidence outcome checks exist | Focused native Proof artifact validates supplied outcomes; standard `compile_source` still rejects Proof operations | Outcomes remain runtime-supplied and are not proofs produced by the compiler; trusted evidence producer identity and per-invocation boundary must be explicit | **Focused runtime path only; standard compiler fail-closed** |
| Phase declaration | Parsed transition edges; typed metadata lowering exists | Explicit initial state and declared-edge checks; invalid transitions reject without state mutation | Standard compiler emits deterministic native phase-start/transition instructions for the supported source operation subset | No general workflow engine, scheduler, or external effect orchestration is implied | **Supported bounded source-to-native subset via PR #371** |
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

1. **Proof/Trace standard compiler integration:** add per-invocation evidence and structured event arguments plus explicit trusted verifier/signer interfaces. Do not synthesize evidence, signatures, actors, or event fields; preserve fail-closed behavior until the host trust boundary is complete.
2. **Construct conformance:** extend combined Cell/Phase/Vault tests and add cross-construct Proof/Trace tests, malformed artifacts, unauthorized-effect regressions, and failure-atomicity checks.
3. **AI-native primitive review:** confirm the bounded GradientTape, ActivationCache, and TokenizerState source subsets against exact-head evidence; keep unsupported tensor autograd, generic model tokenizers, and arbitrary recomputation explicitly excluded.
4. **Independent review and benchmark baseline:** review numerical determinism, bounds, verifier/runtime agreement, and honest capability claims; record reproducible benchmarks without claiming unsupported hardware acceleration.
5. **Self-hosting:** retain the Stage 0 -> 1 -> 2 -> 3 gate, byte-identical Stage 2/3 output, independent holdouts, and forbidden-host-fallback checks.

A source declaration without executable semantics must continue to fail compilation with AIF610 rather than being ignored. A runtime helper does not establish cryptographic trust, OS isolation, formal proof, Rust parity, or self-hosting.
