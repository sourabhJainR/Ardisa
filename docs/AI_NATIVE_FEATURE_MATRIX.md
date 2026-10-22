# AI-native source feature matrix

This matrix separates source-language behavior from helper APIs and from claims that require independent evidence. Status is limited to the implemented subset; it does not imply general autograd, model-tokenizer compatibility, cryptographic trust, OS isolation, accelerator support, or complete self-hosting.

| Feature | Source / AST | Semantic validation | Native artifact/runtime path | Limits and exclusions | Status |
|---|---|---|---|---|---|
| Trace declaration | Parsed into `Module.constructs`; compiler AST retains declaration | Required signature policy is checked by a source-backed host adapter | Source runtime can require a signer, verify bounded receipt chains, and request trusted source-backing verification before evidence-graph creation; native compilation still rejects with AIF610 | Signer, verifier, key trust/rotation, and source-byte authentication are external interfaces; no native Trace instruction/effect integration | **Source host-runtime adapter; native source path fail-closed** |
| Cell declaration | Parsed; AST retains declaration | Source-backed host adapter validates a narrow `value: Int` and integer-comparison invariant subset | Explicit initial values and updates are invariant-checked; rejected updates preserve state; native compilation still rejects with AIF610 | No general expression evaluator, source-level native mutation instruction, or native artifact representation | **Bounded source host-runtime adapter; native source path fail-closed** |
| Vault declaration | Parsed; AST retains declaration | Source-backed host adapter accepts only a closed capability vocabulary and explicit grants | Host operations are gated against the full required capability set before the closure runs; native compilation still rejects with AIF610 | Not an OS sandbox; arbitrary source effects are not automatically intercepted; denied clauses do not establish isolation | **Source host-runtime adapter; native source path fail-closed** |
| Proof declaration | Parsed; AST retains declaration | Source-backed host adapter accepts named requirements, `required: true`, `ensures: true`, and `on_unknown: reject` only | Missing, failed, or unknown required outcomes reject; native compilation still rejects with AIF610 | Outcomes are caller-supplied and not proofs produced by the compiler; non-trivial postconditions and other unknown policies fail closed | **Narrow source host-runtime adapter; native source path fail-closed** |
| Phase declaration | Parsed transition edges; typed metadata lowering exists | Duplicate/self-transition and graph validation | Source-backed host adapter requires explicit initial states and rejects invalid transitions without changing state/history; native source compilation still rejects with AIF610 | No source-level phase instruction or effect-boundary integration in ARDISA-EXEC-V1 | **Source host-runtime adapter; native source path fail-closed** |
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

1. **AI construct native integration:** host-runtime adapters now exist for Phase, Cell, Vault, Proof, and Trace, and are composed by `SourceConstructRuntime`; this is not native execution. Carry each construct into verified ARDISA-EXEC-V1 instructions, artifact serialization, per-invocation state, and actual effect boundaries. Keep declarations fail-closed until each contract is complete.
2. **Conformance strengthening:** ensure every AI-native source built-in has a named end-to-end test covering artifact round trip, invalid inputs, resource ceilings, and failure atomicity. Add mutation tests where coverage is missing.
3. **Independent review and benchmark baseline:** review numerical determinism, bounds, verifier/runtime agreement, and honest capability claims; record reproducible benchmarks without claiming unsupported hardware acceleration.
4. **Self-hosting:** retain the Stage 0 -> 1 -> 2 -> 3 gate, byte-identical Stage 2/3 output, independent holdouts, and forbidden-host-fallback checks.

A source declaration without executable semantics must continue to fail compilation with AIF610 rather than being ignored. A runtime helper does not establish cryptographic trust, OS isolation, formal proof, Rust parity, or self-hosting.
