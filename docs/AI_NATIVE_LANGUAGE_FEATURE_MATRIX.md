# AI-native language feature matrix

This matrix distinguishes source syntax from implemented language semantics. A parsed declaration is not a supported runtime feature.

Status terms: Parsed means represented in the AST; partially validated means only listed static checks exist; fail-closed / unsupported means compilation must not accept the feature as implemented; not evidenced means no end-to-end source-to-execution evidence has been established.

| Feature | Syntax / AST | Semantic checks in this phase | Typed IR | ARDISA-EXEC-V1 runtime | Evidence status |
|---|---|---|---|---|---|
| Trace | Declaration, fields, clauses | Duplicate declarations/members; clause-kind validation | Not represented | Not represented | Fail-closed; AIF610 |
| Cell | Declaration, fields, clauses | Duplicate declarations/members; clause-kind validation | Not represented | Not represented | Fail-closed; AIF610 |
| Vault | Declaration, fields, capability/deny clauses | Duplicate declarations/members; clause-kind validation | Not represented | No effect-boundary enforcement from declarations | Fail-closed; AIF610 |
| Proof | Declaration, fields, proof clauses | Duplicate declarations/members; clause-kind validation | Not represented | No proof-obligation execution from declarations | Fail-closed; AIF610 |
| Phase | Declaration and transition edges | Duplicate edges; self-transition rejection; field/clause-kind rejection | Not represented | No phase-transition runtime enforcement | Fail-closed; AIF610 |
| Probabilistic<T> | Generic type syntax | Existing general function type checks only; no uncertainty semantics | Generic type can appear in signatures | No dedicated native representation established | End-to-end support not evidenced |
| Guaranteed<T> | Generic type syntax | Existing general function type checks only; no guarantee semantics | Generic type can appear in signatures | No proof/guarantee enforcement established | End-to-end support not evidenced |
| MoERouter (bounded built-in subset) | `moe_router(expert_count, top_k)` and `moe_route(router, logits, mask)` calls | Expert count 1..=256; top-k 1..=32 and <= expert count; homogeneous logits/mask types; runtime exact-length, finite-logit, and all-masked checks | Nominal `MoERouter` identity and route call preserved | `MakeMoERouter` / `MoERoute` in ARDISA-EXEC-V1; stable top-k ordering and normalized selected weights | Supported bounded built-in subset after exact-head CI/Native Phases gates; generic `MoERouter<EXPERTS, TOP_K>` syntax and hardware acceleration remain unsupported |
| VectorIndex (bounded built-in subset) | `vector_index`, `vector_index_insert`, and `vector_index_search` calls | Domain/dimension bounds; finite coordinates; exact dimensions and domain isolation; unique non-negative IDs; bounded retained storage/search work; F32 distance output range | Nominal `VectorIndex` identity and `VectorHit` tuple-style fields | `MakeVectorIndex`, `VectorIndexInsert`, `VectorIndexSearch` in ARDISA-EXEC-V1; squared-Euclidean distance with deterministic ascending-ID ties | Supported bounded built-in subset after exact-head CI/Native Phases gates; squared Euclidean only; approximate search and quant4/AVX-512/GPU/NPU acceleration remain unsupported |
| KVCache (bounded built-in subset) | `kv_cache`, `kv_append`, `kv_len`, `kv_capacity`, and attention-cache calls | Capacity/retained-scalar budgets, key/value dimension compatibility, append policy, finite tensor values, and bounded cache reads | Nominal `KvCache` identity retained for supported calls | Deterministic `MakeKvCache`/`KvAppend` and attention-cache instructions in ARDISA-EXEC-V1 | Supported bounded subset via #168/#169; generic `KVCache<T, HEADS, MAX_SEQ>` syntax and broader overwrite semantics remain unsupported unless explicitly covered by the selected built-in policy |
| ContextWindow (bounded built-in subset) | `context_window`, `context_append`, `context_len`, `context_bytes`, `context_tokens` | Segment, byte, and token budgets; supported modality tags; deterministic oldest-first eviction | Nominal `ContextWindow` identity retained | Deterministic context-window native instructions and bounded payload retention | Supported bounded subset via #151/#164/#168/#169; device paging/DMA and schema generics unsupported |
| MCTSTree (bounded built-in subset) | `mcts_tree`, root/expand/child/update/backpropagate/len calls | Branch/node/depth bounds, prior-child indices, finite rewards, mutation atomicity | Runtime metadata preserved by native operations | Deterministic bounded native arena operations in ARDISA-EXEC-V1 | Supported bounded subset; no generic compile-time shape parameters or full MCTS algorithm claim beyond implemented operations |
| AttentionBlock + LoRAAdapter (bounded built-in subset) | `attention_block_new/apply/cached`, `lora_adapter_new/apply` | Checked hidden/head dimensions, stable softmax and causal masking; cache compatibility; LoRA rank/shape/finite-value/aggregate-storage checks | Nominal identities and constructor metadata retained through typed IR | Deterministic bounded native values/instructions; LoRA matrices owned immutably and low-rank delta returned without mutating base weights | Supported bounded subset after PR #181 exact-head CI #2098 and Native Phases #520; compile-time generic shape parameters, fused kernels, and hardware acceleration unsupported |
| GradientTape (bounded scalar autodiff subset) | `gradient_tape`, `gradient_input`, `gradient_add`, `gradient_mul`, `gradient_backward`, `gradient_value`, `gradient_get` | Capacity/slot bounds, prior-slot-only references, finite inputs/intermediates/gradients, explicit backward seeding, deterministic repeated-operand accumulation | Nominal `GradientTape` identity and operation dependencies preserved through typed IR | Deterministic `MakeGradientTape` and scalar reverse-mode operations in ARDISA-EXEC-V1 | Supported only for Input/Add/Mul after PR #188 exact-head CI #2122 and Native Phases #544; tensor gradients and arbitrary autograd unsupported |
| ActivationCache (bounded checkpoint subset) | `activation_cache`, `activation_store`, `activation_restore`, `activation_cache_len` | Capacity/stride/byte budgets, stride-aligned unique checkpoint IDs, finite tensor payloads, shape/byte accounting, missing-checkpoint rejection | Nominal `ActivationCache` identity preserved through typed IR | Deterministic bounded checkpoint storage and exact shape/value restore in ARDISA-EXEC-V1 | Supported storage/restore subset on PR #190; exact-head gates for the matrix-updated commit must pass before final evidence is published; general graph recomputation unsupported |
| GraphTensor | Named type syntax | Existing general function type checks only | Named type can appear in signatures | No tensor/graph ABI established | End-to-end support not evidenced |
| Embeddings, tensors, distributions, quantization, semantic values, trace events | Some Rust-side APIs/types exist | Per-type source syntax and type rules not established as a complete language contract | Not established end to end | No general accelerator/model capability implied | Requires explicit support or exclusion decision |

## Diagnostics introduced by construct structural validation

- AIF611: duplicate construct declaration name.
- AIF612: a member form is not valid for that construct kind.
- AIF613: duplicate field/clause member name.
- AIF614: clause is not permitted for that construct kind.
- AIF615: phase self-transition.
- AIF616: duplicate phase transition.

These checks are intentionally additive. They do not remove AIF610, which continues to reject every construct declaration until semantic output, typed IR, deterministic native encoding, and runtime behavior are implemented and tested end to end. Structural validation is not a security sandbox, proof checker, provenance signer, ownership system, or evidence of memory safety.

## Next implementation phases

1. Carry typed construct declarations into semantic output and typed IR.
2. Define per-construct field types, clauses, capabilities, effects, proof statuses, ownership, and transition graph rules.
3. Implement deterministic native representation and enforce operations at runtime boundaries with resource limits.
4. Add .ardisa end-to-end fixtures, negative/mutation tests, and preserve bootstrap, differential/holdout, Native Phases, workspace, and benchmark gates.
5. For each AI-native data type, either implement source syntax + type rules + ABI/runtime behavior + observable tests, or explicitly document its exclusion and reason.