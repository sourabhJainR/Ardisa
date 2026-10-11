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