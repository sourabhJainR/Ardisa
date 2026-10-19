# AI-native source feature matrix

This matrix records the actual source-to-runtime status, not just parser acceptance or the existence of Rust runtime APIs.

| Feature | Source / AST | Semantic validation | Typed IR | ARDISA-EXEC-V1 / runtime | Current status |
|---|---|---|---|---|---|
| Trace declaration | Parsed into `Module.constructs` | Structural clauses; required signature | Metadata lowering exists | Not encoded as executable behavior; no signing/redaction runtime contract | **Partial / fail-closed** |
| Cell declaration | Parsed | Required invariant clause; duplicate/member checks | Metadata lowering exists | No source-level state binding, mutation, or invariant enforcement | **Partial / fail-closed** |
| Vault declaration | Parsed | Capability/deny clauses structurally checked | Metadata lowering exists | No executable effect-boundary capability enforcement | **Partial / fail-closed** |
| Proof declaration | Parsed | Required/ensures, reject-on-unknown and boolean required clause | Metadata lowering exists | No proof-obligation lifecycle or runtime promotion state | **Partial / fail-closed** |
| Phase declaration | Parsed transition edges | Duplicate/self-transition checks | Metadata lowering exists | No executable state machine or invalid-transition runtime operation | **Partial / fail-closed** |
| Probabilistic / Guaranteed | Rust runtime APIs | Some constructor-level checks | Not first-class source values | Not a general source-language ABI/runtime type | **Runtime API only** |
| GraphTensor | Rust runtime API with node/edge checks | Not first-class source typing | Not source typed IR | No source-level executable construct | **Runtime API only** |
| Tensor / Embedding / QuantizedTensor | Selected bounded source operations exist | Type and shape checks for implemented constructors | Selected native lowering exists | Implemented subset only; no blanket accelerator/model guarantees | **Partial** |
| Trace events / distributions / semantic values | Rust runtime APIs | Not a general source type system | Not generally represented in source typed IR | Not general source-level constructs | **Runtime API only** |

## Promotion rule

A construct must pass all of: source syntax, semantic/type checks, ownership/effect checks, typed IR, deterministic native encoding and artifact verification, runtime enforcement, positive and adversarial source-to-runtime tests, mutation tests, independent differential/holdout corpus, and bootstrap reproducibility. A declaration that has no executable semantics must continue to fail compilation with AIF610 rather than being ignored.

The tests in `crates/aifocus-core/tests/ai_constructs_contract.rs` intentionally prove both sides of the boundary: well-formed declarations reach typed metadata, but regular executable compilation remains rejected until the end-to-end runtime path exists.
