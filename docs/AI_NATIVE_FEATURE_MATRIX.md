# AI-native source feature matrix

This matrix distinguishes three separate levels: source-language support, runtime helper APIs, and independently verified end-to-end behavior. A Rust API or a parsed declaration does not by itself make a construct executable in Ardisa.

| Feature | Source / AST | Semantic validation | Typed IR | Runtime helper APIs | ARDISA-EXEC-V1 source execution | Current status |
|---|---|---|---|---|---|---|
| Trace declaration | Parsed into Module.constructs; compiler AstArena retains declaration node | Structural clauses and required signature | Metadata lowering exists | Bounded TraceLedger; caller-supplied signer and separate TraceVerifier boundary; predecessor-chain and payload checks | Source-level trace is not lowered to executable behavior; production cryptography and trusted key distribution/rotation are external requirements | **Runtime foundation; source path fail-closed** |
| Cell declaration | Parsed; compiler AstArena retains declaration node | Required invariant clause; duplicate/member checks | Metadata lowering exists | InvariantCell validates initial/replacement values and commits updates only after validation | No source-level state binding, mutation, or invariant enforcement | **Runtime foundation; source path fail-closed** |
| Vault declaration | Parsed; compiler AstArena retains declaration node | Capability vocabulary, duplicate/conflicting allow-deny checks | Metadata lowering exists | VaultRuntime checks declared capabilities before invoking routed host operations | Source declarations do not yet intercept arbitrary effects or connect to executable instructions; this is not an OS sandbox | **Runtime foundation; source path fail-closed** |
| Proof declaration | Parsed; compiler AstArena retains declaration node | Required/ensures, reject-on-unknown and boolean required clause | Metadata lowering exists | Bounded proof status ledger; caller-supplied digest verifier can validate evidence bytes before recording | Source-level proof obligations are not lowered or enforced by native execution; digest shape or ledger state alone does not prove a claim | **Runtime foundation; source path fail-closed** |
| Phase declaration | Parsed transition edges; compiler AstArena retains declaration node | Duplicate/self-transition and graph checks | Metadata lowering exists | Bounded deterministic PhaseMachine with transition history | Source-level phase declarations are not encoded as executable state transitions | **Runtime foundation; source path fail-closed** |
| Probabilistic / Guaranteed | Rust runtime APIs | Some constructor-level checks | Not first-class source values | Selected constructor/runtime checks | Not a general source-language ABI/runtime type | **Runtime API only** |
| GraphTensor | Rust runtime API with node/edge checks | Not first-class source typing | Not source typed IR | Bounded metadata/runtime checks | No source-level executable construct | **Runtime API only** |
| Tensor / Embedding / QuantizedTensor | Selected bounded source operations exist | Type and shape checks for implemented constructors | Selected native lowering exists | Supported bounded subset | No blanket accelerator/model guarantees | **Partial** |
| GradientTape / ActivationCache | Not first-class source features | Source declarations remain fail-closed | No executable source lowering | Bounded scalar GradientTape and stride-based ActivationCache helpers are implemented; exact-head workspace, Native Phases, differential/holdout and bootstrap gates passed in PR #296 | No source-level ARDISA-EXEC-V1 operation; caller owns model execution and recomputation | **Runtime foundation validated; source integration open (#148)** |
| TokenizerState | Not a first-class source feature | Source declarations remain fail-closed | No executable source lowering | Bounded incremental UTF-8 ingestion, explicit vocabulary, deterministic IDs, and transactional chunk/finish rejection are implemented; exact-head workspace and Native Phases checks passed in PR #297 | No source-level ARDISA-EXEC-V1 operation; whitespace tokenization and explicit-vocabulary semantics are the current helper contract | **Runtime foundation validated; source integration open (#192)** |

## Promotion rule

A construct may be promoted to supported only after all of the following are demonstrated on the exact commit:

1. Source syntax and AST representation.
2. Semantic/type checks plus ownership and effect rules.
3. Typed IR that preserves the construct without dropping or weakening its meaning.
4. Deterministic ARDISA-EXEC-V1 encoding and artifact verification.
5. Runtime enforcement at the actual operation/effect boundary, with explicit resource limits.
6. Positive, negative, adversarial and mutation tests through the normal .ardisa compilation/execution path.
7. Independent differential/holdout results and bootstrap reproducibility evidence.
8. Updated documentation that states what is statically checked, runtime-checked, unknown or unsupported.

A declaration without executable semantics must continue to fail compilation with AIF610 rather than being ignored. Helper APIs remain useful building blocks but are not evidence of source-level integration, cryptographic trust, OS isolation, formal proof, or self-hosting.

## Current next steps

1. Complete the end-to-end source-to-runtime phases for Trace, Cell, Vault, Proof and Phase tracked by issue #132. Start with one construct and carry it through typed IR, deterministic encoding, verifier/runtime behavior and integration tests before expanding the supported subset.
2. Continue source-level AI data-structure integration under #148; GradientTape and ActivationCache still have runtime helpers only. TokenizerState source-to-runtime semantics are tracked separately by #192. Keep generic shapes and unsupported operations fail-closed.
3. Preserve the native self-hosting gate in issue #176: Stage 0 -> Stage 1 -> Stage 2 -> Stage 3, byte-identical Stage 2/3 artifacts, independent holdout tests, and no trusted host-Rust compiler fallback.
4. Run workspace/all-target, Native Phases, generated differential/holdout, and bootstrap checks on the exact PR head. Update capability claims only from those results.

The tests in crates/aifocus-core/tests/ai_constructs_contract.rs intentionally prove both sides of the current boundary: well-formed declarations reach typed metadata, but regular executable compilation remains rejected until the end-to-end runtime path exists.
