# Ardisa roadmap

The implementation is evidence-gated. A phase is complete only after its executable acceptance checks pass in CI and the status is reflected here.

## Completed foundations

- Tiny .ardisa language surface, lexer, parser, AST, spans, formatter and deterministic diagnostics.
- Flow-sensitive ownership state propagation, explicit borrow regions, and conflict detection.
- Read/write/call effect analysis with resource-level read/write sets and transitive invalidation.
- Static structured-concurrency checking with native scope/spawn/join/cancel execution, deterministic cleanup, cancellation and child-failure propagation.
- Safe Rust interop with scalar and aggregate ABI contracts, ownership declarations, isolated unsafe wrappers, and compile-time Rust ABI fixtures.
- Mutation/adversarial verification with deterministic generated cases, malformed corpus coverage, compiler error/crash invariants, and native/reference mismatch checks.
- Versioned AI-native compiler protocol with AST/type/effect queries, structured responses, atomic edits, source/evidence mapping, deterministic wire framing, and bounded stdio transport.
- Persistent compiler learning with observed/verified-repair states plus deterministic repair, replay, regression, promotion and rollback provenance. V1/V2 learning files remain readable.
- Dependency-free native execution across the supported IR operation/value surface, including aggregate List/Result parameters and deterministic compile/runtime/size measurements.
- Reproducible stage2 Ardisa-authored compiler-pipeline replay with explicit bootstrap evidence.
- Native VM dispatch borrows immutable instructions rather than cloning the instruction enum at every dispatch; a repeatable workload benchmark now reports throughput and a clearly labeled static estimate of avoided embedded-payload copying.

## Production evidence status

### A. Aggregate/reference ABI — complete
- Safe List/Result interop.
- Ownership-aware ABI contracts.
- Compile-time ABI fixtures.

### B. Mutation-based verification — complete
- Source mutations.
- Malformed/adversarial corpus.
- Compiler crash/error invariants.
- Native/reference mismatch detection.

### C. Protocol completion — complete
- Rich AST/type/effect queries.
- Structured response framing.
- Bounded stdio transport.
- Source/evidence mapping.

### D. Compiler learning provenance — complete
- Repair provenance.
- Replay linkage.
- Regression linkage.
- Promotion/rollback evidence.

### E. Native backend completion — supported surface complete
- Backend-independent IR operations are lowered to native instructions.
- Aggregate native values and parameters are executable.
- Differential/generated corpus and deterministic compile/runtime/size measurements are retained.
- Native dispatch benchmark covers arithmetic/branch control flow, short string operands, and 4 KiB embedded string operands. CI checks that it builds and returns correct results; performance is reported without a brittle timing threshold. The static avoided-copy estimate is not an allocator measurement.

### F. Bootstrap — deterministic replay complete; true self-hosting remains open
- Ardisa-authored lexer/parser/AST/semantic/IR sources execute on the native backend.
- Stage2 replay is deterministic and independently checked against the native pipeline.
- The evidence model explicitly records that the Rust host is still required to construct the current compiler pipeline.
- True compiler self-compilation, deterministic self-rebuild from an Ardisa compiler executable, and an independent bootstrap from that executable are not yet complete.

## True self-hosting gate

True self-hosting is intentionally unclaimed.

The gate opens only when all of the following are demonstrated in CI:

1. The Ardisa compiler subset is written in Ardisa.
2. A previously built Ardisa compiler executable compiles that compiler source without invoking the Rust-hosted compiler.
3. The resulting compiler executable recompiles the same source deterministically.
4. A clean, independent bootstrap verifier reproduces the same compiler artifact without relying on the Rust compiler pipeline.
5. The resulting artifact passes the full parser, semantic, ownership, IR, native, protocol, mutation and differential acceptance suite.

Until those checks pass, self_hosting_ready must remain false and release documentation must not describe Ardisa as self-hosting.
