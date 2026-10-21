# Ardisa

Ardisa is an AI-first systems programming language designed to make software development **minimalistic, robust, effective, and traceable**.

Ardisa source files use the `.ardisa` extension. The extension is part of the language contract so tools, editors, agents, and CI can identify Ardisa source unambiguously.

## Design goals

- Rust-inspired performance and safety goals, with supported guarantees limited to independently tested language/runtime subsets.
- A smaller, easier-to-generate language surface.
- Intent-first declarations and explicit effects.
- Less lifetime ceremony in application code where the compiler can prove safety.
- Structured errors and deterministic diagnostics.
- Stable syntax trees and source spans for AI-assisted editing.
- Readable lowering to ordinary Rust.
- No mandatory AI service or runtime dependency.

## Architecture

1. Small readable source language.
2. Stable AST, spans, diagnostics and semantic model.
3. Direct native execution for the supported language subset, with Rust lowering as an interoperability path.
4. Rust interoperability and escape hatches.
5. Native compiler work only after semantics are stable.
6. AI-facing compiler APIs for structural editing, deterministic wire requests, and verification.

## Example

```ardisa
module math

fn add(a: Int, b: Int) -> Int
  a + b

fn divide(a: Int, b: Int) -> Result<Int, MathError>
  if b == 0
    Err(MathError::DivisionByZero)
  else
    Ok(a / b)
```

The syntax is intentionally easy for both people and coding agents to parse.

## Repository layout

- `crates/aifocus-core` - source model and diagnostics
- `crates/aifocus-cli` - compiler frontend CLI
- `examples` - language examples
- `docs` - language and implementation design

## Status

Bootstrap-to-self-hosting development is in progress. The repository now has a deterministic parser and formatter, flow-sensitive ownership checks, resource-level effect analysis, structured-concurrency failure propagation, a narrow safe Rust ABI boundary, differential and generated-program verification, persistent verified-repair learning, a versioned AI-facing compiler protocol with structural transactions and deterministic wire requests, a native execution backend exposed through `ardisa run`, benchmark fingerprints, and a reproducible stage2 compiler-pipeline replay. Full compiler self-hosting is not yet claimed.

The repository is intentionally standalone: it has no runtime or build dependency on an external agent framework. Engineering feedback and learning are repository-local compiler facilities.

See the [AI-native source feature matrix](docs/AI_NATIVE_FEATURE_MATRIX.md) for the distinction between parsed declarations, runtime helper APIs, and verified source-to-runtime support. Also see `docs/ROADMAP.md`, `docs/DESIGN.md`, `docs/IMPLEMENTATION_PLAN.md`, the [security and resource-safety model](docs/SECURITY_MODEL.md), and the proposed [AI Mode language specification](docs/AI_MODE.md). AI Mode is currently a design specification, not yet accepted source syntax.

## Source files

Use `*.ardisa` for Ardisa programs. The CLI accepts Ardisa source files and treats `.ardisa` as the canonical source extension. `ardisa run` executes the supported subset directly through the dependency-free native backend; `ardisa build` emits readable Rust when a Rust artifact is required.
