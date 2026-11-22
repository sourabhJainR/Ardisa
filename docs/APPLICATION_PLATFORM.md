# Ardisa application platform

This document tracks the path from the current compiler/runtime to one Ardisa application core that can run on console, desktop, server and browser targets.

## Design contract

Application logic belongs in Ardisa source. Host adapters provide operating-system or browser services, but must not duplicate business rules.

The shared runtime contract is platform-neutral:
- HTTP-shaped requests and responses with explicit methods, paths, headers and byte bodies.
- Deterministic exact-route dispatch to named Ardisa functions.
- Request, response, header, path and route limits.
- Rejection of path traversal, encoded path aliases, header injection and invalid response status codes.
- A native VM bridge whose handler ABI is `(method: String, path: String, query: String, body: String) -> String`.
- Explicit host capabilities for HTTP listening, desktop windows, browser DOM, filesystem, network client, clock and process access.
- A shared UI tree with accessible names, parent validation, bounded node counts and event-kind validation.
- A `DesktopHost` contract for OS-specific renderers.

## Current implementation

The first runtime contract is in `crates/aifocus-core/src/application.rs`.

The application router can dispatch a request into a function in a validated native Ardisa program. The executable example is `examples/shared_app_logic.ardisa`; the core test compiles it and calls it through `NativeProgramHandler`.

The desktop side currently defines the common UI tree, event model and host trait. It does not yet include a Win32, Cocoa, GTK or other concrete window renderer.

The HTTP side now includes a bounded native TCP listener, an HTTP/1.0 and HTTP/1.1 request parser, query forwarding, duplicate Content-Length and Transfer-Encoding rejection, a close-after-response transport, an absolute request-read deadline, client-disconnect isolation, and an `ardisa serve` CLI command. It also rejects whitespace before header colons and non-digit Content-Length values. The initial server accepts numeric loopback bind addresses only, handles one connection at a time, has no TLS, and rejects chunked transfer encoding. Any supplied Host value must match the bound loopback listener address and port (or its `localhost` alias); HTTP/1.1 requires exactly one Host. Duplicate Host/Origin/Sec-Fetch-Site headers, non-loopback browser Origin values, and cross-site or unknown Fetch Metadata site values are rejected, including cross-site requests that omit Origin. This reduces DNS-rebinding and browser CSRF exposure but is not client authentication: native clients without Fetch Metadata can still connect. Keep GET/HEAD handlers side-effect-free and add CSRF protection to state-changing handlers. Non-loopback and hostname binds are rejected until authenticated TLS support is implemented. It is for local development, not production internet-facing use.

The lower-level APIs remain runtime embedding contracts; the initial HTTP route declaration syntax described below is now accepted by the application-source compiler API. This is not yet a complete native desktop or HTTP language surface.

## First-class source declarations (initial slice)

Application route metadata can now be declared in Ardisa source rather than repeated in host-language setup:

```ardisa
module inventory_app

app inventory
  capability http_listen
  route GET "/" home
  route GET "/health" health

fn home(method: String, path: String, query: String, body: String) -> String
  "Inventory service"

fn health(method: String, path: String, query: String, body: String) -> String
  "ok"
```

The `compile_application_source` API extracts the application block, validates route paths, methods, duplicate routes, handler names and explicit capabilities, then compiles the remaining module through the normal compiler pipeline. `ApplicationDefinition::build_runtime` creates the bounded shared router. Host code must still grant the declared capability through `AppPolicy`; source declarations do not grant permissions. This initial slice covers HTTP route metadata only; UI declarations, typed handler request/response values and target-specific lowering remain future work.


### Concrete native desktop host (initial backend)

The optional `native-desktop` feature provides `MinifbDesktopHost` in
`ardisa_core::desktop_minifb`. It opens a real OS window, renders the shared
`UiTree`, translates button clicks and text edits into validated `UiEvent`
values, and applies `UiAction` updates. It requires an explicit
`AppPolicy` grant for `DesktopWindow`; constructing the host without that
grant fails before opening a window.

Build the backend with:

```sh
cargo check -p ardisa-core --features native-desktop
```

This first backend is a lightweight pixel renderer, not a full native widget
toolkit. It displays accessible names as visible labels, but minifb does not
provide native accessibility-tree integration or screen-reader semantics.
It also has a fixed initial window size and simple vertical layout. Those
limitations remain open work; do not treat this backend as accessibility
complete or as a substitute for platform-native controls.


### WebAssembly browser runtime (initial execution target)

The `ardisa-wasm` crate builds a browser-loadable WebAssembly runtime for
canonical ARDISA-EXEC-V1 artifacts. It validates the artifact before execution,
then invokes Ardisa functions through a bounded string-only browser ABI.
Artifact size, argument count and size, instruction count, call depth and
result size are limited. No DOM, network, filesystem, process or clock
capability is granted by this runtime. Browser code must use explicit host
adapters and safe DOM APIs.

Build and validate the target with:

```sh
rustup target add wasm32-unknown-unknown
cargo check -p ardisa-wasm --target wasm32-unknown-unknown
wasm-pack build crates/ardisa-wasm --target web --out-dir ../../examples/browser_app/pkg
```

A runnable host example is in `examples/browser_app`. This first target runs
the existing Ardisa executable format inside a WebAssembly-compiled VM. It is
not yet direct per-application Ardisa-to-WASM code generation, and its ABI v1
supports string arguments and string results only.

## Next implementation gates

1. **HTTP host hardening:** add bounded concurrent connections, graceful shutdown, keep-alive policy, TLS integration, richer route parameters and production-grade operational controls. The current single-connection local host is only the first transport slice.
2. **Desktop host:** implement one concrete native renderer first, then add Windows/macOS/Linux backends behind `DesktopHost`. Add event-loop lifecycle, layout, focus, accessibility, keyboard navigation and safe teardown.
3. **Source-level API:** define accepted Ardisa declarations for app routes and UI views. Wire parser, semantic checks, effects, ownership, typed IR, native instructions and ARDISA-EXEC-V1 serialization together. Unsupported effects must fail closed.
4. **WebAssembly:** define the stable host ABI and browser bindings separately from generic WASM. Add compiler target selection, browser loader, memory/fuel bounds, DOM/fetch event adapters and browser automation. Generic WASM must not assume filesystem or threads.
5. **Packaging:** build signed/reproducible native compiler and application artifacts for Windows, Linux and macOS. Add installer/uninstaller tests, checksums, SBOM and release provenance.
6. **Shared application proof:** compile the same Ardisa business-logic source for console, desktop, server and browser targets and compare deterministic results.

## Security and compatibility

- All host effects require explicit capability grants; no ambient network, filesystem, process or UI access.
- Keep ARDISA-EXEC-V1 decoding canonical and fail closed on unsupported opcodes or target features.
- Apply resource budgets before allocating or dispatching request, UI or browser data.
- Keep native and browser platform bindings outside the pure shared business-logic module.
- Do not promote self-hosting readiness from application-platform work.
