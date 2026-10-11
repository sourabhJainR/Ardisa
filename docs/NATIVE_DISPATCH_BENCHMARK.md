# Native VM dispatch benchmark

This benchmark was added after the dispatch loop was changed to borrow immutable
`NativeInstr` values rather than clone each instruction before matching it. It
also compares three execution paths: the compatibility entry point, which clones
a `NativeProgram` per call; the shared-program entry point, which reuses an
`Arc` but validates each call; and a `ValidatedNativeProgram` wrapper that
validates immutable code once and reuses that validation. Argument checks,
execution limits, instruction counters, and task state remain per invocation.

## Run

From the repository root, with the stable Rust toolchain installed:

```sh
cargo run --release -p ardisa-core --example native_dispatch_bench -- --iterations 5
```

Use `--iterations 1` for a quick smoke run. The workload and code shape are
fixed across runs; the number of measured passes is configurable. Run the
release-mode command more than once on an otherwise idle machine when comparing
revisions, and compare the same workload and toolchain.

## Workloads and interpretation

- `arithmetic_and_jumps`: an instruction-heavy control workload with arithmetic
  and conditional branches.
- `small_strings`: repeated short embedded string operands.
- `large_embedded_strings`: repeated 4 KiB embedded string operands, exercising
  the payload-heavy instruction case.

Each workload is built before timing and must return the same integer result on
every pass. The report includes code instruction count, embedded string bytes,
elapsed nanoseconds, nanoseconds per executed instruction, and observed
instructions per second. There is deliberately no hard timing threshold in CI:
shared runners are noisy, so timings are evidence for comparison, not a flaky
pass/fail contract.

`estimated_dispatch_payload_bytes_avoided_per_pass` is a static estimate: it
sums string/name operand bytes for the instructions in the workload that are
executed once per pass. It estimates the payload bytes that the former
whole-instruction clone would have copied at dispatch. It is **not** a measured
allocator count, does not include allocator overhead, and does not imply zero
allocations. Runtime values (including the string value pushed by
`PushString`) may still legitimately allocate or clone at ownership boundaries.

Both modes include program validation and VM setup in the timed interval. The
`cloning_entry_point` mode also includes the whole-program clone performed by
`run_program`; `shared_program` receives an already-created `Arc` and only
clones the Arc handle. Both modes still validate the program on every run.
These timings represent end-to-end execution of the public VM entry points, not
an isolated instruction-only cycle counter. Compare like-for-like results and
report observed numbers rather than claiming a guaranteed speedup. Reusing an
Arc is opt-in; callers that use the compatibility entry point retain its
existing behavior.

## Correctness gate

The benchmark asserts the result of every workload. CI compiles and runs a
one-pass smoke invocation, while the workspace, Native Phases, bootstrap
reproducibility, and generated differential/holdout gates remain separate
required checks. The benchmark itself must never replace those correctness
gates.
