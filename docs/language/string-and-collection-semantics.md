# String and collection semantics (current supported subset)

This document records the currently implemented native behavior, not Rust standard-library compatibility.

## Strings

- String values are UTF-8 encoded.
- `len(String)` currently reports the UTF-8 byte length.
- `String[index]` treats the index as a zero-based **byte offset** and returns that byte as an `Int` in the range 0–255. This remains distinct from Unicode scalar iteration.
- A negative index fails with a runtime type error. An index at or beyond the byte length fails with `string index out of bounds`.
- Indexing inside a multi-byte UTF-8 sequence is permitted because indexing is byte-based. It does not return a Unicode scalar value or grapheme cluster.
- `chr(Int)` converts a valid Unicode scalar value to a one-scalar UTF-8 string. It rejects negative values, values above `0x10FFFF`, and the surrogate range `0xD800..=0xDFFF` at runtime. It is not a byte conversion; use string indexing when a UTF-8 byte is specifically required.
- String concatenation is supported for `String + String`; unsupported arithmetic combinations are rejected by semantic analysis.

## Lists

- List values contain a sequence of native values. The supported subset relies on semantic analysis to enforce a consistent element type in source expressions.
- List bindings are move-only under the current ownership checker: assigning a list binding into another binding consumes the source name, and later use of that name is rejected. This is an explicit subset contract, not full Rust move/drop parity.
- `List[index]` uses a zero-based integer index and returns the value at that position.
- Negative indexes and indexes at or beyond the list length fail explicitly; the runtime does not silently clamp or wrap indexes.
- `List[start:end]` supports a half-open range (`start` inclusive, `end` exclusive) and returns a new owned list copy. It is not a view and does not alias the source list; mutating the slice result does not mutate the source.
- Both slice bounds must be non-negative integers with `start <= end <= len(list)`. Negative bounds, reversed ranges, and an end beyond the list length fail explicitly. Empty ranges and ranges ending exactly at the list length are valid.
- Slice output is subject to the same collection item, nested-value, and frame resource limits as other lists. Slicing a string is rejected because byte slicing can split UTF-8 sequences; string indexing remains byte-based.
- The native runtime supports mutation for named list bindings through indexed assignment and append. Explicit index-based iteration with `while` is supported; `for-in` syntax and iterator protocols are unsupported and must fail closed. Aliasing views, borrowing, and Rust-compatible drop semantics are not implemented or claimed.
- List/string length and indexing use byte count for strings and element count for lists. These are distinct units.

## Primitive operations and library boundary

The currently implemented native primitives are `len`, `chr`, list/string indexing, list slicing, and `push` for named mutable list bindings. They are compiler/runtime operations, not evidence of a complete standard collection library. Unsupported collection methods, `for-in` loops, iterator protocols, string slices, and borrowed views are not silently substituted with host-language behavior.

## Deliberate limitations

No claim is made for Rust `String`, `Vec`, slice, iterator, or standard-library parity. Unicode scalar/grapheme indexing and grapheme segmentation, string slicing, list slice views, iterator syntax, borrowing/aliasing views, and a complete collection API remain unsupported or unproven. List slices are explicitly copy-on-slice. `chr` supports Unicode scalar construction but does not provide scalar- or grapheme-based indexing. New behavior must be backed by source-to-runtime tests and adversarial bounds/resource tests before the capability matrix is promoted.

## Regression coverage

- `crates/aifocus-core/tests/list_slices.rs`: source-to-native tests for half-open ranges, copy isolation, empty/full slices, negative and reversed bounds, and out-of-range ends.
- `crates/aifocus-core/tests/string_collection_boundaries.rs`: parser and semantic coverage for slice syntax and explicit string-slice rejection.
- `crates/aifocus-core/src/native.rs`: native execution bounds and resource checks for indexing and owned slices.