# String and collection semantics (current supported subset)

This document records the currently implemented native behavior, not Rust standard-library compatibility.

## Strings

- String values are UTF-8 encoded.
- `len(String)` currently reports the UTF-8 byte length.
- `String[index]` treats the index as a zero-based **byte offset** and returns that byte as an `Int` in the range 0–255. This remains distinct from Unicode scalar indexing.
- `scalar_len(text)` counts Unicode scalar values; `scalar_at(text, index)` returns the selected scalar as an owned one-scalar `String`. Both use zero-based scalar positions, reject negative/out-of-range indexes, and are separate from byte-based indexing.
- `grapheme_len(text)` and `grapheme_at(text, index)` use extended grapheme clusters from the pinned `unicode-segmentation = 1.12.0` dependency. They preserve combining sequences, regional-indicator flags, and emoji ZWJ sequences; returned clusters are owned strings. The pinned dependency version defines the segmentation behavior.
- `String[start:end]` is a half-open UTF-8 byte range and returns an owned `String` copy. Both bounds must be non-negative, ordered, at most the byte length, and on UTF-8 scalar boundaries. A range that splits a multi-byte encoding fails explicitly; it never creates invalid UTF-8. The result obeys the configured string-byte limit.
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
- The native runtime supports mutation for named list bindings through indexed assignment and the `push(List, value)` builtin. `len(String)` and `len(List)` are builtins; list literals, indexing, indexed assignment, and slicing are language syntax. These are a deliberately small primitive surface, not a general collection library.
- List bindings are move-only in the current ownership subset: `let other = values` consumes `values`, so using `values` afterward is a compile-time ownership error. This prevents ordinary source-level mutable aliases; it does not establish complete Rust borrow checking.
- `for`/`in`, iterator methods, borrowed slice views, and string slicing are unsupported and must fail compilation rather than silently lowering to different behavior. No iteration syntax is currently supported; explicit indexing with a checked integer is the supported traversal building block.
- List/string length and indexing use element count for lists and UTF-8 byte count for strings. These are distinct units.

## Deliberate limitations

No claim is made for Rust `String`, `Vec`, slice, iterator, or standard-library parity. Unicode scalar/grapheme indexing, list slice views, iterator syntax/protocol, full borrowing/lifetime/aliasing guarantees, drop semantics, and a complete collection API remain unsupported or unproven. Both list slices and string slices are explicitly owned copies. `chr` constructs one valid Unicode scalar but does not provide scalar- or grapheme-based indexing. New behavior must be backed by source-to-runtime tests and adversarial bounds/resource tests before the capability matrix is promoted.

## Regression coverage

- `crates/aifocus-core/tests/list_slices.rs`: source-to-native tests for half-open list/string ranges, copy isolation, empty/full slices, UTF-8 boundary rejection, negative and out-of-range bounds, byte-for-byte artifact round trips, native `for`/`in` iteration, artifact round trips, and list resource limits.
- `crates/aifocus-core/tests/string_collection_boundaries.rs`: empty/negative/out-of-range indexing, parser and semantic coverage for slice syntax, explicit string-slice rejection, and UTF-8 byte-budget enforcement.
- `crates/aifocus-core/tests/list_mutation_adversarial.rs`: negative index, selected-element mutation, and neighboring-element preservation.
- `crates/aifocus-core/src/native.rs`: native execution bounds and resource checks for indexing and owned slices.