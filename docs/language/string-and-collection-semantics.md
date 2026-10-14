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
- `List[index]` uses a zero-based integer index and returns the value at that position.
- Negative indexes and indexes at or beyond the list length fail explicitly; the runtime does not silently clamp or wrap indexes.
- The native runtime supports mutation for named list bindings through the implemented indexed-assignment and append paths. This does not establish Rust ownership, borrowing, aliasing, slice/view, iterator, or drop semantics.
- List/string length and indexing use byte count for strings and element count for lists. These are distinct units.

## Deliberate limitations

No claim is made for Rust `String`, `Vec`, slice, iterator, or standard-library parity. Unicode scalar/grapheme indexing and grapheme segmentation, slices/views with ownership or lifetime guarantees, collection aliasing guarantees, and a complete collection API remain unsupported or unproven. `chr` supports Unicode scalar construction but does not provide scalar- or grapheme-based indexing. New behavior must be backed by source-to-runtime tests and adversarial bounds/resource tests before the capability matrix is promoted.

## Regression coverage

- `crates/aifocus-core/src/pipeline.rs`: source-to-native test for UTF-8 byte indexing and valid/invalid list indexing.
- `crates/aifocus-core/src/native.rs`: native execution bounds checks for string and list indexing.