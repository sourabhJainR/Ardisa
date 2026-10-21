use ardisa_core::{parse, pipeline::compile_source, sema, typed_ir};

#[test]
fn valid_ai_constructs_reach_typed_metadata_but_not_executable_codegen() {
    let source = "module ai_contract\ntrace Audit\n  signature: hmac-sha256\ncell State\n  invariant: non-empty\nvault IO\n  capabilities: filesystem_read filesystem_write\n  denies: network\nproof Review\n  requires: tests-pass\n  ensures: reviewed\n  on_unknown: reject\n  required: true\nphase Delivery\n  draft -> review\n  review -> release\nfn main() -> Int\n  0\n";
    let module = parse(source).expect("construct syntax should parse");
    let ast_arena = ardisa_core::compiler_engine::AstArena::build(&module);
    let construct_nodes = ast_arena.nodes.iter().filter(|(_, node)| node.kind.ends_with("-declaration")).count();
    assert_eq!(construct_nodes, 5, "compiler AST arena must retain all AI construct declarations");
    let constructs = sema::analyze_constructs(&module)
        .expect("well-formed declarations should produce typed semantic metadata");
    assert_eq!(constructs.len(), 5);
    assert_eq!(typed_ir::lower_constructs(&module).expect("construct metadata should lower").len(), 5);

    let error = compile_source(source)
        .expect_err("construct declarations must not be silently ignored by executable codegen");
    let diagnostics = format!("{error:?}");
    assert!(diagnostics.contains("AIF610"), "executable path must remain fail-closed until runtime semantics exist: {diagnostics}");
}

#[test]
fn malformed_ai_constructs_have_stable_semantic_diagnostics() {
    let cases = [
        ("module duplicate_construct\ntrace Audit\n  signature: hmac\ntrace Audit\n  signature: hmac\n", "AIF611"),
        ("module missing_signature\ntrace Audit\n  ignored: value\n", "AIF617"),
        ("module invalid_proof\nproof Review\n  requires: tests-pass\n  ensures: reviewed\n  on_unknown: accept\n  required: true\n", "AIF618"),
        ("module empty_phase\nphase Delivery\n  draft -> draft\n", "AIF615"),
        ("module duplicate_transition\nphase Delivery\n  draft -> review\n  draft -> review\n", "AIF616"),
    ];
    for (source, code) in cases {
        let module = parse(source).expect("malformed declarations should still parse for semantic diagnostics");
        let errors = sema::analyze_constructs(&module).expect_err("invalid declaration must fail semantic validation");
        assert!(errors.iter().any(|error| error.code == code), "expected {code}, got {errors:?}");
    }
}

#[test]
fn typed_construct_encoding_is_deterministic_and_versioned() {
    let source = "module deterministic_construct\ntrace Audit\n  signature: hmac-sha256\nfn main() -> Int\n  0\n";
    let module = parse(source).expect("source should parse");
    let constructs = typed_ir::lower_constructs(&module).expect("typed metadata should lower");
    let first = typed_ir::encode_construct_metadata(&constructs).expect("metadata should encode");
    let second = typed_ir::encode_construct_metadata(&constructs).expect("metadata should encode twice");
    assert_eq!(first, second);
    assert!(first.starts_with("ARDISA-CONSTRUCT-META-V1\n"));
}


#[test]
fn vault_capability_sets_reject_duplicates_and_allow_deny_conflicts() {
    let cases = [
        (
            "module vault_conflict\nvault IO\n  capabilities: filesystem_read filesystem_write\n  denies: filesystem_write\n",
            "an effect cannot be both allowed and denied",
        ),
        (
            "module vault_duplicate\nvault IO\n  capabilities: filesystem_read filesystem_read\n  denies: network\n",
            "a capability must not be duplicated within a policy clause",
        ),
    ];
    for (source, reason) in cases {
        let module = parse(source).expect("vault policy fixture should parse");
        let errors = sema::analyze_constructs(&module)
            .expect_err("ambiguous Vault policy must fail semantic validation");
        assert!(
            errors.iter().any(|error| error.code == "AIF619"),
            "{reason}; expected AIF619, got {errors:?}"
        );
    }

    let valid = "module vault_disjoint\nvault IO\n  capabilities: filesystem_read network\n  denies: filesystem_write\n";
    let module = parse(valid).expect("disjoint vault policy should parse");
    sema::analyze_constructs(&module)
        .expect("disjoint allow/deny sets should pass structural validation");
    assert!(
        compile_source(valid).is_err(),
        "structural validation must not imply runtime enforcement or bypass AIF610"
    );
}


#[test]
fn vault_capability_names_must_match_runtime_capability_vocabulary() {
    let source = "module vault_unknown\nvault IO\n  capabilities: filesystem_read unknown_capability\n  denies: filesystem_write\n";
    let module = parse(source).expect("unknown capability fixture should parse");
    let errors = sema::analyze_constructs(&module)
        .expect_err("unknown capability names must fail closed");
    assert!(
        errors.iter().any(|error| error.code == "AIF620"),
        "expected AIF620 for unsupported capability name, got {errors:?}"
    );
}
