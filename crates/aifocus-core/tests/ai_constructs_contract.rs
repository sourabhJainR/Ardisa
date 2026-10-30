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

    // The typed CFG must preserve every validated declaration. This is metadata
    // propagation only: native compilation below still rejects unimplemented
    // executable semantics rather than silently dropping the declarations.
    let typed_cfg = typed_ir::lower_cfg(&module).expect("valid declarations should survive typed CFG lowering");
    assert_eq!(typed_cfg.constructs.len(), 5);
    assert_eq!(typed_cfg.constructs[0].name, "Audit");
    assert_eq!(typed_cfg.constructs[4].kind, ardisa_core::ConstructKind::Phase);
    let encoded = typed_ir::encode_construct_metadata(&typed_cfg.constructs)
        .expect("typed CFG construct metadata should remain valid and deterministic");
    assert!(encoded.starts_with("ARDISA-CONSTRUCT-META-V1\n"));

    let error = compile_source(source)
        .expect_err("construct declarations must not be silently ignored by executable codegen");
    let diagnostics = format!("{error:?}");
    assert!(diagnostics.contains("UnsupportedRuntimeBoundary"), "declarations without executable operations or trusted Proof/Trace runtime evidence must remain fail-closed: {diagnostics}");
}

#[test]
fn each_ai_construct_kind_is_individually_blocked_from_native_codegen() {
    // Keep the rejection contract per-kind: a test containing every construct
    // could pass if the compiler rejected only one of them and silently ignored
    // the remaining declarations.
    let cases = [
        (
            "trace",
            "module trace_only\ntrace Audit\n  signature: hmac-sha256\nfn main() -> Int\n  0\n",
        ),
        (
            "cell",
            "module cell_only\ncell State\n  value: Int\n  invariant: non-empty\nfn main() -> Int\n  0\n",
        ),
        (
            "vault",
            "module vault_only\nvault IO\n  capabilities: filesystem_read\n  denies: network\nfn main() -> Int\n  0\n",
        ),
        (
            "proof",
            "module proof_only\nproof Review\n  requires: tests-pass\n  ensures: reviewed\n  on_unknown: reject\n  required: true\nfn main() -> Int\n  0\n",
        ),
        (
            "phase",
            "module phase_only\nphase Delivery\n  draft -> review\nfn main() -> Int\n  0\n",
        ),
    ];

    for (kind, source) in cases {
        let module = parse(source).unwrap_or_else(|error| {
            panic!("{kind} fixture must parse before checking the native boundary: {error:?}")
        });
        sema::analyze_constructs(&module).unwrap_or_else(|errors| {
            panic!("{kind} fixture must be semantically valid: {errors:?}")
        });
        let error = match compile_source(source) {
            Err(error) => error,
            Ok(_) => panic!("{kind} declaration was incorrectly accepted by native codegen"),
        };
        let diagnostics = format!("{error:?}");
        assert!(
            diagnostics.contains("UnsupportedRuntimeBoundary"),
            "{kind} must fail closed at the unsupported runtime boundary until its executable contract is implemented; got {diagnostics}"
        );
    }
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


#[test]
fn phase_typed_metadata_builds_runtime_machine_with_explicit_initial_state() {
    use ardisa_core::ai_mode::{PhaseMachine, PhaseMachineError};

    let source = "module phase_runtime_bridge\nphase Delivery\n  draft -> review\n  review -> release\n";
    let module = parse(source).expect("Phase declaration should parse");
    let constructs = typed_ir::lower_constructs(&module).expect("Phase metadata should validate");
    let phase = constructs.first().expect("Phase metadata should exist");

    let mut machine = PhaseMachine::from_typed_construct(phase, "draft")
        .expect("typed Phase edges should create a runtime machine");
    assert_eq!(machine.current(), "draft");
    assert!(machine.can_transition_to("review"));
    machine.transition("review").expect("declared transition should succeed");
    assert_eq!(machine.current(), "review");

    let error = machine.transition("draft").expect_err("undeclared reverse transition must fail");
    assert!(matches!(error, PhaseMachineError::InvalidTransition { .. }));
    assert_eq!(machine.current(), "review", "rejected transition must not mutate state");
    assert_eq!(machine.history().len(), 1, "rejected transition must not append history");

    assert!(matches!(
        PhaseMachine::from_typed_construct(phase, "unknown"),
        Err(PhaseMachineError::InitialStateUnknown(_))
    ));
}

#[test]
fn phase_runtime_bridge_rejects_non_phase_constructs() {
    use ardisa_core::ai_mode::{PhaseMachine, PhaseMachineError};

    let source = "module wrong_kind\ntrace Audit\n  signature: hmac-sha256\n";
    let module = parse(source).expect("Trace declaration should parse");
    let constructs = typed_ir::lower_constructs(&module).expect("Trace metadata should validate");
    assert!(matches!(
        PhaseMachine::from_typed_construct(&constructs[0], "draft"),
        Err(PhaseMachineError::NotPhaseConstruct)
    ));
}
