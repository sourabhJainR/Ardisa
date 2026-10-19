#!/usr/bin/env python3
"""Fail closed unless evidence proves native Stage 0 -> 3 compiler self-hosting.

A host-compiler rebuild of a probe can demonstrate deterministic artifact
encoding, but it is not evidence that the Ardisa compiler compiles itself.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "bootstrap" / "evidence" / "self-hosting.txt"
REQUIRED = {
    "stage0_native_compiler",
    "stage1_compiled_by_stage0",
    "stage2_compiled_by_stage1",
    "stage3_compiled_by_stage2",
    "stage2_stage3_sha256",
    "independent_positive_corpus",
    "independent_negative_corpus",
    "independent_differential_corpus",
    "independent_holdout_corpus",
    "forbidden_host_fallback_scan",
}


def read_evidence():
    result = {}
    try:
        lines = EVIDENCE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [f"missing self-hosting evidence: {exc}"]
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result, []


def validate():
    data, errors = read_evidence()
    if errors:
        return errors
    ready = data.get("self_hosting_ready", "false").lower() == "true"
    claimed = data.get("compiler_self_hosting", "false").lower() == "true"
    if not ready and not claimed:
        print("SELF-HOSTING-GATE: self_hosting_ready=false; native Stage 0 -> 3 proof is not established.")
        return []
    missing = sorted(key for key in REQUIRED if not data.get(key))
    if missing:
        errors.append("self-hosting promotion missing evidence keys: " + ", ".join(missing))
    if data.get("self_hosting_ready", "").lower() != "true":
        errors.append("compiler_self_hosting=true requires self_hosting_ready=true")
    if data.get("stage2_stage3_byte_identical", "").lower() != "true":
        errors.append("Stage 2 and Stage 3 must be byte-identical")
    if data.get("native_compiler_only", "").lower() != "true":
        errors.append("Stage 0 -> 3 must use the native compiler path, not host recompilation")
    if data.get("host_fallback_detected", "").lower() != "false":
        errors.append("forbidden host compiler fallback must be explicitly absent")
    return errors


if __name__ == "__main__":
    failures = validate()
    if failures:
        for failure in failures:
            print("SELF-HOSTING-GATE: " + failure, file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)
