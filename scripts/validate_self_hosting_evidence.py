#!/usr/bin/env python3
"""Fail closed unless evidence proves native Stage 0 -> 3 compiler self-hosting.

Host-CLI replay evidence is checked for byte/hash consistency but never promoted
to compiler self-hosting. A self-hosting claim requires a separate native-only
stage lineage and independent corpus evidence.
"""
import hashlib
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


def _artifact_path(value):
    if not value:
        return None
    root = ROOT.resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        return None
    return path


def validate_probe_reproducibility(data):
    """Validate probe output hashes without confusing replay with self-hosting."""
    keys = ("probe_stage2", "probe_stage3", "probe_stage2_sha256", "probe_stage3_sha256")
    present = [key in data for key in keys]
    if not any(present):
        if data.get("probe_stage2_stage3_byte_identical", "").lower() == "true":
            return ["probe byte-identity claim is missing artifact paths and SHA-256 values"]
        return []
    missing = [key for key in keys if not data.get(key)]
    if missing:
        return ["probe reproducibility evidence missing: " + ", ".join(missing)]
    if data.get("probe_replay_mode") != "host_compiler_deterministic_rebuild":
        return ["probe replay mode is missing or unrecognized"]
    paths = [_artifact_path(data["probe_stage2"]), _artifact_path(data["probe_stage3"])]
    if any(path is None for path in paths):
        return ["probe artifact paths must remain inside the checked-out workspace"]
    errors = []
    for key, path in zip(("probe_stage2_sha256", "probe_stage3_sha256"), paths):
        if not path.is_file():
            errors.append(f"probe artifact is missing: {path.relative_to(ROOT)}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != data[key].lower():
            errors.append(f"{key} does not match the downloaded artifact bytes")
    if all(path.is_file() for path in paths):
        if paths[0].read_bytes() != paths[1].read_bytes():
            errors.append("probe Stage 2 and Stage 3 artifacts are not byte-identical")
        if data.get("probe_stage2_stage3_byte_identical", "").lower() != "true":
            errors.append("probe byte identity must be explicitly recorded")
    return errors


def validate():
    data, errors = read_evidence()
    if errors:
        return errors
    errors.extend(validate_probe_reproducibility(data))
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
