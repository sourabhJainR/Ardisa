#!/usr/bin/env python3
"""Fail-closed consistency checks for Ardisa's capability evidence manifest."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "rust_capabilities.json"
ALLOWED_STATUS = {"supported", "partial", "unsupported", "interop-only"}


def validate(data):
    errors = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    policy = data.get("policy")
    if not isinstance(policy, dict):
        errors.append("policy must be an object")
        policy = {}
    required = policy.get("supported_requires", [])
    if not isinstance(required, list) or not required or any(not isinstance(x, str) for x in required):
        errors.append("policy.supported_requires must be a non-empty list of evidence keys")
        required = []
    if policy.get("self_hosting_requires_stage_0_to_3_reproducibility") is not True:
        errors.append("self-hosting must require Stage 0-to-3 reproducibility")
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        errors.append("capabilities must be a non-empty list")
        return errors
    seen = set()
    for index, item in enumerate(capabilities):
        prefix = f"capabilities[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        ident = item.get("id")
        if not isinstance(ident, str) or not ident.strip():
            errors.append(f"{prefix}.id must be a non-empty string")
        elif ident in seen:
            errors.append(f"duplicate capability id: {ident}")
        else:
            seen.add(ident)
        status = item.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{prefix}.status must be one of {sorted(ALLOWED_STATUS)}")
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict):
            errors.append(f"{prefix}.evidence must be an object")
            continue
        keys = required if status == "supported" else [key for key in evidence if key in required]
        for key in keys:
            paths = evidence.get(key)
            if not isinstance(paths, list) or (status == "supported" and not paths):
                errors.append(f"{ident}: {'supported capability missing' if status == 'supported' else 'invalid'} evidence list '{key}'")
                continue
            for value in paths:
                if not isinstance(value, str) or not value:
                    errors.append(f"{ident}: invalid evidence path for '{key}'")
                elif not (ROOT / value).is_file():
                    errors.append(f"{ident}: evidence path does not exist: {value}")
        if status == "supported" and not isinstance(item.get("limitations", ""), str):
            errors.append(f"{ident}: limitations must be a string when supplied")
    return errors


def main():
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"capability manifest error: {exc}", file=sys.stderr)
        return 2
    errors = validate(data)
    if errors:
        for error in errors:
            print(f"CAPABILITY-EVIDENCE: {error}", file=sys.stderr)
        return 1
    print(f"Validated {len(data['capabilities'])} capability records; no unsupported feature is promoted implicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
