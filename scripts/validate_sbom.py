#!/usr/bin/env python3
"""Fail when a CycloneDX dependency graph references absent or duplicate components."""
import json
import sys
from pathlib import Path

bom = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if bom.get("bomFormat") != "CycloneDX" or bom.get("specVersion") != "1.5":
    raise SystemExit("unexpected CycloneDX format/version")
root = bom.get("metadata", {}).get("component")
if not isinstance(root, dict) or not root.get("bom-ref"):
    raise SystemExit("SBOM root component is missing a bom-ref")
refs = [component.get("bom-ref") for component in bom.get("components", [])]
refs.append(root["bom-ref"])
if any(not ref for ref in refs) or len(refs) != len(set(refs)):
    raise SystemExit("SBOM component references are missing or duplicated")
known = set(refs)
dependency_refs = [item.get("ref") for item in bom.get("dependencies", [])]
if len(dependency_refs) != len(set(dependency_refs)):
    raise SystemExit("SBOM dependency entries are duplicated")
if set(dependency_refs) != known:
    raise SystemExit("SBOM dependency entries do not cover all components")
for item in bom.get("dependencies", []):
    if item["ref"] not in known or any(ref not in known for ref in item.get("dependsOn", [])):
        raise SystemExit("SBOM dependency graph contains a dangling reference")
print("CycloneDX dependency references validated")
