#!/usr/bin/env python3
"""Create a CycloneDX 1.5 SBOM with package-ID-bound dependency references."""
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

output = Path(sys.argv[1] if len(sys.argv) > 1 else "ardisa.cdx.json")
metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--format-version", "1"], text=True))
packages = metadata.get("packages", [])
root = next((package for package in packages if package["name"] == "ardisa"), None)
if root is None:
    raise SystemExit("Cargo metadata did not contain the Ardisa CLI root package")

refs = {package["id"]: "cargo:" + package["id"] for package in packages}
components = []
for package in packages:
    if package["id"] == root["id"]:
        continue
    purl = f"pkg:cargo/{package[\"name\"]}@{package[\"version\"]}"
    if package.get("source"):
        purl += "?repository_url=" + quote(package["source"], safe="")
    components.append({
        "type": "library",
        "bom-ref": refs[package["id"]],
        "name": package["name"],
        "version": package["version"],
        "purl": purl,
        "properties": [{"name": "cargo.package.id", "value": package["id"]}],
    })

dependencies = []
for node in (metadata.get("resolve") or {}).get("nodes", []):
    dependencies.append({
        "ref": refs[node["id"]],
        "dependsOn": [refs[dependency["pkg"]] for dependency in node.get("deps", [])],
    })
root_ref = refs[root["id"]]
known_refs = set(refs.values())
if len(known_refs) != len(refs):
    raise SystemExit("Cargo package reference collision")
for dependency in dependencies:
    if dependency["ref"] not in known_refs or any(ref not in known_refs for ref in dependency["dependsOn"]):
        raise SystemExit("SBOM dependency graph contains an unknown package reference")

bom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "serialNumber": f"urn:uuid:{uuid.uuid4()}",
    "version": 1,
    "metadata": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": {
            "type": "application",
            "bom-ref": root_ref,
            "name": root["name"],
            "version": root["version"],
            "purl": f"pkg:cargo/{root[\"name\"]}@{root[\"version\"]}",
        },
    },
    "components": components,
    "dependencies": dependencies,
}
output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
