#!/usr/bin/env python3
"""Create a CycloneDX 1.5 SBOM from Cargo's resolved package graph."""
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1] if len(sys.argv) > 1 else "ardisa.cdx.json")
metadata = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1"], text=True
))
components = []
for package in metadata.get("packages", []):
    components.append({
        "type": "application" if package["name"] == "ardisa" else "library",
        "bom-ref": f"pkg:cargo/{package['name']}@{package['version']}",
        "name": package["name"],
        "version": package["version"],
        "purl": f"pkg:cargo/{package['name']}@{package['version']}",
    })
dependencies = []
for node in (metadata.get("resolve") or {}).get("nodes", []):
    dependencies.append({
        "ref": node["id"],
        "dependsOn": [dependency["pkg"] for dependency in node.get("deps", [])],
    })

bom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.5",
    "serialNumber": f"urn:uuid:{uuid.uuid4()}",
    "version": 1,
    "metadata": {"timestamp": datetime.now(timezone.utc).isoformat()},
    "components": components,
    "dependencies": dependencies,
}
output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")
