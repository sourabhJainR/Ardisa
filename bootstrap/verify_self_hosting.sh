#!/usr/bin/env bash
set -euo pipefail

mkdir -p bootstrap/evidence target

ROOT="$(pwd)"
PROBE="$ROOT/bootstrap/repro_probe.ardisa"
STAGE1="$ROOT/target/bootstrap-probe-stage1.aexe"
STAGE2="$ROOT/target/bootstrap-probe-stage2.aexe"
STAGE3="$ROOT/target/bootstrap-probe-stage3.aexe"
HOST_COMPILER="$ROOT/target/debug/ardisa"

if [[ ! -f "$PROBE" ]]; then
  echo "BOOTSTRAP-GATE: missing bootstrap probe file"
  exit 70
fi

# Build once and pin the exact host compiler used for all three probe outputs.
# This remains a host-CLI replay probe, not a self-hosting claim.
cargo build -q -p ardisa
if [[ ! -x "$HOST_COMPILER" ]]; then
  echo "BOOTSTRAP-GATE: expected host CLI binary is missing: $HOST_COMPILER"
  exit 72
fi
HOST_COMPILER_SHA256="$(sha256sum "$HOST_COMPILER" | cut -d ' ' -f 1)"
PROBE_SHA256="$(sha256sum "$PROBE" | cut -d ' ' -f 1)"
"$HOST_COMPILER" bootstrap compile "$PROBE" "$STAGE1"
"$HOST_COMPILER" bootstrap compile "$PROBE" "$STAGE2"
"$HOST_COMPILER" bootstrap compile "$PROBE" "$STAGE3"

cmp -s "$STAGE2" "$STAGE3" || {
  echo "BOOTSTRAP-GATE: probe stage-2/stage-3 mismatch"
  exit 71
}

cargo run -q -p ardisa -- bootstrap verify "$STAGE2"
cargo run -q -p ardisa -- bootstrap verify "$STAGE3"

STAGE2_SHA256="$(sha256sum "$STAGE2" | cut -d ' ' -f 1)"
STAGE3_SHA256="$(sha256sum "$STAGE3" | cut -d ' ' -f 1)"
{
  echo "probe_source=bootstrap/repro_probe.ardisa"
  echo "probe_source_sha256=$PROBE_SHA256"
  echo "host_compiler=target/debug/ardisa"
  echo "host_compiler_sha256=$HOST_COMPILER_SHA256"
  echo "probe_stage2_stage3_byte_identical=true"
  echo "probe_replay_mode=host_compiler_deterministic_rebuild"
  echo "probe_stage2=target/bootstrap-probe-stage2.aexe"
  echo "probe_stage3=target/bootstrap-probe-stage3.aexe"
  echo "probe_stage2_sha256=$STAGE2_SHA256"
  echo "probe_stage3_sha256=$STAGE3_SHA256"
  echo "compiler_self_hosting=false"
  echo "self_hosting_ready=false"
  echo "compiler_self_hosting_requires_native_pipeline=true"
} > bootstrap/evidence/self-hosting.txt

echo "BOOTSTRAP-GATE: host-CLI probe replay verified; native self-hosting remains unproven."
