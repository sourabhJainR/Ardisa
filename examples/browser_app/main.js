import init, { ArdisaBrowserRuntime } from "../pkg/ardisa_wasm.js";

const status = document.querySelector("#status");
const output = document.querySelector("#output");
const nameInput = document.querySelector("#name");
const runButton = document.querySelector("#run");
const MAX_ARTIFACT_BYTES = 8 * 1024 * 1024;

let runtime;

async function readArtifactWithLimit(response) {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_ARTIFACT_BYTES) {
    throw new Error("Ardisa artifact exceeds the browser download limit.");
  }
  if (!response.body) {
    throw new Error("Streaming response support is required to load an Ardisa artifact safely.");
  }

  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      totalBytes += value.byteLength;
      if (totalBytes > MAX_ARTIFACT_BYTES) {
        await reader.cancel();
        throw new Error("Ardisa artifact exceeds the browser download limit.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}

async function start() {
  runButton.disabled = true;
  status.textContent = "Loading WebAssembly runtime...";
  try {
    await init();
    const response = await fetch("./app.aexe", { cache: "no-store" });
    if (!response.ok) throw new Error(`Unable to load Ardisa artifact: HTTP ${response.status}`);
    const artifact = await readArtifactWithLimit(response);
    runtime = new ArdisaBrowserRuntime(artifact);
    status.textContent = "Ardisa WebAssembly runtime ready.";
    runButton.disabled = false;
  } catch (error) {
    status.textContent = String(error);
  }
}

runButton.addEventListener("click", () => {
  if (!runtime) return;
  try {
    output.textContent = runtime.invoke("greet", [nameInput.value]);
  } catch (error) {
    output.textContent = String(error);
  }
});

void start();
