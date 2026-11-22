import init, { ArdisaBrowserRuntime } from "../pkg/ardisa_wasm.js";

const status = document.querySelector("#status");
const output = document.querySelector("#output");
const nameInput = document.querySelector("#name");
const runButton = document.querySelector("#run");

let runtime;

async function start() {
  runButton.disabled = true;
  status.textContent = "Loading WebAssembly runtime...";
  try {
    await init();
    const response = await fetch("./app.aexe", { cache: "no-store" });
    if (!response.ok) throw new Error(`Unable to load Ardisa artifact: HTTP ${response.status}`);
    const artifact = await response.text();
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
