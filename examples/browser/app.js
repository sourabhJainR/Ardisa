import init, { run_artifact, runtime_abi_version } from "./pkg/ardisa_wasm.js";

const result = document.querySelector("#result");
const runButton = document.querySelector("#run");
let ready = false;
try {
  await init();
  ready = true;
  result.textContent = "Ardisa WASM ABI " + runtime_abi_version() + " ready.";
} catch (error) {
  result.textContent = "Runtime load failed: " + String(error);
}
runButton.addEventListener("click", async () => {
  if (!ready) return;
  runButton.disabled = true;
  try {
    const file = document.querySelector("#artifact").files[0];
    if (!file) throw new Error("Select an Ardisa executable artifact first.");
    if (file.size > 1048576) throw new Error("Artifact exceeds the 1 MiB browser limit.");
    const entry = document.querySelector("#entry").value;
    const args = JSON.parse(document.querySelector("#args").value);
    if (!Array.isArray(args)) throw new Error("Arguments must be a JSON array.");
    const value = run_artifact(await file.text(), entry, args);
    result.textContent = value;
  } catch (error) {
    result.textContent = "Execution failed: " + String(error);
  } finally {
    runButton.disabled = false;
  }
});
