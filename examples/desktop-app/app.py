#!/usr/bin/env python3
"""Thin Tk desktop host for the Ardisa-owned application."""
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import messagebox

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ["cargo", "run", "--quiet", "-p", "ardisa", "--", "run", "examples/desktop_app.ardisa"]

def run_ardisa() -> str:
    result = subprocess.run(COMMAND, cwd=ROOT, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Ardisa execution failed")
    return result.stdout.strip()

def main() -> None:
    window = tk.Tk()
    window.title("Ardisa Desktop Demo")
    window.geometry("520x260")
    window.minsize(420, 220)
    tk.Label(window, text="Ardisa Desktop Showcase", font=("Segoe UI", 18, "bold")).pack(pady=(28, 8))
    tk.Label(window, text="The message below is returned by a compiled Ardisa program.", wraplength=440).pack(pady=8)
    output = tk.Label(window, text="Click Run Ardisa to execute the program.", font=("Segoe UI", 12), wraplength=440)
    output.pack(pady=18)
    def execute() -> None:
        try:
            output.config(text=run_ardisa())
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            messagebox.showerror("Ardisa execution failed", str(error))
    tk.Button(window, text="Run Ardisa", command=execute, padx=18, pady=8).pack()
    window.mainloop()

if __name__ == "__main__":
    main()
