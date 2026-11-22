# Ardisa desktop app demo

A small desktop window built with Python's Tkinter GUI toolkit. The application message and behaviour come from desktop_app.ardisa; Python only starts Ardisa and displays its output. Ardisa does not yet expose native desktop-window APIs, so this is an integration demo, not a native Ardisa GUI toolkit.

## Requirements

- Rust stable toolchain and Cargo
- Python 3.10+
- Tkinter (included with most Python installers; on Debian/Ubuntu install python3-tk if missing)
- Run commands from the Ardisa repository root

## Download and install the compiler

```sh
git clone https://github.com/sourabhJainR/Ardisa.git
cd Ardisa
cargo build --release -p ardisa
```

To install the CLI into Cargo's user binary directory:

```sh
cargo install --path crates/aifocus-cli --force
```

## Compile and run the Ardisa application

```sh
cargo run -p ardisa -- check examples/desktop_app.ardisa
cargo run -p ardisa -- run examples/desktop_app.ardisa
python examples/desktop-app/app.py
```

The window opens. Select Run Ardisa to execute the .ardisa program again. Close the window to exit. The demo uses the repository-local compiler via Cargo, so it works without a globally installed ardisa command. The first run builds the compiler and may take longer.
