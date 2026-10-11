use std::{env, fs, process::ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("check") => {
            let first = args.next();
            match first.as_deref() {
                Some("--json") => match args.next() {
                    Some(path) => check(&path, true),
                    None => {
                        eprintln!("error: check --json requires a source file");
                        ExitCode::from(2)
                    }
                },
                Some(path) => check(path, false),
                None => {
                    eprintln!("error: check requires a source file");
                    ExitCode::from(2)
                }
            }
        }
        Some("build") => match args.next() {
            Some(path) => build_file(&path),
            None => {
                eprintln!("error: build requires a source file");
                ExitCode::from(2)
            }
        },
        Some("run") => match args.next() {
            Some(path) => run_file(&path, args.collect()),
            None => {
                eprintln!("error: run requires a source file");
                ExitCode::from(2)
            }
        },
        Some("bootstrap") => match args.next().as_deref() {
            Some("compile") => match (args.next(), args.next()) {
                (Some(source), Some(output)) => bootstrap_compile(&source, &output),
                _ => { eprintln!("error: bootstrap compile requires source and output"); ExitCode::from(2) }
            },
            Some("run") => match args.next() {
                Some(output) => bootstrap_run(&output, args.collect()),
                None => { eprintln!("error: bootstrap run requires an executable artifact"); ExitCode::from(2) }
            },
            Some("compile-from-executable") => match (args.next(), args.next(), args.next()) {
                (Some(compiler), Some(source), Some(output)) => bootstrap_compile_from_executable(&compiler, &source, &output),
                _ => { eprintln!("error: bootstrap compile-from-executable requires compiler, source, and output"); ExitCode::from(2) }
            },
            Some("verify") => match args.next() {
                Some(output) => bootstrap_verify(&output),
                None => { eprintln!("error: bootstrap verify requires an executable artifact"); ExitCode::from(2) }
            },
            _ => { eprintln!("error: bootstrap requires compile, run, or verify"); ExitCode::from(2) }
        },
        Some("fmt") => match args.next() {
            Some(path) => format_file(&path),
            None => {
                eprintln!("error: fmt requires a source file");
                ExitCode::from(2)
            }
        },
        None | Some("help") | Some("--help") | Some("-h") => {
            println!("ardisa check [--json] <file>");
            println!("  Parse, type-check, and validate an Ardisa source file.");
            println!("ardisa build <file>");
            println!("  Lower Ardisa to readable Rust.");
            println!("ardisa run <file> [args...]");
            println!("  Compile and execute the module natively without Rust.");
            println!("  The entry function is 'main'; arguments are typed from its signature.");
            println!("ardisa bootstrap compile <source.ardisa> <output.aexe>");
            println!("  Produce a deterministic native executable artifact without invoking Rust at run time.");
            println!("ardisa bootstrap run <output.aexe> [args...]");
            println!("ardisa bootstrap compile-from-executable <compiler.aexe> <source.ardisa> <output.aexe>");
            println!("  Run an already-built Ardisa compiler executable and require an ARDISA-EXEC-V1 result.");
            println!("  Execute a previously produced Ardisa executable artifact.");
            println!("ardisa fmt <file>");
            println!("  Print canonical Ardisa source.");
            ExitCode::SUCCESS
        }
        Some(command) => {
            eprintln!("error: unknown command '{command}'");
            ExitCode::from(2)
        }
    }
}

fn bootstrap_compile_from_executable(compiler_path: &str, source_path: &str, output_path: &str) -> ExitCode {
    require_ardisa_extension(source_path);
    let compiler_artifact = match fs::read_to_string(compiler_path) {
        Ok(value) => value,
        Err(error) => { eprintln!("{compiler_path}: error[AIF000]: {error}"); return ExitCode::from(1); }
    };
    let compiler = match ardisa_core::native::decode_program(&compiler_artifact) {
        Ok(value) => value,
        Err(error) => { eprintln!("{compiler_path}: error[AIF603]: invalid compiler executable: {error:?}"); return ExitCode::from(1); }
    };
    let source = match fs::read_to_string(source_path) {
        Ok(value) => value,
        Err(error) => { eprintln!("{source_path}: error[AIF000]: {error}"); return ExitCode::from(1); }
    };
    let output = match ardisa_core::native::run_program(&compiler, "main", &[ardisa_core::NativeValue::String(source)]) {
        Ok(ardisa_core::NativeValue::String(value)) => value,
        Ok(value) => { eprintln!("{compiler_path}: error[AIF606]: compiler returned non-string value: {}", display_value(&value)); return ExitCode::from(1); }
        Err(error) => { eprintln!("{compiler_path}: error[AIF607]: compiler execution failed: {error:?}"); return ExitCode::from(1); }
    };
    let program = match ardisa_core::native::decode_program(&output) {
        Ok(value) => value,
        Err(error) => { eprintln!("{compiler_path}: error[AIF608]: compiler output is not ARDISA-EXEC-V1: {error:?}"); return ExitCode::from(1); }
    };
    if !program.functions.contains_key("main") {
        eprintln!("{compiler_path}: error[AIF609]: compiler output has no main entry");
        return ExitCode::from(1);
    }
    let canonical = ardisa_core::native::encode_program(&program);
    if canonical != output {
        eprintln!("{compiler_path}: error[AIF610]: compiler output is not canonically encoded");
        return ExitCode::from(1);
    }
    match fs::write(output_path, canonical) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("{output_path}: error[AIF000]: {error}"); ExitCode::from(1) }
    }
}

fn bootstrap_verify(path: &str) -> ExitCode {
    let artifact = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(error) => { eprintln!("{path}: error[AIF000]: {error}"); return ExitCode::from(1); }
    };
    match ardisa_core::native::decode_program(&artifact) {
        Ok(program) if program.functions.contains_key("main") => ExitCode::SUCCESS,
        Ok(_) => { eprintln!("{path}: error[AIF605]: executable has no main entry"); ExitCode::from(1) }
        Err(error) => { eprintln!("{path}: error[AIF603]: invalid executable: {error:?}"); ExitCode::from(1) }
    }
}

fn bootstrap_compile(path: &str, output: &str) -> ExitCode {
    require_ardisa_extension(path);
    let source = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(error) => { eprintln!("{path}: error[AIF000]: {error}"); return ExitCode::from(1); }
    };
    let compiled = match ardisa_core::compile_source(&source) {
        Ok(value) => value,
        Err(ardisa_core::PipelineError::Parse(errors)) => return emit_diagnostics(path, &source, false, errors),
        Err(ardisa_core::PipelineError::Semantic(errors)) => return emit_diagnostics(path, &source, false, errors),
        Err(ardisa_core::PipelineError::Ownership(errors)) => return emit_diagnostics(path, &source, false, errors),
        Err(ardisa_core::PipelineError::Concurrency(errors)) => return emit_diagnostics(path, &source, false, errors),
        Err(ardisa_core::PipelineError::TypedIr(errors)) => {
            for error in errors { eprintln!("{path}: error[AIF500]: {error}"); }
            return ExitCode::from(1);
        }
        Err(ardisa_core::PipelineError::Native(error)) => {
            eprintln!("{path}: error[AIF603]: native compilation failed: {error:?}");
            return ExitCode::from(1);
        }
        Err(ardisa_core::PipelineError::CompilerContract(errors)) => {
            for error in errors { eprintln!("{path}: error[AIF611]: compiler phase contract failed: {error}"); }
            return ExitCode::from(1);
        }
    };
    if let Err(error) = fs::write(output, compiled.artifact) {
        eprintln!("{output}: error[AIF000]: {error}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn bootstrap_run(path: &str, raw_args: Vec<String>) -> ExitCode {
    let artifact = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(error) => { eprintln!("{path}: error[AIF000]: {error}"); return ExitCode::from(1); }
    };
    let program = match ardisa_core::native::decode_program(&artifact) {
        Ok(value) => value,
        Err(error) => { eprintln!("{path}: error[AIF603]: invalid executable: {error:?}"); return ExitCode::from(1); }
    };
    let Some(main) = program.functions.get("main") else {
        eprintln!("{path}: error[AIF600]: entry function 'main' was not found");
        return ExitCode::from(1);
    };
    if raw_args.len() != main.params.len() {
        eprintln!("{path}: error[AIF601]: main expects {} argument(s), got {}", main.params.len(), raw_args.len());
        return ExitCode::from(1);
    }
    let values = raw_args.into_iter().map(|raw| ardisa_core::NativeValue::String(raw)).collect::<Vec<_>>();
    match ardisa_core::native::run_program(&program, "main", &values) {
        Ok(value) => { println!("{}", display_value(&value)); ExitCode::SUCCESS }
        Err(error) => { eprintln!("{path}: error[AIF604]: native execution failed: {error:?}"); ExitCode::from(1) }
    }
}

fn build_file(path: &str) -> ExitCode {
    require_ardisa_extension(path);
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("{path}: error[AIF000]: {error}");
            return ExitCode::from(1);
        }
    };

    match ardisa_core::parse(&source) {
        Ok(module) => match ardisa_core::sema::check(&module) {
            Ok(()) => match ardisa_core::ownership::infer(&module) {
                Ok(()) => {
                    print!("{}", ardisa_core::lower::lower(&module).rust);
                    ExitCode::SUCCESS
                }
                Err(errors) => emit_diagnostics(path, &source, false, errors),
            },
            Err(errors) => emit_diagnostics(path, &source, false, errors),
        },
        Err(errors) => emit_diagnostics(path, &source, false, errors),
    }
}

fn run_file(path: &str, raw_args: Vec<String>) -> ExitCode {
    require_ardisa_extension(path);
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("{path}: error[AIF000]: {error}");
            return ExitCode::from(1);
        }
    };

    let module = match ardisa_core::parse(&source) {
        Ok(module) => module,
        Err(errors) => return emit_diagnostics(path, &source, false, errors),
    };
    if let Err(errors) = ardisa_core::sema::check(&module) {
        return emit_diagnostics(path, &source, false, errors);
    }
    if let Err(errors) = ardisa_core::ownership::infer(&module) {
        return emit_diagnostics(path, &source, false, errors);
    }

    let Some(main) = module.items.iter().find_map(|item| match item {
        ardisa_core::Item::Function(function) if function.name == "main" => Some(function),
        _ => None,
    }) else {
        eprintln!("{path}: error[AIF600]: entry function 'main' was not found");
        return ExitCode::from(1);
    };

    let mut values = Vec::with_capacity(main.params.len());
    if raw_args.len() != main.params.len() {
        eprintln!(
            "{path}: error[AIF601]: main expects {} argument(s), got {}",
            main.params.len(),
            raw_args.len()
        );
        return ExitCode::from(1);
    }

    for (raw, parameter) in raw_args.iter().zip(&main.params) {
        match parse_value(raw, &parameter.ty.kind) {
            Ok(value) => values.push(value),
            Err(message) => {
                eprintln!(
                    "{path}: error[AIF602]: argument '{}' for '{}' {}",
                    raw, parameter.name, message
                );
                return ExitCode::from(1);
            }
        }
    }

    let ir = ardisa_core::ir::lower(&module);
    let program = match ardisa_core::native::compile_program(&ir) {
        Ok(program) => program,
        Err(error) => {
            eprintln!("{path}: error[AIF603]: native compilation failed: {error:?}");
            return ExitCode::from(1);
        }
    };
    match ardisa_core::native::run_program(&program, "main", &values) {
        Ok(value) => {
            println!("{}", display_value(&value));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{path}: error[AIF604]: native execution failed: {error:?}");
            ExitCode::from(1)
        }
    }
}

fn parse_value(
    raw: &str,
    ty: &ardisa_core::TypeKind,
) -> Result<ardisa_core::NativeValue, &'static str> {
    match ty {
        ardisa_core::TypeKind::Int => raw
            .parse::<i64>()
            .map(ardisa_core::NativeValue::Int)
            .map_err(|_| "must be an Int"),
        ardisa_core::TypeKind::Bool => match raw {
            "true" => Ok(ardisa_core::NativeValue::Bool(true)),
            "false" => Ok(ardisa_core::NativeValue::Bool(false)),
            _ => Err("must be true or false"),
        },
        ardisa_core::TypeKind::String => Ok(ardisa_core::NativeValue::String(raw.to_owned())),
        _ => Err("has a type not supported by the native CLI yet"),
    }
}

fn display_value(value: &ardisa_core::NativeValue) -> String {
    match value {
        ardisa_core::NativeValue::Int(value) => value.to_string(),
        ardisa_core::NativeValue::Bool(value) => value.to_string(),
        ardisa_core::NativeValue::Probability(bits) => f32::from_bits(*bits).to_string(),
        ardisa_core::NativeValue::String(value) => value.clone(),
        ardisa_core::NativeValue::List(values) => {
            let rendered = values.iter().map(display_value).collect::<Vec<_>>();
            format!("[{}]", rendered.join(", "))
        }
        ardisa_core::NativeValue::ResultOk(value) => format!("Ok({})", display_value(value)),
        ardisa_core::NativeValue::ResultErr(value) => format!("Err({})", display_value(value)),
        ardisa_core::NativeValue::Unit => "()".into(),
    }
}

fn format_file(path: &str) -> ExitCode {
    require_ardisa_extension(path);
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("{path}: error[AIF000]: {error}");
            return ExitCode::from(1);
        }
    };

    match ardisa_core::parse(&source) {
        Ok(module) => {
            print!("{}", ardisa_core::format::format_module(&module));
            ExitCode::SUCCESS
        }
        Err(errors) => {
            for error in errors {
                eprintln!("{path}: error[{}]: {}", error.code, error.message);
            }
            ExitCode::from(1)
        }
    }
}

fn check(path: &str, json: bool) -> ExitCode {
    require_ardisa_extension(path);
    let source = match fs::read_to_string(path) {
        Ok(source) if source.trim().is_empty() => {
            emit_error(path, "AIF001", "source file is empty", None, &source, json);
            return ExitCode::from(1);
        }
        Ok(source) => source,
        Err(error) => {
            eprintln!("{path}: error[AIF000]: {error}");
            return ExitCode::from(1);
        }
    };

    match ardisa_core::parse(&source) {
        Ok(module) => match ardisa_core::sema::check(&module) {
            Ok(()) => match ardisa_core::ownership::infer(&module) {
                Ok(()) => {
                    if !json {
                        println!(
                            "{path}: ok (module {}, {} item(s))",
                            module.name,
                            module.items.len()
                        );
                    }
                    ExitCode::SUCCESS
                }
                Err(errors) => emit_diagnostics(path, &source, json, errors),
            },
            Err(errors) => emit_diagnostics(path, &source, json, errors),
        },
        Err(errors) => emit_diagnostics(path, &source, json, errors),
    }
}

fn emit_diagnostics(
    path: &str,
    source: &str,
    json: bool,
    errors: Vec<ardisa_core::source::Diagnostic>,
) -> ExitCode {
    for error in errors {
        if json {
            println!("{}", error.to_json(source));
        } else if let Some(location) = error.location(source) {
            eprintln!(
                "{path}:{}:{}: error[{}]: {}",
                location.line, location.column, error.code, error.message
            );
        } else {
            eprintln!("{path}: error[{}]: {}", error.code, error.message);
        }
    }
    ExitCode::from(1)
}

fn emit_error(
    path: &str,
    code: &'static str,
    message: &str,
    span: Option<ardisa_core::source::Span>,
    source: &str,
    json: bool,
) {
    let diagnostic = ardisa_core::source::Diagnostic::error(code, message, span);
    if json {
        println!("{}", diagnostic.to_json(source));
    } else {
        eprintln!("{path}: error[{code}]: {message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_typed_cli_arguments() {
        assert_eq!(
            parse_value("42", &ardisa_core::TypeKind::Int).unwrap(),
            ardisa_core::NativeValue::Int(42)
        );
        assert_eq!(
            parse_value("true", &ardisa_core::TypeKind::Bool).unwrap(),
            ardisa_core::NativeValue::Bool(true)
        );
        assert!(parse_value("wat", &ardisa_core::TypeKind::Bool).is_err());
    }

    #[test]
    fn renders_native_values_deterministically() {
        let value =
            ardisa_core::NativeValue::ResultOk(Box::new(ardisa_core::NativeValue::List(vec![
                ardisa_core::NativeValue::Int(1),
                ardisa_core::NativeValue::Int(2),
            ])));
        assert_eq!(display_value(&value), "Ok([1, 2])");
    }
}

fn require_ardisa_extension(path: &str) {
    if !path.ends_with(".ardisa") {
        eprintln!("{path}: error[AIF002]: Ardisa source files must use the .ardisa extension");
        std::process::exit(2);
    }
}
