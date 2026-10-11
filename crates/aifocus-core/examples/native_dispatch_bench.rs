use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ardisa_core::native::{NativeFunction, NativeInstr, NativeProgram, NativeValue, run_program, run_shared_program_with_limits, run_validated_program_with_limits, ValidatedNativeProgram, ExecutionLimits};

const SMALL_STRING_REPEATS: usize = 2_048;
const LARGE_STRING_REPEATS: usize = 256;
const LARGE_STRING_BYTES: usize = 4_096;
const ARITHMETIC_REPEATS: usize = 20_000;

struct Workload {
    name: &'static str,
    program: NativeProgram,
    instruction_count: usize,
    embedded_string_bytes: usize,
    estimated_dispatch_payload_bytes_avoided_per_pass: usize,
}

fn main() {
    let iterations = parse_iterations().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!("usage: cargo run -p ardisa-core --example native_dispatch_bench -- [--iterations N]");
        std::process::exit(2);
    });

    println!("Ardisa native VM dispatch benchmark");
    println!("iterations={iterations} timing_threshold=none");
    println!("Note: payload-byte savings are a static estimate of the removed dispatch clone, not an allocator measurement.");
    for workload in workloads() {
        report(&workload, iterations);
    }
}

fn parse_iterations() -> Result<usize, String> {
    let mut args = std::env::args().skip(1);
    let mut iterations = 5usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--iterations" => {
                let value = args.next().ok_or("--iterations requires a positive integer")?;
                iterations = value
                    .parse::<usize>()
                    .map_err(|_| "--iterations requires a positive integer")?;
                if iterations == 0 {
                    return Err("--iterations must be greater than zero".into());
                }
            }
            "--help" | "-h" => {
                return Err("usage: cargo run -p ardisa-core --example native_dispatch_bench -- [--iterations N]".into());
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    Ok(iterations)
}

fn workloads() -> Vec<Workload> {
    vec![
        arithmetic_and_jumps(),
        string_workload("small_strings", "tiny", SMALL_STRING_REPEATS),
        string_workload(
            "large_embedded_strings",
            &"x".repeat(LARGE_STRING_BYTES),
            LARGE_STRING_REPEATS,
        ),
    ]
}

fn arithmetic_and_jumps() -> Workload {
    let mut code = Vec::with_capacity(ARITHMETIC_REPEATS * 6 + 2);
    for _ in 0..ARITHMETIC_REPEATS {
        let start = code.len();
        code.push(NativeInstr::PushBool(true));
        code.push(NativeInstr::JumpIfFalse(start + 6));
        code.push(NativeInstr::PushInt(3));
        code.push(NativeInstr::PushInt(4));
        code.push(NativeInstr::Add);
        code.push(NativeInstr::Pop);
    }
    code.push(NativeInstr::PushInt(7));
    code.push(NativeInstr::Return);
    workload("arithmetic_and_jumps", code)
}

fn string_workload(name: &'static str, value: &str, repeats: usize) -> Workload {
    let mut code = Vec::with_capacity(repeats * 2 + 2);
    for _ in 0..repeats {
        code.push(NativeInstr::PushString(value.to_owned()));
        code.push(NativeInstr::Pop);
    }
    code.push(NativeInstr::PushInt(7));
    code.push(NativeInstr::Return);
    workload(name, code)
}

fn workload(name: &'static str, code: Vec<NativeInstr>) -> Workload {
    let embedded_string_bytes = code
        .iter()
        .map(|instruction| match instruction {
            NativeInstr::PushString(value)
            | NativeInstr::Append(value)
            | NativeInstr::Load(value)
            | NativeInstr::Store(value)
            | NativeInstr::AddAssign(value)
            | NativeInstr::StoreIndex(value)
            | NativeInstr::Join { name: value }
            | NativeInstr::Cancel { name: value } => value.len(),
            NativeInstr::Call { callee, .. } => callee.len(),
            NativeInstr::Spawn { name, callee, .. } => name.len() + callee.len(),
            _ => 0,
        })
        .sum::<usize>();
    let estimated_dispatch_payload_bytes_avoided_per_pass = embedded_string_bytes;
    Workload {
        name,
        instruction_count: code.len(),
        embedded_string_bytes,
        estimated_dispatch_payload_bytes_avoided_per_pass,
        program: NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code },
            )]),
        },
    }
}

fn report(workload: &Workload, iterations: usize) {
    report_mode(workload, iterations, 0);
    report_mode(workload, iterations, 1);
    report_mode(workload, iterations, 2);
}

fn report_mode(workload: &Workload, iterations: usize, mode: u8) {
    // Construct reusable objects before timing so each mode measures its entry point.
    let shared_program = Arc::new(workload.program.clone());
    let validated_program = ValidatedNativeProgram::try_new(workload.program.clone())
        .expect("benchmark workload should validate");
    let mut elapsed = Duration::ZERO;
    for _ in 0..iterations {
        let started = Instant::now();
        let result = match mode {
            1 => run_shared_program_with_limits(
                Arc::clone(&shared_program), "main", &[], ExecutionLimits::default()
            ),
            2 => run_validated_program_with_limits(
                &validated_program, "main", &[], ExecutionLimits::default()
            ),
            _ => run_program(&workload.program, "main", &[]),
        };
        elapsed += started.elapsed();
        assert_eq!(
            result,
            Ok(NativeValue::Int(7)),
            "{} returned an unexpected result (mode={mode})",
            workload.name
        );
    }

    let executed_instructions = workload.instruction_count.saturating_mul(iterations);
    let elapsed_ns = elapsed.as_nanos();
    let ns_per_instruction = if executed_instructions == 0 {
        0.0
    } else {
        elapsed_ns as f64 / executed_instructions as f64
    };
    let instructions_per_second = if elapsed_ns == 0 {
        0.0
    } else {
        executed_instructions as f64 * 1_000_000_000.0 / elapsed_ns as f64
    };
    println!(
        "workload={} mode={} code_instructions={} embedded_string_bytes={} estimated_dispatch_payload_bytes_avoided_per_pass={} iterations={} elapsed_ns={} ns_per_instruction={:.2} instructions_per_second={:.0}",
        workload.name,
        match mode {
            1 => "shared_program_revalidates_each_call",
            2 => "validated_program_reuses_validation",
            _ => "cloning_entry_point",
        },
        workload.instruction_count,
        workload.embedded_string_bytes,
        workload.estimated_dispatch_payload_bytes_avoided_per_pass,
        iterations,
        elapsed_ns,
        ns_per_instruction,
        instructions_per_second
    );
}
