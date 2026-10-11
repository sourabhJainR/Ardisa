use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread::{self, JoinHandle};

use crate::TypeKind;
use crate::ir::{IrFunction, IrModule, IrOp, IrValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeInstr {
    PushInt(i64),
    PushBool(bool),
    PushUnit,
    PushString(String),
    PushList(usize),
    Index,
    Len,
    Append(String),
    MakeOk,
    Chr,
    MakeErr,
    Unwrap,
    Load(String),
    Store(String),
    AddAssign(String),
    StoreIndex(String),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    JumpIfFalse(usize),
    Jump(usize),
    Return,
    Pop,
    Call {
        callee: String,
        argc: usize,
    },
    ScopeStart,
    ScopeEnd,
    Spawn {
        name: String,
        callee: String,
        argc: usize,
    },
    Join {
        name: String,
    },
    Cancel {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Deterministic collection of compiled Ardisa functions.
pub struct NativeProgram {
    pub functions: BTreeMap<String, NativeFunction>,
}

/// An immutable program that has passed native-program validation.
///
/// The wrapped program is never exposed mutably. Keeping it behind an Arc also
/// prevents mutation through Arc::get_mut while this wrapper exists. Execution
/// arguments and limits are deliberately validated per invocation.
#[derive(Debug, Clone)]
pub struct ValidatedNativeProgram {
    program: Arc<NativeProgram>,
}

impl ValidatedNativeProgram {
    /// Validate and take ownership of a native program.
    pub fn try_new(program: NativeProgram) -> Result<Self, NativeError> {
        Self::try_from_shared(Arc::new(program))
    }

    /// Validate a shared program and retain an immutable reference to it.
    pub fn try_from_shared(program: Arc<NativeProgram>) -> Result<Self, NativeError> {
        validate_native_program(&program)?;
        Ok(Self { program })
    }

    /// Borrow the validated program without exposing mutable access.
    pub fn as_program(&self) -> &NativeProgram {
        &self.program
    }

    /// Execute with fresh runtime state and per-call argument/limit checks.
    pub fn run_with_limits(
        &self,
        entry: &str,
        args: &[NativeValue],
        limits: ExecutionLimits,
    ) -> Result<NativeValue, NativeError> {
        run_validated_program_with_limits(self, entry, args, limits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFunction {
    pub params: Vec<String>,
    pub code: Vec<NativeInstr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Native values are intentionally dependency-free so the native compiler can bootstrap incrementally.
pub enum NativeValue {
    Int(i64),
    Bool(bool),
    String(String),
    List(Vec<NativeValue>),
    ResultOk(Box<NativeValue>),
    ResultErr(Box<NativeValue>),
    Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeError {
    Unsupported(String),
    Cancelled(String),
    InvalidProgram(String),
    Type(String),
    ResourceLimit(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub max_instructions: usize,
    pub max_call_depth: usize,
    pub max_tasks: usize,
    pub max_collection_items: usize,
    pub max_string_bytes: usize,
    /// Maximum total nodes in any single nested runtime value.
    pub max_value_nodes: usize,
    /// Maximum nesting depth for List and Result values (root depth is zero).
    pub max_value_depth: usize,
    /// Maximum values held on one VM frame's operand stack.
    pub max_stack_values: usize,
    /// Maximum local bindings in one VM frame.
    pub max_locals: usize,
    /// Aggregate nested value nodes across one frame's locals and operand stack.
    pub max_frame_value_nodes: usize,
    /// Aggregate string payload bytes across one frame's locals and operand stack.
    pub max_frame_string_bytes: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_instructions: 1_000_000,
            max_call_depth: 128,
            max_tasks: 64,
            max_collection_items: 16_384,
            max_string_bytes: 1_048_576,
            max_value_nodes: 16_384,
            max_value_depth: 64,
            max_stack_values: 16_384,
            max_locals: 4_096,
            max_frame_value_nodes: 65_536,
            max_frame_string_bytes: 8_388_608,
        }
    }
}

struct ExecutionState {
    limits: ExecutionLimits,
    instructions: AtomicUsize,
    active_tasks: AtomicUsize,
}

struct ActiveTaskLease(Arc<ExecutionState>);

impl Drop for ActiveTaskLease {
    fn drop(&mut self) {
        self.0.active_tasks.fetch_sub(1, Ordering::AcqRel);
    }
}

fn reserve_task(state: &Arc<ExecutionState>) -> Result<ActiveTaskLease, NativeError> {
    let mut current = state.active_tasks.load(Ordering::Acquire);
    loop {
        if current >= state.limits.max_tasks {
            return Err(NativeError::ResourceLimit(format!(
                "active task limit exceeded (limit {})",
                state.limits.max_tasks
            )));
        }
        match state.active_tasks.compare_exchange_weak(
            current, current + 1, Ordering::AcqRel, Ordering::Acquire
        ) {
            Ok(_) => return Ok(ActiveTaskLease(Arc::clone(state))),
            Err(observed) => current = observed,
        }
    }
}

fn validate_value(value: &NativeValue, limits: ExecutionLimits) -> Result<(), NativeError> {
    let mut pending = vec![(value, 0usize)];
    let mut nodes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes = nodes.checked_add(1).ok_or_else(|| NativeError::ResourceLimit("value node count overflow".into()))?;
        if nodes > limits.max_value_nodes {
            return Err(NativeError::ResourceLimit(format!("nested value node limit exceeded (limit {})", limits.max_value_nodes)));
        }
        if depth > limits.max_value_depth {
            return Err(NativeError::ResourceLimit(format!("value nesting depth exceeded (limit {})", limits.max_value_depth)));
        }
        match value {
            NativeValue::String(s) if s.len() > limits.max_string_bytes => {
                return Err(NativeError::ResourceLimit("string exceeds byte limit".into()));
            }
            NativeValue::List(items) => {
                if items.len() > limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("collection exceeds item limit".into()));
                }
                pending.extend(items.iter().map(|item| (item, depth + 1)));
            }
            NativeValue::ResultOk(inner) | NativeValue::ResultErr(inner) => {
                pending.push((inner, depth + 1));
            }
            _ => {}
        }
    }
    Ok(())
}

fn value_footprint(value: &NativeValue) -> Result<(usize, usize), NativeError> {
    let mut pending = vec![value];
    let mut nodes = 0usize;
    let mut string_bytes = 0usize;
    while let Some(value) = pending.pop() {
        nodes = nodes.checked_add(1)
            .ok_or_else(|| NativeError::ResourceLimit("value node count overflow".into()))?;
        match value {
            NativeValue::String(value) => {
                string_bytes = string_bytes.checked_add(value.len())
                    .ok_or_else(|| NativeError::ResourceLimit("string byte accounting overflow".into()))?;
            }
            NativeValue::List(items) => pending.extend(items.iter()),
            NativeValue::ResultOk(inner) | NativeValue::ResultErr(inner) => pending.push(inner),
            _ => {}
        }
    }
    Ok((nodes, string_bytes))
}

fn check_frame_value_budget(
    locals: &HashMap<String, NativeValue>,
    stack: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<(), NativeError> {
    let mut nodes = 0usize;
    let mut string_bytes = 0usize;
    for value in locals.values().chain(stack.iter()) {
        validate_value(value, limits)?;
        let (value_nodes, value_string_bytes) = value_footprint(value)?;
        nodes = nodes.checked_add(value_nodes)
            .ok_or_else(|| NativeError::ResourceLimit("frame value node accounting overflow".into()))?;
        string_bytes = string_bytes.checked_add(value_string_bytes)
            .ok_or_else(|| NativeError::ResourceLimit("frame string accounting overflow".into()))?;
        if nodes > limits.max_frame_value_nodes {
            return Err(NativeError::ResourceLimit(format!(
                "aggregate frame value node limit exceeded (limit {})", limits.max_frame_value_nodes
            )));
        }
        if string_bytes > limits.max_frame_string_bytes {
            return Err(NativeError::ResourceLimit(format!(
                "aggregate frame string byte limit exceeded (limit {})", limits.max_frame_string_bytes
            )));
        }
    }
    Ok(())
}

fn validate_input_values(values: &[NativeValue], limits: ExecutionLimits) -> Result<(), NativeError> {
    for value in values {
        validate_value(value, limits)?;
    }
    Ok(())
}

/// Compile the first function for the legacy single-function API.
pub fn compile(module: &IrModule) -> Result<Vec<NativeInstr>, NativeError> {
    let function = module
        .functions
        .first()
        .ok_or_else(|| NativeError::InvalidProgram("module has no functions".into()))?;
    compile_function(function)
}

pub fn compile_program(module: &IrModule) -> Result<NativeProgram, NativeError> {
    let mut functions = BTreeMap::new();
    for function in &module.functions {
        functions.insert(
            function.name.clone(),
            NativeFunction {
                params: function
                    .params
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect(),
                code: compile_function(function)?,
            },
        );
    }
    Ok(NativeProgram { functions })
}

pub fn compile_function(function: &IrFunction) -> Result<Vec<NativeInstr>, NativeError> {
    if function.params.iter().any(|(_, ty)| !is_native_type(ty)) {
        return Err(NativeError::Unsupported(
            "native backend does not support this parameter type".into(),
        ));
    }
    let mut code = Vec::new();
    for (index, op) in function.ops.iter().enumerate() {
        let is_last_expression = index + 1 == function.ops.len()
            && matches!(op, IrOp::Expr(_))
            && function.return_type.is_some();
        if is_last_expression {
            if let IrOp::Expr(value) = op {
                emit_value(value, &mut code)?;
                code.push(NativeInstr::Return);
            }
        } else {
            emit_op(op, &mut code)?;
        }
    }
    if !matches!(code.last(), Some(NativeInstr::Return)) {
        code.push(NativeInstr::Return);
    }
    Ok(code)
}

fn is_native_type(ty: &TypeKind) -> bool {
    match ty {
        TypeKind::Int | TypeKind::Bool | TypeKind::String | TypeKind::Unit => true,
        TypeKind::List(element) => is_native_type(&element.kind),
        TypeKind::Result(ok, err) => is_native_type(&ok.kind) && is_native_type(&err.kind),
        TypeKind::Named(_) | TypeKind::Generic(_, _) => false,
    }
}

fn emit_op(op: &IrOp, code: &mut Vec<NativeInstr>) -> Result<(), NativeError> {
    match op {
        IrOp::Let { name, value } => {
            emit_value(value, code)?;
            code.push(NativeInstr::Store(name.clone()));
        }
        IrOp::Set { name, value } => {
            if let IrValue::Binary {
                op: crate::BinaryOp::Add,
                left,
                right,
            } = value
            {
                if matches!(left.as_ref(), IrValue::Name(existing) if existing == name) {
                    emit_value(right, code)?;
                    code.push(NativeInstr::AddAssign(name.clone()));
                    return Ok(());
                }
            }
            emit_value(value, code)?;
            code.push(NativeInstr::Store(name.clone()));
        }
        IrOp::SetIndex {
            collection,
            index,
            value,
        } => {
            let IrValue::Name(name) = collection else {
                return Err(NativeError::Unsupported(
                    "indexed assignment currently requires a named list binding".into(),
                ));
            };
            code.push(NativeInstr::Load(name.clone()));
            emit_value(index, code)?;
            emit_value(value, code)?;
            code.push(NativeInstr::StoreIndex(name.clone()));
        }
        IrOp::Return(value) => {
            if let Some(value) = value {
                emit_value(value, code)?;
            } else {
                code.push(NativeInstr::PushInt(0));
            }
            code.push(NativeInstr::Return);
        }
        IrOp::Expr(value) => {
            emit_value(value, code)?;
            code.push(NativeInstr::Pop);
        }
        IrOp::While { condition, ops } => {
            let loop_start = code.len();
            emit_value(condition, code)?;
            let jump_if = code.len();
            code.push(NativeInstr::JumpIfFalse(usize::MAX));
            for op in ops {
                emit_op(op, code)?;
            }
            code.push(NativeInstr::Jump(loop_start));
            let end = code.len();
            code[jump_if] = NativeInstr::JumpIfFalse(end);
        }
        IrOp::Scope { ops } => {
            code.push(NativeInstr::ScopeStart);
            for op in ops {
                emit_op(op, code)?;
            }
            code.push(NativeInstr::ScopeEnd);
        }
        IrOp::Spawn { name, call } => {
            let IrValue::Call { callee, args } = call else {
                return Err(NativeError::Unsupported(
                    "spawn requires a function call".into(),
                ));
            };
            for arg in args {
                emit_value(arg, code)?;
            }
            code.push(NativeInstr::Spawn {
                name: name.clone(),
                callee: callee.clone(),
                argc: args.len(),
            });
        }
        IrOp::Join { name } => code.push(NativeInstr::Join { name: name.clone() }),
        IrOp::Cancel { name } => code.push(NativeInstr::Cancel { name: name.clone() }),
    }
    Ok(())
}

fn emit_branch(ops: &[IrOp], code: &mut Vec<NativeInstr>) -> Result<(), NativeError> {
    if ops.is_empty() {
        code.push(NativeInstr::PushUnit);
        return Ok(());
    }
    for (index, op) in ops.iter().enumerate() {
        let is_last = index + 1 == ops.len();
        if is_last {
            match op {
                IrOp::Expr(value) => emit_value(value, code)?,
                IrOp::Return(_) => emit_op(op, code)?,
                _ => {
                    emit_op(op, code)?;
                    code.push(NativeInstr::PushUnit);
                }
            }
        } else {
            emit_op(op, code)?;
        }
    }
    Ok(())
}

fn emit_value(value: &IrValue, code: &mut Vec<NativeInstr>) -> Result<(), NativeError> {
    match value {
        IrValue::Int(value) => code.push(NativeInstr::PushInt(*value)),
        IrValue::String(value) => code.push(NativeInstr::PushString(value.clone())),
        IrValue::List(values) => {
            for value in values {
                emit_value(value, code)?;
            }
            code.push(NativeInstr::PushList(values.len()));
        }
        IrValue::Index { collection, index } => {
            emit_value(collection, code)?;
            emit_value(index, code)?;
            code.push(NativeInstr::Index);
        }
        IrValue::Bool(value) => code.push(NativeInstr::PushBool(*value)),
        IrValue::Name(name) => code.push(NativeInstr::Load(name.clone())),
        IrValue::Binary { op, left, right } => {
            emit_value(left, code)?;
            emit_value(right, code)?;
            code.push(match op {
                crate::BinaryOp::Add => NativeInstr::Add,
                crate::BinaryOp::Sub => NativeInstr::Sub,
                crate::BinaryOp::Mul => NativeInstr::Mul,
                crate::BinaryOp::Div => NativeInstr::Div,
                crate::BinaryOp::Mod => NativeInstr::Mod,
                crate::BinaryOp::Equal => NativeInstr::Equal,
                crate::BinaryOp::NotEqual => NativeInstr::NotEqual,
                crate::BinaryOp::Less => NativeInstr::Less,
                crate::BinaryOp::LessEqual => NativeInstr::LessEqual,
                crate::BinaryOp::Greater => NativeInstr::Greater,
                crate::BinaryOp::GreaterEqual => NativeInstr::GreaterEqual,
            });
        }
        IrValue::If {
            condition,
            then_ops,
            else_ops,
        } => {
            emit_value(condition, code)?;
            let jump_if = code.len();
            code.push(NativeInstr::JumpIfFalse(usize::MAX));
            emit_branch(then_ops, code)?;
            let jump_end = code.len();
            code.push(NativeInstr::Jump(usize::MAX));
            let else_start = code.len();
            emit_branch(else_ops, code)?;
            let end = code.len();
            code[jump_if] = NativeInstr::JumpIfFalse(else_start);
            code[jump_end] = NativeInstr::Jump(end);
        }
        IrValue::Call { callee, args } => {
            for arg in args {
                emit_value(arg, code)?;
            }
            if callee == "chr" && args.len() == 1 {
                code.push(NativeInstr::Chr);
            } else if callee == "ok" && args.len() == 1 {
                code.push(NativeInstr::MakeOk);
            } else if callee == "err" && args.len() == 1 {
                code.push(NativeInstr::MakeErr);
            } else if callee == "unwrap" && args.len() == 1 {
                code.push(NativeInstr::Unwrap);
            } else if callee == "len" && args.len() == 1 {
                code.push(NativeInstr::Len);
            } else if callee == "push" && args.len() == 2 {
                let crate::ir::IrValue::Name(name) = &args[0] else {
                    return Err(NativeError::Unsupported(
                        "push currently requires a named list binding".into(),
                    ));
                };
                emit_value(&args[1], code)?;
                code.push(NativeInstr::Append(name.clone()));
            } else {
                code.push(NativeInstr::Call {
                    callee: callee.clone(),
                    argc: args.len(),
                });
            }
        }
    }
    Ok(())
}

pub fn run(
    code: &[NativeInstr],
    args: &[(String, NativeValue)],
) -> Result<NativeValue, NativeError> {
    let params = args.iter().map(|(name, _)| name.clone()).collect();
    let values = args.iter().map(|(_, value)| value.clone()).collect::<Vec<_>>();
    let program = NativeProgram {
        functions: BTreeMap::from([(
            "main".to_string(),
            NativeFunction {
                params,
                code: code.to_vec(),
            },
        )]),
    };
    run_program_with_limits(&program, "main", &values, ExecutionLimits::default())
}

struct NativeTask {
    cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<Result<NativeValue, NativeError>>>,
}

impl NativeTask {
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

impl Drop for NativeTask {
    fn drop(&mut self) {
        self.cancel();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn run_program(
    program: &NativeProgram,
    entry: &str,
    args: &[NativeValue],
) -> Result<NativeValue, NativeError> {
    run_program_with_limits(program, entry, args, ExecutionLimits::default())
}

/// Execute with explicit instruction, recursion, task, collection, and string bounds.
/// Limits are shared by all calls and child tasks in this execution.
///
/// Program representation limits are independent of runtime value limits. They
/// bound function/code counts and embedded names/literals, but do not account for
/// allocator capacity, allocator metadata, host allocations, or concurrent programs.
pub fn run_program_with_limits(
    program: &NativeProgram,
    entry: &str,
    args: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    run_shared_program_with_limits(Arc::new(program.clone()), entry, args, limits)
}

/// Execute an immutable program shared by the caller and all child tasks.
///
/// This compatibility entry point validates on each call. Callers that can
/// retain a validated wrapper should use run_validated_program_with_limits to
/// avoid repeating program-level validation.
pub fn run_shared_program_with_limits(
    program: Arc<NativeProgram>,
    entry: &str,
    args: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    let validated = ValidatedNativeProgram::try_from_shared(program)?;
    run_validated_program_with_limits(&validated, entry, args, limits)
}

/// Execute a previously validated immutable program without revalidating its
/// code and metadata. Caller-specific inputs, execution limits, counters, and
/// task state are still fresh and checked for every invocation.
pub fn run_validated_program_with_limits(
    program: &ValidatedNativeProgram,
    entry: &str,
    args: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    validate_input_values(args, limits)?;
    let function = program.program.functions.get(entry)
        .ok_or_else(|| NativeError::InvalidProgram(format!("unknown function '{entry}'")))?;
    let state = Arc::new(ExecutionState {
        limits,
        instructions: AtomicUsize::new(0),
        active_tasks: AtomicUsize::new(0),
    });
    run_function(&program.program, function, args, None, state, 0)
}

fn run_function(
    program: &Arc<NativeProgram>,
    function: &NativeFunction,
    args: &[NativeValue],
    cancellation: Option<Arc<AtomicBool>>,
    state: Arc<ExecutionState>,
    depth: usize,
) -> Result<NativeValue, NativeError> {
    if depth > state.limits.max_call_depth {
        return Err(NativeError::ResourceLimit(format!(
            "call depth limit exceeded (limit {})", state.limits.max_call_depth
        )));
    }
    if args.len() != function.params.len() {
        return Err(NativeError::InvalidProgram(format!(
            "function expects {} argument(s), got {}",
            function.params.len(),
            args.len()
        )));
    }
    let mut pc = 0usize;
    let mut stack = Vec::new();
    let mut locals = HashMap::new();
    let mut scopes: Vec<BTreeMap<String, NativeTask>> = Vec::new();
    for (name, value) in function.params.iter().zip(args.iter()) {
        if locals.len() >= state.limits.max_locals {
            return Err(NativeError::ResourceLimit("local binding limit exceeded".into()));
        }
        validate_value(value, state.limits)?;
        locals.insert(name.clone(), value.clone());
    }

    check_frame_value_budget(&locals, &stack, state.limits)?;
    while pc < function.code.len() {
        if stack.len() > state.limits.max_stack_values {
            return Err(NativeError::ResourceLimit("operand stack limit exceeded".into()));
        }
        if locals.len() > state.limits.max_locals {
            return Err(NativeError::ResourceLimit("local binding limit exceeded".into()));
        }
        let executed = state.instructions.fetch_add(1, Ordering::AcqRel);
        if executed >= state.limits.max_instructions {
            return Err(NativeError::ResourceLimit(format!(
                "instruction budget exceeded after {} instructions (limit {})", executed + 1, state.limits.max_instructions
            )));
        }
        if cancellation
            .as_ref()
            .is_some_and(|token| token.load(Ordering::Acquire))
        {
            return Err(NativeError::Cancelled("task cancelled".into()));
        }
        let instr = &function.code[pc];
        let check_values_after = matches!(
            instr,
            NativeInstr::PushInt(_)
                | NativeInstr::PushBool(_)
                | NativeInstr::PushUnit
                | NativeInstr::PushString(_)
                | NativeInstr::PushList(_)
                | NativeInstr::Index
                | NativeInstr::Len
                | NativeInstr::Append(_)
                | NativeInstr::MakeOk
                | NativeInstr::MakeErr
                | NativeInstr::Chr
                | NativeInstr::Unwrap
                | NativeInstr::Load(_)
                | NativeInstr::AddAssign(_)
                | NativeInstr::StoreIndex(_)
                | NativeInstr::Add
                | NativeInstr::Call { .. }
                | NativeInstr::Join { .. }
        );
        pc += 1;
        match instr {
            NativeInstr::ScopeStart => scopes.push(BTreeMap::new()),
            NativeInstr::ScopeEnd => {
                let mut tasks = scopes.pop().ok_or_else(|| {
                    NativeError::InvalidProgram("scope end without scope start".into())
                })?;
                let names = tasks.keys().cloned().collect::<Vec<_>>();
                for name in names {
                    let mut task = tasks.remove(&name).expect("task disappeared");
                    let result = task
                        .join
                        .take()
                        .expect("task already joined")
                        .join()
                        .map_err(|_| NativeError::Unsupported(format!("task '{name}' panicked")))?;
                    if let Err(error) = result {
                        if matches!(error, NativeError::Cancelled(_)) {
                            continue;
                        }
                        for sibling in tasks.values() {
                            sibling.cancel();
                        }
                        return Err(error);
                    }
                }
            }
            NativeInstr::Spawn { name, callee, argc } => {
                let scope = scopes.last_mut().ok_or_else(|| {
                    NativeError::InvalidProgram("spawn must occur inside a scope".into())
                })?;
                if scope.contains_key(name) {
                    return Err(NativeError::InvalidProgram(format!(
                        "duplicate task '{name}'"
                    )));
                }
                if stack.len() < *argc {
                    return Err(NativeError::InvalidProgram(
                        "spawn has fewer stack arguments than declared".into(),
                    ));
                }
                let start = stack.len() - *argc;
                let call_args = stack.split_off(start);
                let task_lease = reserve_task(&state)?;
                let child_program = Arc::clone(program);
                let child_state = Arc::clone(&state);
                let child_depth = depth + 1;
                let token = Arc::new(AtomicBool::new(false));
                let child_token = token.clone();
                let task_name = name.clone();
                let callee = callee.clone();
                let join = thread::Builder::new().name(format!("ardisa-{task_name}")).spawn(move || {
                    let _task_lease = task_lease;
                    let function = child_program.functions.get(&callee).ok_or_else(|| {
                        NativeError::InvalidProgram(format!("unknown function '{callee}'"))
                    })?;
                    run_function(&child_program, function, &call_args, Some(child_token), child_state, child_depth)
                }).map_err(|error| NativeError::ResourceLimit(format!("worker creation failed: {error}")))?;
                scope.insert(
                    name.clone(),
                    NativeTask {
                        cancel: token,
                        join: Some(join),
                    },
                );
            }
            NativeInstr::Join { name } => {
                let scope = scopes.last_mut().ok_or_else(|| {
                    NativeError::InvalidProgram("join must occur inside a scope".into())
                })?;
                let mut task = scope
                    .remove(name)
                    .ok_or_else(|| NativeError::InvalidProgram(format!("unknown task '{name}'")))?;
                let result = task
                    .join
                    .take()
                    .expect("task already joined")
                    .join()
                    .map_err(|_| NativeError::Unsupported(format!("task '{name}' panicked")))?;
                match result {
                    Ok(_) => stack.push(NativeValue::Unit),
                    Err(NativeError::Cancelled(_)) => stack.push(NativeValue::Unit),
                    Err(error) => {
                        for sibling in scope.values() {
                            sibling.cancel();
                        }
                        return Err(error);
                    }
                }
            }
            NativeInstr::Cancel { name } => {
                let scope = scopes.last_mut().ok_or_else(|| {
                    NativeError::InvalidProgram("cancel must occur inside a scope".into())
                })?;
                let task = scope
                    .get(name)
                    .ok_or_else(|| NativeError::InvalidProgram(format!("unknown task '{name}'")))?;
                task.cancel();
            }
            NativeInstr::Call { callee, argc } => {
                if stack.len() < *argc {
                    return Err(NativeError::InvalidProgram(
                        "call has fewer stack arguments than declared".into(),
                    ));
                }
                let start = stack.len() - *argc;
                let call_args = stack.split_off(start);
                let callee_fn = program.functions.get(callee).ok_or_else(|| {
                    NativeError::InvalidProgram(format!("unknown function '{callee}'"))
                })?;
                let value = run_function(program, callee_fn, &call_args, cancellation.clone(), Arc::clone(&state), depth + 1)?;
                stack.push(value);
            }
            NativeInstr::PushInt(value) => stack.push(NativeValue::Int(*value)),
            NativeInstr::PushBool(value) => stack.push(NativeValue::Bool(*value)),
            NativeInstr::PushUnit => stack.push(NativeValue::Unit),
            NativeInstr::PushString(value) => {
                if value.len() > state.limits.max_string_bytes {
                    return Err(NativeError::ResourceLimit("string exceeds byte limit".into()));
                }
                stack.push(NativeValue::String(value.clone()));
            }
            NativeInstr::PushList(len) => {
                if *len > state.limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("collection exceeds item limit".into()));
                }
                if stack.len() < *len {
                    return Err(NativeError::InvalidProgram(
                        "list has insufficient stack values".into(),
                    ));
                }
                let start = stack.len() - *len;
                let value = NativeValue::List(stack.drain(start..).collect());
                validate_value(&value, state.limits)?;
                stack.push(value);
            }
            NativeInstr::Index => {
                let index = pop_int(&mut stack)?;
                let collection = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("index from empty stack".into()))?;
                let index = usize::try_from(index)
                    .map_err(|_| NativeError::Type("negative index".into()))?;
                match collection {
                    NativeValue::List(values) => stack.push(
                        values
                            .get(index)
                            .cloned()
                            .ok_or_else(|| NativeError::Type("list index out of bounds".into()))?,
                    ),
                    NativeValue::String(value) => stack.push(NativeValue::Int(i64::from(
                        value.as_bytes().get(index).copied().ok_or_else(|| {
                            NativeError::Type("string index out of bounds".into())
                        })?,
                    ))),
                    _ => {
                        return Err(NativeError::Type(
                            "indexing requires a list or String".into(),
                        ));
                    }
                }
            }
            NativeInstr::Len => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("len from empty stack".into()))?;
                let length = match value {
                    NativeValue::String(value) => value.len(),
                    NativeValue::List(values) => values.len(),
                    _ => return Err(NativeError::Type("len requires String or List".into())),
                };
                stack.push(NativeValue::Int(length as i64));
            }
            NativeInstr::Append(name) => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("push value missing".into()))?;
                {
                    let Some(NativeValue::List(items)) = locals.get_mut(name) else {
                        return Err(NativeError::Type("push requires a List binding".into()));
                    };
                    if items.len() >= state.limits.max_collection_items {
                        return Err(NativeError::ResourceLimit("collection exceeds item limit".into()));
                    }
                    items.push(value);
                }
                validate_value(locals.get(name).expect("list binding retained"), state.limits)?;
                stack.push(NativeValue::Unit);
            }
            NativeInstr::Chr => {
                let value = pop_int(&mut stack)?;
                let byte = u8::try_from(value)
                    .map_err(|_| NativeError::Type("chr requires a byte in 0..=255".into()))?;
                stack.push(NativeValue::String(char::from(byte).to_string()));
            }
            NativeInstr::MakeOk => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("ok value missing".into()))?;
                let result = NativeValue::ResultOk(Box::new(value));
                validate_value(&result, state.limits)?;
                stack.push(result);
            }
            NativeInstr::MakeErr => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("err value missing".into()))?;
                let result = NativeValue::ResultErr(Box::new(value));
                validate_value(&result, state.limits)?;
                stack.push(result);
            }
            NativeInstr::Unwrap => {
                match stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("unwrap value missing".into()))?
                {
                    NativeValue::ResultOk(value) => stack.push(*value),
                    NativeValue::ResultErr(_) => {
                        return Err(NativeError::Type("unwrap on Err".into()));
                    }
                    _ => return Err(NativeError::Type("unwrap requires Result".into())),
                }
            }
            NativeInstr::Load(name) => {
                stack.push(locals.get(name).cloned().ok_or_else(|| {
                    NativeError::InvalidProgram(format!("unknown local '{name}'"))
                })?);
            }
            NativeInstr::Store(name) => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("store from empty stack".into()))?;
                validate_value(&value, state.limits)?;
                if !locals.contains_key(name) && locals.len() >= state.limits.max_locals {
                    return Err(NativeError::ResourceLimit("local binding limit exceeded".into()));
                }
                locals.insert(name.clone(), value);
            }
            NativeInstr::AddAssign(name) => {
                let right = stack.pop().ok_or_else(|| NativeError::InvalidProgram("add assignment value missing".into()))?;
                let left = locals.remove(name).ok_or_else(|| NativeError::InvalidProgram(format!("unknown local '{name}'")))?;
                let value = add_values(left, right, state.limits)?;
                if matches!(&value, NativeValue::String(s) if s.len() > state.limits.max_string_bytes) {
                    return Err(NativeError::ResourceLimit("string exceeds byte limit".into()));
                }
                validate_value(&value, state.limits)?;
                locals.insert(name.clone(), value);
            }
            NativeInstr::StoreIndex(name) => {
                let value = stack.pop().ok_or_else(|| {
                    NativeError::InvalidProgram("indexed store value missing".into())
                })?;
                let index = pop_int(&mut stack)?;
                let collection = stack.pop().ok_or_else(|| {
                    NativeError::InvalidProgram("indexed store collection missing".into())
                })?;
                let NativeValue::List(mut items) = collection else {
                    return Err(NativeError::Type(
                        "indexed assignment requires a list".into(),
                    ));
                };
                let index = usize::try_from(index)
                    .map_err(|_| NativeError::Type("negative list index".into()))?;
                *items
                    .get_mut(index)
                    .ok_or_else(|| NativeError::Type("list index out of bounds".into()))? = value;
                let updated = NativeValue::List(items);
                validate_value(&updated, state.limits)?;
                locals.insert(name.clone(), updated);
            }
            NativeInstr::Add => {
                let right = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let left = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let value = add_values(left, right, state.limits)?;
                validate_value(&value, state.limits)?;
                stack.push(value);
            }
            NativeInstr::Sub | NativeInstr::Mul | NativeInstr::Div | NativeInstr::Mod => {
                let right = pop_int(&mut stack)?;
                let left = pop_int(&mut stack)?;
                let value = match instr {
                    NativeInstr::Sub => left.checked_sub(right)
                        .ok_or_else(|| NativeError::Type("integer overflow in subtraction".into()))?,
                    NativeInstr::Mul => left.checked_mul(right)
                        .ok_or_else(|| NativeError::Type("integer overflow in multiplication".into()))?,
                    NativeInstr::Div => {
                        if right == 0 {
                            return Err(NativeError::Type("division by zero".into()));
                        }
                        left.checked_div(right)
                            .ok_or_else(|| NativeError::Type("integer overflow in division".into()))?
                    }
                    NativeInstr::Mod => {
                        if right == 0 {
                            return Err(NativeError::Type("modulo by zero".into()));
                        }
                        left.checked_rem(right)
                            .ok_or_else(|| NativeError::Type("integer overflow in modulo".into()))?
                    }
                    _ => unreachable!(),
                };
                stack.push(NativeValue::Int(value));
            }
            NativeInstr::Equal
            | NativeInstr::NotEqual
            | NativeInstr::Less
            | NativeInstr::LessEqual
            | NativeInstr::Greater
            | NativeInstr::GreaterEqual => {
                let right = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let left = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let result = match instr {
                    NativeInstr::Equal => left == right,
                    NativeInstr::NotEqual => left != right,
                    NativeInstr::Less => compare_ints(&left, &right, |a, b| a < b)?,
                    NativeInstr::LessEqual => compare_ints(&left, &right, |a, b| a <= b)?,
                    NativeInstr::Greater => compare_ints(&left, &right, |a, b| a > b)?,
                    NativeInstr::GreaterEqual => compare_ints(&left, &right, |a, b| a >= b)?,
                    _ => unreachable!(),
                };
                stack.push(NativeValue::Bool(result));
            }
            NativeInstr::JumpIfFalse(target) => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty condition stack".into()))?;
                if value != NativeValue::Bool(true) {
                    pc = *target;
                }
            }
            NativeInstr::Jump(target) => pc = *target,
            NativeInstr::Return => return Ok(stack.pop().unwrap_or(NativeValue::Unit)),
            NativeInstr::Pop => {
                stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("pop from empty stack".into()))?;
            }
        }
        if check_values_after {
            check_frame_value_budget(&locals, &stack, state.limits)?;
        }
    }
    Err(NativeError::InvalidProgram(
        "program terminated without return".into(),
    ))
}

fn compare_ints(
    left: &NativeValue,
    right: &NativeValue,
    predicate: impl FnOnce(i64, i64) -> bool,
) -> Result<bool, NativeError> {
    let (NativeValue::Int(left), NativeValue::Int(right)) = (left, right) else {
        return Err(NativeError::Type(
            "ordering operators require Int operands".into(),
        ));
    };
    Ok(predicate(*left, *right))
}

fn add_values(
    left: NativeValue,
    right: NativeValue,
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    match (left, right) {
        (NativeValue::Int(left), NativeValue::Int(right)) => Ok(NativeValue::Int(
            left.checked_add(right)
                .ok_or_else(|| NativeError::Type("integer overflow in addition".into()))?,
        )),
        (NativeValue::String(left), NativeValue::String(right)) => {
            // Check the byte limit before allocating the combined buffer. String::len()
            // reports UTF-8 bytes, matching the VM's configured resource limits.
            let combined_len = left.len().checked_add(right.len()).ok_or_else(|| {
                NativeError::ResourceLimit("string concatenation length overflow".into())
            })?;
            if combined_len > limits.max_string_bytes {
                return Err(NativeError::ResourceLimit(format!(
                    "string concatenation exceeds byte limit ({} > {})",
                    combined_len, limits.max_string_bytes
                )));
            }
            let mut combined = String::with_capacity(combined_len);
            combined.push_str(&left);
            combined.push_str(&right);
            Ok(NativeValue::String(combined))
        }
        _ => Err(NativeError::Type(
            "String + String or Int + Int required".into(),
        )),
    }
}


/// Hard limits for the serialized/in-memory executable representation.
/// These are intentionally separate from ExecutionLimits, which bound runtime state.
pub const MAX_PROGRAM_FUNCTIONS: usize = 16_384;
pub const MAX_FUNCTION_INSTRUCTIONS: usize = 250_000;
pub const MAX_PROGRAM_INSTRUCTIONS: usize = 1_000_000;
pub const MAX_PROGRAM_EMBEDDED_STRING_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Validate an executable before cloning it into the shared runtime state.
/// The byte budget covers function names, parameter names, and string operands;
/// it is not a claim about total heap usage or allocator overhead.
fn validate_native_program(program: &NativeProgram) -> Result<(), NativeError> {
    if program.functions.len() > MAX_PROGRAM_FUNCTIONS {
        return Err(NativeError::ResourceLimit(format!(
            "program function limit exceeded (limit {MAX_PROGRAM_FUNCTIONS})"
        )));
    }
    let mut total_instructions = 0usize;
    let mut embedded_bytes = 0usize;
    let mut add_bytes = |value: &str| -> Result<(), NativeError> {
        embedded_bytes = embedded_bytes.checked_add(value.len()).ok_or_else(|| {
            NativeError::ResourceLimit("program embedded string byte count overflow".into())
        })?;
        if embedded_bytes > MAX_PROGRAM_EMBEDDED_STRING_BYTES {
            return Err(NativeError::ResourceLimit(format!(
                "program embedded string byte limit exceeded (limit {MAX_PROGRAM_EMBEDDED_STRING_BYTES})"
            )));
        }
        Ok(())
    };
    for (name, function) in &program.functions {
        add_bytes(name)?;
        if function.code.len() > MAX_FUNCTION_INSTRUCTIONS {
            return Err(NativeError::ResourceLimit(format!(
                "function instruction limit exceeded (limit {MAX_FUNCTION_INSTRUCTIONS})"
            )));
        }
        total_instructions = total_instructions.checked_add(function.code.len()).ok_or_else(|| {
            NativeError::ResourceLimit("program instruction count overflow".into())
        })?;
        if total_instructions > MAX_PROGRAM_INSTRUCTIONS {
            return Err(NativeError::ResourceLimit(format!(
                "program instruction limit exceeded (limit {MAX_PROGRAM_INSTRUCTIONS})"
            )));
        }
        for param in &function.params { add_bytes(param)?; }
        for instr in &function.code {
            match instr {
                NativeInstr::PushString(s) | NativeInstr::Append(s) |
                NativeInstr::Load(s) | NativeInstr::Store(s) |
                NativeInstr::AddAssign(s) | NativeInstr::StoreIndex(s) |
                NativeInstr::Join { name: s } | NativeInstr::Cancel { name: s } => add_bytes(s)?,
                NativeInstr::Call { callee, .. } => add_bytes(callee)?,
                NativeInstr::Spawn { name, callee, .. } => { add_bytes(name)?; add_bytes(callee)?; }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Stable, dependency-free serialization for bootstrap artifacts.
///
/// The format is deliberately textual and line-oriented so an existing
/// Ardisa VM can consume a compiler artifact without invoking the Rust compiler.
pub const ARTIFACT_MAGIC: &str = "ARDISA-EXEC-V1";

pub fn encode_program(program: &NativeProgram) -> String {
    let mut out = String::from(ARTIFACT_MAGIC);
    out.push('\n');
    for (name, function) in &program.functions {
        out.push_str("FN|");
        out.push_str(&escape_artifact(name));
        out.push('|');
        out.push_str(&function.params.len().to_string());
        out.push('|');
        out.push_str(&function.params.iter().map(|p| escape_artifact(p)).collect::<Vec<_>>().join(","));
        out.push('\n');
        for instr in &function.code {
            out.push_str("I|");
            out.push_str(&encode_instr(instr));
            out.push('\n');
        }
        out.push_str("END\n");
    }
    out
}

pub fn decode_program(input: &str) -> Result<NativeProgram, NativeError> {
    if input.len() > MAX_ARTIFACT_BYTES {
        return Err(NativeError::ResourceLimit(format!(
            "executable artifact byte limit exceeded (limit {MAX_ARTIFACT_BYTES})"
        )));
    }
    let mut lines = input.lines();
    if lines.next() != Some(ARTIFACT_MAGIC) {
        return Err(NativeError::InvalidProgram("invalid Ardisa executable magic".into()));
    }
    let mut functions = BTreeMap::new();
    let mut current: Option<(String, Vec<String>, Vec<NativeInstr>)> = None;
    for line in lines {
        if let Some(rest) = line.strip_prefix("FN|") {
            if current.is_some() {
                return Err(NativeError::InvalidProgram("nested function in executable".into()));
            }
            let mut parts = rest.split('|');
            let name = unescape_artifact(parts.next().unwrap_or(""))?;
            let _count = parts.next().unwrap_or("0").parse::<usize>()
                .map_err(|_| NativeError::InvalidProgram("invalid parameter count".into()))?;
            let params = if let Some(raw) = parts.next() {
                if raw.is_empty() { Vec::new() } else {
                    raw.split(',').map(unescape_artifact).collect::<Result<Vec<_>, _>>()?
                }
            } else { Vec::new() };
            current = Some((name, params, Vec::new()));
        } else if line == "END" {
            let (name, params, code) = current.take()
                .ok_or_else(|| NativeError::InvalidProgram("function terminator without function".into()))?;
            functions.insert(name, NativeFunction { params, code });
        } else if let Some(rest) = line.strip_prefix("I|") {
            let (_, _, code) = current.as_mut()
                .ok_or_else(|| NativeError::InvalidProgram("instruction outside function".into()))?;
            code.push(decode_instr(rest)?);
        } else if !line.is_empty() {
            return Err(NativeError::InvalidProgram("unknown executable record".into()));
        }
    }
    if current.is_some() {
        return Err(NativeError::InvalidProgram("unterminated executable function".into()));
    }
    let program = NativeProgram { functions };
    validate_native_program(&program)?;
    Ok(program)
}

fn escape_artifact(value: &str) -> String {
    value.replace('\\', "\\\\").replace('|', r"\p").replace(',', r"\c").replace('\n', r"\n")
}

fn unescape_artifact(value: &str) -> Result<String, NativeError> {
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            out.push(match ch { 'p' => '|', 'c' => ',', 'n' => '\n', '\\' => '\\', other => other });
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped { return Err(NativeError::InvalidProgram("trailing artifact escape".into())); }
    Ok(out)
}

fn encode_instr(instr: &NativeInstr) -> String {
    match instr {
        NativeInstr::PushInt(v) => format!("PushInt:{v}"),
        NativeInstr::PushBool(v) => format!("PushBool:{v}"),
        NativeInstr::PushUnit => "PushUnit".into(),
        NativeInstr::PushString(v) => format!("PushString:{}", escape_artifact(v)),
        NativeInstr::PushList(v) => format!("PushList:{v}"),
        NativeInstr::Index => "Index".into(), NativeInstr::Len => "Len".into(),
        NativeInstr::Append(v) => format!("Append:{}", escape_artifact(v)),
        NativeInstr::MakeOk => "MakeOk".into(), NativeInstr::Chr => "Chr".into(),
        NativeInstr::MakeErr => "MakeErr".into(), NativeInstr::Unwrap => "Unwrap".into(),
        NativeInstr::Load(v) => format!("Load:{}", escape_artifact(v)),
        NativeInstr::Store(v) => format!("Store:{}", escape_artifact(v)),
        NativeInstr::AddAssign(v) => format!("AddAssign:{}", escape_artifact(v)),
        NativeInstr::StoreIndex(v) => format!("StoreIndex:{}", escape_artifact(v)),
        NativeInstr::Add => "Add".into(), NativeInstr::Sub => "Sub".into(),
        NativeInstr::Mul => "Mul".into(), NativeInstr::Div => "Div".into(),
        NativeInstr::Mod => "Mod".into(), NativeInstr::Equal => "Equal".into(),
        NativeInstr::NotEqual => "NotEqual".into(), NativeInstr::Less => "Less".into(),
        NativeInstr::LessEqual => "LessEqual".into(), NativeInstr::Greater => "Greater".into(),
        NativeInstr::GreaterEqual => "GreaterEqual".into(),
        NativeInstr::JumpIfFalse(v) => format!("JumpIfFalse:{v}"),
        NativeInstr::Jump(v) => format!("Jump:{v}"), NativeInstr::Return => "Return".into(),
        NativeInstr::Pop => "Pop".into(),
        NativeInstr::Call { callee, argc } => format!("Call:{},{}", escape_artifact(callee), argc),
        NativeInstr::ScopeStart => "ScopeStart".into(), NativeInstr::ScopeEnd => "ScopeEnd".into(),
        NativeInstr::Spawn { name, callee, argc } => format!("Spawn:{},{},{}", escape_artifact(name), escape_artifact(callee), argc),
        NativeInstr::Join { name } => format!("Join:{}", escape_artifact(name)),
        NativeInstr::Cancel { name } => format!("Cancel:{}", escape_artifact(name)),
    }
}

fn decode_instr(s: &str) -> Result<NativeInstr, NativeError> {
    let mut p=s.splitn(2, ':'); let op=p.next().unwrap_or(""); let arg=p.next().unwrap_or("");
    let bad=||NativeError::InvalidProgram(format!("invalid instruction '{s}'"));
    let int=|v:&str|v.parse::<usize>().map_err(|_|bad());
    Ok(match op {
        "PushInt"=>NativeInstr::PushInt(arg.parse().map_err(|_|bad())?),
        "PushBool"=>NativeInstr::PushBool(arg=="true"),
        "PushUnit"=>NativeInstr::PushUnit,"Index"=>NativeInstr::Index,"Len"=>NativeInstr::Len,
        "PushString"=>NativeInstr::PushString(unescape_artifact(arg)?),
        "PushList"=>NativeInstr::PushList(int(arg)?),"Append"=>NativeInstr::Append(unescape_artifact(arg)?),
        "MakeOk"=>NativeInstr::MakeOk,"Chr"=>NativeInstr::Chr,"MakeErr"=>NativeInstr::MakeErr,"Unwrap"=>NativeInstr::Unwrap,
        "Load"=>NativeInstr::Load(unescape_artifact(arg)?),"Store"=>NativeInstr::Store(unescape_artifact(arg)?),
        "AddAssign"=>NativeInstr::AddAssign(unescape_artifact(arg)?),
        "StoreIndex"=>NativeInstr::StoreIndex(unescape_artifact(arg)?),"Add"=>NativeInstr::Add,"Sub"=>NativeInstr::Sub,
        "Mul"=>NativeInstr::Mul,"Div"=>NativeInstr::Div,"Mod"=>NativeInstr::Mod,"Equal"=>NativeInstr::Equal,
        "NotEqual"=>NativeInstr::NotEqual,"Less"=>NativeInstr::Less,"LessEqual"=>NativeInstr::LessEqual,
        "Greater"=>NativeInstr::Greater,"GreaterEqual"=>NativeInstr::GreaterEqual,
        "JumpIfFalse"=>NativeInstr::JumpIfFalse(int(arg)?),"Jump"=>NativeInstr::Jump(int(arg)?),
        "Return"=>NativeInstr::Return,"Pop"=>NativeInstr::Pop,
        "Call"=>{let mut x=arg.split(','); NativeInstr::Call{callee:unescape_artifact(x.next().ok_or_else(bad)?)?,argc:int(x.next().ok_or_else(bad)?)?}},
        "ScopeStart"=>NativeInstr::ScopeStart,"ScopeEnd"=>NativeInstr::ScopeEnd,
        "Spawn"=>{let mut x=arg.split(','); NativeInstr::Spawn{name:unescape_artifact(x.next().ok_or_else(bad)?)?,callee:unescape_artifact(x.next().ok_or_else(bad)?)?,argc:int(x.next().ok_or_else(bad)?)?}},
        "Join"=>NativeInstr::Join{name:unescape_artifact(arg)?},"Cancel"=>NativeInstr::Cancel{name:unescape_artifact(arg)?},
        _=>return Err(bad()),
    })
}

fn pop_int(stack: &mut Vec<NativeValue>) -> Result<i64, NativeError> {
    match stack.pop() {
        Some(NativeValue::Int(value)) => Ok(value),
        _ => Err(NativeError::Type("expected Int value".into())),
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;

    #[test]
    fn executable_artifact_round_trips_deterministically() {
        let module = crate::parse(
            r#"module artifact
fn main(a: String) -> String
  a + "!"
"#,
        ).unwrap();
        crate::sema::check(&module).unwrap();
        crate::ownership::infer(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let encoded = encode_program(&program);
        assert!(encoded.starts_with(ARTIFACT_MAGIC));
        let decoded = decode_program(&encoded).unwrap();
        assert_eq!(decoded, program);
        assert_eq!(encode_program(&decoded), encoded);
        assert_eq!(
            run_program(&decoded, "main", &[NativeValue::String("Ardisa".into())]).unwrap(),
            NativeValue::String("Ardisa!".into())
        );
    }

    #[test]
    fn malformed_executable_is_rejected() {
        assert!(matches!(
            decode_program("ARDISA-EXEC-V1\nI|Return\n"),
            Err(NativeError::InvalidProgram(_))
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_type_coverage_includes_aggregate_parameters() {
        assert!(is_native_type(&TypeKind::Int));
        assert!(is_native_type(&TypeKind::List(Box::new(crate::ast::Type {
            id: crate::NodeId(1),
            span: crate::source::Span::new(0, 0),
            kind: TypeKind::Int,
        }))));
        assert!(is_native_type(&TypeKind::Result(
            Box::new(crate::ast::Type {
                id: crate::NodeId(2),
                span: crate::source::Span::new(0, 0),
                kind: TypeKind::Int,
            }),
            Box::new(crate::ast::Type {
                id: crate::NodeId(3),
                span: crate::source::Span::new(0, 0),
                kind: TypeKind::String,
            }),
        )));
        assert!(!is_native_type(&TypeKind::Named("Custom".into())));
    }

    #[test]
    fn compiles_and_runs_aggregate_values_without_rust() {
        let module = crate::parse(
            "module x\nfn main(values: List<Int>) -> Int\n  len(values)\n",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(
            &program,
            "main",
            &[NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)])],
        )
        .unwrap();
        assert_eq!(result, NativeValue::Int(2));
    }

    #[test]
    fn compiles_and_runs_result_values_without_rust() {
        let module = crate::parse(
            "module x\nfn main(value: Result<Int, String>) -> Int\n  unwrap(value)\n",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(
            &program,
            "main",
            &[NativeValue::ResultOk(Box::new(NativeValue::Int(9)))],
        )
        .unwrap();
        assert_eq!(result, NativeValue::Int(9));
    }

    #[test]
    fn compiles_and_runs_arithmetic_without_rust() {
        let module = crate::parse(
            "module x
fn main(a: Int, b: Int) -> Int
  a + b * 2
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let code = compile(&ir).unwrap();
        let result = run(
            &code,
            &[
                (String::from("a"), NativeValue::Int(3)),
                (String::from("b"), NativeValue::Int(4)),
            ],
        )
        .unwrap();
        assert_eq!(result, NativeValue::Int(11));
    }

    #[test]
    fn compiles_and_runs_conditionals_without_rust() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  if a == 0
    return 1
  else
    return 2
",
        )
        .unwrap();
        let ir = crate::ir::lower(&module);
        let code = compile(&ir).unwrap();
        let result = run(&code, &[(String::from("a"), NativeValue::Int(0))]).unwrap();
        assert_eq!(result, NativeValue::Int(1));
    }

    #[test]
    fn compiles_and_runs_statement_style_if() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  let value = 0
  if a == 0
    set value = 1
  else
    set value = 2
  value
",
        )
        .unwrap();
        let ir = crate::ir::lower(&module);
        let code = compile(&ir).unwrap();
        let result = run(&code, &[(String::from("a"), NativeValue::Int(0))]).unwrap();
        assert_eq!(result, NativeValue::Int(1));
    }

    #[test]
    fn compiles_and_runs_mutable_assignments() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  let value = a
  set value = value + 2
  value
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[NativeValue::Int(5)]).unwrap();
        assert_eq!(result, NativeValue::Int(7));
    }

    #[test]
    fn compiles_and_runs_while_loops() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  while a == 0
    return 7
  return 9
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[NativeValue::Int(0)]).unwrap();
        assert_eq!(result, NativeValue::Int(7));
    }

    #[test]
    fn child_runtime_error_cancels_and_joins_running_sibling() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("fail".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushBool(true),
                        NativeInstr::PushInt(1),
                        NativeInstr::Add,
                        NativeInstr::Return,
                    ],
                }),
                ("worker".into(), NativeFunction {
                    params: vec![],
                    code: vec![NativeInstr::Jump(0)],
                }),
                ("main".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::ScopeStart,
                        NativeInstr::Spawn { name: "a_fail".into(), callee: "fail".into(), argc: 0 },
                        NativeInstr::Spawn { name: "z_worker".into(), callee: "worker".into(), argc: 0 },
                        NativeInstr::ScopeEnd,
                        NativeInstr::PushInt(1),
                        NativeInstr::Return,
                    ],
                }),
            ]),
        };
        let error = run_program_with_limits(
            &program,
            "main",
            &[],
            ExecutionLimits { max_instructions: usize::MAX, ..ExecutionLimits::default() },
        ).unwrap_err();
        assert!(matches!(error, NativeError::Type(_)), "unexpected result: {error:?}");
    }

    #[test]
    fn cancelled_child_does_not_mask_sibling_runtime_error() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("fail".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushBool(true),
                        NativeInstr::PushInt(1),
                        NativeInstr::Add,
                        NativeInstr::Return,
                    ],
                }),
                ("worker".into(), NativeFunction {
                    params: vec![],
                    code: vec![NativeInstr::Jump(0)],
                }),
                ("main".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::ScopeStart,
                        NativeInstr::Spawn { name: "a_cancel".into(), callee: "worker".into(), argc: 0 },
                        NativeInstr::Spawn { name: "z_fail".into(), callee: "fail".into(), argc: 0 },
                        NativeInstr::Cancel { name: "a_cancel".into() },
                        NativeInstr::ScopeEnd,
                        NativeInstr::PushInt(1),
                        NativeInstr::Return,
                    ],
                }),
            ]),
        };
        let error = run_program_with_limits(
            &program,
            "main",
            &[],
            ExecutionLimits { max_instructions: usize::MAX, ..ExecutionLimits::default() },
        ).unwrap_err();
        assert!(matches!(error, NativeError::Type(_)), "unexpected result: {error:?}");
    }

    #[test]
    fn string_concatenation_checks_limit_before_allocating() {
        let limits = ExecutionLimits {
            max_string_bytes: 5,
            ..ExecutionLimits::default()
        };
        assert_eq!(
            add_values(
                NativeValue::String("ab".into()),
                NativeValue::String("cde".into()),
                limits,
            ).unwrap(),
            NativeValue::String("abcde".into())
        );
        assert!(matches!(
            add_values(
                NativeValue::String("abc".into()),
                NativeValue::String("def".into()),
                limits,
            ),
            Err(NativeError::ResourceLimit(message)) if message.contains("concatenation")
        ));
    }

    #[test]
    fn string_concatenation_limit_uses_utf8_byte_length() {
        let limits = ExecutionLimits {
            max_string_bytes: 4,
            ..ExecutionLimits::default()
        };
        assert_eq!(
            add_values(
                NativeValue::String("é".into()),
                NativeValue::String("ab".into()),
                limits,
            ).unwrap(),
            NativeValue::String("éab".into())
        );
        assert!(matches!(
            add_values(
                NativeValue::String("é".into()),
                NativeValue::String("abc".into()),
                limits,
            ),
            Err(NativeError::ResourceLimit(_))
        ));
    }

    #[test]
    fn compiles_and_runs_string_concatenation() {
        let module = crate::parse(
            r#"module x
fn main() -> String
  "hello " + "ardisa"
"#,
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[]).unwrap();
        assert_eq!(result, NativeValue::String("hello ardisa".into()));
    }

    #[test]
    fn compiles_and_runs_function_calls() {
        let module = crate::parse(
            "module x
fn double(a: Int) -> Int
  a * 2
fn main(a: Int) -> Int
  double(a) + 1
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[NativeValue::Int(3)]).unwrap();
        assert_eq!(result, NativeValue::Int(7));
    }

    #[test]
    fn executes_structured_scope_with_joined_child() {
        let module = crate::parse(
            "module x
fn worker(a: Int) -> Int
  a + 1
fn main() -> Int
  scope
    spawn worker_task = worker(4)
    join worker_task
  7
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        crate::concurrency::analyze(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::Int(7)
        );
    }

    #[test]
    fn executes_structured_scope_with_cancelled_child() {
        let module = crate::parse(
            "module x
fn worker() -> Int
  while true
    1
  return 0
fn main() -> Int
  scope
    spawn worker_task = worker()
    cancel worker_task
  9
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        crate::concurrency::analyze(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::Int(9)
        );
    }

    #[test]
    fn cancellation_is_a_normal_scope_terminal_state() {
        let module = crate::parse(
            "module x
fn worker() -> Int
  while true
    1
  return 0
fn main() -> Int
  scope
    spawn worker_task = worker()
    cancel worker_task
  9
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        crate::concurrency::analyze(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::Int(9)
        );
    }

    #[test]
    fn compiles_and_runs_recursive_calls() {
        let module = crate::parse(
            "module x
fn fact(n: Int) -> Int
  if n == 0
    return 1
  else
    return n * fact(n - 1)
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "fact", &[NativeValue::Int(5)]).unwrap();
        assert_eq!(result, NativeValue::Int(120));
    }

    #[test]
    fn instruction_budget_stops_infinite_loops() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![NativeInstr::Jump(0)] },
            )]),
        };
        let error = run_program_with_limits(&program, "main", &[], ExecutionLimits {
            max_instructions: 12,
            ..ExecutionLimits::default()
        }).unwrap_err();
        assert!(matches!(error, NativeError::ResourceLimit(message) if message.contains("instruction budget")));
    }

    #[test]
    fn call_depth_limit_stops_unbounded_recursion() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![
                    NativeInstr::Call { callee: "main".into(), argc: 0 },
                    NativeInstr::Return,
                ] },
            )]),
        };
        let error = run_program_with_limits(&program, "main", &[], ExecutionLimits {
            max_instructions: 100,
            max_call_depth: 3,
            ..ExecutionLimits::default()
        }).unwrap_err();
        assert!(matches!(error, NativeError::ResourceLimit(message) if message.contains("call depth")));
    }

    #[test]
    fn input_collection_and_string_limits_fail_closed() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec!["value".into()], code: vec![
                    NativeInstr::Load("value".into()),
                    NativeInstr::Return,
                ] },
            )]),
        };
        let limits = ExecutionLimits {
            max_string_bytes: 3,
            max_collection_items: 1,
            ..ExecutionLimits::default()
        };
        assert!(matches!(
            run_program_with_limits(&program, "main", &[NativeValue::String("long".into())], limits),
            Err(NativeError::ResourceLimit(_))
        ));
        assert!(matches!(
            run_program_with_limits(&program, "main", &[NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)])], limits),
            Err(NativeError::ResourceLimit(_))
        ));
    }

    #[test]
    fn task_limit_stops_excess_workers_and_cleans_up() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("worker".into(), NativeFunction { params: vec![], code: vec![NativeInstr::Jump(0)] }),
                ("main".into(), NativeFunction { params: vec![], code: vec![
                    NativeInstr::ScopeStart,
                    NativeInstr::Spawn { name: "one".into(), callee: "worker".into(), argc: 0 },
                    NativeInstr::Spawn { name: "two".into(), callee: "worker".into(), argc: 0 },
                    NativeInstr::ScopeEnd,
                    NativeInstr::PushInt(1),
                    NativeInstr::Return,
                ] }),
            ]),
        };
        let error = run_program_with_limits(&program, "main", &[], ExecutionLimits {
            max_instructions: usize::MAX,
            max_tasks: 1,
            ..ExecutionLimits::default()
        }).unwrap_err();
        assert!(matches!(error, NativeError::ResourceLimit(message) if message.contains("active task limit")));
    }

    #[test]
    fn self_appending_string_assignment_uses_in_place_add_assign() {
        let module = crate::parse(
            "module x\nfn main() -> String\n  let output = \"\"\n  set output = output + \"hello\"\n  output\n",
        ).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let main = &program.functions["main"];
        assert!(main.code.iter().any(|instr| matches!(instr, NativeInstr::AddAssign(name) if name == "output")));
        assert_eq!(run_program(&program, "main", &[]).unwrap(), NativeValue::String("hello".into()));
        let decoded = decode_program(&encode_program(&program)).unwrap();
        assert_eq!(decoded, program);
    }

    #[test]
    fn nested_value_depth_and_node_budgets_reject_adversarial_inputs() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec!["value".into()], code: vec![
                    NativeInstr::Load("value".into()),
                    NativeInstr::Return,
                ] },
            )]),
        };
        let depth_limited = ExecutionLimits {
            max_value_depth: 2,
            ..ExecutionLimits::default()
        };
        let nested = NativeValue::List(vec![NativeValue::List(vec![
            NativeValue::List(vec![NativeValue::Int(1)])
        ])]);
        assert!(matches!(
            run_program_with_limits(&program, "main", &[nested], depth_limited),
            Err(NativeError::ResourceLimit(message)) if message.contains("nesting depth")
        ));

        let node_limited = ExecutionLimits {
            max_value_nodes: 2,
            ..ExecutionLimits::default()
        };
        let broad = NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)]);
        assert!(matches!(
            run_program_with_limits(&program, "main", &[broad], node_limited),
            Err(NativeError::ResourceLimit(message)) if message.contains("node limit")
        ));
    }

    #[test]
    fn vm_frame_stack_and_local_limits_fail_closed() {
        let stack_program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![
                    NativeInstr::PushInt(1),
                    NativeInstr::PushInt(2),
                    NativeInstr::PushInt(3),
                    NativeInstr::Return,
                ] },
            )]),
        };
        assert!(matches!(
            run_program_with_limits(&stack_program, "main", &[], ExecutionLimits {
                max_stack_values: 1,
                ..ExecutionLimits::default()
            }),
            Err(NativeError::ResourceLimit(message)) if message.contains("operand stack")
        ));

        let local_program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![
                    NativeInstr::PushInt(1), NativeInstr::Store("a".into()),
                    NativeInstr::PushInt(2), NativeInstr::Store("b".into()),
                    NativeInstr::Return,
                ] },
            )]),
        };
        assert!(matches!(
            run_program_with_limits(&local_program, "main", &[], ExecutionLimits {
                max_locals: 1,
                ..ExecutionLimits::default()
            }),
            Err(NativeError::ResourceLimit(message)) if message.contains("local binding")
        ));
    }

    #[test]
    fn integer_overflow_returns_errors_instead_of_panicking() {
        let cases = [
            (NativeInstr::Add, i64::MAX, 1, "addition"),
            (NativeInstr::Sub, i64::MIN, 1, "subtraction"),
            (NativeInstr::Mul, i64::MAX, 2, "multiplication"),
            (NativeInstr::Div, i64::MIN, -1, "division"),
            (NativeInstr::Mod, i64::MIN, -1, "modulo"),
        ];
        for (operation, left, right, label) in cases {
            let program = NativeProgram {
                functions: BTreeMap::from([(
                    "main".into(),
                    NativeFunction {
                        params: vec![],
                        code: vec![
                            NativeInstr::PushInt(left),
                            NativeInstr::PushInt(right),
                            operation,
                            NativeInstr::Return,
                        ],
                    },
                )]),
            };
            let result = run_program(&program, "main", &[]);
            assert!(
                matches!(&result, Err(NativeError::Type(message)) if message.contains("overflow")),
                "expected explicit {label} overflow error, got {result:?}"
            );
        }
    }

    #[test]
    fn self_appending_integer_assignment_checks_overflow() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushInt(i64::MAX),
                        NativeInstr::Store("value".into()),
                        NativeInstr::PushInt(1),
                        NativeInstr::AddAssign("value".into()),
                        NativeInstr::Load("value".into()),
                        NativeInstr::Return,
                    ],
                },
            )]),
        };
        assert!(matches!(
            run_program(&program, "main", &[]),
            Err(NativeError::Type(message)) if message.contains("overflow")
        ));
    }

    #[test]
    fn shared_program_execution_reuses_immutable_program_and_returns_result() {
        let program = Arc::new(NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![NativeInstr::PushInt(7), NativeInstr::PushInt(8), NativeInstr::Add, NativeInstr::Return],
                },
            )]),
        });
        assert_eq!(
            run_shared_program_with_limits(
                Arc::clone(&program), "main", &[], ExecutionLimits::default()
            ),
            Ok(NativeValue::Int(15))
        );
        assert_eq!(
            run_shared_program_with_limits(
                program, "main", &[], ExecutionLimits::default()
            ),
            Ok(NativeValue::Int(15))
        );
    }

    #[test]
    fn validated_program_rejects_invalid_program_at_construction() {
        let invalid = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushString("x".repeat(MAX_PROGRAM_EMBEDDED_STRING_BYTES + 1)),
                        NativeInstr::Return,
                    ],
                },
            )]),
        };
        assert!(matches!(
            ValidatedNativeProgram::try_new(invalid),
            Err(NativeError::ResourceLimit(message)) if message.contains("embedded string byte")
        ));
    }

    #[test]
    fn validated_program_reuses_code_but_resets_budgets_and_checks_each_call() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec!["input".into()],
                    code: vec![NativeInstr::Load("input".into()), NativeInstr::Return],
                },
            )]),
        };
        let validated = ValidatedNativeProgram::try_new(program).unwrap();
        let limits = ExecutionLimits {
            max_instructions: 2,
            max_string_bytes: 4,
            ..ExecutionLimits::default()
        };

        assert_eq!(
            validated.run_with_limits("main", &[NativeValue::String("ok".into())], limits),
            Ok(NativeValue::String("ok".into()))
        );
        // A fresh execution budget is created for each call.
        assert_eq!(
            validated.run_with_limits("main", &[NativeValue::String("yes".into())], limits),
            Ok(NativeValue::String("yes".into()))
        );
        // Caller-specific arguments are still validated even though the program is trusted.
        assert!(matches!(
            validated.run_with_limits("main", &[NativeValue::String("too-long".into())], limits),
            Err(NativeError::ResourceLimit(message)) if message.contains("string byte")
        ));
    }

    #[test]
    fn validated_program_rejects_unknown_entry_per_invocation() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![NativeInstr::PushUnit, NativeInstr::Return] },
            )]),
        };
        let validated = ValidatedNativeProgram::try_new(program).unwrap();
        assert!(matches!(
            validated.run_with_limits("missing", &[], ExecutionLimits::default()),
            Err(NativeError::InvalidProgram(message)) if message.contains("unknown function")
        ));
    }

    #[test]
    fn native_program_rejects_oversized_instruction_payload_before_execution() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushString("x".repeat(MAX_PROGRAM_EMBEDDED_STRING_BYTES + 1)),
                        NativeInstr::Return,
                    ],
                },
            )]),
        };
        assert!(matches!(
            run_program(&program, "main", &[]),
            Err(NativeError::ResourceLimit(message)) if message.contains("embedded string byte")
        ));
    }

    #[test]
    fn decoder_rejects_oversized_artifact_before_parsing_records() {
        let input = format!("{}\\n{}", ARTIFACT_MAGIC, "x".repeat(MAX_ARTIFACT_BYTES));
        assert!(matches!(
            decode_program(&input),
            Err(NativeError::ResourceLimit(message)) if message.contains("artifact byte")
        ));
    }

    #[test]
    fn native_program_rejects_excessive_function_count() {
        let functions = (0..=MAX_PROGRAM_FUNCTIONS)
            .map(|i| (format!("f{i}"), NativeFunction { params: vec![], code: vec![] }))
            .collect();
        let program = NativeProgram { functions };
        assert!(matches!(
            run_program(&program, "f0", &[]),
            Err(NativeError::ResourceLimit(message)) if message.contains("function limit")
        ));
    }


    #[test]
    fn borrowed_dispatch_preserves_string_and_nested_task_semantics() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("worker".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushString("child payload".into()),
                        NativeInstr::Return,
                    ],
                }),
                ("main".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::ScopeStart,
                        NativeInstr::PushString("local payload".into()),
                        NativeInstr::Store("message".into()),
                        NativeInstr::Load("message".into()),
                        NativeInstr::Pop,
                        NativeInstr::Spawn {
                            name: "child".into(),
                            callee: "worker".into(),
                            argc: 0,
                        },
                        NativeInstr::Join { name: "child".into() },
                        NativeInstr::ScopeEnd,
                        NativeInstr::PushString("dispatch-ok".into()),
                        NativeInstr::Return,
                    ],
                }),
            ]),
        };
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::String("dispatch-ok".into())
        );
    }

    #[test]
    fn aggregate_frame_value_budget_bounds_combined_locals() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec!["left".into(), "right".into()],
                    code: vec![NativeInstr::PushUnit, NativeInstr::Return],
                },
            )]),
        };
        let list = || NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)]);
        let limits = ExecutionLimits {
            max_value_nodes: 3,
            max_frame_value_nodes: 5,
            ..ExecutionLimits::default()
        };
        assert!(matches!(
            run_program_with_limits(&program, "main", &[list(), list()], limits),
            Err(NativeError::ResourceLimit(message)) if message.contains("aggregate frame value node")
        ));
    }

    #[test]
    fn aggregate_frame_string_budget_bounds_combined_strings() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec!["left".into(), "right".into()],
                    code: vec![NativeInstr::PushUnit, NativeInstr::Return],
                },
            )]),
        };
        let limits = ExecutionLimits {
            max_string_bytes: 8,
            max_frame_string_bytes: 10,
            ..ExecutionLimits::default()
        };
        assert!(matches!(
            run_program_with_limits(
                &program, "main",
                &[NativeValue::String("123456".into()), NativeValue::String("abcdef".into())],
                limits
            ),
            Err(NativeError::ResourceLimit(message)) if message.contains("aggregate frame string byte")
        ));
    }
}
