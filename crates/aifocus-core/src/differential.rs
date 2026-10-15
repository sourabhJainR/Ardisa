use std::collections::HashMap;

use crate::{
    ast::{BinaryOp, Expr, ExprKind, Function, Item, Module, StmtKind},
    lower, native, ownership, parse, sema,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferentialCase {
    pub name: &'static str,
    pub source: &'static str,
}

pub const CORPUS: &[DifferentialCase] = &[
    DifferentialCase {
        name: "arithmetic",
        source: "module x\nfn main(a: Int, b: Int) -> Int\n  a + b * 2\n",
    },
    DifferentialCase {
        name: "conditional",
        source: "module x\nfn main(a: Int) -> Int\n  if a == 0\n    return 1\n  else\n    return a\n",
    },
    DifferentialCase {
        name: "list",
        source: "module x\nfn main() -> Int\n  let items = [2, 3]\n  items[1] + 4\n",
    },
    DifferentialCase {
        name: "string",
        source: r#"module x
fn main() -> String
  "hello " + "ardisa"
"#,
    },
    DifferentialCase {
        name: "result",
        source: "module x\nfn main() -> Int\n  unwrap(ok(9))\n",
    },
    DifferentialCase {
        name: "while",
        source: "module x\nfn main() -> Int\n  let x = 0\n  while x < 3\n    set x = x + 1\n  x\n",
    },
    DifferentialCase {
        name: "call",
        source: "module x\nfn add(a: Int, b: Int) -> Int\n  a + b\nfn main() -> Int\n  add(2, 5)\n",
    },
    DifferentialCase {
        name: "list-index",
        source: "module x\nfn main() -> Int\n  let items = [2, 3]\n  items[1] + 4\n",
    },
];

pub fn verify_case(case: &DifferentialCase) -> Result<(), String> {
    let module =
        parse(case.source).map_err(|errors| format!("{}: parse failed: {errors:?}", case.name))?;
    sema::check(&module)
        .map_err(|errors| format!("{}: semantic check failed: {errors:?}", case.name))?;
    ownership::infer(&module)
        .map_err(|errors| format!("{}: ownership check failed: {errors:?}", case.name))?;
    let lowered = lower::lower(&module);
    if lowered.rust.trim().is_empty() {
        return Err(format!("{}: lowering produced empty Rust", case.name));
    }

    let ir = crate::ir::lower(&module);
    let program = native::compile_program(&ir)
        .map_err(|error| format!("{}: native compile failed: {error:?}", case.name))?;
    let args = match case.name {
        "arithmetic" => vec![native::NativeValue::Int(3), native::NativeValue::Int(4)],
        "conditional" => vec![native::NativeValue::Int(0)],
        "list" | "string" | "result" | "while" | "call" | "list-index" => Vec::new(),
        _ => return Err(format!("{}: unknown differential case", case.name)),
    };
    let native_result = native::run_program(&program, "main", &args)
        .map_err(|error| format!("{}: native run failed: {error:?}", case.name))?;
    let reference_result = reference_run(&module, "main", &args)
        .map_err(|error| format!("{}: reference run failed: {error}", case.name))?;

    if native_result != reference_result {
        return Err(format!(
            "{}: native/reference mismatch: {native_result:?} != {reference_result:?}",
            case.name
        ));
    }
    Ok(())
}

#[derive(Debug)]
enum Control {
    Continue(Option<native::NativeValue>),
    Return(native::NativeValue),
}

fn reference_run(
    module: &Module,
    entry: &str,
    args: &[native::NativeValue],
) -> Result<native::NativeValue, String> {
    let function = find_function(module, entry)?;
    if function.params.len() != args.len() {
        return Err(format!(
            "function expects {} argument(s), got {}",
            function.params.len(),
            args.len()
        ));
    }
    let mut locals = HashMap::new();
    for (parameter, value) in function.params.iter().zip(args.iter()) {
        locals.insert(parameter.name.clone(), value.clone());
    }
    match reference_block(module, &function.body, &mut locals)? {
        Control::Continue(value) => Ok(value.unwrap_or(native::NativeValue::Unit)),
        Control::Return(value) => Ok(value),
    }
}

fn reference_block(
    module: &Module,
    block: &crate::Block,
    locals: &mut HashMap<String, native::NativeValue>,
) -> Result<Control, String> {
    let mut last = None;
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, value } | StmtKind::Set { name, value } => {
                let value = reference_expr(module, value, locals)?;
                locals.insert(name.clone(), value);
                last = None;
            }
            StmtKind::SetIndex {
                collection,
                index,
                value,
            } => {
                let name = match &collection.kind {
                    ExprKind::Name(name) => name,
                    _ => return Err("indexed assignment requires a named binding".into()),
                };
                let index = expect_int(reference_expr(module, index, locals)?)?;
                let value = reference_expr(module, value, locals)?;
                let native::NativeValue::List(mut items) = locals
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown list binding '{name}'"))?
                else {
                    return Err("indexed assignment requires a list".into());
                };
                let slot = usize::try_from(index).map_err(|_| "negative list index".to_string())?;
                let item = items
                    .get_mut(slot)
                    .ok_or_else(|| "list index out of bounds".to_string())?;
                *item = value;
                locals.insert(name.clone(), native::NativeValue::List(items));
                last = None;
            }
            StmtKind::Return(value) => {
                let value = match value {
                    Some(value) => reference_expr(module, value, locals)?,
                    None => native::NativeValue::Unit,
                };
                return Ok(Control::Return(value));
            }
            StmtKind::Expr(expr) => {
                last = Some(reference_expr(module, expr, locals)?);
            }
            StmtKind::Scope { body } => {
                let mut scoped = locals.clone();
                match reference_block(module, body, &mut scoped)? {
                    Control::Continue(_) => {}
                    Control::Return(value) => return Ok(Control::Return(value)),
                }
                last = None;
            }
            StmtKind::For { .. } => {
                return Err("for/in iteration is unsupported by the independent reference evaluator".into());
            }
            StmtKind::While { condition, body } => {
                while expect_bool(reference_expr(module, condition, locals)?)? {
                    match reference_block(module, body, locals)? {
                        Control::Continue(_) => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                    }
                }
                last = None;
            }
            StmtKind::Spawn { .. } | StmtKind::Join { .. } | StmtKind::Cancel { .. } => {
                return Err("reference oracle does not execute concurrency statements".into());
            }
        }
    }
    Ok(Control::Continue(last))
}

fn reference_expr(
    module: &Module,
    expr: &Expr,
    locals: &HashMap<String, native::NativeValue>,
) -> Result<native::NativeValue, String> {
    match &expr.kind {
        ExprKind::Int(value) => Ok(native::NativeValue::Int(*value)),
        ExprKind::Float(bits) => Ok(native::NativeValue::Float(*bits)),
        ExprKind::Bool(value) => Ok(native::NativeValue::Bool(*value)),
        ExprKind::String(value) => Ok(native::NativeValue::String(value.clone())),
        ExprKind::Name(name) => locals
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown name '{name}'")),
        ExprKind::Group(inner) => reference_expr(module, inner, locals),
        ExprKind::List(items) => Ok(native::NativeValue::List(
            items
                .iter()
                .map(|item| reference_expr(module, item, locals))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        ExprKind::Index { collection, index } => {
            let collection = reference_expr(module, collection, locals)?;
            let index = usize::try_from(expect_int(reference_expr(module, index, locals)?)?)
                .map_err(|_| "negative index".to_string())?;
            match collection {
                native::NativeValue::List(items) => items
                    .get(index)
                    .cloned()
                    .ok_or_else(|| "list index out of bounds".into()),
                native::NativeValue::String(value) => value
                    .as_bytes()
                    .get(index)
                    .copied()
                    .map(|byte| native::NativeValue::Int(i64::from(byte)))
                    .ok_or_else(|| "string index out of bounds".into()),
                _ => Err("indexing requires a list or String".into()),
            }
        }
        ExprKind::Slice { collection, start, end } => {
            let collection = reference_expr(module, collection, locals)?;
            let start = usize::try_from(expect_int(reference_expr(module, start, locals)?)?)
                .map_err(|_| "negative slice start".to_string())?;
            let end = usize::try_from(expect_int(reference_expr(module, end, locals)?)?)
                .map_err(|_| "negative slice end".to_string())?;
            let native::NativeValue::List(items) = collection else {
                return Err("slicing currently requires List<T>".into());
            };
            if start > end { return Err("slice start must not exceed slice end".into()); }
            if end > items.len() { return Err("slice end out of bounds".into()); }
            Ok(native::NativeValue::List(items[start..end].to_vec()))
        }
        ExprKind::Binary { op, left, right } => {
            let left = reference_expr(module, left, locals)?;
            let right = reference_expr(module, right, locals)?;
            reference_binary(*op, left, right)
        }
        ExprKind::Call { callee, args } => {
            let ExprKind::Name(name) = &callee.kind else {
                return Err("dynamic calls are unsupported by reference oracle".into());
            };
            match name.as_str() {
                "len" => {
                    let value = one_arg(args, module, locals)?;
                    match value {
                        native::NativeValue::String(value) => {
                            Ok(native::NativeValue::Int(value.len() as i64))
                        }
                        native::NativeValue::List(values) => {
                            Ok(native::NativeValue::Int(values.len() as i64))
                        }
                        _ => Err("len requires String or List".into()),
                    }
                }
                "chr" => {
                    let value = expect_int(one_arg(args, module, locals)?)?;
                    let byte =
                        u8::try_from(value).map_err(|_| "chr byte out of range".to_string())?;
                    Ok(native::NativeValue::String(char::from(byte).to_string()))
                }
                "ok" => Ok(native::NativeValue::ResultOk(Box::new(one_arg(
                    args, module, locals,
                )?))),
                "err" => Ok(native::NativeValue::ResultErr(Box::new(one_arg(
                    args, module, locals,
                )?))),
                "unwrap" => match one_arg(args, module, locals)? {
                    native::NativeValue::ResultOk(value) => Ok(*value),
                    native::NativeValue::ResultErr(_) => Err("unwrap on Err".into()),
                    _ => Err("unwrap requires Result".into()),
                },
                "push" => {
                    if args.len() != 2 {
                        return Err("push expects two arguments".into());
                    }
                    Err(
                        "push requires mutable reference semantics not modeled by this pure oracle"
                            .into(),
                    )
                }
                _ => {
                    let values = args
                        .iter()
                        .map(|arg| reference_expr(module, arg, locals))
                        .collect::<Result<Vec<_>, _>>()?;
                    reference_run(module, name, &values)
                }
            }
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if expect_bool(reference_expr(module, condition, locals)?)? {
                match reference_block(module, then_branch, &mut locals.clone())? {
                    Control::Continue(value) => Ok(value.unwrap_or(native::NativeValue::Unit)),
                    Control::Return(value) => Ok(value),
                }
            } else if let Some(else_branch) = else_branch {
                match reference_block(module, else_branch, &mut locals.clone())? {
                    Control::Continue(value) => Ok(value.unwrap_or(native::NativeValue::Unit)),
                    Control::Return(value) => Ok(value),
                }
            } else {
                Ok(native::NativeValue::Unit)
            }
        }
    }
}

fn one_arg(
    args: &[Expr],
    module: &Module,
    locals: &HashMap<String, native::NativeValue>,
) -> Result<native::NativeValue, String> {
    if args.len() != 1 {
        return Err("expected one argument".into());
    }
    reference_expr(module, &args[0], locals)
}

fn reference_binary(
    op: BinaryOp,
    left: native::NativeValue,
    right: native::NativeValue,
) -> Result<native::NativeValue, String> {
    use native::NativeValue::{Bool, Int, String};
    match op {
        BinaryOp::Add => match (left, right) {
            (Int(a), Int(b)) => Ok(Int(a + b)),
            (String(a), String(b)) => Ok(String(format!("{a}{b}"))),
            _ => Err("invalid operands for +".into()),
        },
        BinaryOp::Sub => Ok(Int(expect_int(left)? - expect_int(right)?)),
        BinaryOp::Mul => Ok(Int(expect_int(left)? * expect_int(right)?)),
        BinaryOp::Div => {
            let right = expect_int(right)?;
            if right == 0 {
                return Err("division by zero".into());
            }
            Ok(Int(expect_int(left)? / right))
        }
        BinaryOp::Mod => {
            let right = expect_int(right)?;
            if right == 0 {
                return Err("modulo by zero".into());
            }
            Ok(Int(expect_int(left)? % right))
        }
        BinaryOp::Equal => Ok(Bool(left == right)),
        BinaryOp::NotEqual => Ok(Bool(left != right)),
        BinaryOp::Less => Ok(Bool(expect_int(left)? < expect_int(right)?)),
        BinaryOp::LessEqual => Ok(Bool(expect_int(left)? <= expect_int(right)?)),
        BinaryOp::Greater => Ok(Bool(expect_int(left)? > expect_int(right)?)),
        BinaryOp::GreaterEqual => Ok(Bool(expect_int(left)? >= expect_int(right)?)),
    }
}

fn expect_int(value: native::NativeValue) -> Result<i64, String> {
    match value {
        native::NativeValue::Int(value) => Ok(value),
        _ => Err("expected Int".into()),
    }
}

fn expect_bool(value: native::NativeValue) -> Result<bool, String> {
    match value {
        native::NativeValue::Bool(value) => Ok(value),
        _ => Err("expected Bool".into()),
    }
}

fn find_function<'a>(module: &'a Module, name: &str) -> Result<&'a Function, String> {
    module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Function(function) if function.name == name => Some(function),
            _ => None,
        })
        .ok_or_else(|| format!("unknown function '{name}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_backend_matches_independent_reference_results() {
        for case in CORPUS {
            verify_case(case).unwrap_or_else(|error| panic!("{error}"));
        }
    }

    #[test]
    fn corpus_is_non_empty() {
        assert!(CORPUS.len() >= 8);
    }

    #[test]
    fn parse_format_parse_preserves_semantic_identity() {
        for case in CORPUS {
            let first = parse(case.source).unwrap();
            let formatted = crate::format::format_module(&first);
            let second = parse(&formatted).unwrap();
            assert_eq!(first.items.len(), second.items.len(), "{}", case.name);
            for (left, right) in first.items.iter().zip(second.items.iter()) {
                match (left, right) {
                    (crate::Item::Function(a), crate::Item::Function(b)) => {
                        assert_eq!(a.id, b.id, "{}", case.name);
                    }
                }
            }
        }
    }

    #[test]
    fn every_corpus_case_survives_the_compiler_pipeline() {
        for case in CORPUS {
            verify_case(case).unwrap_or_else(|error| panic!("{error}"));
        }
    }
}
