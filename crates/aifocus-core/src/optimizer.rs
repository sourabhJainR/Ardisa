use crate::ir::{IrModule, IrOp, IrValue};

pub fn optimize(mut module: IrModule) -> IrModule {
    for function in &mut module.functions {
        let folded = std::mem::take(&mut function.ops).into_iter().filter_map(fold_op).collect::<Vec<_>>();
        function.ops = eliminate_dead_ops(simplify_branches(folded));
    }
    module
}

fn eliminate_dead_ops(ops: Vec<IrOp>) -> Vec<IrOp> {
    let mut out = Vec::with_capacity(ops.len());
    let last = ops.len().saturating_sub(1);
    for (index, op) in ops.into_iter().enumerate() {
        let terminal = matches!(op, IrOp::Return(_));
        if index != last && !terminal && is_pure_expr(&op) {
            continue;
        }
        out.push(op);
        if terminal { break; }
    }
    out
}

fn is_pure_expr(op: &IrOp) -> bool {
    match op {
        IrOp::Expr(v) => is_pure_value(v),
        IrOp::Scope { ops } => ops.iter().all(is_pure_expr),
        _ => false,
    }
}

fn is_pure_value(v: &IrValue) -> bool {
    match v {
        IrValue::Int(_) | IrValue::Float(_) | IrValue::Bool(_) | IrValue::String(_) | IrValue::Name(_) => true,
        IrValue::List(xs) => xs.iter().all(is_pure_value),
        IrValue::Index { collection, index } => is_pure_value(collection) && is_pure_value(index),
        IrValue::Binary { left, right, .. } => is_pure_value(left) && is_pure_value(right),
        IrValue::If { condition, then_ops, else_ops } =>
            is_pure_value(condition) && then_ops.iter().all(is_pure_expr) && else_ops.iter().all(is_pure_expr),
        IrValue::Call { .. } => false,
    }
}

fn simplify_branches(ops: Vec<IrOp>) -> Vec<IrOp> {
    ops.into_iter().filter_map(|op| match op {
        IrOp::While { condition, ops } => match condition {
            IrValue::Bool(false) => None,
            condition => Some(IrOp::While { condition, ops: simplify_branches(ops) }),
        },
        IrOp::Scope { ops } => Some(IrOp::Scope { ops: simplify_branches(ops) }),
        other => Some(other),
    }).collect()
}

fn fold_op(op: IrOp) -> Option<IrOp> {
    match op {
        IrOp::Expr(value) => Some(IrOp::Expr(fold_value(value))),
        IrOp::Let { name, value } => Some(IrOp::Let { name, value: fold_value(value) }),
        IrOp::Set { name, value } => Some(IrOp::Set { name, value: fold_value(value) }),
        IrOp::SetIndex { collection, index, value } => Some(IrOp::SetIndex { collection: fold_value(collection), index: fold_value(index), value: fold_value(value) }),
        IrOp::Return(value) => Some(IrOp::Return(value.map(fold_value))),
        IrOp::Scope { ops } => Some(IrOp::Scope { ops: ops.into_iter().filter_map(fold_op).collect() }),
        IrOp::Spawn { name, call } => Some(IrOp::Spawn { name, call: fold_value(call) }),
        IrOp::Join { name } => Some(IrOp::Join { name }),
        IrOp::Cancel { name } => Some(IrOp::Cancel { name }),
        IrOp::While { condition, ops } => Some(IrOp::While { condition: fold_value(condition), ops: ops.into_iter().filter_map(fold_op).collect() }),
    }
}

fn fold_value(value: IrValue) -> IrValue {
    match value {
        IrValue::Binary { op, left, right } => {
            let left = fold_value(*left);
            let right = fold_value(*right);
            match (&op, &left, &right) {
                (crate::BinaryOp::Add, IrValue::Int(a), IrValue::Int(b)) => a.checked_add(*b).map(IrValue::Int).unwrap_or_else(|| IrValue::Binary { op, left: Box::new(left), right: Box::new(right) }),
                (crate::BinaryOp::Sub, IrValue::Int(a), IrValue::Int(b)) => a.checked_sub(*b).map(IrValue::Int).unwrap_or_else(|| IrValue::Binary { op, left: Box::new(left), right: Box::new(right) }),
                (crate::BinaryOp::Mul, IrValue::Int(a), IrValue::Int(b)) => a.checked_mul(*b).map(IrValue::Int).unwrap_or_else(|| IrValue::Binary { op, left: Box::new(left), right: Box::new(right) }),
                (crate::BinaryOp::Div, IrValue::Int(a), IrValue::Int(b)) if *b != 0 => a.checked_div(*b).map(IrValue::Int).unwrap_or_else(|| IrValue::Binary { op, left: Box::new(left), right: Box::new(right) }),
                (crate::BinaryOp::Mod, IrValue::Int(a), IrValue::Int(b)) if *b != 0 => a.checked_rem(*b).map(IrValue::Int).unwrap_or_else(|| IrValue::Binary { op, left: Box::new(left), right: Box::new(right) }),
                (crate::BinaryOp::Equal, _, _) => IrValue::Bool(left == right),
                (crate::BinaryOp::NotEqual, _, _) => IrValue::Bool(left != right),
                (crate::BinaryOp::Less, IrValue::Int(a), IrValue::Int(b)) => IrValue::Bool(a < b),
                (crate::BinaryOp::LessEqual, IrValue::Int(a), IrValue::Int(b)) => IrValue::Bool(a <= b),
                (crate::BinaryOp::Greater, IrValue::Int(a), IrValue::Int(b)) => IrValue::Bool(a > b),
                (crate::BinaryOp::GreaterEqual, IrValue::Int(a), IrValue::Int(b)) => IrValue::Bool(a >= b),
                _ => IrValue::Binary { op, left: Box::new(left), right: Box::new(right) },
            }
        }
        IrValue::If { condition, then_ops, else_ops } => {
            let condition = fold_value(*condition);
            match condition {
                IrValue::Bool(true) => branch_value(then_ops).unwrap_or(IrValue::Bool(false)),
                IrValue::Bool(false) => branch_value(else_ops).unwrap_or(IrValue::Bool(false)),
                condition => IrValue::If { condition: Box::new(condition), then_ops, else_ops },
            }
        }
        IrValue::List(values) => IrValue::List(values.into_iter().map(fold_value).collect()),
        IrValue::Index { collection, index } => IrValue::Index { collection: Box::new(fold_value(*collection)), index: Box::new(fold_value(*index)) },
        IrValue::Call { callee, args } => IrValue::Call { callee, args: args.into_iter().map(fold_value).collect() },
        other => other,
    }
}

fn branch_value(ops: Vec<IrOp>) -> Option<IrValue> {
    ops.into_iter().rev().find_map(|op| match op {
        IrOp::Expr(value) => Some(fold_value(value)),
        IrOp::Return(Some(value)) => Some(fold_value(value)),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn folds_constant_arithmetic_without_dropping_terminal_expression() {
        let module = crate::parse("module x\nfn main() -> Int\n  2 + 3 * 4\n").unwrap();
        let optimized = optimize(crate::ir::lower(&module));
        assert!(matches!(optimized.functions[0].ops[0], IrOp::Expr(IrValue::Int(14))));
    }
    #[test]
    fn removes_dead_pure_expression_but_keeps_calls() {
        let module = crate::parse("module x
fn main(a: Int) -> Int
  1 + 2
  a + 1
").unwrap();
        let optimized = optimize(crate::ir::lower(&module));
        assert_eq!(optimized.functions[0].ops.len(), 1);
    }

    #[test]
    fn preserves_overflowing_constant_arithmetic_for_runtime_diagnostics() {
        let module = crate::parse("module x\nfn main() -> Int\n  9223372036854775807 + 1\n").unwrap();
        let optimized = optimize(crate::ir::lower(&module));
        assert!(matches!(optimized.functions[0].ops[0], IrOp::Expr(IrValue::Binary { .. })));
    }

    #[test]
    fn removes_constant_false_loop() {
        let module = crate::parse("module x
fn main() -> Int
  while false
    1 + 2
  7
").unwrap();
        let optimized = optimize(crate::ir::lower(&module));
        assert!(optimized.functions[0].ops.iter().all(|op| !matches!(op, IrOp::While { .. })));
    }

    #[test]
    fn preserves_non_constant_expression() {
        let module = crate::parse("module x\nfn main(a: Int) -> Int\n  a + 1\n").unwrap();
        let optimized = optimize(crate::ir::lower(&module));
        assert!(matches!(optimized.functions[0].ops[0], IrOp::Expr(IrValue::Binary { .. })));
    }
}