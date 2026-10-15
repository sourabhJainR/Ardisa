use crate::ast::{BinaryOp, Expr, ExprKind, Item, Module, StmtKind, TypeKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrModule {
    pub name: String,
    pub functions: Vec<IrFunction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrFunction {
    pub name: String,
    pub params: Vec<(String, TypeKind)>,
    pub return_type: Option<TypeKind>,
    pub ops: Vec<IrOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Backend-independent operations include mutable assignment for compiler state.
pub enum IrOp {
    Let {
        name: String,
        value: IrValue,
    },
    Set {
        name: String,
        value: IrValue,
    },
    SetIndex {
        collection: IrValue,
        index: IrValue,
        value: IrValue,
    },
    Return(Option<IrValue>),
    Expr(IrValue),
    Scope {
        ops: Vec<IrOp>,
    },
    Spawn {
        name: String,
        call: IrValue,
    },
    Join {
        name: String,
    },
    Cancel {
        name: String,
    },
    While {
        condition: IrValue,
        ops: Vec<IrOp>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrValue {
    Int(i64),
    Float(u32),
    Bool(bool),
    String(String),
    List(Vec<IrValue>),
    Index {
        collection: Box<IrValue>,
        index: Box<IrValue>,
    },
    Slice {
        collection: Box<IrValue>,
        start: Box<IrValue>,
        end: Box<IrValue>,
    },
    Name(String),
    Binary {
        op: BinaryOp,
        left: Box<IrValue>,
        right: Box<IrValue>,
    },
    Call {
        callee: String,
        args: Vec<IrValue>,
    },
    If {
        condition: Box<IrValue>,
        then_ops: Vec<IrOp>,
        else_ops: Vec<IrOp>,
    },
}

pub fn lower(module: &Module) -> IrModule {
    IrModule {
        name: module.name.clone(),
        functions: module
            .items
            .iter()
            .map(|item| {
                let Item::Function(function) = item;
                let mut ops = Vec::new();
                for stmt in &function.body.stmts {
                    match &stmt.kind {
                        StmtKind::Set { name, value } => ops.push(IrOp::Set {
                            name: name.clone(),
                            value: value_to_ir(value),
                        }),
                        StmtKind::SetIndex {
                            collection,
                            index,
                            value,
                        } => ops.push(IrOp::SetIndex {
                            collection: value_to_ir(collection),
                            index: value_to_ir(index),
                            value: value_to_ir(value),
                        }),
                        StmtKind::Let { name, value } => ops.push(IrOp::Let {
                            name: name.clone(),
                            value: value_to_ir(value),
                        }),
                        StmtKind::Return(value) => {
                            ops.push(IrOp::Return(value.as_ref().map(value_to_ir)))
                        }
                        StmtKind::Expr(expr) => ops.push(IrOp::Expr(value_to_ir(expr))),
                        StmtKind::Scope { body } => ops.push(IrOp::Scope {
                            ops: block_to_ops(body),
                        }),
                        StmtKind::Spawn { name, call } => ops.push(IrOp::Spawn {
                            name: name.clone(),
                            call: value_to_ir(call),
                        }),
                        StmtKind::Join { name } => ops.push(IrOp::Join { name: name.clone() }),
                        StmtKind::Cancel { name } => ops.push(IrOp::Cancel { name: name.clone() }),
                        StmtKind::While { condition, body } => ops.push(IrOp::While {
                            condition: value_to_ir(condition),
                            ops: block_to_ops(body),
                        }),
                    }
                }
                IrFunction {
                    name: function.name.clone(),
                    params: function
                        .params
                        .iter()
                        .map(|param| (param.name.clone(), param.ty.kind.clone()))
                        .collect(),
                    return_type: function.return_type.as_ref().map(|ty| ty.kind.clone()),
                    ops,
                }
            })
            .collect(),
    }
}

fn block_to_ops(block: &crate::Block) -> Vec<IrOp> {
    block
        .stmts
        .iter()
        .map(|stmt| match &stmt.kind {
            StmtKind::Set { name, value } => IrOp::Set {
                name: name.clone(),
                value: value_to_ir(value),
            },
            StmtKind::SetIndex {
                collection,
                index,
                value,
            } => IrOp::SetIndex {
                collection: value_to_ir(collection),
                index: value_to_ir(index),
                value: value_to_ir(value),
            },
            StmtKind::Let { name, value } => IrOp::Let {
                name: name.clone(),
                value: value_to_ir(value),
            },
            StmtKind::Return(value) => IrOp::Return(value.as_ref().map(value_to_ir)),
            StmtKind::Expr(expr) => IrOp::Expr(value_to_ir(expr)),
            StmtKind::Scope { body } => IrOp::Scope {
                ops: block_to_ops(body),
            },
            StmtKind::Spawn { name, call } => IrOp::Spawn {
                name: name.clone(),
                call: value_to_ir(call),
            },
            StmtKind::Join { name } => IrOp::Join { name: name.clone() },
            StmtKind::Cancel { name } => IrOp::Cancel { name: name.clone() },
            StmtKind::While { condition, body } => IrOp::While {
                condition: value_to_ir(condition),
                ops: block_to_ops(body),
            },
        })
        .collect()
}

fn value_to_ir(expr: &Expr) -> IrValue {
    match &expr.kind {
        ExprKind::Int(value) => IrValue::Int(*value),
        ExprKind::Float(bits) => IrValue::Float(*bits),
        ExprKind::Bool(value) => IrValue::Bool(*value),
        ExprKind::String(value) => IrValue::String(value.clone()),
        ExprKind::List(values) => IrValue::List(values.iter().map(value_to_ir).collect()),
        ExprKind::Index { collection, index } => IrValue::Index {
            collection: Box::new(value_to_ir(collection)),
            index: Box::new(value_to_ir(index)),
        },
        ExprKind::Slice { collection, start, end } => IrValue::Slice {
            collection: Box::new(value_to_ir(collection)),
            start: Box::new(value_to_ir(start)),
            end: Box::new(value_to_ir(end)),
        },
        ExprKind::Name(name) => IrValue::Name(name.clone()),
        ExprKind::Group(inner) => value_to_ir(inner),
        ExprKind::Binary { op, left, right } => IrValue::Binary {
            op: *op,
            left: Box::new(value_to_ir(left)),
            right: Box::new(value_to_ir(right)),
        },
        ExprKind::Call { callee, args } => {
            let callee = match &callee.kind {
                ExprKind::Name(name) => name.clone(),
                _ => "<dynamic>".into(),
            };
            IrValue::Call {
                callee,
                args: args.iter().map(value_to_ir).collect(),
            }
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => IrValue::If {
            condition: Box::new(value_to_ir(condition)),
            then_ops: block_to_ops(then_branch),
            else_ops: else_branch.as_ref().map(block_to_ops).unwrap_or_default(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn lowers_without_rust_backend_types() {
        let module = parse("module x\nfn add(a: Int, b: Int) -> Int\n  a + b\n").unwrap();
        let ir = lower(&module);
        assert_eq!(ir.name, "x");
        assert_eq!(ir.functions[0].params.len(), 2);
        assert!(matches!(
            ir.functions[0].ops[0],
            IrOp::Expr(IrValue::Binary { .. })
        ));
    }

    #[test]
    fn represents_conditionals_without_backend_placeholders() {
        let module = parse(
            "module x\nfn choose(a: Int) -> Int\n  if a == 0\n    return 1\n  else\n    return 2\n",
        )
        .unwrap();
        let ir = lower(&module);
        assert!(matches!(
            ir.functions[0].ops[0],
            IrOp::Expr(IrValue::If { .. })
        ));
    }
}
