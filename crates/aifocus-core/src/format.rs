use crate::ast::*;

/// Canonical formatter for the growing control-flow surface.
pub fn format_module(module: &Module) -> String {
    let mut out = format!("module {}\n", module.name);
    for item in &module.items {
        format_item(item, &mut out);
    }
    for declaration in &module.constructs {
        format_construct(declaration, &mut out);
    }
    out
}

fn format_item(item: &Item, out: &mut String) {
    match item {
        Item::Function(function) => {
            out.push_str("\nfn ");
            out.push_str(&function.name);
            out.push('(');
            for (index, param) in function.params.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&param.name);
                out.push_str(": ");
                out.push_str(&param.ty.display_name());
            }
            out.push(')');
            if let Some(ty) = &function.return_type {
                out.push_str(" -> ");
                out.push_str(&ty.display_name());
            }
            out.push('\n');
            format_block(&function.body, 2, out);
        }
    }
}

fn format_construct(decl: &ConstructDeclaration, out: &mut String) {
    out.push_str("\n");
    out.push_str(match decl.kind {
        ConstructKind::Trace => "trace",
        ConstructKind::Cell => "cell",
        ConstructKind::Vault => "vault",
        ConstructKind::Proof => "proof",
        ConstructKind::Phase => "phase",
    });
    out.push(' ');
    out.push_str(&decl.name);
    out.push('\n');
    for member in &decl.members {
        out.push_str("  ");
        match member {
            ConstructMember::Field { name, ty, .. } => {
                out.push_str(name);
                out.push_str(": ");
                out.push_str(&ty.display_name());
            }
            ConstructMember::Clause { name, value, .. } => {
                out.push_str(name);
                out.push_str(": ");
                out.push_str(value);
            }
            ConstructMember::Transition { from, to, .. } => {
                out.push_str(from);
                out.push_str(" -> ");
                out.push_str(to);
            }
        }
        out.push('\n');
    }
}

fn format_block(block: &Block, indent: usize, out: &mut String) {
    for stmt in &block.stmts {
        format_indent(indent, out);
        format_stmt(stmt, indent, out);
    }
}

fn format_stmt(stmt: &Stmt, indent: usize, out: &mut String) {
    match &stmt.kind {
        StmtKind::Set { name, value } => {
            out.push_str("set ");
            out.push_str(name);
            out.push_str(" = ");
            format_expr(value, indent, out);
            out.push('\n');
        }
        StmtKind::SetIndex {
            collection,
            index,
            value,
        } => {
            out.push_str("set ");
            format_expr(collection, indent, out);
            out.push('[');
            format_expr(index, indent, out);
            out.push_str("] = ");
            format_expr(value, indent, out);
            out.push('\n');
        }
        StmtKind::Let { name, value } => {
            out.push_str("let ");
            out.push_str(name);
            out.push_str(" = ");
            format_expr(value, indent, out);
            out.push('\n');
        }
        StmtKind::Return(value) => {
            out.push_str("return");
            if let Some(value) = value {
                out.push(' ');
                format_expr(value, indent, out);
            }
            out.push('\n');
        }
        StmtKind::Expr(expr) => {
            format_expr(expr, indent, out);
            if !matches!(expr.kind, ExprKind::If { .. }) {
                out.push('\n');
            }
        }
        StmtKind::Scope { body } => {
            out.push_str("scope\n");
            format_block(body, indent + 2, out);
        }
        StmtKind::Spawn { name, call } => {
            out.push_str("spawn ");
            out.push_str(name);
            out.push_str(" = ");
            format_expr(call, indent, out);
            out.push('\n');
        }
        StmtKind::Join { name } => {
            out.push_str("join ");
            out.push_str(name);
            out.push('\n');
        }
        StmtKind::Cancel { name } => {
            out.push_str("cancel ");
            out.push_str(name);
            out.push('\n');
        }
        StmtKind::While { condition, body } => {
            out.push_str("while ");
            format_expr(condition, indent, out);
            out.push('\n');
            format_block(body, indent + 2, out);
        }
        StmtKind::For { name, iterable, body } => {
            out.push_str("for ");
            out.push_str(name);
            out.push_str(" in ");
            format_expr(iterable, indent, out);
            out.push('\n');
            format_block(body, indent + 2, out);
        }
    }
}

fn format_expr(expr: &Expr, indent: usize, out: &mut String) {
    match &expr.kind {
        ExprKind::Int(value) => out.push_str(&value.to_string()),
        ExprKind::Float(bits) => out.push_str(&format!("{:?}", f32::from_bits(*bits))),
        ExprKind::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
        ExprKind::String(value) => {
            out.push('"');
            out.push_str(value);
            out.push('"');
        }
        ExprKind::Name(name) => out.push_str(name),
        ExprKind::List(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                format_expr(item, indent, out);
            }
            out.push(']');
        }
        ExprKind::Index { collection, index } => {
            format_expr(collection, indent, out);
            out.push('[');
            format_expr(index, indent, out);
            out.push(']');
        }
        ExprKind::Slice { collection, start, end } => {
            format_expr(collection, indent, out);
            out.push('[');
            format_expr(start, indent, out);
            out.push(':');
            format_expr(end, indent, out);
            out.push(']');
        }
        ExprKind::Group(inner) => {
            out.push('(');
            format_expr(inner, indent, out);
            out.push(')');
        }
        ExprKind::Binary { op, left, right } => {
            format_expr(left, indent, out);
            out.push(' ');
            out.push_str(match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                BinaryOp::Mod => "%",
                BinaryOp::Equal => "==",
                BinaryOp::NotEqual => "!=",
                BinaryOp::Less => "<",
                BinaryOp::LessEqual => "<=",
                BinaryOp::Greater => ">",
                BinaryOp::GreaterEqual => ">=",
            });
            out.push(' ');
            format_expr(right, indent, out);
        }
        ExprKind::Call { callee, args } => {
            format_expr(callee, indent, out);
            out.push('(');
            for (index, arg) in args.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                format_expr(arg, indent, out);
            }
            out.push(')');
        }
        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            out.push_str("if ");
            format_expr(condition, indent, out);
            out.push('\n');
            format_block(then_branch, indent + 2, out);
            format_indent(indent, out);
            if let Some(else_branch) = else_branch {
                out.push_str("else\n");
                format_block(else_branch, indent + 2, out);
            }
        }
    }
}

fn format_indent(indent: usize, out: &mut String) {
    for _ in 0..indent {
        out.push(' ');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn canonicalizes_spacing() {
        let module = parse("module x\nfn f(a:Int,b:Int)->Int\n  a+b*2\n").unwrap();
        assert_eq!(
            format_module(&module),
            "module x\n\nfn f(a: Int, b: Int) -> Int\n  a + b * 2\n"
        );
    }

    #[test]
    fn preserves_grouping() {
        let module = parse("module x\nfn f(a: Int,b: Int)->Int\n  (a+b)*2\n").unwrap();
        assert_eq!(
            format_module(&module),
            "module x\n\nfn f(a: Int, b: Int) -> Int\n  (a + b) * 2\n"
        );
    }
}
