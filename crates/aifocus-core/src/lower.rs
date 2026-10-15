use crate::{
    NodeId,
    ast::{BinaryOp, Block, Expr, ExprKind, Item, Module, StmtKind, Type, TypeKind},
    source::Span,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMapEntry {
    pub node_id: NodeId,
    pub source_span: Span,
    pub rust_line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweredModule {
    pub rust: String,
    pub source_map: Vec<SourceMapEntry>,
}

pub fn lower(module: &Module) -> LoweredModule {
    let mut lowerer = Lowerer {
        out: String::new(),
        source_map: Vec::new(),
    };
    for item in &module.items {
        if let Item::Function(function) = item {
            lowerer.function(function);
        }
    }
    LoweredModule {
        rust: lowerer.out,
        source_map: lowerer.source_map,
    }
}

struct Lowerer {
    out: String,
    source_map: Vec<SourceMapEntry>,
}

impl Lowerer {
    fn function(&mut self, function: &crate::ast::Function) {
        self.record(function.id, function.span);
        self.out.push_str("fn ");
        self.out.push_str(&function.name);
        self.out.push('(');
        for (index, param) in function.params.iter().enumerate() {
            if index > 0 {
                self.out.push_str(", ");
            }
            self.out.push_str(&param.name);
            self.out.push_str(": ");
            self.ty(&param.ty);
        }
        self.out.push(')');
        if let Some(ty) = &function.return_type {
            self.out.push_str(" -> ");
            self.ty(ty);
        }
        self.out.push_str(" {\n");
        self.block(&function.body, 1, function.return_type.is_some());
        self.out.push_str("}\n");
    }

    fn block(&mut self, block: &Block, indent: usize, returning: bool) {
        for (index, stmt) in block.stmts.iter().enumerate() {
            let last = index + 1 == block.stmts.len();
            self.indent(indent);
            match &stmt.kind {
                StmtKind::Set { name, value } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str(name);
                    self.out.push_str(" = ");
                    self.expr(value, indent);
                    self.out.push_str(";\n");
                }
                StmtKind::SetIndex {
                    collection,
                    index,
                    value,
                } => {
                    self.record(stmt.id, stmt.span);
                    self.expr(collection, indent);
                    self.out.push('[');
                    self.expr(index, indent);
                    self.out.push_str("] = ");
                    self.expr(value, indent);
                    self.out.push_str(";\n");
                }
                StmtKind::Let { name, value } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str("let mut ");
                    self.out.push_str(name);
                    self.out.push_str(" = ");
                    self.expr(value, indent);
                    self.out.push_str(";\n");
                }
                StmtKind::Return(value) => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str("return");
                    if let Some(value) = value {
                        self.out.push(' ');
                        self.expr(value, indent);
                    }
                    self.out.push_str(";\n");
                }
                StmtKind::Scope { body } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str("{\n");
                    self.block(body, indent + 1, false);
                    self.indent(indent);
                    self.out.push_str("}\n");
                }
                StmtKind::Spawn { name, call } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str("let ");
                    self.out.push_str(name);
                    self.out.push_str(" = std::thread::spawn(move || ");
                    self.expr(call, indent);
                    self.out.push_str(");\n");
                }
                StmtKind::Join { name } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str(name);
                    self.out
                        .push_str(".join().expect(\"Ardisa task panicked\");\n");
                }
                StmtKind::Cancel { name } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str("let _ = &");
                    self.out.push_str(name);
                    self.out
                        .push_str("; // cooperative cancellation is runtime-defined\n");
                }
                StmtKind::While { condition, body } => {
                    self.record(stmt.id, stmt.span);
                    self.out.push_str("while ");
                    self.expr(condition, indent);
                    self.out.push_str(" {\n");
                    self.block(body, indent + 1, false);
                    self.indent(indent);
                    self.out.push_str("}\n");
                }
                StmtKind::Expr(expr) => {
                    self.record(stmt.id, stmt.span);
                    self.expr(expr, indent);
                    if !(last && returning) {
                        self.out.push(';');
                    }
                    self.out.push('\n');
                }
            }
        }
    }

    fn expr(&mut self, expr: &Expr, indent: usize) {
        self.record(expr.id, expr.span);
        match &expr.kind {
            ExprKind::Int(value) => self.out.push_str(&value.to_string()),
            ExprKind::Float(bits) => self.out.push_str(&format!("{:?}", f32::from_bits(*bits))),
            ExprKind::Bool(value) => self.out.push_str(if *value { "true" } else { "false" }),
            ExprKind::String(value) => {
                self.out.push('"');
                for ch in value.chars() {
                    match ch {
                        '\\' => self.out.push_str("\\\\"),
                        '"' => self.out.push_str("\\\""),
                        '\n' => self.out.push_str("\\n"),
                        '\r' => self.out.push_str("\\r"),
                        '\t' => self.out.push_str("\\t"),
                        c => self.out.push(c),
                    }
                }
                self.out.push('"');
            }
            ExprKind::Name(name) => self.out.push_str(name),
            ExprKind::List(items) => {
                self.out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        self.out.push_str(", ");
                    }
                    self.expr(item, indent);
                }
                self.out.push(']');
            }
            ExprKind::Index { collection, index } => {
                self.expr(collection, indent);
                self.out.push('[');
                self.expr(index, indent);
                self.out.push(']');
            }
            ExprKind::Slice { collection, start, end } => {
                self.expr(collection, indent);
                self.out.push('[');
                self.expr(start, indent);
                self.out.push_str(":");
                self.expr(end, indent);
                self.out.push(']');
            }
            ExprKind::Group(inner) => {
                self.out.push('(');
                self.expr(inner, indent);
                self.out.push(')');
            }
            ExprKind::Binary { op, left, right } => {
                self.expr(left, indent);
                self.out.push(' ');
                self.out.push_str(match op {
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
                self.out.push(' ');
                self.expr(right, indent);
            }
            ExprKind::Call { callee, args } => {
                self.expr(callee, indent);
                self.out.push('(');
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        self.out.push_str(", ");
                    }
                    self.expr(arg, indent);
                }
                self.out.push(')');
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.out.push_str("if ");
                self.expr(condition, indent);
                self.out.push_str(" {\n");
                self.block(then_branch, indent + 1, false);
                self.indent(indent);
                if let Some(else_branch) = else_branch {
                    self.out.push_str("} else {\n");
                    self.block(else_branch, indent + 1, false);
                    self.indent(indent);
                }
                self.out.push('}');
            }
        }
    }

    fn ty(&mut self, ty: &Type) {
        self.record(ty.id, ty.span);
        match &ty.kind {
            TypeKind::Int => self.out.push_str("i64"),
            TypeKind::Bool => self.out.push_str("bool"),
            TypeKind::String => self.out.push_str("String"),
            TypeKind::Unit => self.out.push_str("()"),
            TypeKind::Named(name) => self.out.push_str(name),
            TypeKind::Generic(name, args) => {
                self.out.push_str(name);
                self.out.push('<');
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        self.out.push_str(", ");
                    }
                    self.ty(arg);
                }
                self.out.push('>');
            }
            TypeKind::List(element) => {
                self.out.push_str("Vec<");
                self.ty(element);
                self.out.push('>');
            }
            TypeKind::Result(ok, err) => {
                self.out.push_str("Result<");
                self.ty(ok);
                self.out.push_str(", ");
                self.ty(err);
                self.out.push('>');
            }
        }
    }

    fn indent(&mut self, count: usize) {
        for _ in 0..count {
            self.out.push_str("    ");
        }
    }

    fn record(&mut self, node_id: NodeId, source_span: Span) {
        let rust_line = self.out.bytes().filter(|byte| *byte == b'\n').count() + 1;
        self.source_map.push(SourceMapEntry {
            node_id,
            source_span,
            rust_line,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn lowers_function_to_readable_rust() {
        let module = parse("module x\nfn add(a: Int, b: Int) -> Int\n  a + b\n").unwrap();
        let lowered = lower(&module);
        assert_eq!(
            lowered.rust,
            "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n"
        );
        assert!(!lowered.source_map.is_empty());
    }

    #[test]
    fn lowers_grouped_expression_without_changing_shape() {
        let module = parse("module x\nfn calc(a: Int, b: Int) -> Int\n  (a + b) * 2\n").unwrap();
        let lowered = lower(&module);
        assert!(lowered.rust.contains("(a + b) * 2"));
    }

    #[test]
    fn generated_rust_supports_mutable_bindings() {
        let module = parse(
            "module x\nfn update(a: Int) -> Int\n  let value = a\n  set value = value + 1\n  value\n",
        )
        .unwrap();
        let lowered = lower(&module);
        assert!(lowered.rust.contains("let mut value = a;"));
    }

    #[test]
    fn generated_rust_is_accepted_by_rustc() {
        let module = parse("module x\nfn add(a: Int, b: Int) -> Int\n  a + b\n").unwrap();
        let lowered = lower(&module);
        let base = std::env::temp_dir().join(format!("aifocus-lowering-{}", std::process::id()));
        let source_path = base.with_extension("rs");
        let output_path = base.with_extension("rlib");
        std::fs::write(&source_path, lowered.rust).unwrap();

        let status = std::process::Command::new("rustc")
            .arg("--crate-type=lib")
            .arg("--emit=metadata")
            .arg("-o")
            .arg(&output_path)
            .arg(&source_path)
            .status()
            .expect("rustc must be available for lowering verification");
        let _ = std::fs::remove_file(&source_path);
        let _ = std::fs::remove_file(&output_path);
        assert!(status.success());
    }
}
