use crate::ast::{BinaryOp, Expr, ExprKind, Module, StmtKind, TypeKind};
use crate::source::Span;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIrModule {
    pub name: String,
    pub functions: Vec<TypedIrFunction>,
    /// Typed declaration metadata. The executable backend must not ignore
    /// non-empty constructs; compile_source still rejects them until runtime
    /// operations and effect enforcement are implemented.
    pub constructs: Vec<TypedIrConstruct>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIrConstruct {
    pub kind: crate::ast::ConstructKind,
    pub name: String,
    pub members: Vec<TypedIrConstructMember>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedIrConstructMember {
    Field { name: String, ty: TypeKind, span: Span },
    Clause { name: String, value: String, span: Span },
    Transition { from: String, to: String, span: Span },
}

/// Lower only construct declarations for compiler tooling and focused tests.
/// The regular function lowerer remains fail-closed through sema::analyze.
pub fn lower_constructs(module: &Module) -> Result<Vec<TypedIrConstruct>, Vec<String>> {
    let constructs = crate::sema::analyze_constructs(module).map_err(|errors| {
        errors.into_iter().map(|error| format!("{}: {}", error.code, error.message)).collect::<Vec<_>>()
    })?;
    Ok(constructs.into_iter().map(|construct| TypedIrConstruct {
        kind: construct.kind,
        name: construct.name,
        span: construct.span,
        members: construct.members.into_iter().map(|member| match member {
            crate::sema::TypedConstructMember::Field { name, ty, span } => TypedIrConstructMember::Field { name, ty, span },
            crate::sema::TypedConstructMember::Clause { name, value, span } => TypedIrConstructMember::Clause { name, value, span },
            crate::sema::TypedConstructMember::Transition { from, to, span } => TypedIrConstructMember::Transition { from, to, span },
        }).collect(),
    }).collect())

}

/// Validate typed construct metadata before any serializer or downstream consumer
/// accepts it. This is a tooling contract only; it does not make constructs executable.
pub fn validate_construct_metadata(constructs: &[TypedIrConstruct]) -> Result<(), Vec<String>> {
    use std::collections::BTreeSet;

    let mut errors = Vec::new();
    let mut construct_names = BTreeSet::new();
    for construct in constructs {
        if construct.name.trim().is_empty() {
            errors.push("construct name must not be empty".to_string());
        }
        if !construct_names.insert((construct.kind as u8, construct.name.clone())) {
            errors.push(format!("duplicate construct declaration: {}", construct.name));
        }

        let mut member_names = BTreeSet::new();
        let mut transitions = BTreeSet::new();
        for member in &construct.members {
            match member {
                TypedIrConstructMember::Field { name, .. } => {
                    if name.trim().is_empty() {
                        errors.push(format!("{}: field name must not be empty", construct.name));
                    }
                    if !member_names.insert(name.clone()) {
                        errors.push(format!("{}: duplicate member name: {}", construct.name, name));
                    }
                }
                TypedIrConstructMember::Clause { name, value, .. } => {
                    if name.trim().is_empty() || value.trim().is_empty() {
                        errors.push(format!("{}: clause name and value must not be empty", construct.name));
                    }
                    if !member_names.insert(name.clone()) {
                        errors.push(format!("{}: duplicate member name: {}", construct.name, name));
                    }
                }
                TypedIrConstructMember::Transition { from, to, .. } => {
                    if from.trim().is_empty() || to.trim().is_empty() {
                        errors.push(format!("{}: transition endpoints must not be empty", construct.name));
                    }
                    if !transitions.insert((from.clone(), to.clone())) {
                        errors.push(format!("{}: duplicate transition: {} -> {}", construct.name, from, to));
                    }
                }
            }
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Encode validated construct metadata in a stable, versioned representation for
/// tooling and differential tests. Source spans are intentionally excluded. This
/// format is not an executable artifact and is not accepted by the native runtime.
pub fn encode_construct_metadata(constructs: &[TypedIrConstruct]) -> Result<String, Vec<String>> {
    validate_construct_metadata(constructs)?;
    let mut output = String::from("ARDISA-CONSTRUCT-META-V1\n");
    for construct in constructs {
        output.push_str(&format!("K|{}|{}\n", construct_kind_name(construct.kind), encode_metadata_atom(&construct.name)));
        for member in &construct.members {
            match member {
                TypedIrConstructMember::Field { name, ty, .. } => output.push_str(&format!("F|{}|{}\n", encode_metadata_atom(name), encode_metadata_atom(&ty.display_name()))),
                TypedIrConstructMember::Clause { name, value, .. } => output.push_str(&format!("C|{}|{}\n", encode_metadata_atom(name), encode_metadata_atom(value))),
                TypedIrConstructMember::Transition { from, to, .. } => output.push_str(&format!("T|{}|{}\n", encode_metadata_atom(from), encode_metadata_atom(to))),
            }
        }
    }
    Ok(output)
}

fn construct_kind_name(kind: crate::ast::ConstructKind) -> &'static str {
    match kind {
        crate::ast::ConstructKind::Trace => "Trace",
        crate::ast::ConstructKind::Cell => "Cell",
        crate::ast::ConstructKind::Vault => "Vault",
        crate::ast::ConstructKind::Proof => "Proof",
        crate::ast::ConstructKind::Phase => "Phase",
    }
}

fn encode_metadata_atom(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedIrFunction {
    pub name: String,
    pub params: Vec<(String, TypeKind)>,
    pub return_type: TypeKind,
    pub blocks: Vec<TypedBasicBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedBasicBlock {
    pub id: u32,
    pub ops: Vec<TypedIrOp>,
    pub terminator: TypedTerminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedIrOp {
    Bind { name: String, value: TypedValue },
    Assign { name: String, value: TypedValue },
    AssignIndex { collection: TypedValue, index: TypedValue, value: TypedValue },
    Expr(TypedValue),
    Scope(Vec<TypedIrOp>),
    Spawn { name: String, call: TypedValue },
    Join(String),
    Cancel(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedTerminator {
    Fallthrough,
    Return(Option<TypedValue>),
    Branch { condition: TypedValue, then_block: u32, else_block: u32 },
    Loop { condition: TypedValue, body_block: u32, exit_block: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedValue {
    pub ty: TypeKind,
    pub span: Span,
    pub kind: TypedValueKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedValueKind {
    Int(i64),
    Float(u32),
    Bool(bool),
    String(String),
    Name(String),
    List(Vec<TypedValue>),
    Index(Box<TypedValue>, Box<TypedValue>),
    Binary { op: BinaryOp, left: Box<TypedValue>, right: Box<TypedValue> },
    Call { callee: String, args: Vec<TypedValue> },
    If { condition: Box<TypedValue>, then_ops: Vec<TypedIrOp>, else_ops: Vec<TypedIrOp> },
}

pub fn lower(module: &Module) -> Result<TypedIrModule, Vec<String>> {
    let analysis = crate::sema::analyze(module).map_err(|errors| {
        errors.into_iter().map(|e| format!("{}: {}", e.code, e.message)).collect::<Vec<_>>()
    })?;
    let mut functions = Vec::new();
    let mut errors = Vec::new();

    for item in &module.items {
        let crate::ast::Item::Function(function) = item;
        let return_type = function.return_type.as_ref()
            .map(|t| t.kind.clone())
            .or_else(|| analysis.function_returns.get(&function.name).map(|t| t.kind.clone()))
            .unwrap_or(TypeKind::Unit);
        let mut locals = BTreeMap::new();
        for p in &function.params { locals.insert(p.name.clone(), p.ty.kind.clone()); }
        let mut ops = Vec::new();
        for stmt in &function.body.stmts {
            lower_stmt(stmt, &analysis.inferred_types, &mut locals, &mut ops, &mut errors);
        }
        let terminator = match ops.last() {
            Some(TypedIrOp::Expr(value)) if return_type != TypeKind::Unit => TypedTerminator::Return(Some(value.clone())),
            _ => TypedTerminator::Return(None),
        };
        functions.push(TypedIrFunction {
            name: function.name.clone(),
            params: function.params.iter().map(|p| (p.name.clone(), p.ty.kind.clone())).collect(),
            return_type,
            blocks: vec![TypedBasicBlock { id: 0, ops, terminator }],
        });
    }
    if errors.is_empty() { Ok(TypedIrModule { name: module.name.clone(), functions, constructs: analysis.constructs.iter().map(|construct| TypedIrConstruct {
        kind: construct.kind,
        name: construct.name.clone(),
        span: construct.span,
        members: construct.members.iter().map(|member| match member {
            crate::sema::TypedConstructMember::Field { name, ty, span } => TypedIrConstructMember::Field { name: name.clone(), ty: ty.clone(), span: *span },
            crate::sema::TypedConstructMember::Clause { name, value, span } => TypedIrConstructMember::Clause { name: name.clone(), value: value.clone(), span: *span },
            crate::sema::TypedConstructMember::Transition { from, to, span } => TypedIrConstructMember::Transition { from: from.clone(), to: to.clone(), span: *span },
        }).collect(),
    }).collect() }) } else { Err(errors) }
}

fn lower_stmt(
    stmt: &crate::ast::Stmt,
    types: &HashMap<crate::NodeId, crate::ast::Type>,
    locals: &mut BTreeMap<String, TypeKind>,
    ops: &mut Vec<TypedIrOp>,
    errors: &mut Vec<String>,
) {
    match &stmt.kind {
        StmtKind::Let { name, value } => {
            if let Some(v) = lower_expr(value, types, locals, errors) {
                locals.insert(name.clone(), v.ty.clone());
                ops.push(TypedIrOp::Bind { name: name.clone(), value: v });
            }
        }
        StmtKind::Set { name, value } => {
            if let Some(v) = lower_expr(value, types, locals, errors) {
                if let Some(expected) = locals.get(name) {
                    if expected != &v.ty { errors.push(format!("assignment type mismatch for {name}")); }
                } else { errors.push(format!("assignment to unknown binding {name}")); }
                ops.push(TypedIrOp::Assign { name: name.clone(), value: v });
            }
        }
        StmtKind::SetIndex { collection, index, value } => {
            if let (Some(c), Some(i), Some(v)) = (
                lower_expr(collection, types, locals, errors),
                lower_expr(index, types, locals, errors),
                lower_expr(value, types, locals, errors),
            ) { ops.push(TypedIrOp::AssignIndex { collection:c,index:i,value:v }); }
        }
        StmtKind::Return(value) => {
            let v = value.as_ref().and_then(|e| lower_expr(e, types, locals, errors));
            ops.push(TypedIrOp::Expr(v.unwrap_or(TypedValue { ty:TypeKind::Unit, span:stmt.span, kind:TypedValueKind::Bool(false) })));
        }
        StmtKind::Expr(e) => if let Some(v)=lower_expr(e,types,locals,errors){ ops.push(TypedIrOp::Expr(v)); },
        StmtKind::Scope { body } => {
            let mut nested=Vec::new();
            let mut scoped=locals.clone();
            for s in &body.stmts { lower_stmt(s,types,&mut scoped,&mut nested,errors); }
            ops.push(TypedIrOp::Scope(nested));
        }
        StmtKind::Spawn { name, call } => if let Some(v)=lower_expr(call,types,locals,errors){ops.push(TypedIrOp::Spawn{name:name.clone(),call:v});},
        StmtKind::Join { name } => ops.push(TypedIrOp::Join(name.clone())),
        StmtKind::Cancel { name } => ops.push(TypedIrOp::Cancel(name.clone())),
        StmtKind::While { condition, body } => {
            if let Some(c)=lower_expr(condition,types,locals,errors) {
                let mut nested=Vec::new(); let mut scoped=locals.clone();
                for s in &body.stmts { lower_stmt(s,types,&mut scoped,&mut nested,errors); }
                ops.push(TypedIrOp::Scope(vec![TypedIrOp::Expr(c), TypedIrOp::Scope(nested)]));
            }
        }
    }
}

fn lower_expr(
    expr:&Expr,
    types:&HashMap<crate::NodeId, crate::ast::Type>,
    locals:&BTreeMap<String,TypeKind>,
    errors:&mut Vec<String>
)->Option<TypedValue>{
    let ty=types.get(&expr.id).map(|t|t.kind.clone()).or_else(||match &expr.kind {
        ExprKind::Int(_)=>Some(TypeKind::Int), ExprKind::Float(_)=>Some(TypeKind::Named("F32".into())), ExprKind::Bool(_)=>Some(TypeKind::Bool),
        ExprKind::String(_)=>Some(TypeKind::String), ExprKind::Name(n)=>locals.get(n).cloned(),
        _=>None
    })?;
    let kind=match &expr.kind {
        ExprKind::Int(v)=>TypedValueKind::Int(*v), ExprKind::Float(bits)=>TypedValueKind::Float(*bits), ExprKind::Bool(v)=>TypedValueKind::Bool(*v),
        ExprKind::String(v)=>TypedValueKind::String(v.clone()), ExprKind::Name(v)=>TypedValueKind::Name(v.clone()),
        ExprKind::Group(e)=>return lower_expr(e,types,locals,errors),
        ExprKind::List(xs)=>TypedValueKind::List(xs.iter().filter_map(|e|lower_expr(e,types,locals,errors)).collect()),
        ExprKind::Index{collection,index}=>TypedValueKind::Index(Box::new(lower_expr(collection,types,locals,errors)?),Box::new(lower_expr(index,types,locals,errors)?)),
        ExprKind::Binary{op,left,right}=>TypedValueKind::Binary{op:*op,left:Box::new(lower_expr(left,types,locals,errors)?),right:Box::new(lower_expr(right,types,locals,errors)?)},
        ExprKind::Call{callee,args}=>{
            let ExprKind::Name(name)=&callee.kind else { errors.push("dynamic call in typed IR".into()); return None; };
            TypedValueKind::Call{callee:name.clone(),args:args.iter().filter_map(|e|lower_expr(e,types,locals,errors)).collect()}
        }
        ExprKind::If{condition,then_branch,else_branch}=>{
            let condition=Box::new(lower_expr(condition,types,locals,errors)?);
            let mut then_ops=Vec::new(); let mut then_locals=locals.clone();
            for s in &then_branch.stmts { lower_stmt(s,types,&mut then_locals,&mut then_ops,errors); }
            let mut else_ops=Vec::new(); let mut else_locals=locals.clone();
            if let Some(branch)=else_branch { for s in &branch.stmts { lower_stmt(s,types,&mut else_locals,&mut else_ops,errors); } }
            TypedValueKind::If{condition,then_ops,else_ops}
        },
    };
    Some(TypedValue{ty,span:expr.span,kind})
}


pub fn lower_cfg(module: &Module) -> Result<TypedIrModule, Vec<String>> {
    let analysis = crate::sema::analyze(module).map_err(|errors| {
        errors.into_iter().map(|e| format!("{}: {}", e.code, e.message)).collect::<Vec<_>>()
    })?;
    let mut functions = Vec::new();
    let mut errors = Vec::new();
    for item in &module.items {
        let crate::ast::Item::Function(function) = item;
        let return_type = function.return_type.as_ref()
            .map(|t| t.kind.clone())
            .or_else(|| analysis.function_returns.get(&function.name).map(|t| t.kind.clone()))
            .unwrap_or(TypeKind::Unit);
        let mut locals = BTreeMap::new();
        for p in &function.params { locals.insert(p.name.clone(), p.ty.kind.clone()); }
        let mut blocks = Vec::new();
        let entry = build_cfg_block(&function.body, &analysis.inferred_types, &mut locals, &mut blocks, &mut errors, true);
        if entry != 0 { errors.push(format!("internal CFG error: entry block is {entry}")); }
        functions.push(TypedIrFunction {
            name: function.name.clone(),
            params: function.params.iter().map(|p| (p.name.clone(), p.ty.kind.clone())).collect(),
            return_type,
            blocks,
        });
    }
    if errors.is_empty() { Ok(TypedIrModule { name: module.name.clone(), functions, constructs: Vec::new() }) } else { Err(errors) }
}

fn new_cfg_block(blocks: &mut Vec<TypedBasicBlock>) -> u32 {
    let id = blocks.len() as u32;
    blocks.push(TypedBasicBlock { id, ops: Vec::new(), terminator: TypedTerminator::Fallthrough });
    id
}

fn build_cfg_block(
    body: &crate::Block,
    types: &HashMap<crate::NodeId, crate::ast::Type>,
    locals: &mut BTreeMap<String, TypeKind>,
    blocks: &mut Vec<TypedBasicBlock>,
    errors: &mut Vec<String>,
    finalize_return: bool,
) -> u32 {
    let current = new_cfg_block(blocks);
    let mut cursor = current;
    for stmt in &body.stmts {
        match &stmt.kind {
            StmtKind::While { condition, body } => {
                let header = cursor;
                let body_block = new_cfg_block(blocks);
                let exit_block = new_cfg_block(blocks);
                if let Some(cond) = lower_expr(condition, types, locals, errors) {
                    blocks[header as usize].terminator = TypedTerminator::Loop {
                        condition: cond, body_block, exit_block,
                    };
                }
                let mut scoped = locals.clone();
                let nested_entry = build_cfg_block(body, types, &mut scoped, blocks, errors, false);
                if nested_entry != body_block {
                    blocks[body_block as usize] = blocks[nested_entry as usize].clone();
                    blocks[body_block as usize].id = body_block;
                }
                blocks[exit_block as usize].terminator = TypedTerminator::Fallthrough;
                cursor = exit_block;
            }
            StmtKind::Return(value) => {
                let value = value.as_ref().and_then(|e| lower_expr(e, types, locals, errors));
                blocks[cursor as usize].terminator = TypedTerminator::Return(value);
                cursor = new_cfg_block(blocks);
            }
            StmtKind::Let { name, value } => {
                if let Some(v) = lower_expr(value, types, locals, errors) {
                    locals.insert(name.clone(), v.ty.clone());
                    blocks[cursor as usize].ops.push(TypedIrOp::Bind { name: name.clone(), value: v });
                }
            }
            StmtKind::Set { name, value } => {
                if let Some(v) = lower_expr(value, types, locals, errors) {
                    blocks[cursor as usize].ops.push(TypedIrOp::Assign { name: name.clone(), value: v });
                }
            }
            StmtKind::SetIndex { collection, index, value } => {
                if let (Some(c), Some(i), Some(v)) = (lower_expr(collection, types, locals, errors), lower_expr(index, types, locals, errors), lower_expr(value, types, locals, errors)) {
                    blocks[cursor as usize].ops.push(TypedIrOp::AssignIndex { collection: c, index: i, value: v });
                }
            }
            StmtKind::Expr(expr) => {
                if let Some(v) = lower_expr(expr, types, locals, errors) { blocks[cursor as usize].ops.push(TypedIrOp::Expr(v)); }
            }
            StmtKind::Scope { body } => {
                let mut nested = Vec::new();
                let mut scoped = locals.clone();
                for s in &body.stmts { lower_stmt(s, types, &mut scoped, &mut nested, errors); }
                blocks[cursor as usize].ops.push(TypedIrOp::Scope(nested));
            }
            StmtKind::Spawn { name, call } => {
                if let Some(v) = lower_expr(call, types, locals, errors) { blocks[cursor as usize].ops.push(TypedIrOp::Spawn { name: name.clone(), call: v }); }
            }
            StmtKind::Join { name } => blocks[cursor as usize].ops.push(TypedIrOp::Join(name.clone())),
            StmtKind::Cancel { name } => blocks[cursor as usize].ops.push(TypedIrOp::Cancel(name.clone())),
        }
    }
    if finalize_return && matches!(blocks[cursor as usize].terminator, TypedTerminator::Fallthrough) {
        let value = blocks[cursor as usize].ops.last().and_then(|op| match op { TypedIrOp::Expr(v) => Some(v.clone()), _ => None });
        blocks[cursor as usize].terminator = TypedTerminator::Return(value);
    }
    current
}


pub fn optimize(module: TypedIrModule) -> TypedIrModule {
    let functions = module.functions.into_iter().map(|mut function| {
        for block in &mut function.blocks {
            for op in &mut block.ops {
                optimize_op(op);
            }
            block.terminator = optimize_terminator(block.terminator.clone());
        }
        function
    }).collect();
    TypedIrModule { name: module.name, functions, constructs: module.constructs }
}

fn optimize_op(op: &mut TypedIrOp) {
    match op {
        TypedIrOp::Bind { value, .. }
        | TypedIrOp::Assign { value, .. }
        | TypedIrOp::Expr(value) => optimize_value(value),
        TypedIrOp::AssignIndex { collection, index, value } => {
            optimize_value(collection);
            optimize_value(index);
            optimize_value(value);
        }
        TypedIrOp::Scope(ops) => ops.iter_mut().for_each(optimize_op),
        TypedIrOp::Spawn { call, .. } => optimize_value(call),
        TypedIrOp::Join(_) | TypedIrOp::Cancel(_) => {}
    }
}

fn optimize_terminator(terminator: TypedTerminator) -> TypedTerminator {
    match terminator {
        TypedTerminator::Return(value) => TypedTerminator::Return(value.map(|mut v| {
            optimize_value(&mut v);
            v
        })),
        TypedTerminator::Branch { condition, then_block, else_block } => {
            let mut condition = condition;
            optimize_value(&mut condition);
            TypedTerminator::Branch { condition, then_block, else_block }
        }
        TypedTerminator::Loop { condition, body_block, exit_block } => {
            let mut condition = condition;
            optimize_value(&mut condition);
            TypedTerminator::Loop { condition, body_block, exit_block }
        }
        TypedTerminator::Fallthrough => TypedTerminator::Fallthrough,
    }
}

fn optimize_value(value: &mut TypedValue) {
    match &mut value.kind {
        TypedValueKind::Binary { left, right, op } => {
            optimize_value(left);
            optimize_value(right);
            if let (TypedValueKind::Int(a), TypedValueKind::Int(b)) = (&left.kind, &right.kind) {
                let folded = match op {
                    BinaryOp::Add => Some(TypedValueKind::Int(a + b)),
                    BinaryOp::Sub => Some(TypedValueKind::Int(a - b)),
                    BinaryOp::Mul => Some(TypedValueKind::Int(a * b)),
                    BinaryOp::Div if *b != 0 => Some(TypedValueKind::Int(a / b)),
                    BinaryOp::Mod if *b != 0 => Some(TypedValueKind::Int(a % b)),
                    BinaryOp::Equal => Some(TypedValueKind::Bool(a == b)),
                    BinaryOp::NotEqual => Some(TypedValueKind::Bool(a != b)),
                    BinaryOp::Less => Some(TypedValueKind::Bool(a < b)),
                    BinaryOp::LessEqual => Some(TypedValueKind::Bool(a <= b)),
                    BinaryOp::Greater => Some(TypedValueKind::Bool(a > b)),
                    BinaryOp::GreaterEqual => Some(TypedValueKind::Bool(a >= b)),
                    _ => None,
                };
                if let Some(kind) = folded {
                    value.kind = kind;
                    value.ty = match &value.kind {
                        TypedValueKind::Bool(_) => TypeKind::Bool,
                        _ => TypeKind::Int,
                    };
                }
            }
        }
        TypedValueKind::List(values) => values.iter_mut().for_each(optimize_value),
        TypedValueKind::Index(collection, index) => {
            optimize_value(collection);
            optimize_value(index);
        }
        TypedValueKind::Call { args, .. } => args.iter_mut().for_each(optimize_value),
        TypedValueKind::If { condition, then_ops, else_ops } => {
            optimize_value(condition);
            then_ops.iter_mut().for_each(optimize_op);
            else_ops.iter_mut().for_each(optimize_op);
        }
        TypedValueKind::Float(_)
        | TypedValueKind::Int(_)
        | TypedValueKind::Bool(_)
        | TypedValueKind::String(_)
        | TypedValueKind::Name(_) => {}
    }
}


pub fn to_legacy_ir(module: &TypedIrModule) -> crate::ir::IrModule {
    crate::ir::IrModule {
        name: module.name.clone(),
        functions: module.functions.iter().map(|function| crate::ir::IrFunction {
            name: function.name.clone(),
            params: function.params.clone(),
            return_type: Some(function.return_type.clone()),
            ops: flatten_cfg(function),
        }).collect(),
    }
}

fn flatten_cfg(function: &TypedIrFunction) -> Vec<crate::ir::IrOp> {
    let mut ops = Vec::new();
    for block in &function.blocks {
        ops.extend(block.ops.iter().map(typed_op_to_legacy));
        match &block.terminator {
            TypedTerminator::Return(value) => ops.push(crate::ir::IrOp::Return(value.as_ref().map(typed_value_to_legacy))),
            TypedTerminator::Branch { condition, .. } => ops.push(crate::ir::IrOp::Expr(typed_value_to_legacy(condition))),
            TypedTerminator::Loop { condition, .. } => ops.push(crate::ir::IrOp::While { condition: typed_value_to_legacy(condition), ops: Vec::new() }),
            TypedTerminator::Fallthrough => {}
        }
    }
    ops
}

fn typed_op_to_legacy(op: &TypedIrOp) -> crate::ir::IrOp {
    match op {
        TypedIrOp::Bind { name, value } => crate::ir::IrOp::Let { name: name.clone(), value: typed_value_to_legacy(value) },
        TypedIrOp::Assign { name, value } => crate::ir::IrOp::Set { name: name.clone(), value: typed_value_to_legacy(value) },
        TypedIrOp::AssignIndex { collection, index, value } => crate::ir::IrOp::SetIndex { collection: typed_value_to_legacy(collection), index: typed_value_to_legacy(index), value: typed_value_to_legacy(value) },
        TypedIrOp::Expr(value) => crate::ir::IrOp::Expr(typed_value_to_legacy(value)),
        TypedIrOp::Scope(ops) => crate::ir::IrOp::Scope { ops: ops.iter().map(typed_op_to_legacy).collect() },
        TypedIrOp::Spawn { name, call } => crate::ir::IrOp::Spawn { name: name.clone(), call: typed_value_to_legacy(call) },
        TypedIrOp::Join(name) => crate::ir::IrOp::Join { name: name.clone() },
        TypedIrOp::Cancel(name) => crate::ir::IrOp::Cancel { name: name.clone() },
    }
}

fn typed_value_to_legacy(value: &TypedValue) -> crate::ir::IrValue {
    match &value.kind {
        TypedValueKind::Int(v) => crate::ir::IrValue::Int(*v),
        TypedValueKind::Float(bits) => crate::ir::IrValue::Float(*bits),
        TypedValueKind::Bool(v) => crate::ir::IrValue::Bool(*v),
        TypedValueKind::String(v) => crate::ir::IrValue::String(v.clone()),
        TypedValueKind::Name(v) => crate::ir::IrValue::Name(v.clone()),
        TypedValueKind::List(values) => crate::ir::IrValue::List(values.iter().map(typed_value_to_legacy).collect()),
        TypedValueKind::Index(c, i) => crate::ir::IrValue::Index { collection: Box::new(typed_value_to_legacy(c)), index: Box::new(typed_value_to_legacy(i)) },
        TypedValueKind::Binary { op, left, right } => crate::ir::IrValue::Binary { op: *op, left: Box::new(typed_value_to_legacy(left)), right: Box::new(typed_value_to_legacy(right)) },
        TypedValueKind::Call { callee, args } => crate::ir::IrValue::Call { callee: callee.clone(), args: args.iter().map(typed_value_to_legacy).collect() },
        TypedValueKind::If { condition, then_ops, else_ops } => crate::ir::IrValue::If {
            condition: Box::new(typed_value_to_legacy(condition)),
            then_ops: then_ops.iter().map(typed_op_to_legacy).collect(),
            else_ops: else_ops.iter().map(typed_op_to_legacy).collect(),
        },
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lowers_structurally_validated_constructs_to_typed_ir_without_enabling_execution() {
        let module = crate::parse(
            "module constructs\ncell State\n  value: Int\n  invariant: valid\nphase Lifecycle\n  Ready -> Done\n",
        ).unwrap();
        let constructs = lower_constructs(&module).unwrap();
        assert_eq!(constructs.len(), 2);
        assert!(matches!(
            constructs[0].members.first(),
            Some(TypedIrConstructMember::Field { name, ty: TypeKind::Int, .. }) if name == "value"
        ));
        assert!(matches!(
            constructs[1].members.first(),
            Some(TypedIrConstructMember::Transition { from, to, .. }) if from == "Ready" && to == "Done"
        ));
        assert!(lower(&module).is_err(), "regular compiler lowering must remain fail-closed");
    }

    #[test]
    fn construct_metadata_encoding_is_deterministic_and_span_independent() {
        let module = crate::parse(
            "module constructs\ncell State\n  value: Int\n  invariant: valid\nphase Lifecycle\n  Ready -> Done\n",
        ).unwrap();
        let constructs = lower_constructs(&module).unwrap();
        let encoded = encode_construct_metadata(&constructs).unwrap();
        assert!(encoded.starts_with("ARDISA-CONSTRUCT-META-V1\n"));
        let mut relocated = constructs.clone();
        for construct in &mut relocated {
            construct.span = crate::source::Span::new(100, 200);
            for member in &mut construct.members {
                match member {
                    TypedIrConstructMember::Field { span, .. }
                    | TypedIrConstructMember::Clause { span, .. }
                    | TypedIrConstructMember::Transition { span, .. } => *span = crate::source::Span::new(101, 102),
                }
            }
        }
        assert_eq!(encode_construct_metadata(&relocated).unwrap(), encoded);
        assert_eq!(encoded, encode_construct_metadata(&constructs).unwrap());
    }

    #[test]
    fn construct_metadata_rejects_duplicate_members_before_encoding() {
        let module = crate::parse(
            "module constructs\ncell State\n  value: Int\n  invariant: valid\n",
        ).unwrap();
        let mut constructs = lower_constructs(&module).unwrap();
        constructs[0].members.push(TypedIrConstructMember::Field {
            name: "value".into(),
            ty: TypeKind::Int,
            span: crate::source::Span::new(0, 0),
        });
        let errors = validate_construct_metadata(&constructs).unwrap_err();
        assert!(errors.iter().any(|error| error.contains("duplicate member name: value")));
        assert!(encode_construct_metadata(&constructs).is_err());
    }

    #[test]
    fn lowers_typed_function() {
        let m=crate::parse("module x\nfn main(a: Int) -> Int\n  a + 1\n").unwrap();
        let ir=lower(&m).unwrap();
        assert_eq!(ir.functions[0].return_type,TypeKind::Int);
        assert!(matches!(ir.functions[0].blocks[0].ops[0],TypedIrOp::Expr(_)));
    }
    #[test]
    fn mcts_type_and_resource_bounds_survive_typed_ir_lowering() {
        let module = crate::parse("module mcts_typed\nfn main() -> Int\n  let tree = mcts_tree(16, 4)\n  let root = mcts_root(tree, 1)\n  let child = mcts_expand(tree, root, 2)\n  mcts_update(tree, child, 0.5)\n  mcts_backpropagate(tree, child, 0.25)\n  mcts_child(tree, root, 0)\n  mcts_len(tree)\n").unwrap();
        let typed = lower(&module).unwrap();
        let call = match &typed.functions[0].blocks[0].ops[0] {
            TypedIrOp::Bind { value, .. } => value,
            other => panic!("expected MCTS tree binding, got {other:?}"),
        };
        assert_eq!(call.ty, TypeKind::Named("MctsTree".into()));
        match &call.kind {
            TypedValueKind::Call { callee, args } => {
                assert_eq!(callee, "mcts_tree");
                assert!(matches!(args[0].kind, TypedValueKind::Int(16)));
                assert!(matches!(args[1].kind, TypedValueKind::Int(4)));
            }
            other => panic!("expected constructor call with resource metadata, got {other:?}"),
        }
    }

    #[test]
    fn attention_and_lora_tensor_types_and_rank_metadata_survive_typed_ir() {
        let module = crate::parse(r#"module attention_lora_typed
fn main() -> Tensor<F32>
  let input = tensor([1, 2], [1.0, 2.0])
  let a = tensor([1, 2], [1.0, 1.0])
  let b = tensor([2, 1], [2.0, 3.0])
  let delta = lora_apply(input, a, b, 2.0, 1)
  let q = tensor([1, 2], [1.0, 0.0])
  let k = tensor([1, 2], [1.0, 0.0])
  let v = tensor([1, 1], [2.0])
  attention_block(q, k, v, true)
"#).unwrap();
        let typed = lower(&module).unwrap();
        let ops = &typed.functions[0].blocks[0].ops;
        let lora = match &ops[3] { TypedIrOp::Bind { value, .. } => value, other => panic!("expected LoRA binding, got {other:?}") };
        assert!(matches!(&lora.ty, TypeKind::Generic(name, args) if name == "Tensor" && args.len() == 1 && args[0].kind == TypeKind::Named("F32".into())));
        assert!(matches!(&lora.kind, TypedValueKind::Call { callee, args } if callee == "lora_apply" && matches!(args[4].kind, TypedValueKind::Int(1))));
        let attention = match &ops[7] { TypedIrOp::Expr(value) => value, other => panic!("expected attention expression, got {other:?}") };
        assert!(matches!(&attention.ty, TypeKind::Generic(name, args) if name == "Tensor" && args.len() == 1 && args[0].kind == TypeKind::Named("F32".into())));
        assert!(matches!(&attention.kind, TypedValueKind::Call { callee, args } if callee == "attention_block" && matches!(args[3].kind, TypedValueKind::Bool(true))));
    }


    #[test]
    fn nominal_attention_and_lora_types_and_runtime_metadata_survive_typed_ir() {
        let module = crate::parse(r#"module nominal_typed_ir
fn make_attention() -> AttentionBlock
  attention_block_new(4, 2, 2, true)
fn make_adapter() -> LoRAAdapter
  let a = tensor([1, 2], [1.0, 1.0])
  let b = tensor([3, 1], [1.0, 2.0, 3.0])
  lora_adapter_new(a, b, 2.0, 1)
"#).unwrap();
        let typed = lower(&module).expect("nominal types lower to typed IR");
        assert_eq!(typed.functions[0].return_type, TypeKind::Named("AttentionBlock".into()));
        assert_eq!(typed.functions[1].return_type, TypeKind::Named("LoRAAdapter".into()));
        let attention = match &typed.functions[0].blocks[0].ops[0] {
            TypedIrOp::Expr(value) => value,
            other => panic!("expected AttentionBlock constructor expression, got {other:?}"),
        };
        assert_eq!(attention.ty, TypeKind::Named("AttentionBlock".into()));
        assert!(matches!(&attention.kind, TypedValueKind::Call { callee, args }
            if callee == "attention_block_new"
            && matches!(args[0].kind, TypedValueKind::Int(4))
            && matches!(args[1].kind, TypedValueKind::Int(2))
            && matches!(args[2].kind, TypedValueKind::Int(2))
            && matches!(args[3].kind, TypedValueKind::Bool(true))));
        let adapter = match &typed.functions[1].blocks[0].ops[2] {
            TypedIrOp::Expr(value) => value,
            other => panic!("expected LoRAAdapter constructor expression, got {other:?}"),
        };
        assert_eq!(adapter.ty, TypeKind::Named("LoRAAdapter".into()));
        assert!(matches!(&adapter.kind, TypedValueKind::Call { callee, args }
            if callee == "lora_adapter_new" && matches!(args[3].kind, TypedValueKind::Int(1))));
    }

    #[test]
    fn nominal_moe_router_identity_and_route_metadata_survive_typed_ir() {
        let module = crate::parse(r#"module nominal_moe_typed_ir
fn make_router() -> MoERouter
  moe_router(4, 2)
fn route() -> Int
  let router = moe_router(4, 2)
  let logits = [0.0, 1.0, 1.0, 0.0]
  let mask = [false, false, false, false]
  let routed = moe_route(router, logits, mask)
  routed[0][0]
"#).unwrap();
        crate::sema::check(&module).expect("nominal MoERouter source type-checks");
        let typed = lower(&module).expect("MoERouter lowers to typed IR");
        assert_eq!(typed.functions[0].return_type, TypeKind::Named("MoERouter".into()));
        let constructor = match &typed.functions[0].blocks[0].ops[0] {
            TypedIrOp::Expr(value) => value,
            other => panic!("expected MoERouter constructor expression, got {other:?}"),
        };
        assert_eq!(constructor.ty, TypeKind::Named("MoERouter".into()));
        assert!(matches!(&constructor.kind, TypedValueKind::Call { callee, args }
            if callee == "moe_router" && matches!(args[0].kind, TypedValueKind::Int(4)) && matches!(args[1].kind, TypedValueKind::Int(2))));
        let route = typed.functions[1].blocks[0].ops.iter().find_map(|op| match op {
            TypedIrOp::Bind { value, .. } if matches!(&value.kind, TypedValueKind::Call { callee, .. } if callee == "moe_route") => Some(value),
            _ => None,
        }).expect("typed IR retains the nominal route operation");
        assert!(matches!(&route.kind, TypedValueKind::Call { callee, args }
            if callee == "moe_route" && args.len() == 3 && args[0].ty == TypeKind::Named("MoERouter".into())));
    }

    #[test]
    fn nominal_vector_index_identity_and_search_metadata_survive_typed_ir() {
        let module = crate::parse(r#"module nominal_vector_typed_ir
fn make_index() -> VectorIndex
  vector_index("semantic", 2)
fn search() -> Int
  let index = vector_index("semantic", 2)
  vector_index_insert(index, 3, "semantic", [1.0, 0.0])
  let hits = vector_index_search(index, "semantic", [1.0, 0.0], 1)
  hits[0][0]
"#).unwrap();
        crate::sema::check(&module).expect("nominal VectorIndex source type-checks");
        let typed = lower(&module).expect("VectorIndex lowers to typed IR");
        assert_eq!(typed.functions[0].return_type, TypeKind::Named("VectorIndex".into()));
        let constructor = match &typed.functions[0].blocks[0].ops[0] {
            TypedIrOp::Expr(value) => value,
            other => panic!("expected VectorIndex constructor expression, got {other:?}"),
        };
        assert_eq!(constructor.ty, TypeKind::Named("VectorIndex".into()));
        assert!(matches!(&constructor.kind, TypedValueKind::Call { callee, args }
            if callee == "vector_index" && args.len() == 2));
        let search = typed.functions[1].blocks[0].ops.iter().find_map(|op| match op {
            TypedIrOp::Bind { value, .. } if matches!(&value.kind, TypedValueKind::Call { callee, .. } if callee == "vector_index_search") => Some(value),
            _ => None,
        }).expect("typed IR retains VectorIndex search");
        assert!(matches!(&search.kind, TypedValueKind::Call { callee, args }
            if callee == "vector_index_search" && args.len() == 4 && args[0].ty == TypeKind::Named("VectorIndex".into())));
    }

    #[test]
    fn tokenizer_state_nominal_identity_and_vocabulary_metadata_survive_typed_ir() {
        let module = crate::parse(r#"module tokenizer_typed_ir
fn main() -> TokenizerState
  tokenizer_state(["a", "é"], 4, 16, 0)
"#).unwrap();
        let typed = lower(&module).expect("TokenizerState lowers to typed IR");
        assert_eq!(typed.functions[0].return_type, TypeKind::Named("TokenizerState".into()));
        let op = match &typed.functions[0].blocks[0].ops[0] {
            TypedIrOp::Expr(value) => value,
            other => panic!("expected TokenizerState constructor expression, got {other:?}"),
        };
        assert_eq!(op.ty, TypeKind::Named("TokenizerState".into()));
        assert!(matches!(&op.kind, TypedValueKind::Call { callee, args }
            if callee == "tokenizer_state"
            && matches!(&args[0].kind, TypedValueKind::List(pieces) if pieces.len()==2
                && matches!(&pieces[0].kind, TypedValueKind::String(a) if a=="a")
                && matches!(&pieces[1].kind, TypedValueKind::String(e) if e=="é"))
            && matches!(args[1].kind, TypedValueKind::Int(4))
            && matches!(args[2].kind, TypedValueKind::Int(16))
            && matches!(args[3].kind, TypedValueKind::Int(0))));
    }
}
