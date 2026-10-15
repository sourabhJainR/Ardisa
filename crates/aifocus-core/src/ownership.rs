use std::collections::HashMap;

use crate::{
    ast::{BinaryOp, Block, Expr, ExprKind, Function, Item, Module, StmtKind, Type, TypeKind},
    source::{Diagnostic, Span},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipClass {
    Copy,
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Copy,
    Move,
    SharedBorrow,
    MutableBorrow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    Shared,
    Mutable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipTransition {
    Move,
    BorrowStart(BorrowKind),
    BorrowEnd(BorrowKind),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OwnershipModel {
    pub accesses: HashMap<crate::NodeId, AccessKind>,
    pub transitions: Vec<(crate::NodeId, OwnershipTransition)>,
    pub borrow_regions: Vec<BorrowRegion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowRegion {
    pub local: String,
    pub kind: BorrowKind,
    pub span: Span,
    pub scope_depth: usize,
}

impl BorrowRegion {
    pub fn contains(&self, position: usize) -> bool {
        self.span.start <= position && position < self.span.end
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.local == other.local
            && self.span.start < other.span.end
            && other.span.start < self.span.end
    }
}

impl OwnershipModel {
    pub fn validate_borrow_transitions(
        transitions: &[(crate::NodeId, OwnershipTransition)],
    ) -> Result<(), &'static str> {
        let mut shared = 0usize;
        let mut mutable = false;
        for (_, transition) in transitions {
            match transition {
                OwnershipTransition::Move if shared > 0 || mutable => {
                    return Err("cannot move while a borrow is active");
                }
                OwnershipTransition::Move => {}
                OwnershipTransition::BorrowStart(BorrowKind::Shared) if mutable => {
                    return Err("cannot shared-borrow while a mutable borrow is active");
                }
                OwnershipTransition::BorrowStart(BorrowKind::Shared) => shared += 1,
                OwnershipTransition::BorrowStart(BorrowKind::Mutable) if mutable || shared > 0 => {
                    return Err("cannot create a mutable borrow while another borrow is active");
                }
                OwnershipTransition::BorrowStart(BorrowKind::Mutable) => mutable = true,
                OwnershipTransition::BorrowEnd(BorrowKind::Shared) if shared == 0 => {
                    return Err("shared borrow ended without a matching borrow");
                }
                OwnershipTransition::BorrowEnd(BorrowKind::Shared) => shared -= 1,
                OwnershipTransition::BorrowEnd(BorrowKind::Mutable) if !mutable => {
                    return Err("mutable borrow ended without a matching borrow");
                }
                OwnershipTransition::BorrowEnd(BorrowKind::Mutable) => mutable = false,
            }
        }
        if mutable || shared > 0 {
            return Err("borrow remains active at scope exit");
        }
        Ok(())
    }
}

impl OwnershipClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Move => "move",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Available,
    Moved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessMode {
    Move,
    SharedBorrow,
    MutableBorrow,
}

pub fn infer(module: &Module) -> Result<(), Vec<Diagnostic>> {
    analyze(module).map(|_| ())
}

pub fn analyze(module: &Module) -> Result<OwnershipModel, Vec<Diagnostic>> {
    let mut checker = Checker {
        functions: module
            .items
            .iter()
            .map(|item| {
                let Item::Function(function) = item;
                (function.name.clone(), function.clone())
            })
            .collect(),
        errors: Vec::new(),
        accesses: HashMap::new(),
        transitions: Vec::new(),
        borrow_regions: Vec::new(),
        scope_depth: 0,
    };

    for item in &module.items {
        let Item::Function(function) = item;
        checker.check_function(function);
    }

    if checker.errors.is_empty() {
        OwnershipModel::validate_borrow_transitions(&checker.transitions)
            .map_err(|message| vec![Diagnostic::error("AIF403", message, None)])?;
        validate_borrow_regions(&checker.borrow_regions)
            .map_err(|message| vec![Diagnostic::error("AIF404", message, None)])?;
        Ok(OwnershipModel {
            accesses: checker.accesses,
            transitions: checker.transitions,
            borrow_regions: checker.borrow_regions,
        })
    } else {
        Err(checker.errors)
    }
}

struct Checker {
    functions: HashMap<String, Function>,
    errors: Vec<Diagnostic>,
    accesses: HashMap<crate::NodeId, AccessKind>,
    transitions: Vec<(crate::NodeId, OwnershipTransition)>,
    borrow_regions: Vec<BorrowRegion>,
    scope_depth: usize,
}

impl Checker {
    fn check_function(&mut self, function: &Function) {
        let mut locals = HashMap::new();
        for param in &function.params {
            locals.insert(param.name.clone(), (param.ty.clone(), State::Available));
        }
        self.check_block(&function.body, &mut locals);
    }

    fn check_block(&mut self, block: &Block, locals: &mut HashMap<String, (Type, State)>) {
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Set { name, value } => {
                    self.check_expr(value, locals, AccessMode::Move);
                    if let Some((_, state)) = locals.get_mut(name) {
                        *state = State::Available;
                    }
                }
                StmtKind::SetIndex {
                    collection,
                    index,
                    value,
                } => {
                    self.check_expr(collection, locals, AccessMode::MutableBorrow);
                    self.check_expr(index, locals, AccessMode::Move);
                    self.check_expr(value, locals, AccessMode::Move);
                }
                StmtKind::Let { name, value } => {
                    if let Some(ty) = self.check_expr(value, locals, AccessMode::Move) {
                        locals.insert(name.clone(), (ty, State::Available));
                    }
                }
                StmtKind::Return(value) => {
                    if let Some(value) = value {
                        self.check_expr(value, locals, AccessMode::Move);
                    }
                }
                StmtKind::Expr(expr) => {
                    self.check_expr(expr, locals, AccessMode::Move);
                }
                StmtKind::Scope { body } => {
                    self.scope_depth += 1;
                    let mut scoped = locals.clone();
                    self.check_block(body, &mut scoped);
                    self.scope_depth -= 1;
                    merge_states(locals, &scoped, false);
                }
                StmtKind::Spawn { call, .. } => {
                    self.check_expr(call, locals, AccessMode::Move);
                }
                StmtKind::While { condition, body } => {
                    self.check_expr(condition, locals, AccessMode::Move);
                    self.scope_depth += 1;
                    let mut scoped = locals.clone();
                    self.check_block(body, &mut scoped);
                    self.scope_depth -= 1;
                    merge_states(locals, &scoped, true);
                }
                StmtKind::For { iterable, .. } => {
                    // Treat the source expression conservatively as a shared read; semantic
                    // analysis rejects the loop before code generation.
                    self.check_expr(iterable, locals, AccessMode::SharedBorrow);
                }
                StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
            }
        }
    }

    fn check_expr(
        &mut self,
        expr: &Expr,
        locals: &mut HashMap<String, (Type, State)>,
        mode: AccessMode,
    ) -> Option<Type> {
        match &expr.kind {
            ExprKind::Int(_) => Some(type_node(TypeKind::Int, expr)),
            ExprKind::Float(_) => Some(type_node(TypeKind::Named("F32".into()), expr)),
            ExprKind::Bool(_) => Some(type_node(TypeKind::Bool, expr)),
            ExprKind::String(_) => Some(type_node(TypeKind::String, expr)),
            ExprKind::Name(name) => {
                let Some((ty, state)) = locals.get_mut(name) else {
                    return None;
                };
                if *state == State::Moved {
                    self.errors.push(Diagnostic::error(
                        "AIF400",
                        format!("use of moved value '{name}'"),
                        Some(expr.span),
                    ));
                    return None;
                }
                let access = if ownership_of(ty) == OwnershipClass::Copy {
                    AccessKind::Copy
                } else if mode == AccessMode::SharedBorrow {
                    AccessKind::SharedBorrow
                } else if mode == AccessMode::MutableBorrow {
                    AccessKind::MutableBorrow
                } else {
                    AccessKind::Move
                };
                self.accesses.insert(expr.id, access);
                match access {
                    AccessKind::Move => {
                        self.transitions.push((expr.id, OwnershipTransition::Move));
                        *state = State::Moved;
                    }
                    AccessKind::SharedBorrow => {
                        self.borrow_regions.push(BorrowRegion {
                            local: name.clone(),
                            kind: BorrowKind::Shared,
                            span: expr.span,
                            scope_depth: self.scope_depth,
                        });
                        self.transitions.push((
                            expr.id,
                            OwnershipTransition::BorrowStart(BorrowKind::Shared),
                        ));
                        self.transitions
                            .push((expr.id, OwnershipTransition::BorrowEnd(BorrowKind::Shared)));
                    }
                    AccessKind::MutableBorrow => {
                        self.borrow_regions.push(BorrowRegion {
                            local: name.clone(),
                            kind: BorrowKind::Mutable,
                            span: expr.span,
                            scope_depth: self.scope_depth,
                        });
                        self.transitions.push((
                            expr.id,
                            OwnershipTransition::BorrowStart(BorrowKind::Mutable),
                        ));
                        self.transitions
                            .push((expr.id, OwnershipTransition::BorrowEnd(BorrowKind::Mutable)));
                    }
                    _ => {}
                }
                Some(ty.clone())
            }
            ExprKind::List(items) => {
                let mut element_type = None;
                for item in items {
                    if let Some(ty) = self.check_expr(item, locals, AccessMode::Move) {
                        if element_type.is_none() {
                            element_type = Some(ty);
                        }
                    }
                }
                Some(type_node(
                    TypeKind::List(Box::new(
                        element_type.unwrap_or_else(|| type_node(TypeKind::Unit, expr)),
                    )),
                    expr,
                ))
            }
            ExprKind::Index { collection, index } => {
                self.check_expr(collection, locals, AccessMode::SharedBorrow);
                self.check_expr(index, locals, AccessMode::Move)
            }
            ExprKind::Slice { collection, start, end } => {
                self.check_expr(collection, locals, AccessMode::SharedBorrow);
                self.check_expr(start, locals, AccessMode::Move);
                self.check_expr(end, locals, AccessMode::Move)
            }
            ExprKind::Group(inner) => self.check_expr(inner, locals, mode),
            ExprKind::Binary { op, left, right } => {
                let operand_mode = if matches!(
                    op,
                    BinaryOp::Equal
                        | BinaryOp::NotEqual
                        | BinaryOp::Less
                        | BinaryOp::LessEqual
                        | BinaryOp::Greater
                        | BinaryOp::GreaterEqual
                ) {
                    AccessMode::SharedBorrow
                } else {
                    AccessMode::Move
                };
                let left_type = self.check_expr(left, locals, operand_mode);
                let right_type = self.check_expr(right, locals, operand_mode);
                match op {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Mod => left_type.or(right_type),
                    BinaryOp::Equal
                    | BinaryOp::NotEqual
                    | BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual => Some(type_node(TypeKind::Bool, expr)),
                }
            }
            ExprKind::Call { callee, args } => {
                let ExprKind::Name(name) = &callee.kind else {
                    self.check_expr(callee, locals, AccessMode::Move);
                    for arg in args {
                        self.check_expr(arg, locals, AccessMode::Move);
                    }
                    return None;
                };
                let Some(function) = self.functions.get(name).cloned() else {
                    for (index, arg) in args.iter().enumerate() {
                        let mode = if name == "len" || name == "unwrap" {
                            AccessMode::SharedBorrow
                        } else if name == "push" && index == 0 {
                            AccessMode::MutableBorrow
                        } else {
                            AccessMode::Move
                        };
                        self.check_expr(arg, locals, mode);
                    }
                    return None;
                };
                for (index, arg) in args.iter().enumerate() {
                    let mode = function
                        .params
                        .get(index)
                        .map(|param| {
                            if matches!(param.ty.kind, TypeKind::String) {
                                AccessMode::SharedBorrow
                            } else {
                                AccessMode::Move
                            }
                        })
                        .unwrap_or(AccessMode::Move);
                    self.check_expr(arg, locals, mode);
                }
                function.return_type
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expr(condition, locals, AccessMode::Move);
                let mut then_locals = locals.clone();
                self.check_block(then_branch, &mut then_locals);
                if let Some(else_branch) = else_branch {
                    let mut else_locals = locals.clone();
                    self.check_block(else_branch, &mut else_locals);
                    for (name, (_, state)) in locals.iter_mut() {
                        let then_state = then_locals.get(name).map(|entry| entry.1);
                        let else_state = else_locals.get(name).map(|entry| entry.1);
                        if then_state == Some(State::Moved) || else_state == Some(State::Moved) {
                            *state = State::Moved;
                        }
                    }
                } else {
                    // The branch may execute, so a move in it is conservatively
                    // considered visible after the conditional.
                    merge_states(locals, &then_locals, false);
                }
                Some(type_node(TypeKind::Unit, expr))
            }
        }
    }
}

fn validate_borrow_regions(regions: &[BorrowRegion]) -> Result<(), &'static str> {
    for (index, region) in regions.iter().enumerate() {
        for other in regions.iter().skip(index + 1) {
            if region.overlaps(other)
                && (region.kind == BorrowKind::Mutable || other.kind == BorrowKind::Mutable)
            {
                return Err("overlapping borrow regions conflict");
            }
        }
    }
    Ok(())
}

fn ownership_of(ty: &Type) -> OwnershipClass {
    match ty.kind {
        TypeKind::Int | TypeKind::Bool | TypeKind::Unit => OwnershipClass::Copy,
        TypeKind::String | TypeKind::Named(_) | TypeKind::Generic(_, _) | TypeKind::Result(_, _) | TypeKind::List(_) => {
            OwnershipClass::Move
        }
    }
}

fn merge_states(
    outer: &mut HashMap<String, (Type, State)>,
    inner: &HashMap<String, (Type, State)>,
    loop_body: bool,
) {
    for (name, (_, state)) in outer.iter_mut() {
        if inner.get(name).map(|entry| entry.1) == Some(State::Moved) {
            *state = State::Moved;
        }
        if loop_body && inner.get(name).map(|entry| entry.1) == Some(State::Available) {
            // A loop body may execute zero or many times; availability is safe to
            // retain only when the body itself did not consume the value.
        }
    }
}

fn type_node(kind: TypeKind, expr: &Expr) -> Type {
    Type {
        id: expr.id,
        span: expr.span,
        kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn string_parameters_are_shared_borrows() {
        let source = "module text
fn consume(value: String) -> String
  value
fn reuse(source: String) -> String
  let first = consume(source)
  let second = consume(source)
  first + second
";
        let module = crate::parse(source).unwrap();
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn validates_borrow_conflicts_and_lifetimes() {
        let id = crate::NodeId(1);
        assert!(
            OwnershipModel::validate_borrow_transitions(&[
                (id, OwnershipTransition::BorrowStart(BorrowKind::Shared)),
                (id, OwnershipTransition::BorrowStart(BorrowKind::Shared)),
                (id, OwnershipTransition::BorrowEnd(BorrowKind::Shared)),
                (id, OwnershipTransition::BorrowEnd(BorrowKind::Shared)),
            ])
            .is_ok()
        );
        assert_eq!(
            OwnershipModel::validate_borrow_transitions(&[
                (id, OwnershipTransition::BorrowStart(BorrowKind::Mutable)),
                (id, OwnershipTransition::BorrowStart(BorrowKind::Shared)),
            ]),
            Err("cannot shared-borrow while a mutable borrow is active")
        );
        assert_eq!(
            OwnershipModel::validate_borrow_transitions(&[
                (id, OwnershipTransition::BorrowStart(BorrowKind::Shared)),
                (id, OwnershipTransition::Move),
            ]),
            Err("cannot move while a borrow is active")
        );
    }

    #[test]
    fn treats_primitives_as_copy() {
        let module = parse("module x\nfn f(a: Int) -> Int\n  a + a\n").unwrap();
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn equality_shared_borrows_owned_values() {
        let module =
            parse("module x\nfn f(a: String) -> String\n  let same = a == a\n  a\n").unwrap();
        let model = analyze(&module).unwrap();
        assert!(
            model
                .accesses
                .values()
                .any(|access| *access == AccessKind::SharedBorrow)
        );
    }

    #[test]
    fn conservatively_tracks_move_through_if_without_else() {
        let module = parse(
            "module x\nfn f(flag: Bool, value: String) -> String\n  if flag\n    let consumed = value\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn tracks_move_through_loop_body() {
        let module = parse(
            "module x\nfn f(flag: Bool, value: String) -> String\n  while flag\n    let consumed = value\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn models_push_as_a_mutable_borrow() {
        let module = parse(
            "module x\nfn f(value: Int) -> Int\n  let items = [1]\n  push(items, value)\n  len(items)\n",
        )
        .unwrap();
        let model = analyze(&module).unwrap();
        assert!(
            model
                .accesses
                .values()
                .any(|access| *access == AccessKind::MutableBorrow)
        );
    }

    #[test]
    fn records_borrow_regions_with_scope_depth() {
        let module = parse(
            "module x
fn f(value: String) -> String
  let same = value == value
  value
",
        )
        .unwrap();
        let model = analyze(&module).unwrap();
        assert_eq!(model.borrow_regions.len(), 2);
        assert!(
            model
                .borrow_regions
                .iter()
                .all(|region| region.scope_depth == 0)
        );
        assert!(
            model
                .borrow_regions
                .iter()
                .all(|region| region.contains(region.span.start))
        );
    }

    #[test]
    fn rejects_overlapping_shared_and_mutable_regions() {
        let span = Span::new(10, 20);
        let regions = [
            BorrowRegion {
                local: "value".into(),
                kind: BorrowKind::Shared,
                span,
                scope_depth: 0,
            },
            BorrowRegion {
                local: "value".into(),
                kind: BorrowKind::Mutable,
                span: Span::new(15, 25),
                scope_depth: 0,
            },
        ];
        assert_eq!(
            validate_borrow_regions(&regions),
            Err("overlapping borrow regions conflict")
        );
    }

    #[test]
    fn catches_use_after_move() {
        let module = parse("module x\nfn f(a: String) -> String\n  let b = a\n  a\n").unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn list_binding_move_invalidates_the_source_name() {
        let module = parse(
            "module list_move\nfn f() -> Int\n  let values = [1, 2]\n  let moved = values\n  len(values)\n",
        ).unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn active_borrow_cannot_escape_its_scope() {
        let id = crate::NodeId(2);
        assert_eq!(
            OwnershipModel::validate_borrow_transitions(&[
                (id, OwnershipTransition::BorrowStart(BorrowKind::Shared)),
            ]),
            Err("borrow remains active at scope exit")
        );
    }
}
