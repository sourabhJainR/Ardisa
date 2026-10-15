use std::collections::{HashMap, HashSet};

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
    MaybeMoved,
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
        terminated_exprs: HashSet::new(),
        borrow_context: Vec::new(),
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
    terminated_exprs: HashSet<crate::NodeId>,
    /// Extent of the operation evaluating a borrowed expression.
    borrow_context: Vec<Span>,
}

impl Checker {
    fn check_function(&mut self, function: &Function) {
        let mut locals = HashMap::new();
        for param in &function.params {
            locals.insert(param.name.clone(), (param.ty.clone(), State::Available));
        }
        self.check_block(&function.body, &mut locals);
    }

    fn check_block(&mut self, block: &Block, locals: &mut HashMap<String, (Type, State)>) -> bool {
        let mut reachable = true;
        for stmt in &block.stmts {
            if !reachable {
                break;
            }
            match &stmt.kind {
                StmtKind::Set { name, value } => {
                    self.check_expr(value, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&value.id) {
                        reachable = false;
                    } else if let Some((_, state)) = locals.get_mut(name) {
                        *state = State::Available;
                    }
                }
                StmtKind::SetIndex {
                    collection,
                    index,
                    value,
                } => {
                    // The target collection is mutably borrowed for the full
                    // indexed assignment operation, not only while its name is read.
                    self.check_expr_in_region(
                        collection,
                        locals,
                        AccessMode::MutableBorrow,
                        stmt.span,
                    );
                    if self.terminated_exprs.contains(&collection.id) {
                        reachable = false;
                        continue;
                    }
                    self.check_expr(index, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&index.id) {
                        reachable = false;
                        continue;
                    }
                    self.check_expr(value, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&value.id) {
                        reachable = false;
                    }
                }
                StmtKind::Let { name, value } => {
                    let shadows_existing = locals.contains_key(name);
                    let value_type = self.check_expr(value, locals, AccessMode::Move);
                    if shadows_existing {
                        self.errors.push(Diagnostic::error(
                            "AIF406",
                            format!("shadowing local '{name}' is unsupported until ownership bindings have stable identities"),
                            Some(stmt.span),
                        ));
                    }
                    if self.terminated_exprs.contains(&value.id) {
                        reachable = false;
                    } else if !shadows_existing {
                        if let Some(ty) = value_type {
                            locals.insert(name.clone(), (ty, State::Available));
                        }
                    }
                }
                StmtKind::Return(value) => {
                    if let Some(value) = value {
                        self.check_expr(value, locals, AccessMode::Move);
                    }
                    reachable = false;
                }
                StmtKind::Expr(expr) => {
                    self.check_expr(expr, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&expr.id) {
                        reachable = false;
                    }
                }
                StmtKind::Scope { body } => {
                    self.scope_depth += 1;
                    let mut scoped = locals.clone();
                    let scope_reachable = self.check_block(body, &mut scoped);
                    self.scope_depth -= 1;
                    // Only a scope that can fall through contributes state to the
                    // following statement. A returning scope terminates this path.
                    if scope_reachable {
                        merge_scope_states(locals, &scoped);
                    } else {
                        reachable = false;
                    }
                }
                StmtKind::Spawn { call, .. } => {
                    self.check_expr(call, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&call.id) {
                        reachable = false;
                    }
                }
                StmtKind::While { condition, body } => {
                    self.check_expr(condition, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&condition.id) {
                        reachable = false;
                        continue;
                    }
                    self.scope_depth += 1;
                    let mut scoped = locals.clone();
                    self.check_block(body, &mut scoped);
                    self.scope_depth -= 1;
                    merge_states(locals, &scoped);
                }
                StmtKind::For { name, iterable, body } => {
                    // Binding identity is not represented in the current checker.
                    // Reject shadowing here rather than letting the loop-local name
                    // overwrite an outer binding in the string-keyed state map.
                    let shadows_existing = locals.contains_key(name);
                    if shadows_existing {
                        self.errors.push(Diagnostic::error(
                            "AIF303",
                            format!("loop binding '{name}' shadows an existing local"),
                            Some(stmt.span),
                        ));
                    }
                    // Iteration consumes an owned List<T>. This aligns static move state
                    // with the native loop's single evaluation of the iterable.
                    let iterable_type = self.check_expr(iterable, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&iterable.id) {
                        reachable = false;
                        continue;
                    }
                    let element_type = iterable_type.and_then(|ty| match ty.kind {
                        TypeKind::List(element) => Some(*element),
                        _ => None,
                    });
                    let mut scoped = locals.clone();
                    if !shadows_existing {
                        if let Some(element) = element_type {
                            scoped.insert(name.clone(), (element, State::Available));
                        }
                    }
                    self.scope_depth += 1;
                    self.check_block(body, &mut scoped);
                    self.scope_depth -= 1;
                    // Merge only pre-existing bindings. A rejected shadow binding is
                    // never allowed to overwrite an outer local's state.
                    if !shadows_existing {
                        scoped.remove(name);
                    }
                    merge_states(locals, &scoped);
                }
                StmtKind::Join { .. } | StmtKind::Cancel { .. } => {}
            }
        }
        reachable
    }

    fn check_expr_in_region(
        &mut self,
        expr: &Expr,
        locals: &mut HashMap<String, (Type, State)>,
        mode: AccessMode,
        span: Span,
    ) -> Option<Type> {
        self.borrow_context.push(span);
        let result = self.check_expr(expr, locals, mode);
        self.borrow_context.pop();
        result
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
                if *state != State::Available {
                    let message = if *state == State::Moved {
                        format!("use of moved value '{name}'")
                    } else {
                        format!("use of possibly moved value '{name}'")
                    };
                    self.errors.push(Diagnostic::error("AIF400", message, Some(expr.span)));
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
                let operation_span = self
                    .borrow_context
                    .last()
                    .copied()
                    .unwrap_or(expr.span);
                if access == AccessKind::Move
                    && self.borrow_regions.iter().any(|region| {
                        region.local == *name
                            && region.span.start < expr.span.end
                            && expr.span.start < region.span.end
                    })
                {
                    self.errors.push(Diagnostic::error(
                        "AIF405",
                        format!("cannot move '{name}' while a borrow is active in this expression"),
                        Some(expr.span),
                    ));
                    return None;
                }
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
                            span: operation_span,
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
                            span: operation_span,
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
                    if self.terminated_exprs.contains(&item.id) {
                        self.terminated_exprs.insert(expr.id);
                        break;
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
                let collection_type =
                    self.check_expr(collection, locals, AccessMode::SharedBorrow);
                if self.terminated_exprs.contains(&collection.id) {
                    self.terminated_exprs.insert(expr.id);
                    return collection_type;
                }
                self.check_expr(index, locals, AccessMode::Move);
                if self.terminated_exprs.contains(&index.id) {
                    self.terminated_exprs.insert(expr.id);
                    return collection_type;
                }
                collection_type.and_then(|ty| match ty.kind {
                    TypeKind::List(element) => Some(*element),
                    // String indexing is byte-based in the current language contract.
                    TypeKind::String => Some(type_node(TypeKind::Int, expr)),
                    _ => None,
                })
            }
            ExprKind::Slice { collection, start, end } => {
                let collection_type =
                    self.check_expr(collection, locals, AccessMode::SharedBorrow);
                if self.terminated_exprs.contains(&collection.id) {
                    self.terminated_exprs.insert(expr.id);
                    return collection_type;
                }
                self.check_expr(start, locals, AccessMode::Move);
                if self.terminated_exprs.contains(&start.id) {
                    self.terminated_exprs.insert(expr.id);
                    return collection_type;
                }
                self.check_expr(end, locals, AccessMode::Move);
                if self.terminated_exprs.contains(&end.id) {
                    self.terminated_exprs.insert(expr.id);
                }
                // Slices are owned copies with the same collection type, not views.
                collection_type
            }
            ExprKind::Group(inner) => {
                let ty = self.check_expr(inner, locals, mode);
                if self.terminated_exprs.contains(&inner.id) {
                    self.terminated_exprs.insert(expr.id);
                }
                ty
            },
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
                let left_type = if operand_mode == AccessMode::SharedBorrow {
                    self.check_expr_in_region(left, locals, operand_mode, expr.span)
                } else {
                    self.check_expr(left, locals, operand_mode)
                };
                if self.terminated_exprs.contains(&left.id) {
                    self.terminated_exprs.insert(expr.id);
                    return left_type;
                }
                let right_type = if operand_mode == AccessMode::SharedBorrow {
                    self.check_expr_in_region(right, locals, operand_mode, expr.span)
                } else {
                    self.check_expr(right, locals, operand_mode)
                };
                if self.terminated_exprs.contains(&right.id) {
                    self.terminated_exprs.insert(expr.id);
                    return left_type.or(right_type);
                }
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
                        self.check_expr_in_region(arg, locals, mode, expr.span);
                        if self.terminated_exprs.contains(&arg.id) {
                            self.terminated_exprs.insert(expr.id);
                            return None;
                        }
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
                    self.check_expr_in_region(arg, locals, mode, expr.span);
                    if self.terminated_exprs.contains(&arg.id) {
                        self.terminated_exprs.insert(expr.id);
                        return function.return_type;
                    }
                }
                function.return_type
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expr(condition, locals, AccessMode::Move);
                if self.terminated_exprs.contains(&condition.id) {
                    self.terminated_exprs.insert(expr.id);
                    return Some(type_node(TypeKind::Unit, expr));
                }
                let mut then_locals = locals.clone();
                let then_reachable = self.check_block(then_branch, &mut then_locals);
                if let Some(else_branch) = else_branch {
                    let mut else_locals = locals.clone();
                    let else_reachable = self.check_block(else_branch, &mut else_locals);
                    match (then_reachable, else_reachable) {
                        (true, true) => {
                            for (name, (_, state)) in locals.iter_mut() {
                                if let (Some(then_state), Some(else_state)) = (
                                    then_locals.get(name).map(|entry| entry.1),
                                    else_locals.get(name).map(|entry| entry.1),
                                ) {
                                    *state = join_state(then_state, else_state);
                                }
                            }
                        }
                        (true, false) => merge_scope_states(locals, &then_locals),
                        (false, true) => merge_scope_states(locals, &else_locals),
                        (false, false) => { self.terminated_exprs.insert(expr.id); }
                    }
                } else if then_reachable {
                    // The implicit false path is always reachable, so join it with
                    // the then path. If the then branch returns, only the false path
                    // reaches the next statement and its original state is retained.
                    merge_states(locals, &then_locals);
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
) {
    for (name, (_, state)) in outer.iter_mut() {
        if let Some((_, inner_state)) = inner.get(name) {
            *state = join_state(*state, *inner_state);
        }
    }
}

fn join_state(left: State, right: State) -> State {
    if left == right { left } else { State::MaybeMoved }
}

fn merge_scope_states(
    outer: &mut HashMap<String, (Type, State)>,
    inner: &HashMap<String, (Type, State)>,
) {
    for (name, (_, state)) in outer.iter_mut() {
        if let Some((_, inner_state)) = inner.get(name) {
            *state = *inner_state;
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
    fn rejects_reading_an_item_while_its_collection_is_mutably_borrowed() {
        let module = parse(
            "module alias_conflict\nfn f(items: List<List<Int>>) -> Int\n  push(items, items[0])\n  0\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF404"),
            "the mutable collection borrow and nested shared read overlap for the whole call"
        );
    }

    #[test]
    fn ends_nested_read_borrows_before_a_later_move_in_the_same_call() {
        let module = parse(
            "module nested_read_then_move\nfn consume(flag: Bool, values: List<Int>) -> Int\n  len(values)\nfn f(values: List<Int>) -> Int\n  consume(values == values, values)\n",
        )
        .unwrap();
        assert!(
            infer(&module).is_ok(),
            "a read-only comparison completes before the later argument moves the collection"
        );
    }

    #[test]
    fn permits_separate_read_only_calls_on_the_same_owned_value() {
        let module = parse(
            "module separate_reads\nfn f(items: List<Int>) -> Int\n  len(items) + len(items)\n",
        )
        .unwrap();
        assert!(infer(&module).is_ok());
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

    #[test]
    fn rejects_loop_binding_shadowing_until_bindings_have_stable_identity() {
        let module = parse(
            "module loop_shadow\nfn f(values: List<String>, item: String) -> String\n  for item in values\n    len(item)\n  item\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF303"),
            "loop binding shadowing must fail closed, got {errors:?}"
        );
    }

    #[test]
    fn moving_a_loop_binding_does_not_move_unrelated_outer_bindings() {
        let module = parse(
            "module loop_binding\nfn f(values: List<String>, value: String) -> String\n  for item in values\n    let consumed = item\n  value\n",
        ).unwrap();
        assert!(
            infer(&module).is_ok(),
            "consuming a loop-local binding must not mark an unrelated outer value moved"
        );
        assert!(crate::sema::check(&module).is_ok());
    }

    #[test]
    fn a_move_on_one_if_branch_is_visible_after_the_join() {
        let module = parse(
            "module branch_join\nfn f(flag: Bool, value: String) -> String\n  if flag\n    let consumed = value\n  else\n    let other = \"safe\"\n  value\n",
        ).unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn assignment_reinitializes_a_previously_moved_local() {
        let module = parse(
            "module reinit\nfn f(value: String) -> String\n  let consumed = value\n  set value = \"restored\"\n  value\n",
        ).unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn indexing_infers_the_element_ownership_class() {
        let module = parse(
            "module index_move\nfn f(values: List<String>) -> String\n  let selected = values[0]\n  let moved = selected\n  selected\n",
        ).unwrap();
        assert!(crate::sema::check(&module).is_ok());
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn slicing_preserves_the_owned_collection_type_for_move_analysis() {
        let module = parse(
            "module slice_move\nfn f(values: List<String>) -> Int\n  let selected = values[0:1]\n  let moved = selected\n  len(selected)\n",
        ).unwrap();
        assert!(crate::sema::check(&module).is_ok());
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }



    #[test]
    fn scope_reinitialization_restores_outer_local_availability() {
        let module = parse(
            "module scope_reinit\nfn f(value: String) -> String\n  let consumed = value\n  scope\n    set value = \"restored\"\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn reinitialization_on_both_if_paths_restores_availability_after_join() {
        let module = parse(
            "module branch_reinit\nfn f(flag: Bool, value: String) -> String\n  let consumed = value\n  if flag\n    set value = \"left\"\n  else\n    set value = \"right\"\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn reinitialization_on_only_one_if_path_remains_path_dependent() {
        let module = parse(
            "module branch_partial_reinit\nfn f(flag: Bool, value: String) -> String\n  let consumed = value\n  if flag\n    set value = \"restored\"\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
        assert!(errors.iter().any(|error| error.message.contains("possibly moved")));
    }


    #[test]
    fn loop_reinitialization_does_not_assume_the_loop_executes() {
        let module = parse(
            "module loop_reinit\nfn f(flag: Bool, value: String) -> String\n  let consumed = value\n  while flag\n    set value = \"restored\"\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
        assert!(errors.iter().any(|error| error.message.contains("possibly moved")));
    }

    #[test]
    fn a_move_on_both_if_branches_is_definitely_moved_after_join() {
        let module = parse(
            "module branch_both_move\nfn f(flag: Bool, value: String) -> String\n  if flag\n    let left = value\n  else\n    let right = value\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
        assert!(errors.iter().any(|error| error.message.contains("use of moved value")));
    }

    #[test]
    fn ownership_join_ignores_a_branch_that_returns_early() {
        let module = parse(
            "module early_return_join\nfn f(flag: Bool, value: String) -> String\n  if flag\n    return value\n  else\n    set value = \"restored\"\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(
            infer(&module).is_ok(),
            "the returning branch must not poison the state of the only path that reaches the final use"
        );
    }

    #[test]
    fn propagates_termination_through_let_initializer() {
        let mut module = parse(
            "module terminated_binding\nfn f(flag: Bool, value: String) -> String\n  if flag\n    return value\n  else\n    return value\n  value\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let StmtKind::Expr(terminating_if) = &function.body.stmts[0].kind else {
            panic!("expected if expression statement");
        };
        let terminating_if = terminating_if.clone();
        function.body.stmts[0].kind = StmtKind::Let {
            name: "unreachable".into(),
            value: terminating_if,
        };
        let StmtKind::Expr(final_expr) = &function.body.stmts.last().unwrap().kind else {
            panic!("expected final expression");
        };
        let final_expr_id = final_expr.id;
        let model = analyze(&module).unwrap();
        assert!(
            !model.accesses.contains_key(&final_expr_id),
            "an expression after an initializer that always returns must be unreachable to ownership analysis"
        );
    }

    #[test]
    fn propagates_termination_through_assignment_value() {
        let mut module = parse(
            "module terminated_assignment\nfn f(flag: Bool, value: String) -> String\n  let result = \"start\"\n  if flag\n    return value\n  else\n    return value\n  value\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let StmtKind::Expr(terminating_if) = &function.body.stmts[1].kind else {
            panic!("expected if expression statement");
        };
        let terminating_if = terminating_if.clone();
        function.body.stmts[1].kind = StmtKind::Set {
            name: "result".into(),
            value: terminating_if,
        };
        let StmtKind::Expr(final_expr) = &function.body.stmts.last().unwrap().kind else {
            panic!("expected final expression");
        };
        let final_expr_id = final_expr.id;
        let model = analyze(&module).unwrap();
        assert!(
            !model.accesses.contains_key(&final_expr_id),
            "an expression after an assignment that always returns must be unreachable to ownership analysis"
        );
    }

    #[test]
    fn propagates_termination_through_grouped_expression() {
        let mut module = parse(
            "module terminated_group\nfn f(flag: Bool, value: String) -> String\n  if flag\n    return value\n  else\n    return value\n  value\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let StmtKind::Expr(terminating_if) = &function.body.stmts[0].kind else {
            panic!("expected if expression statement");
        };
        let terminating_if = terminating_if.clone();
        let grouped = Expr {
            id: crate::NodeId(u64::MAX),
            span: terminating_if.span,
            kind: ExprKind::Group(Box::new(terminating_if)),
        };
        function.body.stmts[0].kind = StmtKind::Let {
            name: "unreachable".into(),
            value: grouped,
        };
        let StmtKind::Expr(final_expr) = &function.body.stmts.last().unwrap().kind else {
            panic!("expected final expression");
        };
        let final_expr_id = final_expr.id;
        let model = analyze(&module).unwrap();
        assert!(
            !model.accesses.contains_key(&final_expr_id),
            "grouping must preserve a non-fallthrough expression's termination state"
        );
    }

    #[test]
    fn termination_in_binary_operand_skips_later_operands_and_statements() {
        let mut module = parse(
            "module nested_termination\nfn f(flag: Bool, value: String) -> String\n  if flag\n    return value\n  else\n    return value\n  value\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let StmtKind::Expr(terminating_if) = &function.body.stmts[0].kind else {
            panic!("expected if expression");
        };
        let terminating_if = terminating_if.clone();
        function.body.stmts[0].kind = StmtKind::Let {
            name: "unreachable".into(),
            value: Expr {
                id: crate::NodeId(9_000_001),
                span: terminating_if.span,
                kind: ExprKind::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(terminating_if),
                    right: Box::new(Expr {
                        id: crate::NodeId(9_000_002),
                        span: function.body.stmts[1].span,
                        kind: ExprKind::Name("value".into()),
                    }),
                },
            },
        };
        let StmtKind::Expr(final_expr) = &function.body.stmts.last().unwrap().kind else {
            panic!("expected final expression");
        };
        let final_expr_id = final_expr.id;
        let model = analyze(&module).unwrap();
        assert!(
            !model.accesses.contains_key(&final_expr_id),
            "statements after a nested non-fallthrough operand must be unreachable"
        );
    }

    #[test]
    fn termination_in_index_assignment_value_makes_following_statements_unreachable() {
        let mut module = parse(
            "module terminated_index_assignment\nfn f(flag: Bool, value: String) -> String\n  let values = [1]\n  if flag\n    return value\n  else\n    return value\n  value\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let StmtKind::Expr(terminating_if) = &function.body.stmts[1].kind else {
            panic!("expected if expression");
        };
        let terminating_if = terminating_if.clone();
        let span = terminating_if.span;
        function.body.stmts[1].kind = StmtKind::SetIndex {
            collection: Expr {
                id: crate::NodeId(9_100_001),
                span,
                kind: ExprKind::Name("values".into()),
            },
            index: Expr {
                id: crate::NodeId(9_100_002),
                span,
                kind: ExprKind::Int(0),
            },
            value: terminating_if,
        };
        let StmtKind::Expr(final_expr) = &function.body.stmts.last().unwrap().kind else {
            panic!("expected final expression");
        };
        let final_expr_id = final_expr.id;
        let model = analyze(&module).unwrap();
        assert!(
            !model.accesses.contains_key(&final_expr_id),
            "statements after a non-fallthrough indexed assignment value must be unreachable"
        );
    }



    #[test]
    fn rejects_shadowing_until_binding_identity_is_tracked() {
        let module = parse(
            "module reject_shadow\nfn f(value: String) -> String\n  scope\n    let value = \"fresh\"\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF406"));
    }

    #[test]
    fn nested_scope_shadow_cannot_resurrect_moved_outer_binding() {
        let module = parse(
            "module scope_shadow_move\nfn f(value: String) -> String\n  let consumed = value\n  scope\n    let value = \"fresh\"\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn conditional_shadow_cannot_resurrect_moved_outer_binding() {
        let module = parse(
            "module branch_shadow_move\nfn f(flag: Bool, value: String) -> String\n  let consumed = value\n  if flag\n    let value = \"fresh\"\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn loop_body_shadow_cannot_resurrect_moved_outer_binding() {
        let module = parse(
            "module loop_shadow_move\nfn f(flag: Bool, value: String) -> String\n  let consumed = value\n  while flag\n    let value = \"fresh\"\n  value\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF400"));
    }

    #[test]
    fn rejects_reading_same_collection_during_indexed_mutation() {
        let mut module = parse(
            "module index_assignment_alias\nfn f(values: List<List<Int>>) -> Int\n  0\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let span = function.body.stmts[0].span;
        let name = |id, span| Expr {
            id: crate::NodeId(id),
            span,
            kind: ExprKind::Name("values".into()),
        };
        let integer = |id, span| Expr {
            id: crate::NodeId(id),
            span,
            kind: ExprKind::Int(0),
        };
        function.body.stmts[0].kind = StmtKind::SetIndex {
            collection: name(9_200_001, span),
            index: integer(9_200_002, span),
            value: Expr {
                id: crate::NodeId(9_200_003),
                span,
                kind: ExprKind::Index {
                    collection: Box::new(name(9_200_004, span)),
                    index: Box::new(integer(9_200_005, span)),
                },
            },
        };
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF404"),
            "the mutable target borrow must overlap reads from the same collection"
        );
    }



    #[test]
    fn indexed_assignment_rejects_reading_the_mutably_borrowed_collection() {
        let module = parse(
            "module indexed_assignment_alias
fn f(values: List<List<Int>>) -> Int
  set values[0] = values[0]
  0
",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF404"),
            "the mutable borrow must span the entire indexed assignment, including its value"
        );
    }

    #[test]
    fn indexed_assignment_allows_an_independent_value() {
        let module = parse(
            "module indexed_assignment_safe
fn f(values: List<Int>) -> Int
  set values[0] = 7
  len(values)
",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(infer(&module).is_ok());
    }

    fn run_owned_collection_case(source: &str) -> crate::NativeValue {
        let compiled = crate::compile_source(source)
            .expect("source should compile through the complete native pipeline");
        let program = crate::native::decode_program(&compiled.artifact)
            .expect("compiled artifact should decode");
        crate::native::run_program_with_limits(
            &program,
            "f",
            &[crate::NativeValue::List(vec![
                crate::NativeValue::String("item".into()),
            ])],
            crate::ExecutionLimits::default(),
        )
        .expect("native execution should succeed")
    }

    #[test]
    fn moving_an_indexed_element_does_not_move_the_source_collection() {
        let module = parse(
            "module indexed_value_is_owned\nfn f(values: List<String>) -> Int\n  let selected = values[0]\n  let consumed = selected\n  len(values)\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(
            infer(&module).is_ok(),
            "indexing yields an owned element in the current runtime contract; moving it must not invalidate the source collection"
        );
    }

    #[test]
    fn moving_an_owned_slice_does_not_move_the_source_collection() {
        let module = parse(
            "module slice_value_is_owned\nfn f(values: List<String>) -> Int\n  let selected = values[0:1]\n  let consumed = selected\n  len(values)\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(
            infer(&module).is_ok(),
            "slicing currently creates an owned copy rather than a borrowed view"
        );
    }

    #[test]
    fn indexed_element_move_preserves_source_collection_at_runtime() {
        let source = "module indexed_runtime
fn f(values: List<String>) -> Int
  let selected = values[0]
  let consumed = selected
  len(values)
";
        assert_eq!(
            run_owned_collection_case(source),
            crate::NativeValue::Int(1),
            "moving the copied indexed element must not consume the source list"
        );
    }

    #[test]
    fn slice_result_move_preserves_source_collection_at_runtime() {
        let source = "module slice_runtime
fn f(values: List<String>) -> Int
  let selected = values[0:1]
  let consumed = selected
  len(values)
";
        assert_eq!(
            run_owned_collection_case(source),
            crate::NativeValue::Int(1),
            "moving the owned slice must not consume the source list"
        );
    }

}
