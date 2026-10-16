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
    /// Stable identity of the declaration, independent of its source-level name.
    pub binding_id: BindingId,
    pub local: String,
    pub kind: BorrowKind,
    /// Place path accessed within the binding; an empty path borrows the whole value.
    pub path: Vec<PlaceProjection>,
    pub span: Span,
    pub scope_depth: usize,
}

impl BorrowRegion {
    pub fn contains(&self, position: usize) -> bool {
        self.span.start <= position && position < self.span.end
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.binding_id == other.binding_id
            && paths_may_alias(&self.path, &other.path)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BindingId(pub crate::NodeId);

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PlaceProjection {
    Index(usize),
    /// An unknown index may alias any element at this depth.
    DynamicIndex,
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
    /// Borrow the aggregate only to access one projection; other moved projections
    /// remain independently tracked.
    PathBorrow,
    PathMutableBorrow,
}

#[derive(Debug, Clone)]
struct LocalBinding {
    id: BindingId,
    ty: Type,
    state: State,
    path_states: HashMap<Vec<PlaceProjection>, State>,
}

impl LocalBinding {
    fn new(id: BindingId, ty: Type) -> Self {
        Self { id, ty, state: State::Available, path_states: HashMap::new() }
    }

    fn path_state(&self, path: &[PlaceProjection]) -> State {
        self.joined_path_state(path, false)
    }

    /// State needed to traverse an aggregate projection to reach a child.
    /// A moved descendant does not make its parent unusable for disjoint child
    /// access, but a moved ancestor or a potentially aliasing dynamic path does.
    fn path_state_for_projection(&self, path: &[PlaceProjection]) -> State {
        self.joined_path_state(path, true)
    }

    fn joined_path_state(&self, path: &[PlaceProjection], prefixes_only: bool) -> State {
        let mut result: Option<State> = None;
        for (existing, state) in &self.path_states {
            if *state == State::Available
                || (prefixes_only && existing.len() > path.len())
                || !paths_may_alias(existing, path)
            {
                continue;
            }
            result = Some(match result {
                Some(previous) => join_state(previous, *state),
                None => *state,
            });
        }
        result.unwrap_or(State::Available)
    }

    fn has_moved_paths(&self) -> bool {
        self.path_states.values().any(|state| *state != State::Available)
    }

    fn set_path_state(&mut self, path: Vec<PlaceProjection>, state: State) {
        self.path_states.retain(|existing, _| !existing.starts_with(&path) || existing == &path);
        if state == State::Available {
            self.path_states.remove(&path);
        } else {
            self.path_states.insert(path, state);
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Locals {
    visible: HashMap<String, BindingId>,
    bindings: HashMap<BindingId, LocalBinding>,
}

impl Locals {
    fn insert(&mut self, name: String, binding: LocalBinding) {
        self.visible.insert(name, binding.id);
        self.bindings.insert(binding.id, binding);
    }
    fn get(&self, name: &str) -> Option<&LocalBinding> {
        self.visible.get(name).and_then(|id| self.bindings.get(id))
    }
    fn get_mut(&mut self, name: &str) -> Option<&mut LocalBinding> {
        let id = *self.visible.get(name)?;
        self.bindings.get_mut(&id)
    }
    fn contains_key(&self, name: &str) -> bool { self.visible.contains_key(name) }
    fn remove(&mut self, name: &str) -> Option<BindingId> { self.visible.remove(name) }
}

fn index_projection(index: &Expr) -> PlaceProjection {
    match index.kind {
        ExprKind::Int(value) if value >= 0 => PlaceProjection::Index(value as usize),
        _ => PlaceProjection::DynamicIndex,
    }
}

fn place_path(expr: &Expr) -> Option<(String, Vec<PlaceProjection>)> {
    match &expr.kind {
        ExprKind::Name(name) => Some((name.clone(), Vec::new())),
        ExprKind::Index { collection, index } => {
            let (name, mut path) = place_path(collection)?;
            path.push(index_projection(index));
            Some((name, path))
        }
        _ => None,
    }
}

fn paths_may_alias(left: &[PlaceProjection], right: &[PlaceProjection]) -> bool {
    for (a, b) in left.iter().zip(right.iter()) {
        match (a, b) {
            (PlaceProjection::DynamicIndex, _) | (_, PlaceProjection::DynamicIndex) => return true,
            (a, b) if a != b => return false,
            _ => {}
        }
    }
    true
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
        place_context: Vec::new(),
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
    /// Most-specific place currently being evaluated, preserved through nested projections.
    place_context: Vec<(String, Vec<PlaceProjection>)>,
}

impl Checker {
    fn check_function(&mut self, function: &Function) {
        let mut locals = Locals::default();
        for param in &function.params {
            locals.insert(
                param.name.clone(),
                LocalBinding::new(BindingId(param.id), param.ty.clone()),
            );
        }
        self.check_block(&function.body, &mut locals);
    }

    fn check_block(&mut self, block: &Block, locals: &mut Locals) -> bool {
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
                    } else if let Some(binding) = locals.get_mut(name) {
                        binding.state = State::Available;
                        binding.path_states.clear();
                    }
                }
                StmtKind::SetIndex {
                    collection,
                    index,
                    value,
                } => {
                    // The mutable place is the collection projection plus its target index.
                    // Carry that full path while analyzing the intermediate collection so
                    // alias checks can distinguish a sibling path from the actual target.
                    let has_target_place = if let Some((name, mut path)) = place_path(collection) {
                        path.push(index_projection(index));
                        self.place_context.push((name, path));
                        true
                    } else {
                        false
                    };
                    self.check_expr_in_region(
                        collection, locals, AccessMode::PathMutableBorrow, stmt.span,
                    );
                    if has_target_place {
                        self.place_context.pop();
                    }
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
                        continue;
                    }
                    if let Some((name, mut path)) = place_path(collection) {
                        if !matches!(index_projection(index), PlaceProjection::DynamicIndex) {
                            path.push(index_projection(index));
                            if let Some(binding) = locals.get_mut(&name) {
                                binding.set_path_state(path, State::Available);
                            }
                        }
                    }
                }
                StmtKind::Let { name, value } => {
                    let value_type = self.check_expr(value, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&value.id) {
                        reachable = false;
                    } else if let Some(ty) = value_type {
                        // The declaration node is a stable identity. Reusing a source
                        // name creates a new binding, not an alias for the old one.
                        locals.insert(name.clone(), LocalBinding::new(BindingId(stmt.id), ty));
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
                    // Evaluate the iterable before snapshotting the outer binding:
                    // the iterable itself may move a binding with the same name as
                    // the loop variable (for example, \`for values in values\`).
                    let had_previous_binding = locals.contains_key(name);
                    let iterable_type = self.check_expr(iterable, locals, AccessMode::Move);
                    if self.terminated_exprs.contains(&iterable.id) {
                        reachable = false;
                        continue;
                    }
                    let previous_binding = had_previous_binding.then(|| locals.get(name).cloned()).flatten();
                    let element_type = iterable_type.and_then(|ty| match ty.kind {
                        TypeKind::List(element) => Some(*element),
                        _ => None,
                    });
                    let mut scoped = locals.clone();
                    if let Some(element) = element_type {
                        scoped.insert(name.clone(), LocalBinding::new(BindingId(stmt.id), element));
                    }
                    self.scope_depth += 1;
                    self.check_block(body, &mut scoped);
                    self.scope_depth -= 1;
                    if let Some(previous) = previous_binding {
                        scoped.insert(name.clone(), previous);
                    } else {
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
        locals: &mut Locals,
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
        locals: &mut Locals,
        mode: AccessMode,
    ) -> Option<Type> {
        // The outermost projection is the most precise place for this access.
        // Do not replace it with an intermediate parent projection while recursively
        // evaluating a chain such as values[0][1].
        let place = place_path(expr);
        let should_push_place = match (place.as_ref(), self.place_context.last()) {
            (Some((root, _)), Some((current_root, _))) if root == current_root => false,
            (Some(_), _) => true,
            (None, _) => false,
        };
        if should_push_place {
            self.place_context.push(place.expect("place was checked above"));
        }
        let result = self.check_expr_inner(expr, locals, mode);
        if should_push_place {
            self.place_context.pop();
        }
        result
    }

    fn check_expr_inner(
        &mut self,
        expr: &Expr,
        locals: &mut Locals,
        mode: AccessMode,
    ) -> Option<Type> {
        match &expr.kind {
            ExprKind::Int(_) => Some(type_node(TypeKind::Int, expr)),
            ExprKind::Float(_) => Some(type_node(TypeKind::Named("F32".into()), expr)),
            ExprKind::Bool(_) => Some(type_node(TypeKind::Bool, expr)),
            ExprKind::String(_) => Some(type_node(TypeKind::String, expr)),
            ExprKind::Name(name) => {
                let Some(binding) = locals.get_mut(name) else {
                    return None;
                };
                let blocked_by_partial_move = !matches!(mode, AccessMode::PathBorrow | AccessMode::PathMutableBorrow)
                    && binding.has_moved_paths();
                if binding.state != State::Available || blocked_by_partial_move {
                    let message = if binding.state == State::Moved {
                        format!("use of moved value '{name}'")
                    } else if binding.state == State::MaybeMoved {
                        format!("use of possibly moved value '{name}'")
                    } else if blocked_by_partial_move
                        && binding.path_states.values().any(|state| *state == State::MaybeMoved)
                    {
                        format!("use of possibly moved or partially moved value '{name}'")
                    } else if blocked_by_partial_move {
                        format!("use of partially moved value '{name}'")
                    } else {
                        format!("use of possibly moved value '{name}'")
                    };
                    self.errors.push(Diagnostic::error("AIF400", message, Some(expr.span)));
                    return None;
                }
                let ty = &binding.ty;
                let binding_id = binding.id;
                let access = if ownership_of(ty) == OwnershipClass::Copy {
                    AccessKind::Copy
                } else if mode == AccessMode::SharedBorrow || mode == AccessMode::PathBorrow {
                    AccessKind::SharedBorrow
                } else if mode == AccessMode::MutableBorrow || mode == AccessMode::PathMutableBorrow {
                    AccessKind::MutableBorrow
                } else {
                    AccessKind::Move
                };
                let operation_span = self
                    .borrow_context
                    .last()
                    .copied()
                    .unwrap_or(expr.span);
                let borrow_path = self
                    .place_context
                    .last()
                    .filter(|(root, _)| root == name)
                    .map(|(_, path)| path.clone())
                    .unwrap_or_default();
                if access == AccessKind::Move
                    && self.borrow_regions.iter().any(|region| {
                        region.binding_id == binding_id
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
                        binding.state = State::Moved;
                    }
                    AccessKind::SharedBorrow => {
                        self.borrow_regions.push(BorrowRegion {
                            binding_id,
                            local: name.clone(),
                            kind: BorrowKind::Shared,
                            path: borrow_path.clone(),
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
                            binding_id,
                            local: name.clone(),
                            kind: BorrowKind::Mutable,
                            path: borrow_path.clone(),
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
                // Preserve a mutable place borrow through every projection in an
                // indexed assignment. Downgrading an intermediate projection to a
                // shared borrow lets nested writes alias reads of the same root.
                let collection_mode = if mode == AccessMode::PathMutableBorrow {
                    AccessMode::PathMutableBorrow
                } else {
                    AccessMode::PathBorrow
                };
                let collection_type = self.check_expr(collection, locals, collection_mode);
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
                    TypeKind::List(element) => {
                        let element_type = *element;
                        if mode == AccessMode::Move && ownership_of(&element_type) == OwnershipClass::Move {
                            if let Some((name, path)) = place_path(expr) {
                                if let Some(binding) = locals.get_mut(&name) {
                                    let path_state = binding.path_state(&path);
                                    if path_state != State::Available {
                                        let message = if path_state == State::Moved {
                                            format!("use of moved indexed place '{name}'")
                                        } else {
                                            format!("use of possibly moved indexed place '{name}'")
                                        };
                                        self.errors.push(Diagnostic::error("AIF400", message, Some(expr.span)));
                                        return None;
                                    }
                                    let next_state = if path.iter().any(|part| *part == PlaceProjection::DynamicIndex) {
                                        State::MaybeMoved
                                    } else {
                                        State::Moved
                                    };
                                    binding.set_path_state(path, next_state);
                                    self.accesses.insert(expr.id, AccessKind::Move);
                                    self.transitions.push((expr.id, OwnershipTransition::Move));
                                }
                            }
                        } else if let Some((name, path)) = place_path(expr) {
                            if let Some(binding) = locals.get(&name) {
                                let path_state = if matches!(
                                    mode,
                                    AccessMode::PathBorrow | AccessMode::PathMutableBorrow
                                ) {
                                    binding.path_state_for_projection(&path)
                                } else {
                                    binding.path_state(&path)
                                };
                                if path_state != State::Available {
                                    let message = if path_state == State::Moved {
                                        format!("use of moved indexed place '{name}'")
                                    } else {
                                        format!("use of possibly moved indexed place '{name}'")
                                    };
                                    self.errors.push(Diagnostic::error("AIF400", message, Some(expr.span)));
                                    return None;
                                }
                            }
                        }
                        Some(element_type)
                    }
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
                        let mode = if name == "len" {
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
                            for (id, binding) in locals.bindings.iter_mut() {
                                if let (Some(then_binding), Some(else_binding)) = (
                                    then_locals.bindings.get(id),
                                    else_locals.bindings.get(id),
                                ) {
                                    binding.state = join_state(then_binding.state, else_binding.state);
                                    binding.path_states = join_path_states(
                                        &then_binding.path_states,
                                        &else_binding.path_states,
                                    );
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

fn merge_states(outer: &mut Locals, inner: &Locals) {
    for (id, binding) in outer.bindings.iter_mut() {
        if let Some(inner_binding) = inner.bindings.get(id) {
            binding.state = join_state(binding.state, inner_binding.state);
            binding.path_states = join_path_states(&binding.path_states, &inner_binding.path_states);
        }
    }
}

fn join_path_states(
    left: &HashMap<Vec<PlaceProjection>, State>,
    right: &HashMap<Vec<PlaceProjection>, State>,
) -> HashMap<Vec<PlaceProjection>, State> {
    let mut paths = HashSet::new();
    paths.extend(left.keys().cloned());
    paths.extend(right.keys().cloned());
    paths.into_iter().filter_map(|path| {
        let state = join_state(
            left.get(&path).copied().unwrap_or(State::Available),
            right.get(&path).copied().unwrap_or(State::Available),
        );
        (state != State::Available).then_some((path, state))
    }).collect()
}

fn join_state(left: State, right: State) -> State {
    if left == right { left } else { State::MaybeMoved }
}

fn merge_scope_states(outer: &mut Locals, inner: &Locals) {
    for (id, binding) in outer.bindings.iter_mut() {
        if let Some(inner_binding) = inner.bindings.get(id) {
            binding.state = inner_binding.state;
            binding.path_states = inner_binding.path_states.clone();
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
    fn unwrap_consumes_result_values() {
        let module = parse(
            "module unwrap_moves_result\nfn f(value: Result<Int, String>) -> Int\n  let extracted = unwrap(value)\n  unwrap(value)\n",
        )
        .unwrap();
        let errors = infer(&module).expect_err("unwrap must consume its Result argument");
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "reusing a Result after unwrap must be rejected, got {errors:?}"
        );
    }

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
                binding_id: BindingId(crate::NodeId(10)),
                local: "value".into(),
                kind: BorrowKind::Shared,
                path: Vec::new(),
                span,
                scope_depth: 0,
            },
            BorrowRegion {
                binding_id: BindingId(crate::NodeId(10)),
                local: "value".into(),
                kind: BorrowKind::Mutable,
                path: Vec::new(),
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
    fn loop_binding_shadowing_preserves_the_outer_binding_identity() {
        let module = parse(
            "module loop_shadow\nfn f(values: List<String>, item: String) -> String\n  for item in values\n    len(item)\n  item\n",
        )
        .unwrap();
        assert!(
            infer(&module).is_ok(),
            "the loop binding must shadow only inside the loop and preserve the outer binding"
        );
    }

    #[test]
    fn same_scope_shadowing_creates_a_new_binding_identity() {
        let module = parse(
            "module local_shadow\nfn f(value: String) -> String\n  let value = value\n  value\n",
        )
        .unwrap();
        assert!(
            infer(&module).is_ok(),
            "the new binding must not inherit the moved state of the binding used to initialize it"
        );
    }

    #[test]
    fn moving_one_list_index_does_not_move_a_disjoint_index() {
        let module = parse(
            "module distinct_paths\nfn f(values: List<String>) -> String\n  let first = values[0]\n  let second = values[1]\n  first\n",
        )
        .unwrap();
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn moving_the_same_list_index_twice_is_rejected() {
        let module = parse(
            "module repeated_path\nfn f(values: List<String>) -> String\n  let first = values[0]\n  let second = values[0]\n  first\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "re-reading a moved path must be rejected, got {errors:?}"
        );
        assert!(
            errors.iter().any(|error| error.message.contains("use of moved indexed place")),
            "a statically identified repeated path is definitely moved, got {errors:?}"
        );
    }

    #[test]
    fn dynamic_index_move_conservatively_blocks_other_indices() {
        let module = parse(
            "module dynamic_path\nfn f(values: List<String>, index: Int) -> String\n  let first = values[index]\n  let second = values[0]\n  first\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "an unknown index may alias any concrete element, got {errors:?}"
        );
    }

    #[test]
    fn indexed_reinitialization_restores_only_the_assigned_path() {
        let module = parse(
            "module path_reinit\nfn f(values: List<String>) -> String\n  let first = values[0]\n  set values[0] = \"restored\"\n  let again = values[0]\n  again\n",
        )
        .unwrap();
        assert!(infer(&module).is_ok());
    }

    #[test]
    fn partial_move_on_one_branch_is_path_dependent_after_join() {
        let module = parse(
            "module path_join\nfn f(flag: Bool, values: List<String>) -> String\n  if flag\n    let selected = values[0]\n  else\n    let other = \"safe\"\n  values[0]\n",
        )
        .unwrap();
        let errors = infer(&module).unwrap_err();
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "a path moved on only one branch must be maybe-moved after the join, got {errors:?}"
        );
    }

    #[test]
    fn moving_an_iterable_before_same_name_loop_shadow_is_preserved() {
        let module = parse(
            "module iterable_shadow_move\nfn f(values: List<String>) -> Int\n  for values in values\n    len(values)\n  len(values)\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        let errors = infer(&module).expect_err(
            "the iterable is consumed before the loop-local binding shadows the same name",
        );
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "the outer list must remain moved after iteration, got {errors:?}"
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
        assert!(
            errors.iter().any(|error| error.message.contains("moved")),
            "the joined state should be definitely moved, got {errors:?}"
        );
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
    fn nested_scope_shadowing_uses_distinct_binding_identity() {
        let module = parse(
            "module scope_shadow\nfn f(value: String) -> String\n  scope\n    let value = \"fresh\"\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(
            infer(&module).is_ok(),
            "the nested shadow must not replace or move the outer binding"
        );
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
                crate::NativeValue::String("next".into()),
            ])],
            crate::ExecutionLimits::default(),
        )
        .expect("native execution should succeed")
    }

    #[test]
    fn moving_an_indexed_element_partially_moves_the_source_collection() {
        let module = parse(
            "module indexed_value_is_owned\nfn f(values: List<String>) -> Int\n  let selected = values[0]\n  let consumed = selected\n  len(values)\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        let errors = infer(&module).expect_err(
            "using the whole list after moving an element must be rejected as a partial move",
        );
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "expected a partial-move diagnostic, got {errors:?}"
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
    fn indexed_element_move_allows_a_disjoint_path_at_runtime() {
        let source = "module indexed_runtime\nfn f(values: List<String>) -> Int\n  let selected = values[0]\n  let consumed = selected\n  len(values[1])\n";
        assert_eq!(
            run_owned_collection_case(source),
            crate::NativeValue::Int(4),
            "moving one element must leave a disjoint indexed path usable"
        );
    }

    #[test]
    fn slice_result_move_preserves_source_collection_at_runtime() {
        let source = "module slice_runtime\nfn f(values: List<String>) -> Int\n  let selected = values[0:1]\n  let consumed = selected\n  len(values)\n";
        assert_eq!(
            run_owned_collection_case(source),
            crate::NativeValue::Int(2),
            "moving the owned slice must not consume the source list"
        );
    }

    #[test]
    fn shadowed_inner_binding_does_not_mutate_outer_binding_state() {
        let module = parse(
            "module scope_shadow\nfn f(value: String) -> String\n  scope\n    let value = \"inner\"\n    len(value)\n  value\n",
        )
        .unwrap();
        assert!(crate::sema::check(&module).is_ok());
        assert!(infer(&module).is_ok());
    }


    #[test]
    fn nested_index_moves_track_disjoint_leaf_paths() {
        let module = parse(
            "module nested_path_siblings\nfn f(values: List<List<String>>) -> String\n  let first = values[0][0]\n  let second = values[0][1]\n  first\n",
        )
        .unwrap();
        assert!(
            infer(&module).is_ok(),
            "moving [0][0] must not mark the disjoint [0][1] place as moved"
        );
    }

    #[test]
    fn nested_index_assignment_keeps_mutable_borrow_on_root_binding() {
        let mut module = parse(
            "module nested_index_alias\nfn f(values: List<List<Int>>) -> Int\n  0\n",
        )
        .unwrap();
        let Item::Function(function) = &mut module.items[0];
        let span = function.body.stmts[0].span;
        let name = |id| Expr {
            id: crate::NodeId(id),
            span,
            kind: ExprKind::Name("values".into()),
        };
        let int = |id, value| Expr {
            id: crate::NodeId(id),
            span,
            kind: ExprKind::Int(value),
        };
        let index = |id, collection, subscript| Expr {
            id: crate::NodeId(id),
            span,
            kind: ExprKind::Index {
                collection: Box::new(collection),
                index: Box::new(subscript),
            },
        };
        let target_collection = index(9_300_001, name(9_300_002), int(9_300_003, 0));
        let aliased_value = index(
            9_300_004,
            index(9_300_005, name(9_300_006), int(9_300_007, 0)),
            int(9_300_008, 1),
        );
        function.body.stmts[0].kind = StmtKind::SetIndex {
            collection: target_collection,
            index: int(9_300_009, 0),
            value: aliased_value,
        };
        let errors = infer(&module).expect_err(
            "a nested indexed write must conflict with a read from the same root collection",
        );
        assert!(
            errors.iter().any(|error| error.code == "AIF404"),
            "expected overlapping mutable/shared borrow diagnostics, got {errors:?}"
        );
    }

    #[test]
    fn shadowed_borrow_regions_keep_distinct_binding_identities() {
        let module = parse(
            "module shadow_binding_identity\nfn f(value: String) -> String\n  scope\n    let value = \"inner\"\n    let equal = value == value\n  value\n",
        )
        .unwrap();
        let Item::Function(function) = &module.items[0];
        let parameter_id = BindingId(function.params[0].id);
        let model = analyze(&module).unwrap();
        assert!(!model.borrow_regions.is_empty());
        assert!(
            model.borrow_regions.iter().all(|region| region.binding_id != parameter_id),
            "borrow regions for the inner shadow must point at the declaration identity, not the outer parameter"
        );
    }




    #[test]
    fn nested_index_move_preserves_disjoint_path_at_runtime() {
        let source = "module nested_index_runtime\nfn f(values: List<List<String>>) -> Int\n  let selected = values[0][0]\n  let sibling = values[0][1]\n  len(sibling)\n";
        let compiled = crate::compile_source(source)
            .expect("nested disjoint paths should compile through the native pipeline");
        let program = crate::native::decode_program(&compiled.artifact)
            .expect("compiled nested-path artifact should decode");
        let input = crate::NativeValue::List(vec![crate::NativeValue::List(vec![
            crate::NativeValue::String("first".into()),
            crate::NativeValue::String("next".into()),
        ])]);
        assert_eq!(
            crate::native::run_program_with_limits(
                &program,
                "f",
                &[input],
                crate::ExecutionLimits::default(),
            )
            .expect("the disjoint nested path remains usable at runtime"),
            crate::NativeValue::Int(4),
        );
    }


    #[test]
    fn moving_a_nested_child_still_blocks_moving_its_parent() {
        let module = parse(
            "module nested_parent_after_child_move\nfn f(values: List<List<String>>) -> Int\n  let selected = values[0][0]\n  let parent = values[0]\n  len(parent)\n",
        )
        .unwrap();
        let errors = infer(&module).expect_err(
            "a partially moved nested collection cannot be moved or read as a whole",
        );
        assert!(
            errors.iter().any(|error| error.code == "AIF400"),
            "the parent place must remain partially moved, got {errors:?}"
        );
    }

}
