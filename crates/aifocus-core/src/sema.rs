use std::collections::HashMap;

use crate::{
    ast::{BinaryOp, Block, ConstructKind, ConstructMember, Expr, ExprKind, Function, Item, Module, StmtKind, Type, TypeKind},
    source::{Diagnostic, Span},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    pub params: Vec<Type>,
    pub return_type: Option<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SemanticModel {
    pub inferred_types: HashMap<crate::NodeId, Type>,
    pub function_returns: HashMap<String, Type>,
}

/// Validate semantic types including String concatenation.
pub fn check(module: &Module) -> Result<(), Vec<Diagnostic>> {
    analyze(module).map(|_| ())
}

pub fn analyze(module: &Module) -> Result<SemanticModel, Vec<Diagnostic>> {
    let mut checker = Checker {
        functions: HashMap::new(),
        inferred_types: HashMap::new(),
        function_returns: HashMap::new(),
        errors: Vec::new(),
    };

    validate_construct_declarations(module, &mut checker);

    for declaration in &module.constructs {
        checker.error(
            "AIF610",
            format!("AI Mode declaration '{}' is parsed but not yet semantically enforced", declaration.name),
            declaration.span,
        );
    }

    for item in &module.items {
        let Item::Function(function) = item;
        let signature = FunctionSignature {
            params: function.params.iter().map(|p| p.ty.clone()).collect(),
            return_type: function.return_type.clone(),
        };
        if checker
            .functions
            .insert(function.name.clone(), signature)
            .is_some()
        {
            checker.error(
                "AIF300",
                format!("duplicate function '{}'", function.name),
                function.span,
            );
        }
    }

    // Infer omitted return types to a fixed point. This makes call sites benefit
    // from information discovered in functions declared later in the module.
    for _ in 0..module.items.len().max(1) {
        let before = checker.function_returns.clone();
        for item in &module.items {
            let Item::Function(function) = item else { continue; };
            if function.return_type.is_none() {
                let mut locals = function
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), p.ty.clone()))
                    .collect::<HashMap<_, _>>();
                if let Some(ty) = checker.check_block(&function.body, &mut locals) {
                    checker.function_returns.insert(function.name.clone(), ty);
                }
            } else if let Some(ty) = function.return_type.clone() {
                checker.function_returns.insert(function.name.clone(), ty);
            }
        }
        if before == checker.function_returns {
            break;
        }
    }

    for item in &module.items {
        if let Item::Function(function) = item {
            checker.check_function(function);
        }
    }

    if checker.errors.is_empty() {
        Ok(SemanticModel {
            inferred_types: checker.inferred_types,
            function_returns: checker.function_returns,
        })
    } else {
        Err(checker.errors)
    }
}


/// Validate the declaration structure we can currently prove, while retaining
/// AIF610 until declarations are carried through typed IR and native execution.
fn validate_construct_declarations(module: &Module, checker: &mut Checker) {
    use std::collections::HashSet;

    let mut declaration_names = HashSet::new();
    for declaration in &module.constructs {
        if !declaration_names.insert(declaration.name.as_str()) {
            checker.error(
                "AIF611",
                format!("duplicate AI Mode declaration '{}'", declaration.name),
                declaration.span,
            );
        }

        let mut member_names = HashSet::new();
        let mut transitions = HashSet::new();
        for member in &declaration.members {
            match member {
                ConstructMember::Field { name, span, .. } => {
                    if declaration.kind == ConstructKind::Phase {
                        checker.error(
                            "AIF612",
                            format!("phase '{}' cannot declare field '{}'; use transitions", declaration.name, name),
                            *span,
                        );
                    }
                    if !member_names.insert(name.as_str()) {
                        checker.error(
                            "AIF613",
                            format!("duplicate member '{}' in declaration '{}'", name, declaration.name),
                            *span,
                        );
                    }
                }
                ConstructMember::Clause { name, span, .. } => {
                    let allowed = match declaration.kind {
                        ConstructKind::Trace => matches!(name.as_str(), "signature"),
                        ConstructKind::Cell => matches!(name.as_str(), "invariant"),
                        ConstructKind::Vault => matches!(name.as_str(), "capabilities" | "denies"),
                        ConstructKind::Proof => matches!(name.as_str(), "requires" | "ensures" | "on_unknown" | "required"),
                        ConstructKind::Phase => false,
                    };
                    if !allowed {
                        checker.error(
                            "AIF614",
                            format!("clause '{}' is not supported for {:?} declaration '{}'", name, declaration.kind, declaration.name),
                            *span,
                        );
                    }
                    if !member_names.insert(name.as_str()) {
                        checker.error(
                            "AIF613",
                            format!("duplicate member '{}' in declaration '{}'", name, declaration.name),
                            *span,
                        );
                    }
                }
                ConstructMember::Transition { from, to, span } => {
                    if declaration.kind != ConstructKind::Phase {
                        checker.error(
                            "AIF612",
                            format!("{:?} declaration '{}' cannot declare phase transitions", declaration.kind, declaration.name),
                            *span,
                        );
                    }
                    if from == to {
                        checker.error(
                            "AIF615",
                            format!("phase '{}' contains a self-transition for '{}'", declaration.name, from),
                            *span,
                        );
                    }
                    if !transitions.insert((from.as_str(), to.as_str())) {
                        checker.error(
                            "AIF616",
                            format!("duplicate phase transition '{} -> {}' in '{}'", from, to, declaration.name),
                            *span,
                        );
                    }
                }
            }
        }
    }
}

struct Checker {
    functions: HashMap<String, FunctionSignature>,
    inferred_types: HashMap<crate::NodeId, Type>,
    function_returns: HashMap<String, Type>,
    errors: Vec<Diagnostic>,
}

impl Checker {
    fn check_function(&mut self, function: &Function) {
        let mut locals = HashMap::new();
        for param in &function.params {
            if locals
                .insert(param.name.clone(), param.ty.clone())
                .is_some()
            {
                self.error(
                    "AIF301",
                    format!("duplicate parameter '{}'", param.name),
                    param.span,
                );
            }
            self.inferred_types.insert(param.id, param.ty.clone());
        }

        let block_type = self.check_block(&function.body, &mut locals);
        if let Some(expected) = &function.return_type {
            if let Some(actual) = block_type {
                if !same_type(&actual, expected) {
                    self.error(
                        "AIF302",
                        format!(
                            "function '{}' returns {}, expected {}",
                            function.name,
                            actual.display_name(),
                            expected.display_name()
                        ),
                        function.body.span,
                    );
                }
            }
        }
    }

    fn check_block(&mut self, block: &Block, locals: &mut HashMap<String, Type>) -> Option<Type> {
        let mut last = None;
        for stmt in &block.stmts {
            match &stmt.kind {
                StmtKind::Set { name, value } => {
                    let Some(expected) = locals.get(name).cloned() else {
                        self.error(
                            "AIF314",
                            format!("unknown mutable binding '{name}'"),
                            stmt.span,
                        );
                        last = None;
                        continue;
                    };
                    if let Some(actual) = self.check_expr(value, locals) {
                        if !same_type(&actual, &expected) {
                            self.error(
                                "AIF315",
                                format!(
                                    "binding '{}' has type {}, assigned {}",
                                    name,
                                    expected.display_name(),
                                    actual.display_name()
                                ),
                                stmt.span,
                            );
                        }
                    }
                    last = None;
                }
                StmtKind::SetIndex {
                    collection,
                    index,
                    value,
                } => {
                    let Some(collection_type) = self.check_expr(collection, locals) else {
                        last = None;
                        continue;
                    };
                    let Some(index_type) = self.check_expr(index, locals) else {
                        last = None;
                        continue;
                    };
                    if !is_kind(&index_type, &TypeKind::Int) {
                        self.error("AIF318", "list index must be Int", index.span);
                    }
                    let Some(element_type) = (match &collection_type.kind {
                        TypeKind::List(element) => Some((**element).clone()),
                        _ => None,
                    }) else {
                        self.error(
                            "AIF319",
                            "indexed assignment requires List<T>",
                            collection.span,
                        );
                        last = None;
                        continue;
                    };
                    if let Some(actual) = self.check_expr(value, locals) {
                        if !same_type(&actual, &element_type) {
                            self.error(
                                "AIF320",
                                format!(
                                    "list element has type {}, assigned {}",
                                    element_type.display_name(),
                                    actual.display_name()
                                ),
                                value.span,
                            );
                        }
                    }
                    last = None;
                }
                StmtKind::Let { name, value } => {
                    if locals.contains_key(name) {
                        self.error(
                            "AIF303",
                            format!("binding '{}' shadows an existing local", name),
                            stmt.span,
                        );
                    }
                    if let Some(ty) = self.check_expr(value, locals) {
                        locals.insert(name.clone(), ty);
                    }
                    last = None;
                }
                StmtKind::Return(value) => {
                    last = value.as_ref().and_then(|e| self.check_expr(e, locals));
                }
                StmtKind::Expr(expr) => {
                    last = self.check_expr(expr, locals);
                }
                StmtKind::Scope { body } => {
                    let mut scoped = locals.clone();
                    self.check_block(body, &mut scoped);
                    last = None;
                }
                StmtKind::Spawn { call, .. } => {
                    self.check_expr(call, locals);
                    last = None;
                }
                StmtKind::Join { .. } | StmtKind::Cancel { .. } => {
                    last = None;
                }
                StmtKind::While { condition, body } => {
                    if let Some(condition_type) = self.check_expr(condition, locals) {
                        if !is_kind(&condition_type, &TypeKind::Bool) {
                            self.error("AIF313", "while condition must be Bool", condition.span);
                        }
                    }
                    let mut scoped = locals.clone();
                    self.check_block(body, &mut scoped);
                    last = None;
                }
            }
        }
        last
    }

    fn check_expr(&mut self, expr: &Expr, locals: &HashMap<String, Type>) -> Option<Type> {
        let result = match &expr.kind {
            ExprKind::Int(_) => Some(type_node(TypeKind::Int, expr.span)),
            ExprKind::Bool(_) => Some(type_node(TypeKind::Bool, expr.span)),
            ExprKind::String(_) => Some(type_node(TypeKind::String, expr.span)),
            ExprKind::List(elements) => {
                let mut element_type: Option<Type> = None;
                for element in elements {
                    if let Some(actual) = self.check_expr(element, locals) {
                        if let Some(expected) = &element_type {
                            if !same_type(&actual, expected) {
                                self.error(
                                    "AIF317",
                                    "list elements must have the same type",
                                    element.span,
                                );
                            }
                        } else {
                            element_type = Some(actual);
                        }
                    }
                }
                Some(type_node(
                    TypeKind::List(Box::new(
                        element_type.unwrap_or_else(|| type_node(TypeKind::Unit, expr.span)),
                    )),
                    expr.span,
                ))
            }
            ExprKind::Index { collection, index } => {
                let collection_type = self.check_expr(collection, locals)?;
                let index_type = self.check_expr(index, locals)?;
                if !is_kind(&index_type, &TypeKind::Int) {
                    self.error("AIF318", "list index must be Int", index.span);
                }
                match collection_type.kind {
                    TypeKind::List(element) => Some((*element).clone()),
                    TypeKind::String => Some(type_node(TypeKind::Int, expr.span)),
                    _ => {
                        self.error(
                            "AIF319",
                            "indexing requires List<T> or String",
                            collection.span,
                        );
                        None
                    }
                }
            }
            ExprKind::Name(name) => {
                if let Some(ty) = locals.get(name) {
                    Some(ty.clone())
                } else if let Some(signature) = self.functions.get(name) {
                    signature
                        .return_type
                        .clone()
                        .or_else(|| self.function_returns.get(name).cloned())
                } else {
                    self.error("AIF304", format!("unknown name '{name}'"), expr.span);
                    None
                }
            }
            ExprKind::Group(inner) => self.check_expr(inner, locals),
            ExprKind::Binary { op, left, right } => {
                let left_type = self.check_expr(left, locals)?;
                let right_type = self.check_expr(right, locals)?;
                match op {
                    BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Mod => {
                        let int_operands = is_kind(&left_type, &TypeKind::Int)
                            && is_kind(&right_type, &TypeKind::Int);
                        let string_add = matches!(op, BinaryOp::Add)
                            && is_kind(&left_type, &TypeKind::String)
                            && is_kind(&right_type, &TypeKind::String);
                        if int_operands {
                            Some(type_node(TypeKind::Int, expr.span))
                        } else if string_add {
                            Some(type_node(TypeKind::String, expr.span))
                        } else {
                            self.error(
                                "AIF305",
                                "arithmetic operators require Int operands, except String + String",
                                expr.span,
                            );
                            None
                        }
                    }
                    BinaryOp::Equal | BinaryOp::NotEqual => {
                        if !same_type(&left_type, &right_type) {
                            self.error(
                                "AIF306",
                                "equality operands must have the same type",
                                expr.span,
                            );
                            None
                        } else {
                            Some(type_node(TypeKind::Bool, expr.span))
                        }
                    }
                    BinaryOp::Less
                    | BinaryOp::LessEqual
                    | BinaryOp::Greater
                    | BinaryOp::GreaterEqual => {
                        if is_kind(&left_type, &TypeKind::Int)
                            && is_kind(&right_type, &TypeKind::Int)
                        {
                            Some(type_node(TypeKind::Bool, expr.span))
                        } else {
                            self.error(
                                "AIF321",
                                "ordering operators require Int operands",
                                expr.span,
                            );
                            None
                        }
                    }
                }
            }
            ExprKind::Call { callee, args } => {
                let ExprKind::Name(name) = &callee.kind else {
                    self.error(
                        "AIF307",
                        "only named functions are callable in this language version",
                        callee.span,
                    );
                    return None;
                };
                if name == "chr" {
                    if args.len() != 1 {
                        self.error("AIF330", "chr expects one argument", expr.span);
                        return None;
                    }
                    let argument_type = self.check_expr(&args[0], locals)?;
                    if !is_kind(&argument_type, &TypeKind::Int) {
                        self.error("AIF331", "chr requires an Int byte value", args[0].span);
                        return None;
                    }
                    return Some(type_node(TypeKind::String, expr.span));
                }
                if name == "ok" || name == "err" {
                    if args.len() != 1 {
                        self.error("AIF327", format!("{name} expects one argument"), expr.span);
                        return None;
                    }
                    let value_type = self.check_expr(&args[0], locals)?;
                    let unit = type_node(TypeKind::Unit, expr.span);
                    return Some(if name == "ok" {
                        type_node(
                            TypeKind::Result(Box::new(value_type), Box::new(unit)),
                            expr.span,
                        )
                    } else {
                        type_node(
                            TypeKind::Result(Box::new(unit), Box::new(value_type)),
                            expr.span,
                        )
                    });
                }
                if name == "unwrap" {
                    if args.len() != 1 {
                        self.error("AIF328", "unwrap expects one argument", expr.span);
                        return None;
                    }
                    let result_type = self.check_expr(&args[0], locals)?;
                    let TypeKind::Result(ok, _) = result_type.kind else {
                        self.error("AIF329", "unwrap requires Result<T, E>", args[0].span);
                        return None;
                    };
                    return Some(*ok);
                }
                if name == "push" {
                    if args.len() != 2 {
                        self.error("AIF324", "push expects a List<T> and one value", expr.span);
                        return None;
                    }
                    let list_type = self.check_expr(&args[0], locals)?;
                    let value_type = self.check_expr(&args[1], locals)?;
                    let TypeKind::List(element) = &list_type.kind else {
                        self.error(
                            "AIF325",
                            "push requires List<T> as its first argument",
                            args[0].span,
                        );
                        return None;
                    };
                    if !same_type(element, &value_type) {
                        self.error(
                            "AIF326",
                            "push value must match the list element type",
                            args[1].span,
                        );
                        return None;
                    }
                    return Some(type_node(TypeKind::Unit, expr.span));
                }
                if name == "len" {
                    if args.len() != 1 {
                        self.error("AIF322", "len expects exactly one argument", expr.span);
                        return None;
                    }
                    let argument_type = self.check_expr(&args[0], locals)?;
                    if !matches!(argument_type.kind, TypeKind::String | TypeKind::List(_)) {
                        self.error("AIF323", "len requires String or List<T>", args[0].span);
                        return None;
                    }
                    return Some(type_node(TypeKind::Int, expr.span));
                }
                let Some(signature) = self.functions.get(name).cloned() else {
                    self.error("AIF308", format!("unknown function '{name}'"), callee.span);
                    return None;
                };
                if args.len() != signature.params.len() {
                    self.error(
                        "AIF309",
                        format!(
                            "function '{}' expects {} argument(s), got {}",
                            name,
                            signature.params.len(),
                            args.len()
                        ),
                        expr.span,
                    );
                }
                for (index, arg) in args.iter().enumerate() {
                    if let Some(actual) = self.check_expr(arg, locals) {
                        if let Some(expected) = signature.params.get(index) {
                            if !same_type(&actual, expected) {
                                self.error(
                                    "AIF310",
                                    format!(
                                        "argument {} to '{}' has type {}, expected {}",
                                        index + 1,
                                        name,
                                        actual.display_name(),
                                        expected.display_name()
                                    ),
                                    arg.span,
                                );
                            }
                        }
                    }
                }
                signature
                    .return_type
                    .or_else(|| self.function_returns.get(name).cloned())
            }
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition_type = self.check_expr(condition, locals)?;
                if !is_kind(&condition_type, &TypeKind::Bool) {
                    self.error("AIF311", "if condition must be Bool", condition.span);
                }
                let mut then_locals = locals.clone();
                let then_type = self.check_block(then_branch, &mut then_locals);
                let else_branch = match else_branch {
                    Some(branch) => branch,
                    None => return Some(type_node(TypeKind::Unit, expr.span)),
                };
                let mut else_locals = locals.clone();
                let else_type = self.check_block(else_branch, &mut else_locals);
                match (then_type, else_type) {
                    (Some(left), Some(right)) if same_type(&left, &right) => Some(left),
                    (Some(_), Some(_)) => {
                        self.error(
                            "AIF312",
                            "if branches must produce the same type",
                            expr.span,
                        );
                        None
                    }
                    _ => Some(type_node(TypeKind::Unit, expr.span)),
                }
            }
        };
        if let Some(ref ty) = result {
            self.inferred_types.insert(expr.id, ty.clone());
        }
        result
    }

    fn error(&mut self, code: &'static str, message: impl Into<String>, span: Span) {
        self.errors
            .push(Diagnostic::error(code, message, Some(span)));
    }
}

fn type_node(kind: TypeKind, span: Span) -> Type {
    Type {
        id: crate::NodeId(0),
        span,
        kind,
    }
}

fn is_kind(ty: &Type, kind: &TypeKind) -> bool {
    &ty.kind == kind
}

fn same_type(left: &Type, right: &Type) -> bool {
    match (&left.kind, &right.kind) {
        (TypeKind::Int, TypeKind::Int)
        | (TypeKind::Bool, TypeKind::Bool)
        | (TypeKind::String, TypeKind::String)
        | (TypeKind::Unit, TypeKind::Unit) => true,
        (TypeKind::Named(left), TypeKind::Named(right)) => left == right,
        (TypeKind::Generic(left_name, left_args), TypeKind::Generic(right_name, right_args)) => {
            left_name == right_name
                && left_args.len() == right_args.len()
                && left_args.iter().zip(right_args).all(|(left, right)| same_type(left, right))
        }
        (TypeKind::List(left), TypeKind::List(right)) => same_type(left, right),
        (TypeKind::Result(left_ok, left_err), TypeKind::Result(right_ok, right_err)) => {
            same_type(left_ok, right_ok) && same_type(left_err, right_err)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn accepts_well_typed_function() {
        let module = parse("module x\nfn add(a: Int, b: Int) -> Int\n  a + b\n").unwrap();
        assert!(check(&module).is_ok());
    }

    #[test]
    fn catches_unknown_name() {
        let module = parse("module x\nfn add(a: Int) -> Int\n  a + missing\n").unwrap();
        let errors = check(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF304"));
    }

    #[test]
    fn catches_argument_type_mismatch() {
        let module = parse(
            "module x\nfn add(a: Int, b: Int) -> Int\n  a + b\nfn main() -> Int\n  add(true, 1)\n",
        )
        .unwrap();
        let errors = check(&module).unwrap_err();
        assert!(errors.iter().any(|error| error.code == "AIF310"));
    }

    #[test]
    fn infers_omitted_return_type_and_propagates_it_to_calls() {
        let module = parse("module x\nfn value()\n  42\nfn main() -> Int\n  value()\n").unwrap();
        let model = analyze(&module).unwrap();
        assert_eq!(model.function_returns["value"].kind, TypeKind::Int);
        assert_eq!(
            model
                .inferred_types
                .values()
                .filter(|t| t.kind == TypeKind::Int)
                .count(),
            2
        );
    }

    #[test]
    fn infers_return_through_later_declaration() {
        let module = parse("module x\nfn main() -> Int\n  value()\nfn value()\n  42\n").unwrap();
        assert!(check(&module).is_ok());
    }
}
