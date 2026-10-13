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
    /// Structurally validated declarations. Compilation remains fail-closed.
    pub constructs: Vec<TypedConstructDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedConstructDeclaration {
    pub kind: ConstructKind,
    pub name: String,
    pub members: Vec<TypedConstructMember>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedConstructMember {
    Field { name: String, ty: TypeKind, span: Span },
    Clause { name: String, value: String, span: Span },
    Transition { from: String, to: String, span: Span },
}


/// Validate built-in AI type constructor signatures at source level. This does
/// not claim runtime values or executable VM instructions are implemented.
fn validate_ai_type(ty: &Type, checker: &mut Checker) {
    match &ty.kind {
        TypeKind::Generic(name, _) if matches!(name.as_str(), "AttentionBlock" | "LoRAAdapter") => {
            checker.error(
                "AIF451",
                format!("{} compile-time shape parameters are not supported yet; use the validated runtime nominal constructor", name),
                ty.span,
            );
        }
        TypeKind::Generic(name, args) if matches!(name.as_str(), "Tensor" | "Embedding" | "QuantizedTensor" | "Probabilistic" | "Guaranteed") => {
            if args.len() != 1 {
                checker.error("AIF320", format!("AI type '{}' expects one type argument", name), ty.span);
                return;
            }
            if name == "Tensor" || name == "QuantizedTensor" {
                let TypeKind::Named(element) = &args[0].kind else {
                    checker.error("AIF320", format!("AI type '{}' requires a named numeric element type", name), args[0].span);
                    return;
                };
                let allowed = if name == "Tensor" {
                    matches!(element.as_str(), "F32" | "F16" | "BF16" | "I8" | "I4" | "Quant4")
                } else {
                    matches!(element.as_str(), "I8" | "I4" | "Quant4")
                };
                if !allowed {
                    checker.error("AIF320", format!("unsupported element type '{}' for {}", element, name), args[0].span);
                }
            } else if name == "Embedding" && !matches!(&args[0].kind, TypeKind::Named(_)) {
                checker.error("AIF320", "Embedding domain must be a named domain type", args[0].span);
            }
            for arg in args { validate_ai_type(arg, checker); }
        }
        TypeKind::Generic(name, args) if matches!(name.as_str(), "Probability" | "GraphTensor") => {
            checker.error("AIF320", format!("AI type '{}' does not accept type arguments", name), ty.span);
            for arg in args { validate_ai_type(arg, checker); }
        }
        TypeKind::Generic(_, args) => for arg in args { validate_ai_type(arg, checker); },
        TypeKind::List(element) => validate_ai_type(element, checker),
        TypeKind::Result(ok, err) => {
            validate_ai_type(ok, checker);
            validate_ai_type(err, checker);
        }
        _ => {}
    }
}

/// Validate declaration structure and produce a typed semantic representation.
/// This inspection API omits AIF610; normal analyze() still rejects compilation.
pub fn analyze_constructs(module: &Module) -> Result<Vec<TypedConstructDeclaration>, Vec<Diagnostic>> {
    let mut checker = Checker {
        functions: HashMap::new(),
        inferred_types: HashMap::new(),
        function_returns: HashMap::new(),
        errors: Vec::new(),
    };
    validate_construct_declarations(module, &mut checker);
    if !checker.errors.is_empty() {
        return Err(checker.errors);
    }
    Ok(module.constructs.iter().map(|declaration| TypedConstructDeclaration {
        kind: declaration.kind,
        name: declaration.name.clone(),
        span: declaration.span,
        members: declaration.members.iter().map(|member| match member {
            ConstructMember::Field { name, ty, span } => TypedConstructMember::Field {
                name: name.clone(), ty: ty.kind.clone(), span: *span,
            },
            ConstructMember::Clause { name, value, span } => TypedConstructMember::Clause {
                name: name.clone(), value: value.clone(), span: *span,
            },
            ConstructMember::Transition { from, to, span } => TypedConstructMember::Transition {
                from: from.clone(), to: to.clone(), span: *span,
            },
        }).collect(),
    }).collect())
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
        for parameter in &function.params {
            validate_ai_type(&parameter.ty, &mut checker);
        }
        if let Some(return_type) = &function.return_type {
            validate_ai_type(return_type, &mut checker);
        }
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
            constructs: module.constructs.iter().map(|declaration| TypedConstructDeclaration {
                kind: declaration.kind,
                name: declaration.name.clone(),
                span: declaration.span,
                members: declaration.members.iter().map(|member| match member {
                    ConstructMember::Field { name, ty, span } => TypedConstructMember::Field { name: name.clone(), ty: ty.kind.clone(), span: *span },
                    ConstructMember::Clause { name, value, span } => TypedConstructMember::Clause { name: name.clone(), value: value.clone(), span: *span },
                    ConstructMember::Transition { from, to, span } => TypedConstructMember::Transition { from: from.clone(), to: to.clone(), span: *span },
                }).collect(),
            }).collect(),
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
                ConstructMember::Clause { name, value, span } => {
                    if value.trim().is_empty() {
                        checker.error(
                            "AIF618",
                            format!("clause '{}' in '{}' must have a non-empty value", name, declaration.name),
                            *span,
                        );
                    }
                    if name == "required" && !matches!(value.trim(), "true" | "false") {
                        checker.error(
                            "AIF618",
                            format!("clause 'required' in '{}' must be 'true' or 'false'", declaration.name),
                            *span,
                        );
                    }
                    // Until an explicit unknown-policy lattice is implemented, only
                    // the conservative reject policy is accepted.
                    if name == "on_unknown" && value.trim() != "reject" {
                        checker.error(
                            "AIF618",
                            format!("unsupported on_unknown policy '{}' in '{}'; only 'reject' is currently supported", value.trim(), declaration.name),
                            *span,
                        );
                    }
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

        let required_members: &[&str] = match declaration.kind {
            ConstructKind::Trace => &["signature"],
            ConstructKind::Cell => &["invariant"],
            ConstructKind::Vault => &["capabilities"],
            ConstructKind::Proof => &["requires", "ensures", "on_unknown", "required"],
            ConstructKind::Phase => &[],
        };
        for required in required_members {
            if !member_names.contains(required) {
                checker.error(
                    "AIF617",
                    format!(
                        "{:?} declaration '{}' is missing required member '{}'",
                        declaration.kind, declaration.name, required
                    ),
                    declaration.span,
                );
            }
        }
        if declaration.kind == ConstructKind::Phase && transitions.is_empty() {
            checker.error(
                "AIF617",
                format!("phase '{}' must declare at least one transition", declaration.name),
                declaration.span,
            );
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
            ExprKind::Float(_) => Some(type_node(TypeKind::Named("F32".into()), expr.span)),
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
                    TypeKind::Named(name) if matches!(name.as_str(), "ExpertWeight" | "VectorHit") => {
                        match &index.kind {
                            ExprKind::Int(0) => Some(type_node(TypeKind::Int, expr.span)),
                            ExprKind::Int(1) => Some(type_node(TypeKind::Named("F32".into()), expr.span)),
                            _ => {
                                self.error("AIF403", format!("{} supports only tuple-style fields [0] (id/index) and [1] (weight/distance)", name), index.span);
                                None
                            }
                        }
                    }
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
                if name == "moe_router" {
                    if args.len() != 2 { self.error("AIF475", "moe_router expects expert_count and top_k Int", expr.span); return None; }
                    let experts = self.check_expr(&args[0], locals)?;
                    let top_k = self.check_expr(&args[1], locals)?;
                    if !is_kind(&experts, &TypeKind::Int) { self.error("AIF476", "moe_router expert_count must be Int", args[0].span); }
                    if !is_kind(&top_k, &TypeKind::Int) { self.error("AIF477", "moe_router top_k must be Int", args[1].span); }
                    if let ExprKind::Int(value) = args[0].kind { if value <= 0 || value > 256 { self.error("AIF478", "moe_router expert_count must be in 1..=256", args[0].span); } }
                    if let ExprKind::Int(value) = args[1].kind { if value <= 0 || value > 32 { self.error("AIF479", "moe_router top_k must be in 1..=32", args[1].span); } }
                    if let (ExprKind::Int(experts), ExprKind::Int(top_k)) = (&args[0].kind, &args[1].kind) {
                        if *experts > 0 && *top_k > *experts { self.error("AIF484", "moe_router top_k cannot exceed expert_count", args[1].span); }
                    }
                    let ty = type_node(TypeKind::Named("MoERouter".into()), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "moe_route" {
                    if args.len() != 3 { self.error("AIF480", "moe_route expects MoERouter, logits List<F32>, and masked List<Bool>", expr.span); return None; }
                    let router = self.check_expr(&args[0], locals)?;
                    let logits = self.check_expr(&args[1], locals)?;
                    let masked = self.check_expr(&args[2], locals)?;
                    if !matches!(&router.kind, TypeKind::Named(ty) if ty == "MoERouter") { self.error("AIF481", "moe_route requires a MoERouter value", args[0].span); }
                    let expected_logits = TypeKind::List(Box::new(type_node(TypeKind::Named("F32".into()), args[1].span)));
                    let expected_masked = TypeKind::List(Box::new(type_node(TypeKind::Bool, args[2].span)));
                    if !same_type_kind(&logits.kind, &expected_logits) { self.error("AIF482", "moe_route logits must be a homogeneous List<F32>", args[1].span); }
                    if !same_type_kind(&masked.kind, &expected_masked) { self.error("AIF483", "moe_route mask must be a homogeneous List<Bool>", args[2].span); }
                    let ty = type_node(TypeKind::List(Box::new(type_node(TypeKind::Named("ExpertWeight".into()), expr.span))), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "route_top_k" {
                    if args.len() != 3 {
                        self.error("AIF380", "route_top_k expects logits: List<F32>, masked: List<Bool>, and top_k: Int", expr.span);
                        return None;
                    }
                    let logits = self.check_expr(&args[0], locals)?;
                    let masked = self.check_expr(&args[1], locals)?;
                    let top_k = self.check_expr(&args[2], locals)?;
                    let expected_logits = TypeKind::List(Box::new(type_node(TypeKind::Named("F32".into()), args[0].span)));
                    let expected_masked = TypeKind::List(Box::new(type_node(TypeKind::Bool, args[1].span)));
                    if !same_type_kind(&logits.kind, &expected_logits) {
                        self.error("AIF381", "route_top_k logits must be a homogeneous List<F32>", args[0].span);
                    }
                    if !same_type_kind(&masked.kind, &expected_masked) {
                        self.error("AIF382", "route_top_k mask must be a homogeneous List<Bool>", args[1].span);
                    }
                    if !is_kind(&top_k, &TypeKind::Int) {
                        self.error("AIF383", "route_top_k top_k must be Int", args[2].span);
                    }
                    if let ExprKind::Int(value) = args[2].kind {
                        if value <= 0 || value > 32 {
                            self.error("AIF384", "route_top_k top_k must be in 1..=32", args[2].span);
                        }
                    }
                    let ty = type_node(TypeKind::List(Box::new(type_node(TypeKind::Named("ExpertWeight".into()), expr.span))), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "vector_index" {
                    if args.len() != 2 {
                        self.error("AIF385", "vector_index expects a non-empty domain String and positive dimensions Int", expr.span);
                        return None;
                    }
                    let domain = self.check_expr(&args[0], locals)?;
                    let dimensions = self.check_expr(&args[1], locals)?;
                    if !is_kind(&domain, &TypeKind::String) {
                        self.error("AIF386", "vector_index domain must be String", args[0].span);
                    }
                    if !is_kind(&dimensions, &TypeKind::Int) {
                        self.error("AIF387", "vector_index dimensions must be Int", args[1].span);
                    }
                    if let ExprKind::String(value) = &args[0].kind {
                        if value.trim().is_empty() || value.len() > 1024 {
                            self.error("AIF388", "vector_index domain must contain 1..=1024 UTF-8 bytes", args[0].span);
                        }
                    }
                    if let ExprKind::Int(value) = args[1].kind {
                        if value <= 0 || value > 16384 {
                            self.error("AIF389", "vector_index dimensions must be in 1..=16384", args[1].span);
                        }
                    }
                    let ty = type_node(TypeKind::Named("VectorIndex".into()), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "vector_index_insert" {
                    if args.len() != 4 {
                        self.error("AIF390", "vector_index_insert expects index, non-negative id, domain String, and List<F32> coordinates", expr.span);
                        return None;
                    }
                    let index = self.check_expr(&args[0], locals)?;
                    let id = self.check_expr(&args[1], locals)?;
                    let domain = self.check_expr(&args[2], locals)?;
                    let coordinates = self.check_expr(&args[3], locals)?;
                    if !matches!(&index.kind, TypeKind::Named(ty) if ty == "VectorIndex") {
                        self.error("AIF391", "vector_index_insert requires VectorIndex", args[0].span);
                    }
                    if !is_kind(&id, &TypeKind::Int) {
                        self.error("AIF392", "vector_index_insert id must be Int", args[1].span);
                    }
                    if let ExprKind::Int(value) = args[1].kind {
                        if value < 0 { self.error("AIF393", "vector_index_insert id must be non-negative", args[1].span); }
                    }
                    if !is_kind(&domain, &TypeKind::String) { self.error("AIF394", "vector_index_insert domain must be String", args[2].span); }
                    let expected = TypeKind::List(Box::new(type_node(TypeKind::Named("F32".into()), args[3].span)));
                    if !same_type_kind(&coordinates.kind, &expected) {
                        self.error("AIF401", "vector_index_insert coordinates must be List<F32>", args[3].span);
                    }
                    if !matches!(&args[0].kind, ExprKind::Name(_)) {
                        self.error("AIF395", "vector_index_insert requires a named mutable VectorIndex binding", args[0].span);
                    }
                    return Some(type_node(TypeKind::Unit, expr.span));
                }
                if name == "vector_index_search" {
                    if args.len() != 4 {
                        self.error("AIF396", "vector_index_search expects index, domain String, List<F32> query, and non-negative Int limit", expr.span);
                        return None;
                    }
                    let index = self.check_expr(&args[0], locals)?;
                    let domain = self.check_expr(&args[1], locals)?;
                    let query = self.check_expr(&args[2], locals)?;
                    let limit = self.check_expr(&args[3], locals)?;
                    if !matches!(&index.kind, TypeKind::Named(ty) if ty == "VectorIndex") {
                        self.error("AIF397", "vector_index_search requires VectorIndex", args[0].span);
                    }
                    if !is_kind(&domain, &TypeKind::String) { self.error("AIF398", "vector_index_search domain must be String", args[1].span); }
                    let expected = TypeKind::List(Box::new(type_node(TypeKind::Named("F32".into()), args[2].span)));
                    if !same_type_kind(&query.kind, &expected) {
                        self.error("AIF402", "vector_index_search query must be List<F32>", args[2].span);
                    }
                    if !is_kind(&limit, &TypeKind::Int) {
                        self.error("AIF399", "vector_index_search limit must be Int", args[3].span);
                    }
                    if let ExprKind::Int(value) = args[3].kind {
                        if value < 0 || value > 65536 {
                            self.error("AIF400", "vector_index_search limit must be in 0..=65536", args[3].span);
                        }
                    }
                    let ty = type_node(TypeKind::List(Box::new(type_node(TypeKind::Named("VectorHit".into()), expr.span))), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "mcts_tree" {
                    if args.len() != 2 { self.error("AIF410", "mcts_tree expects max_nodes and max_branches Int bounds", expr.span); return None; }
                    let nodes = self.check_expr(&args[0], locals)?;
                    let branches = self.check_expr(&args[1], locals)?;
                    if !is_kind(&nodes, &TypeKind::Int) { self.error("AIF411", "mcts_tree max_nodes must be Int", args[0].span); }
                    if !is_kind(&branches, &TypeKind::Int) { self.error("AIF412", "mcts_tree max_branches must be Int", args[1].span); }
                    if let ExprKind::Int(v) = args[0].kind { if v <= 0 || v > 4096 { self.error("AIF413", "mcts_tree max_nodes must be in 1..=4096", args[0].span); } }
                    if let ExprKind::Int(v) = args[1].kind { if v <= 0 || v > 64 { self.error("AIF414", "mcts_tree max_branches must be in 1..=64", args[1].span); } }
                    let ty = type_node(TypeKind::Named("MctsTree".into()), expr.span); self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "mcts_root" {
                    if args.len() != 2 { self.error("AIF415", "mcts_root expects a named MctsTree and non-negative state id", expr.span); return None; }
                    let tree = self.check_expr(&args[0], locals)?; let state = self.check_expr(&args[1], locals)?;
                    if !matches!(&tree.kind, TypeKind::Named(t) if t == "MctsTree") { self.error("AIF416", "mcts_root requires MctsTree", args[0].span); }
                    if !is_kind(&state, &TypeKind::Int) { self.error("AIF417", "MCTS state id must be Int", args[1].span); }
                    if !matches!(&args[0].kind, ExprKind::Name(_)) { self.error("AIF418", "MCTS mutations require a named mutable binding", args[0].span); }
                    let ty = type_node(TypeKind::Int, expr.span); self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "mcts_expand" {
                    if args.len() != 3 { self.error("AIF419", "mcts_expand expects tree, parent index, and state id", expr.span); return None; }
                    let tree = self.check_expr(&args[0], locals)?; let parent = self.check_expr(&args[1], locals)?; let state = self.check_expr(&args[2], locals)?;
                    if !matches!(&tree.kind, TypeKind::Named(t) if t == "MctsTree") { self.error("AIF420", "mcts_expand requires MctsTree", args[0].span); }
                    if !is_kind(&parent, &TypeKind::Int) || !is_kind(&state, &TypeKind::Int) { self.error("AIF421", "MCTS parent index and state id must be Int", expr.span); }
                    if !matches!(&args[0].kind, ExprKind::Name(_)) { self.error("AIF422", "MCTS mutations require a named mutable binding", args[0].span); }
                    let ty = type_node(TypeKind::Int, expr.span); self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "mcts_child" {
                    if args.len() != 3 { self.error("AIF423", "mcts_child expects tree, parent index, and child ordinal", expr.span); return None; }
                    let tree = self.check_expr(&args[0], locals)?; let parent = self.check_expr(&args[1], locals)?; let ordinal = self.check_expr(&args[2], locals)?;
                    if !matches!(&tree.kind, TypeKind::Named(t) if t == "MctsTree") { self.error("AIF424", "mcts_child requires MctsTree", args[0].span); }
                    if !is_kind(&parent, &TypeKind::Int) || !is_kind(&ordinal, &TypeKind::Int) { self.error("AIF425", "MCTS parent index and child ordinal must be Int", expr.span); }
                    let ty = type_node(TypeKind::Int, expr.span); self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "mcts_update" || name == "mcts_backpropagate" {
                    if args.len() != 3 { self.error("AIF426", "MCTS update expects tree, node index, and finite F32 reward", expr.span); return None; }
                    let tree = self.check_expr(&args[0], locals)?; let node = self.check_expr(&args[1], locals)?; let reward = self.check_expr(&args[2], locals)?;
                    if !matches!(&tree.kind, TypeKind::Named(t) if t == "MctsTree") { self.error("AIF427", "MCTS update requires MctsTree", args[0].span); }
                    if !is_kind(&node, &TypeKind::Int) { self.error("AIF428", "MCTS node index must be Int", args[1].span); }
                    if !is_kind(&reward, &TypeKind::Named("F32".into())) { self.error("AIF429", "MCTS reward must be F32", args[2].span); }
                    if !matches!(&args[0].kind, ExprKind::Name(_)) { self.error("AIF430", "MCTS mutations require a named mutable binding", args[0].span); }
                    let ty = type_node(TypeKind::Unit, expr.span); self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "mcts_len" {
                    if args.len() != 1 { self.error("AIF431", "mcts_len expects one MctsTree", expr.span); return None; }
                    let tree = self.check_expr(&args[0], locals)?; if !matches!(&tree.kind, TypeKind::Named(t) if t == "MctsTree") { self.error("AIF432", "mcts_len requires MctsTree", args[0].span); }
                    let ty = type_node(TypeKind::Int, expr.span); self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "kv_cache" {
                    if args.len() != 1 { self.error("AIF360", "kv_cache expects one positive Int capacity", expr.span); return None; }
                    let capacity = self.check_expr(&args[0], locals)?;
                    if !is_kind(&capacity, &TypeKind::Int) { self.error("AIF361", "kv_cache capacity must be Int", args[0].span); }
                    let ty = type_node(TypeKind::Named("KvCache".into()), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "kv_len" || name == "kv_capacity" {
                    if args.len() != 1 { self.error("AIF362", format!("{name} expects one KvCache"), expr.span); return None; }
                    let actual = self.check_expr(&args[0], locals)?;
                    if !matches!(&actual.kind, TypeKind::Named(ty) if ty == "KvCache") { self.error("AIF363", format!("{name} requires KvCache"), args[0].span); }
                    return Some(type_node(TypeKind::Int, expr.span));
                }
                if name == "kv_append" {
                    if args.len() != 4 { self.error("AIF364", "kv_append expects cache, key list, value list, and policy", expr.span); return None; }
                    let cache = self.check_expr(&args[0], locals)?;
                    let key = self.check_expr(&args[1], locals)?;
                    let value = self.check_expr(&args[2], locals)?;
                    let policy = self.check_expr(&args[3], locals)?;
                    if !matches!(&cache.kind, TypeKind::Named(ty) if ty == "KvCache") { self.error("AIF365", "kv_append requires a KvCache binding", args[0].span); }
                    if !is_numeric_list(&key) { self.error("AIF366", "KVCache key must be a numeric list", args[1].span); }
                    if !is_numeric_list(&value) { self.error("AIF367", "KVCache value must be a numeric list", args[2].span); }
                    if !is_kind(&policy, &TypeKind::String) || !matches!(&args[3].kind, ExprKind::String(value) if value == "reject" || value == "evict_oldest") { self.error("AIF368", "KVCache policy must be the literal \"reject\" or \"evict_oldest\"", args[3].span); }
                    if !matches!(&args[0].kind, ExprKind::Name(_)) { self.error("AIF369", "kv_append requires a named mutable KvCache binding", args[0].span); }
                    return Some(type_node(TypeKind::Unit, expr.span));
                }
                if name == "context_window" {
                    if args.len() != 3 { self.error("AIF370", "context_window expects segment, byte, and token budgets", expr.span); return None; }
                    for arg in args { let actual = self.check_expr(arg, locals)?; if !is_kind(&actual, &TypeKind::Int) { self.error("AIF371", "ContextWindow budgets must be Int", arg.span); } }
                    let ty = type_node(TypeKind::Named("ContextWindow".into()), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "context_len" || name == "context_bytes" || name == "context_tokens" {
                    if args.len() != 1 { self.error("AIF372", format!("{name} expects one ContextWindow"), expr.span); return None; }
                    let actual = self.check_expr(&args[0], locals)?;
                    if !matches!(&actual.kind, TypeKind::Named(ty) if ty == "ContextWindow") { self.error("AIF373", format!("{name} requires ContextWindow"), args[0].span); }
                    return Some(type_node(TypeKind::Int, expr.span));
                }
                if name == "context_append" {
                    if args.len() != 4 { self.error("AIF374", "context_append expects window, modality, payload, and token count", expr.span); return None; }
                    let window = self.check_expr(&args[0], locals)?;
                    let modality = self.check_expr(&args[1], locals)?;
                    let payload = self.check_expr(&args[2], locals)?;
                    let tokens = self.check_expr(&args[3], locals)?;
                    if !matches!(&window.kind, TypeKind::Named(ty) if ty == "ContextWindow") { self.error("AIF375", "context_append requires a ContextWindow binding", args[0].span); }
                    if !is_kind(&modality, &TypeKind::String) || !matches!(&args[1].kind, ExprKind::String(value) if ["text", "image", "audio", "video", "structured"].contains(&value.as_str())) { self.error("AIF376", "context modality must be a supported string literal", args[1].span); }
                    if !is_kind(&payload, &TypeKind::String) { self.error("AIF377", "context payload must be String bytes", args[2].span); }
                    if !is_kind(&tokens, &TypeKind::Int) { self.error("AIF378", "context token count must be Int", args[3].span); }
                    if !matches!(&args[0].kind, ExprKind::Name(_)) { self.error("AIF379", "context_append requires a named mutable ContextWindow binding", args[0].span); }
                    return Some(type_node(TypeKind::Unit, expr.span));
                }
                if name == "activation_cache" {
                    if args.len()!=3 { self.error("AIF510","activation_cache expects capacity, checkpoint_stride, and max_bytes Int",expr.span); return None; }
                    let capacity=self.check_expr(&args[0],locals)?; let stride=self.check_expr(&args[1],locals)?; let bytes=self.check_expr(&args[2],locals)?;
                    for (ty,arg) in [(&capacity,&args[0]),(&stride,&args[1]),(&bytes,&args[2])] { if !is_kind(ty,&TypeKind::Int) { self.error("AIF511","ActivationCache limits must be Int",arg.span); } }
                    for (arg,label,max) in [(&args[0],"capacity",4096i64),(&args[1],"checkpoint_stride",4096i64),(&args[2],"max_bytes",16777216i64)] { if let ExprKind::Int(v)=arg.kind { if v<=0||v>max { self.error("AIF512",format!("ActivationCache {label} must be in 1..={max}"),arg.span); } } }
                    let ty=type_node(TypeKind::Named("ActivationCache".into()),expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "activation_store" {
                    if args.len()!=3 { self.error("AIF513","activation_store expects named cache, checkpoint ID, and Tensor<F32>",expr.span); return None; }
                    let cache=self.check_expr(&args[0],locals)?; let step=self.check_expr(&args[1],locals)?; let tensor=self.check_expr(&args[2],locals)?;
                    if !matches!(&cache.kind,TypeKind::Named(n) if n=="ActivationCache") { self.error("AIF514","activation_store requires ActivationCache",args[0].span); }
                    if !is_kind(&step,&TypeKind::Int) { self.error("AIF515","ActivationCache checkpoint ID must be Int",args[1].span); }
                    if !matches!(&tensor.kind,TypeKind::Generic(n,a) if n=="Tensor"&&a.len()==1&&matches!(&a[0].kind,TypeKind::Named(e) if e=="F32")) { self.error("AIF516","ActivationCache values must be Tensor<F32>",args[2].span); }
                    if !matches!(&args[0].kind,ExprKind::Name(_)) { self.error("AIF517","ActivationCache mutation requires a named binding",args[0].span); }
                    let ty=type_node(TypeKind::Unit,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "activation_restore" {
                    if args.len()!=2 { self.error("AIF518","activation_restore expects ActivationCache and checkpoint ID",expr.span); return None; }
                    let cache=self.check_expr(&args[0],locals)?; let step=self.check_expr(&args[1],locals)?;
                    if !matches!(&cache.kind,TypeKind::Named(n) if n=="ActivationCache") { self.error("AIF519","activation_restore requires ActivationCache",args[0].span); }
                    if !is_kind(&step,&TypeKind::Int) { self.error("AIF520","ActivationCache checkpoint ID must be Int",args[1].span); }
                    let ty=type_node(TypeKind::Generic("Tensor".into(),vec![type_node(TypeKind::Named("F32".into()),expr.span)]),expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "activation_cache_len" {
                    if args.len()!=1 { self.error("AIF521","activation_cache_len expects ActivationCache",expr.span); return None; }
                    let cache=self.check_expr(&args[0],locals)?;
                    if !matches!(&cache.kind,TypeKind::Named(n) if n=="ActivationCache") { self.error("AIF522","activation_cache_len requires ActivationCache",args[0].span); }
                    let ty=type_node(TypeKind::Int,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "tokenizer_state" {
                    if args.len()!=4 { self.error("AIF540","tokenizer_state expects vocabulary List<String>, max_pending Int, max_tokens Int, and unknown_id Int (-1 rejects unknowns)",expr.span); return None; }
                    let vocab=self.check_expr(&args[0],locals)?; let pending=self.check_expr(&args[1],locals)?; let tokens=self.check_expr(&args[2],locals)?; let unknown=self.check_expr(&args[3],locals)?;
                    if !matches!(&vocab.kind,TypeKind::List(element) if matches!(&element.kind,TypeKind::String)) { self.error("AIF541","TokenizerState vocabulary must be List<String>",args[0].span); }
                    for (ty,arg,label,max) in [(&pending,&args[1],"max_pending",4i64),(&tokens,&args[2],"max_tokens",65536i64)] { if !is_kind(ty,&TypeKind::Int) { self.error("AIF542",format!("TokenizerState {label} must be Int"),arg.span); } if let ExprKind::Int(v)=arg.kind { if v<=0||v>max { self.error("AIF543",format!("TokenizerState {label} must be in 1..={max}"),arg.span); } } }
                    if !is_kind(&unknown,&TypeKind::Int) { self.error("AIF544","TokenizerState unknown_id must be Int (-1 or non-negative)",args[3].span); }
                    if let ExprKind::Int(v)=args[3].kind { if v < -1 { self.error("AIF545","TokenizerState unknown_id must be -1 or non-negative",args[3].span); } }
                    if let ExprKind::List(items)=&args[0].kind { let mut seen=std::collections::BTreeSet::new(); for item in items { if let ExprKind::String(piece)=&item.kind { if piece.chars().count()!=1 || !seen.insert(piece.clone()) { self.error("AIF546","TokenizerState vocabulary entries must be unique single Unicode scalars",item.span); } } } }
                    let ty=type_node(TypeKind::Named("TokenizerState".into()),expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "tokenizer_feed" {
                    if args.len()!=2 { self.error("AIF547","tokenizer_feed expects named TokenizerState and List<Int> bytes",expr.span); return None; }
                    let state=self.check_expr(&args[0],locals)?; let bytes=self.check_expr(&args[1],locals)?;
                    if !matches!(&state.kind,TypeKind::Named(n) if n=="TokenizerState") { self.error("AIF548","tokenizer_feed requires TokenizerState",args[0].span); }
                    if !matches!(&bytes.kind,TypeKind::List(element) if matches!(&element.kind,TypeKind::Int)) { self.error("AIF549","tokenizer_feed bytes must be List<Int>",args[1].span); }
                    if !matches!(&args[0].kind,ExprKind::Name(_)) { self.error("AIF550","TokenizerState mutation requires a named binding",args[0].span); }
                    let ty=type_node(TypeKind::Unit,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "tokenizer_flush" {
                    if args.len()!=1 { self.error("AIF551","tokenizer_flush expects named TokenizerState",expr.span); return None; }
                    let state=self.check_expr(&args[0],locals)?;
                    if !matches!(&state.kind,TypeKind::Named(n) if n=="TokenizerState") { self.error("AIF552","tokenizer_flush requires TokenizerState",args[0].span); }
                    if !matches!(&args[0].kind,ExprKind::Name(_)) { self.error("AIF553","TokenizerState mutation requires a named binding",args[0].span); }
                    let ty=type_node(TypeKind::Unit,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "tokenizer_token_count" {
                    if args.len()!=1 { self.error("AIF554","tokenizer_token_count expects TokenizerState",expr.span); return None; }
                    let state=self.check_expr(&args[0],locals)?; if !matches!(&state.kind,TypeKind::Named(n) if n=="TokenizerState") { self.error("AIF555","tokenizer_token_count requires TokenizerState",args[0].span); }
                    let ty=type_node(TypeKind::Int,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "tokenizer_token" {
                    if args.len()!=2 { self.error("AIF556","tokenizer_token expects TokenizerState and token index Int",expr.span); return None; }
                    let state=self.check_expr(&args[0],locals)?; let index=self.check_expr(&args[1],locals); if !matches!(&state.kind,TypeKind::Named(n) if n=="TokenizerState") { self.error("AIF557","tokenizer_token requires TokenizerState",args[0].span); }
                    if let Some(index)=index { if !is_kind(&index,&TypeKind::Int) { self.error("AIF558","tokenizer_token index must be Int",args[1].span); } }
                    let ty=type_node(TypeKind::Int,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "gradient_tape" {
                    if args.len()!=1 { self.error("AIF490","gradient_tape expects capacity Int",expr.span); return None; }
                    let capacity=self.check_expr(&args[0],locals)?;
                    if !is_kind(&capacity,&TypeKind::Int) { self.error("AIF491","GradientTape capacity must be Int",args[0].span); }
                    if let ExprKind::Int(v)=args[0].kind { if v<=0||v>4096 { self.error("AIF492","GradientTape capacity must be in 1..=4096",args[0].span); } }
                    let ty=type_node(TypeKind::Named("GradientTape".into()),expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "gradient_input" {
                    if args.len()!=2 { self.error("AIF493","gradient_input expects named tape and finite F32/Int value",expr.span); return None; }
                    let tape=self.check_expr(&args[0],locals)?; let value=self.check_expr(&args[1],locals)?;
                    if !matches!(&tape.kind,TypeKind::Named(n) if n=="GradientTape") { self.error("AIF494","gradient_input requires GradientTape",args[0].span); }
                    if !is_kind(&value,&TypeKind::Int)&&!matches!(&value.kind,TypeKind::Named(n) if n=="F32") { self.error("AIF495","GradientTape values must be F32 or Int",args[1].span); }
                    if !matches!(&args[0].kind,ExprKind::Name(_)) { self.error("AIF496","GradientTape mutation requires a named binding",args[0].span); }
                    let ty=type_node(TypeKind::Int,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "gradient_add" || name == "gradient_mul" {
                    if args.len()!=3 { self.error("AIF497","gradient_add/mul expects named tape and two prior slot IDs",expr.span); return None; }
                    let tape=self.check_expr(&args[0],locals)?; let left=self.check_expr(&args[1],locals)?; let right=self.check_expr(&args[2],locals)?;
                    if !matches!(&tape.kind,TypeKind::Named(n) if n=="GradientTape") { self.error("AIF498","gradient_add/mul requires GradientTape",args[0].span); }
                    if !is_kind(&left,&TypeKind::Int)||!is_kind(&right,&TypeKind::Int) { self.error("AIF499","GradientTape slot IDs must be Int",expr.span); }
                    if !matches!(&args[0].kind,ExprKind::Name(_)) { self.error("AIF496","GradientTape mutation requires a named binding",args[0].span); }
                    let ty=type_node(TypeKind::Int,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "gradient_backward" {
                    if args.len()!=3 { self.error("AIF500","gradient_backward expects named tape, output slot ID, and seed",expr.span); return None; }
                    let tape=self.check_expr(&args[0],locals)?; let output=self.check_expr(&args[1],locals)?; let seed=self.check_expr(&args[2],locals)?;
                    if !matches!(&tape.kind,TypeKind::Named(n) if n=="GradientTape") { self.error("AIF501","gradient_backward requires GradientTape",args[0].span); }
                    if !is_kind(&output,&TypeKind::Int) { self.error("AIF502","GradientTape output slot must be Int",args[1].span); }
                    if !is_kind(&seed,&TypeKind::Int)&&!matches!(&seed.kind,TypeKind::Named(n) if n=="F32") { self.error("AIF503","GradientTape seed must be F32 or Int",args[2].span); }
                    if !matches!(&args[0].kind,ExprKind::Name(_)) { self.error("AIF496","GradientTape mutation requires a named binding",args[0].span); }
                    let ty=type_node(TypeKind::Unit,expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "gradient_value" || name == "gradient_get" {
                    if args.len()!=2 { self.error("AIF504","gradient_value/get expects GradientTape and slot ID",expr.span); return None; }
                    let tape=self.check_expr(&args[0],locals)?; let slot=self.check_expr(&args[1],locals)?;
                    if !matches!(&tape.kind,TypeKind::Named(n) if n=="GradientTape") { self.error("AIF505","gradient query requires GradientTape",args[0].span); }
                    if !is_kind(&slot,&TypeKind::Int) { self.error("AIF506","GradientTape slot ID must be Int",args[1].span); }
                    let ty=type_node(TypeKind::Named("F32".into()),expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "attention_block_new" {
                    if args.len() != 4 { self.error("AIF452", "attention_block_new expects hidden, heads, head_dim Int and causal Bool", expr.span); return None; }
                    let hidden=self.check_expr(&args[0],locals)?; let heads=self.check_expr(&args[1],locals)?; let dim=self.check_expr(&args[2],locals)?; let causal=self.check_expr(&args[3],locals)?;
                    for (ty,arg) in [(&hidden,&args[0]),(&heads,&args[1]),(&dim,&args[2])] { if !is_kind(ty,&TypeKind::Int) { self.error("AIF453","AttentionBlock dimensions must be Int",arg.span); } }
                    if !is_kind(&causal,&TypeKind::Bool) { self.error("AIF454","AttentionBlock causal flag must be Bool",args[3].span); }
                    if let (ExprKind::Int(h),ExprKind::Int(n),ExprKind::Int(d))=(&args[0].kind,&args[1].kind,&args[2].kind) {
                        if *h<=0 || *n<=0 || *d<=0 || *n>256 || n.checked_mul(*d) != Some(*h) { self.error("AIF455","AttentionBlock requires positive dimensions and hidden == heads * head_dim (heads <= 256)",expr.span); }
                    }
                    let ty=type_node(TypeKind::Named("AttentionBlock".into()),expr.span); self.inferred_types.insert(expr.id,ty.clone()); return Some(ty);
                }
                if name == "attention_block_apply" {
                    if args.len()!=4 { self.error("AIF456","attention_block_apply expects AttentionBlock, Q, K, V tensors",expr.span); return None; }
                    let block=self.check_expr(&args[0],locals)?; let q=self.check_expr(&args[1],locals)?; let k=self.check_expr(&args[2],locals)?; let v=self.check_expr(&args[3],locals)?;
                    if !matches!(&block.kind,TypeKind::Named(n) if n=="AttentionBlock"){self.error("AIF457","attention_block_apply requires an AttentionBlock value",args[0].span);}
                    let tensor=|ty:&Type|matches!(&ty.kind,TypeKind::Generic(n,a) if n=="Tensor"&&a.len()==1&&matches!(&a[0].kind,TypeKind::Named(e) if e=="F32"));
                    for (ty,arg) in [(&q,&args[1]),(&k,&args[2]),(&v,&args[3])] { if !tensor(ty){self.error("AIF458","AttentionBlock Q/K/V must be Tensor<F32>",arg.span);} }
                    let ty=type_node(TypeKind::Generic("Tensor".into(),vec![type_node(TypeKind::Named("F32".into()),expr.span)]),expr.span);self.inferred_types.insert(expr.id,ty.clone());return Some(ty);
                }
                if name == "attention_block_cached" {
                    if args.len()!=3 { self.error("AIF459","attention_block_cached expects AttentionBlock, query Tensor<F32>, and KvCache",expr.span);return None; }
                    let block=self.check_expr(&args[0],locals)?;let q=self.check_expr(&args[1],locals)?;let cache=self.check_expr(&args[2],locals)?;
                    if !matches!(&block.kind,TypeKind::Named(n) if n=="AttentionBlock"){self.error("AIF460","attention_block_cached requires an AttentionBlock",args[0].span);}
                    if !matches!(&q.kind,TypeKind::Generic(n,a) if n=="Tensor"&&a.len()==1&&matches!(&a[0].kind,TypeKind::Named(e) if e=="F32")){self.error("AIF461","attention_block_cached query must be Tensor<F32>",args[1].span);}
                    if !matches!(&cache.kind,TypeKind::Named(n) if n=="KvCache"){self.error("AIF462","attention_block_cached requires KvCache",args[2].span);}
                    let ty=type_node(TypeKind::Generic("Tensor".into(),vec![type_node(TypeKind::Named("F32".into()),expr.span)]),expr.span);self.inferred_types.insert(expr.id,ty.clone());return Some(ty);
                }
                if name == "lora_adapter_new" {
                    if args.len()!=4 { self.error("AIF463","lora_adapter_new expects A, B Tensor<F32>, alpha F32/Int, rank Int",expr.span);return None; }
                    let a=self.check_expr(&args[0],locals)?;let b=self.check_expr(&args[1],locals)?;let alpha=self.check_expr(&args[2],locals)?;let rank=self.check_expr(&args[3],locals)?;
                    let tensor=|ty:&Type|matches!(&ty.kind,TypeKind::Generic(n,a) if n=="Tensor"&&a.len()==1&&matches!(&a[0].kind,TypeKind::Named(e) if e=="F32"));
                    for (ty,arg) in [(&a,&args[0]),(&b,&args[1])] {if !tensor(ty){self.error("AIF464","LoRAAdapter A/B must be Tensor<F32>",arg.span);}}
                    if !is_kind(&alpha,&TypeKind::Int)&&!matches!(&alpha.kind,TypeKind::Named(n) if n=="F32"){self.error("AIF465","LoRAAdapter alpha must be F32 or Int",args[2].span);}
                    if !is_kind(&rank,&TypeKind::Int){self.error("AIF466","LoRAAdapter rank must be Int",args[3].span);}
                    if let ExprKind::Int(v)=args[3].kind {if v<=0||v>256{self.error("AIF467","LoRAAdapter rank must be in 1..=256",args[3].span);}}
                    let ty=type_node(TypeKind::Named("LoRAAdapter".into()),expr.span);self.inferred_types.insert(expr.id,ty.clone());return Some(ty);
                }
                if name == "lora_adapter_apply" {
                    if args.len()!=2 {self.error("AIF468","lora_adapter_apply expects LoRAAdapter and input Tensor<F32>",expr.span);return None;}
                    let adapter=self.check_expr(&args[0],locals)?;let input=self.check_expr(&args[1],locals)?;
                    if !matches!(&adapter.kind,TypeKind::Named(n) if n=="LoRAAdapter"){self.error("AIF469","lora_adapter_apply requires a LoRAAdapter value",args[0].span);}
                    if !matches!(&input.kind,TypeKind::Generic(n,a) if n=="Tensor"&&a.len()==1&&matches!(&a[0].kind,TypeKind::Named(e) if e=="F32")){self.error("AIF470","LoRAAdapter input must be Tensor<F32>",args[1].span);}
                    let ty=type_node(TypeKind::Generic("Tensor".into(),vec![type_node(TypeKind::Named("F32".into()),expr.span)]),expr.span);self.inferred_types.insert(expr.id,ty.clone());return Some(ty);
                }
                if name == "attention_block" {
                    if args.len() != 4 { self.error("AIF440", "attention_block expects Q, K, V Tensor<F32> values and a causal Bool", expr.span); return None; }
                    let q = self.check_expr(&args[0], locals)?;
                    let k = self.check_expr(&args[1], locals)?;
                    let v = self.check_expr(&args[2], locals)?;
                    let causal = self.check_expr(&args[3], locals)?;
                    let tensor = |ty: &Type| matches!(&ty.kind, TypeKind::Generic(name, args) if name == "Tensor" && args.len() == 1 && matches!(&args[0].kind, TypeKind::Named(element) if element == "F32"));
                    for (ty, arg) in [(&q, &args[0]), (&k, &args[1]), (&v, &args[2])] {
                        if !tensor(ty) { self.error("AIF441", "attention_block Q/K/V must be Tensor<F32>", arg.span); }
                    }
                    if !is_kind(&causal, &TypeKind::Bool) { self.error("AIF442", "attention_block causal flag must be Bool", args[3].span); }
                    let ty = type_node(TypeKind::Generic("Tensor".into(), vec![type_node(TypeKind::Named("F32".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "attention_cached" {
                    if args.len() != 2 { self.error("AIF448", "attention_cached expects query Tensor<F32> and a KvCache", expr.span); return None; }
                    let query = self.check_expr(&args[0], locals)?;
                    let cache = self.check_expr(&args[1], locals)?;
                    if !matches!(&query.kind, TypeKind::Generic(name, args) if name == "Tensor" && args.len() == 1 && matches!(&args[0].kind, TypeKind::Named(element) if element == "F32")) {
                        self.error("AIF449", "attention_cached query must be Tensor<F32>", args[0].span);
                    }
                    if !matches!(&cache.kind, TypeKind::Named(name) if name == "KvCache") {
                        self.error("AIF450", "attention_cached requires a KvCache", args[1].span);
                    }
                    let ty = type_node(TypeKind::Generic("Tensor".into(), vec![type_node(TypeKind::Named("F32".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "lora_apply" {
                    if args.len() != 5 { self.error("AIF443", "lora_apply expects input, A, B Tensor<F32>, alpha F32, and rank Int", expr.span); return None; }
                    let input = self.check_expr(&args[0], locals)?;
                    let a = self.check_expr(&args[1], locals)?;
                    let b = self.check_expr(&args[2], locals)?;
                    let alpha = self.check_expr(&args[3], locals)?;
                    let rank = self.check_expr(&args[4], locals)?;
                    let tensor = |ty: &Type| matches!(&ty.kind, TypeKind::Generic(name, args) if name == "Tensor" && args.len() == 1 && matches!(&args[0].kind, TypeKind::Named(element) if element == "F32"));
                    for (ty, arg) in [(&input, &args[0]), (&a, &args[1]), (&b, &args[2])] {
                        if !tensor(ty) { self.error("AIF444", "lora_apply input/A/B must be Tensor<F32>", arg.span); }
                    }
                    if !is_kind(&alpha, &TypeKind::Int) && !matches!(&alpha.kind, TypeKind::Named(name) if name == "F32") { self.error("AIF445", "lora_apply alpha must be F32 or Int", args[3].span); }
                    if !is_kind(&rank, &TypeKind::Int) { self.error("AIF446", "lora_apply rank must be Int", args[4].span); }
                    if let ExprKind::Int(value) = args[4].kind { if value <= 0 || value > 256 { self.error("AIF447", "lora_apply rank must be in 1..=256", args[4].span); } }
                    let ty = type_node(TypeKind::Generic("Tensor".into(), vec![type_node(TypeKind::Named("F32".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone()); return Some(ty);
                }
                if name == "tensor" {
                    if args.len() != 2 { self.error("AIF340", "tensor expects shape and value lists", expr.span); return None; }
                    let shape = self.check_expr(&args[0], locals)?;
                    let values = self.check_expr(&args[1], locals)?;
                    if !is_int_list(&shape) { self.error("AIF341", "tensor shape must be a List<Int>", args[0].span); }
                    if !is_numeric_list(&values) { self.error("AIF342", "tensor values must be a homogeneous List<Int> or List<F32>", args[1].span); }
                    let ty = type_node(TypeKind::Generic("Tensor".into(), vec![type_node(TypeKind::Named("F32".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "quantized_tensor" {
                    if args.len() != 4 { self.error("AIF350", "quantized_tensor expects shape, scale, zero point, and values", expr.span); return None; }
                    let shape = self.check_expr(&args[0], locals)?;
                    let scale = self.check_expr(&args[1], locals)?;
                    let zero_point = self.check_expr(&args[2], locals)?;
                    let values = self.check_expr(&args[3], locals)?;
                    if !is_int_list(&shape) { self.error("AIF351", "quantized tensor shape must be a List<Int>", args[0].span); }
                    if !matches!(&scale.kind, TypeKind::Named(name) if name == "F32") && !is_kind(&scale, &TypeKind::Int) {
                        self.error("AIF352", "quantization scale must be an F32 or positive Int", args[1].span);
                    }
                    if !is_kind(&zero_point, &TypeKind::Int) { self.error("AIF353", "quantized tensor zero point must be an Int in [-128, 127]", args[2].span); }
                    if !is_int_list(&values) { self.error("AIF354", "quantized tensor values must be a List<Int> in [-128, 127]", args[3].span); }
                    let ty = type_node(TypeKind::Generic("QuantizedTensor".into(), vec![type_node(TypeKind::Named("I8".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "dequantize" {
                    if args.len() != 1 { self.error("AIF355", "dequantize expects one QuantizedTensor argument", expr.span); return None; }
                    let actual = self.check_expr(&args[0], locals)?;
                    let expected = TypeKind::Generic("QuantizedTensor".into(), vec![type_node(TypeKind::Named("I8".into()), expr.span)]);
                    if !same_type_kind(&actual.kind, &expected) { self.error("AIF356", "dequantize requires a QuantizedTensor<I8> value", args[0].span); }
                    let ty = type_node(TypeKind::Generic("Tensor".into(), vec![type_node(TypeKind::Named("F32".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "tensor_add" || name == "tensor_matmul" {
                    if args.len() != 2 { self.error("AIF343", format!("{name} expects two Tensor<F32> arguments"), expr.span); return None; }
                    let expected = TypeKind::Generic("Tensor".into(), vec![type_node(TypeKind::Named("F32".into()), expr.span)]);
                    for arg in args { if let Some(actual) = self.check_expr(arg, locals) { if !same_type_kind(&actual.kind, &expected) { self.error("AIF344", format!("{name} requires Tensor<F32> arguments"), arg.span); } } }
                    let ty = type_node(expected, expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "embedding" {
                    if args.len() != 2 { self.error("AIF345", "embedding expects a domain string and a value list", expr.span); return None; }
                    let domain = self.check_expr(&args[0], locals)?;
                    let values = self.check_expr(&args[1], locals)?;
                    if !is_kind(&domain, &TypeKind::String) { self.error("AIF346", "embedding domain must be a String", args[0].span); }
                    if !is_numeric_list(&values) { self.error("AIF347", "embedding coordinates must be a homogeneous List<Int> or List<F32>", args[1].span); }
                    let ty = type_node(TypeKind::Generic("Embedding".into(), vec![type_node(TypeKind::Named("DynamicDomain".into()), expr.span)]), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
                if name == "embedding_dot" {
                    if args.len() != 2 { self.error("AIF348", "embedding_dot expects two Embedding arguments", expr.span); return None; }
                    let expected = TypeKind::Generic("Embedding".into(), vec![type_node(TypeKind::Named("DynamicDomain".into()), expr.span)]);
                    for arg in args { if let Some(actual) = self.check_expr(arg, locals) { if !same_type_kind(&actual.kind, &expected) { self.error("AIF349", "embedding_dot requires Embedding values created by embedding(...)", arg.span); } } }
                    let ty = type_node(TypeKind::Named("F32".into()), expr.span);
                    self.inferred_types.insert(expr.id, ty.clone());
                    return Some(ty);
                }
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

fn same_type_kind(left: &TypeKind, right: &TypeKind) -> bool {
    match (left, right) {
        (TypeKind::Generic(left_name, left_args), TypeKind::Generic(right_name, right_args)) => {
            left_name == right_name
                && left_args.len() == right_args.len()
                && left_args.iter().zip(right_args).all(|(left, right)| same_type_kind(&left.kind, &right.kind))
        }
        (TypeKind::List(left), TypeKind::List(right)) => same_type_kind(&left.kind, &right.kind),
        (TypeKind::Result(left_ok, left_err), TypeKind::Result(right_ok, right_err)) => {
            same_type_kind(&left_ok.kind, &right_ok.kind) && same_type_kind(&left_err.kind, &right_err.kind)
        }
        _ => left == right,
    }
}

fn is_int_list(ty: &Type) -> bool {
    matches!(&ty.kind, TypeKind::List(element) if matches!(&element.kind, TypeKind::Int))
}

fn is_numeric_list(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::List(element) => matches!(&element.kind, TypeKind::Int) || matches!(&element.kind, TypeKind::Named(name) if name == "F32"),
        _ => false,
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
    fn validates_ai_native_type_constructor_arity_and_numeric_elements() {
        let valid = parse("module ai\nfn tensor(value: Tensor<F32>) -> Tensor<F32>\n  value\nfn embed(value: Embedding<TextDomain>) -> Embedding<TextDomain>\n  value\nfn prob(value: Probability) -> Probability\n  value\nfn quant(value: QuantizedTensor<I8>) -> QuantizedTensor<I8>\n  value\n").unwrap();
        assert!(check(&valid).is_ok(), "valid AI type signatures should pass: {:?}", check(&valid).err());

        let invalid = parse("module bad_ai\nfn tensor(value: Tensor<String>) -> Tensor<String>\n  value\nfn embed(value: Embedding<TextDomain, ImageDomain>) -> Embedding<TextDomain, ImageDomain>\n  value\nfn quant(value: QuantizedTensor<F32>) -> QuantizedTensor<F32>\n  value\n").unwrap();
        let errors = check(&invalid).expect_err("invalid AI type parameters must be rejected");
        assert!(errors.iter().filter(|error| error.code == "AIF320").count() >= 3);
    }

    #[test]
    fn construct_analysis_produces_typed_metadata_but_normal_analysis_fails_closed() {
        let module = parse(
            "module constructs\ntrace Evidence\n  signature: sha256\ncell State\n  value: Int\n  invariant: valid\nvault NoNetwork\n  denies: network\n  capabilities: filesystem\nproof P\n  requires: valid\n  ensures: true\n  on_unknown: reject\n  required: true\nphase Lifecycle\n  Ready -> Done\n",
        ).unwrap();

        let constructs = analyze_constructs(&module).unwrap();
        assert_eq!(constructs.len(), 5);
        assert!(matches!(constructs[0].kind, ConstructKind::Trace));
        assert!(matches!(
            constructs[1].members.first(),
            Some(TypedConstructMember::Field { name, ty: TypeKind::Int, .. }) if name == "value"
        ));
        assert!(matches!(
            constructs[4].members.first(),
            Some(TypedConstructMember::Transition { from, to, .. }) if from == "Ready" && to == "Done"
        ));

        let errors = analyze(&module).expect_err("constructs must remain fail-closed until runtime enforcement exists");
        assert!(errors.iter().any(|error| error.code == "AIF610"));
    }

    #[test]
    fn construct_analysis_rejects_missing_required_members() {
        let module = parse(
            "module missing_construct_members\ntrace MissingTrace\n  value: Int\ncell MissingCell\n  value: Int\nvault MissingVault\n  denies: network\nproof MissingProof\n  requires: valid\nphase PresentPhase\n  ready -> done\n",
        )
        .expect("constructs with valid syntax should parse for semantic diagnostics");

        let errors = analyze_constructs(&module)
            .expect_err("every construct must declare its minimum required members");
        let required_errors = errors.iter().filter(|error| error.code == "AIF617").count();
        assert!(
            required_errors >= 6,
            "expected missing signature, invariant, capabilities, and three proof clauses; got {:?}",
            errors
        );
    }

    #[test]
    fn construct_analysis_rejects_empty_clauses_and_unsupported_unknown_policy() {
        let module = parse(
            "module invalid_clause_values\ntrace Evidence\n  signature: \ncell State\n  value: Int\n  invariant: \nvault Safe\n  capabilities: clock\nproof Contract\n  requires: valid\n  ensures: true\n  on_unknown: allow\n  required: maybe\nphase Lifecycle\n  Ready -> Done\n",
        )
        .expect("empty clause values remain parseable for semantic diagnostics");

        let errors = analyze_constructs(&module)
            .expect_err("empty or unsupported policy values must be rejected");
        assert!(
            errors.iter().filter(|error| error.code == "AIF618").count() >= 4,
            "expected empty signature, empty invariant, unsupported on_unknown, and invalid required diagnostics; got {:?}",
            errors
        );
    }

    #[test]
    fn infers_return_through_later_declaration() {
        let module = parse("module x\nfn main() -> Int\n  value()\nfn value()\n  42\n").unwrap();
        assert!(check(&module).is_ok());
    }
}

#[cfg(test)]
mod ai_routing_semantic_tests {
    use super::*;

    fn rejects(source: &str, code: &str) {
        let module = crate::parse(source).expect("fixture parses");
        let errors = analyze(&module).expect_err("invalid AI builtin use must be rejected");
        assert!(errors.iter().any(|error| error.code == code), "expected {code}, got {errors:?}");
    }

    #[test]
    fn route_top_k_rejects_invalid_top_k_type_and_static_bounds() {
        rejects("module bad\nfn main()\n  route_top_k([1.0], [false], \"bad\")\n", "AIF383");
        rejects("module bad\nfn main()\n  route_top_k([1.0], [false], 0)\n", "AIF384");
        rejects("module bad\nfn main()\n  route_top_k([1.0], [false], 33)\n", "AIF384");
    }

    #[test]
    fn vector_index_rejects_invalid_static_dimensions_and_domain() {
        rejects("module bad\nfn main()\n  vector_index(\"\", 2)\n", "AIF388");
        rejects("module bad\nfn main()\n  vector_index(\"text\", 0)\n", "AIF389");
    }

    #[test]
    fn vector_index_insert_and_search_reject_wrong_types() {
        rejects("module bad\nfn main()\n  let index = vector_index(\"text\", 2)\n  vector_index_insert(index, true, \"text\", [1.0, 2.0])\n", "AIF392");
        rejects("module bad\nfn main()\n  let index = vector_index(\"text\", 2)\n  vector_index_search(index, 7, [1.0, 2.0], 1)\n", "AIF398");
    }
}
