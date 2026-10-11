use crate::source::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub id: NodeId,
    pub span: Span,
    pub name: String,
    pub items: Vec<Item>,
    pub constructs: Vec<ConstructDeclaration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    Function(Function),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstructKind {
    Trace,
    Cell,
    Vault,
    Proof,
    Phase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructDeclaration {
    pub id: NodeId,
    pub span: Span,
    pub kind: ConstructKind,
    pub name: String,
    pub members: Vec<ConstructMember>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstructMember {
    Field { name: String, ty: Type, span: Span },
    Clause { name: String, value: String, span: Span },
    Transition { from: String, to: String, span: Span },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub id: NodeId,
    pub span: Span,
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Option<Type>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub id: NodeId,
    pub span: Span,
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub id: NodeId,
    pub span: Span,
    pub kind: TypeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Int,
    Float(u32),
    Bool,
    String,
    Unit,
    Named(String),
    Generic(String, Vec<Type>),
    Result(Box<Type>, Box<Type>),
    List(Box<Type>),
}

impl Type {
    pub fn display_name(&self) -> String {
        self.kind.display_name()
    }
}

impl TypeKind {
    pub fn display_name(&self) -> String {
        match self {
            Self::Int => "Int".into(),
            Self::Bool => "Bool".into(),
            Self::String => "String".into(),
            Self::Unit => "()".into(),
            Self::Named(name) => name.clone(),
            Self::Generic(name, args) => format!("{}<{}>", name, args.iter().map(Type::display_name).collect::<Vec<_>>().join(", ")),
            Self::Result(ok, err) => {
                format!("Result<{}, {}>", ok.display_name(), err.display_name())
            }
            Self::List(element) => format!("List<{}>", element.display_name()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub id: NodeId,
    pub span: Span,
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stmt {
    pub id: NodeId,
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Statement forms include explicit mutable assignment via `set`.
pub enum StmtKind {
    Let {
        name: String,
        value: Expr,
    },
    Set {
        name: String,
        value: Expr,
    },
    SetIndex {
        collection: Expr,
        index: Expr,
        value: Expr,
    },
    Return(Option<Expr>),
    Expr(Expr),
    Scope {
        body: Block,
    },
    Spawn {
        name: String,
        call: Expr,
    },
    Join {
        name: String,
    },
    Cancel {
        name: String,
    },
    While {
        condition: Expr,
        body: Block,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub id: NodeId,
    pub span: Span,
    pub kind: ExprKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprKind {
    Int(i64),
    Bool(bool),
    String(String),
    Name(String),
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Group(Box<Expr>),
    List(Vec<Expr>),
    Index {
        collection: Box<Expr>,
        index: Box<Expr>,
    },
    If {
        condition: Box<Expr>,
        then_branch: Block,
        else_branch: Option<Block>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}
