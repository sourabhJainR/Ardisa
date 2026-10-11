//! Core language model, parser, and AI-native structural editing API for Ardisa.

pub mod ai;
pub mod ai_types;
pub mod ai_primitives;
pub mod ai_mode;
pub mod arena;
pub mod ast;
pub mod benchmark;
pub mod bootstrap;
pub mod concurrency;
pub mod compiler_engine;
pub mod diagnostic_memory;
pub mod differential;
pub mod edit;
pub mod effects;
pub mod engineering_feedback;
pub mod evidence;
pub mod format;
pub mod fuzz;
pub mod interop;
pub mod ir;
pub mod learning;
pub mod lower;
pub mod module_graph;
pub mod native;
pub mod ownership;
pub mod optimizer;
pub mod pipeline;
pub mod typed_ir;
pub mod protocol;
pub mod resource_guard;
pub mod sema;
pub mod source;
pub mod token;

pub use ai_mode::{AiModePolicy, Capability, PolicyViolation, ProofPolicy, ProofRequirement, ProofStatus, TracePolicy, UnknownProofPolicy, VaultPolicy};
pub use ai_types::{Alternative, Guaranteed, GraphConflictPolicy, GraphTensor, GraphTensorEdge, GraphTensorNode, Probabilistic};
pub use ai_primitives::{AiNumericType, AiTypeError, DomainEmbedding, Probability as AiProbability, QuantizationScale, QuantizedTensor, TensorShape, TensorValue, encode_tensor_descriptor};
pub use ai::{
    Distribution, Embedding, Modality, Probability, Quantization, RuntimeError, SemanticValue,
    Tensor, TraceBuffer, TraceEvent, TraceLevel,
};
pub use arena::{Arena, ArenaId, RecordField, RecordSchema, RecordValue};
pub use compiler_engine::{AstArena,BackendReport,CompilerEvidence,SafetyGraph,SymbolTables,TokenArena};
pub use concurrency::{
    CancellationToken, ScopeEvent, ScopeReport, StructuredScope, TaskHandle, TaskSpec, TaskTerminal,
};
pub use diagnostic_memory::{DiagnosticHistoryEntry, DiagnosticMemory};
pub use differential::{CORPUS as DIFFERENTIAL_CORPUS, DifferentialCase};
pub use effects::{EffectKind, EffectModel, FunctionEffects};
pub use engineering_feedback::{CapabilityFeedback, EpisodeOutcome, VerificationDepth};
pub use evidence::{CapabilityDecision, CapabilityEvaluation, EvidenceEnvelope, EvidenceGraph, EvidenceSourceVerifier};
pub use fuzz::{GeneratedCase, generate as generate_fuzz_case};
pub use interop::{
    AbiParameterContract, InteropType, OwnershipContract, RustFunction, SafeRustBoundary,
};
pub use ir::{IrFunction, IrModule, IrOp, IrValue};
pub use optimizer::optimize;
pub use pipeline::{compile_source, CompiledArtifact, PipelineError, PipelineEvidence, PipelineTiming};
pub use typed_ir::{TypedBasicBlock, TypedIrFunction, TypedIrModule, TypedIrOp, TypedTerminator, TypedValue, TypedValueKind};
pub use learning::{
    LearningEntry, LearningKey, LearningProvenance, LearningStatus, PersistentCompilerLearning,
    ProvenanceKind,
};
pub use module_graph::{ModuleGraph, ModuleGraphError, ModuleNode};
pub use native::{run_program_with_limits, ExecutionLimits, NativeError, NativeFunction, NativeInstr, NativeProgram, NativeValue};
pub use ownership::{AccessKind, BorrowKind, OwnershipClass, OwnershipModel, OwnershipTransition};
pub use resource_guard::{ResourceBudget, ResourceKind, ResourceLease, ResourceLimitExceeded, ResourceLimits, ResourceSnapshot};
pub use protocol::{
    CompilerRequest, CompilerResponse, CompilerSession, CompilerSnapshot, CompilerTrace,
    PROTOCOL_VERSION, ProtocolError, VerificationRequirement,
};

pub use ast::{
    BinaryOp, Block, ConstructDeclaration, ConstructKind, ConstructMember, Expr, ExprKind, Function, Item, Module, NodeId, Parameter, Stmt, StmtKind,
    Type, TypeKind,
};
pub use token::{Token, TokenKind, lex};

use std::collections::HashMap;

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<source::Diagnostic>,
    occurrences: HashMap<String, u64>,
}

pub fn parse(input: &str) -> Result<Module, Vec<source::Diagnostic>> {
    let tokens = token::lex(input)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        errors: Vec::new(),
        occurrences: HashMap::new(),
    };
    let module = parser.parse_module();

    if parser.errors.is_empty() {
        module.ok_or_else(|| {
            vec![source::Diagnostic::error(
                "AIF199",
                "parser produced no module",
                None,
            )]
        })
    } else {
        Err(parser.errors)
    }
}

impl Parser {
    fn parse_module(&mut self) -> Option<Module> {
        self.skip_newlines();
        let start = self.current().span.start;
        self.expect(TokenKind::Module, "module declaration")?;
        let name_token = self.expect(TokenKind::Ident, "module name")?;
        let name = name_token.lexeme.clone();
        self.expect(TokenKind::Newline, "end of module declaration")?;

        let mut items = Vec::new();
        let mut constructs = Vec::new();
        while !self.at(TokenKind::Eof) {
            self.skip_newlines();
            if self.at(TokenKind::Eof) {
                break;
            }
            if self.at(TokenKind::Ident) && matches!(self.current().lexeme.as_str(), "trace" | "cell" | "vault" | "proof" | "phase") {
                if let Some(declaration) = self.parse_construct() {
                    constructs.push(declaration);
                } else {
                    self.recover_top_level();
                }
            } else if let Some(item) = self.parse_item() {
                items.push(item);
            } else {
                self.recover_top_level();
            }
        }

        Some(Module {
            id: self.id("module", &name),
            span: source::Span::new(start, self.previous_span().end),
            name,
            items,
            constructs,
        })
    }

    fn parse_item(&mut self) -> Option<Item> {
        if self.at(TokenKind::Fn) {
            self.parse_function().map(Item::Function)
        } else {
            self.error("AIF201", "expected a top-level function");
            None
        }
    }

    fn parse_construct(&mut self) -> Option<ConstructDeclaration> {
        let start_token = self.bump();
        let kind = start_token.lexeme.as_str();
        let name_token = self.expect(TokenKind::Ident, "AI Mode declaration name")?;
        let name = name_token.lexeme.clone();
        self.expect(TokenKind::Newline, "end of AI Mode declaration header")?;
        if !matches!(self.current().kind, TokenKind::Indent(_)) {
            self.error("AIF610", format!("expected an indented body for {kind} declaration"));
            return None;
        }
        self.bump();
        let mut members = Vec::new();
        self.skip_newlines();
        while !self.at(TokenKind::Dedent) && !self.at(TokenKind::Eof) {
            let member_start = self.current().span.start;
            if kind == "phase" {
                let from = self.expect(TokenKind::Ident, "source phase")?;
                self.expect(TokenKind::Arrow, "'->' in phase transition")?;
                let to = self.expect(TokenKind::Ident, "destination phase")?;
                let span = source::Span::new(member_start, to.span.end);
                members.push(ConstructMember::Transition {
                    from: from.lexeme,
                    to: to.lexeme,
                    span,
                });
                self.expect(TokenKind::Newline, "end of phase transition")?;
            } else {
                let member_name = self.expect(TokenKind::Ident, "field or policy clause name")?;
                self.expect(TokenKind::Colon, "':' after field or policy clause name")?;
                let member_key = member_name.lexeme.clone();
                if matches!(member_key.as_str(), "invariant" | "capabilities" | "denies" | "requires" | "ensures" | "on_unknown" | "required" | "signature") {
                    let mut parts = Vec::new();
                    while !self.at(TokenKind::Newline) && !self.at(TokenKind::Dedent) && !self.at(TokenKind::Eof) {
                        parts.push(self.bump().lexeme);
                    }
                    let span = source::Span::new(member_start, self.previous_span().end);
                    members.push(ConstructMember::Clause {
                        name: member_key,
                        value: parts.join(" "),
                        span,
                    });
                    self.expect(TokenKind::Newline, "end of policy clause")?;
                } else {
                    let ty = self.parse_type()?;
                    let span = source::Span::new(member_start, self.previous_span().end);
                    members.push(ConstructMember::Field { name: member_key, ty, span });
                    self.expect(TokenKind::Newline, "end of field declaration")?;
                }
            }
            self.skip_newlines();
        }
        self.expect(TokenKind::Dedent, "end of AI Mode declaration")?;
        Some(ConstructDeclaration {
            id: self.id("construct", &name),
            span: source::Span::new(start_token.span.start, self.previous_span().end),
            kind: match kind {
                "trace" => ConstructKind::Trace,
                "cell" => ConstructKind::Cell,
                "vault" => ConstructKind::Vault,
                "proof" => ConstructKind::Proof,
                _ => ConstructKind::Phase,
            },
            name,
            members,
        })
    }
