use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::thread::{self, JoinHandle};

use crate::TypeKind;
use crate::context_window::{ContextBudget, ContextSegment, ContextWindow, SegmentModality};
use crate::kv_cache::{CacheFullPolicy, KvCache};
use crate::ir::{IrFunction, IrModule, IrOp, IrValue};

const MAX_TENSOR_ELEMENTS: usize = 16_384;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeInstr {
    PushInt(i64),
    PushFloat(u32),
    PushBool(bool),
    /// Push a validated IEEE-754 f32 probability encoded as its raw bits.
    PushProbability(u32),
    ProbabilityAnd,
    ProbabilityOr,
    ProbabilityNot,
    ProbabilityMeets,
    /// Push a rank-1 or rank-2 finite f32 tensor; bits are IEEE-754 f32 payloads.
    PushTensor { shape: Vec<usize>, values: Vec<u32> },
    TensorAdd,
    TensorMatMul,
    MakeActivationCache, ActivationStore { name: String }, ActivationRestore, ActivationCacheLen,
    MakeGradientTape,
    MakeTokenizerState, TokenizerFeed { name: String }, TokenizerFlush { name: String }, TokenizerTokenCount, TokenizerToken,
    GradientInput { name: String }, GradientAdd { name: String }, GradientMul { name: String }, GradientBackward { name: String }, GradientValue, GradientGet,
    /// Construct a validated nominal multi-head attention configuration.
    MakeAttentionBlock,
    /// Apply a nominal attention configuration to rank-2 Q/K/V tensors.
    AttentionBlockApply,
    /// Apply nominal attention to a query and compatible retained KVCache.
    AttentionBlockCached,
    /// Stable scaled dot-product attention over rank-2 Q/K/V tensors.
    AttentionBlock,
    /// Attend over the retained key/value entries of an existing KVCache.
    AttentionCached,
    /// Construct an immutable nominal LoRA adapter from A/B tensors and scale/rank.
    MakeLoRAAdapter,
    /// Apply an immutable LoRA adapter to input, returning the low-rank delta only.
    LoRAAdapterApply,
    /// Apply a bounded low-rank update: input [batch,in], A [rank,in], B [out,rank].
    LoRAApply,
    /// Build a finite f32 tensor from source-level shape/value integer lists.
    MakeTensor,
    /// Build a domain-tagged embedding from a source-level domain and integer list.
    MakeEmbedding,
    /// Build a validated signed-int8 quantized tensor from source-level lists and metadata.
    MakeQuantizedTensor,
    /// Push signed-int8 quantized tensor storage with finite positive scale.
    PushQuantizedTensor { shape: Vec<usize>, scale_bits: u32, zero_point: i8, values: Vec<i8> },
    /// Convert a quantized tensor to a finite f32 tensor.
    DequantizeTensor,
    /// Push a domain-tagged finite embedding with deterministic f32-bit coordinates.
    PushEmbedding { domain: String, values: Vec<u32> },
    EmbeddingDot,
    /// Construct a nominal bounded expert router with a fixed expert count and top-k.
    MakeMoERouter,
    /// Route finite logits through a validated nominal MoERouter and mask.
    MoERoute,
    /// Execute bounded deterministic MoE top-k routing from source-level lists.
    RouteTopK,
    /// Create a domain-isolated bounded vector index.
    MakeVectorIndex,
    /// Insert a vector into a named VectorIndex binding.
    VectorIndexInsert { name: String },
    /// Search a VectorIndex by squared Euclidean distance.
    VectorIndexSearch,
    MakeMctsTree,
    MctsRoot { name: String },
    MctsExpand { name: String },
    MctsChild,
    MctsUpdate { name: String },
    MctsBackpropagate { name: String },
    MctsLen,
    MakeKvCache,
    KvLen,
    KvCapacity,
    KvAppend { name: String, evict_oldest: bool },
    MakeContextWindow,
    ContextLen,
    ContextBytes,
    ContextTokens,
    ContextAppend { name: String },
    PushUnit,
    PushString(String),
    PushList(usize),
    Index,
    Len,
    Append(String),
    MakeOk,
    Chr,
    MakeErr,
    Unwrap,
    Load(String),
    Store(String),
    AddAssign(String),
    StoreIndex(String),
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
    JumpIfFalse(usize),
    Jump(usize),
    Return,
    Pop,
    Call {
        callee: String,
        argc: usize,
    },
    ScopeStart,
    ScopeEnd,
    Spawn {
        name: String,
        callee: String,
        argc: usize,
    },
    Join {
        name: String,
    },
    Cancel {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Deterministic collection of compiled Ardisa functions.
pub struct NativeProgram {
    pub functions: BTreeMap<String, NativeFunction>,
}

/// An immutable program that has passed native-program validation.
///
/// The wrapped program is never exposed mutably. Keeping it behind an Arc also
/// prevents mutation through Arc::get_mut while this wrapper exists. Execution
/// arguments and limits are deliberately validated per invocation.
#[derive(Debug, Clone)]
pub struct ValidatedNativeProgram {
    program: Arc<NativeProgram>,
}

impl ValidatedNativeProgram {
    /// Validate and take ownership of a native program.
    pub fn try_new(program: NativeProgram) -> Result<Self, NativeError> {
        Self::try_from_shared(Arc::new(program))
    }

    /// Validate a shared program and retain an immutable reference to it.
    pub fn try_from_shared(program: Arc<NativeProgram>) -> Result<Self, NativeError> {
        validate_native_program(&program)?;
        Ok(Self { program })
    }

    /// Borrow the validated program without exposing mutable access.
    pub fn as_program(&self) -> &NativeProgram {
        &self.program
    }

    /// Execute with fresh runtime state and per-call argument/limit checks.
    pub fn run_with_limits(
        &self,
        entry: &str,
        args: &[NativeValue],
        limits: ExecutionLimits,
    ) -> Result<NativeValue, NativeError> {
        run_validated_program_with_limits(self, entry, args, limits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeFunction {
    pub params: Vec<String>,
    pub code: Vec<NativeInstr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientOp { Input, Add { left: usize, right: usize }, Mul { left: usize, right: usize } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradientSlot { pub value_bits: u32, pub op: GradientOp }

/// Native values are intentionally dependency-free so the native compiler can bootstrap incrementally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeValue {
    Int(i64),
    /// Finite f32 runtime scalar encoded as bits for deterministic Eq and serialization.
    Float(u32),
    Bool(bool),
    /// A probability is stored as f32 bits to preserve Eq and deterministic artifacts.
    Probability(u32),
    /// Tensor dimensions and IEEE-754 f32 values encoded as bits.
    Tensor { shape: Vec<usize>, values: Vec<u32> },
    ActivationCache { capacity: usize, checkpoint_stride: usize, max_bytes: usize, retained_bytes: usize, entries: Vec<(i64, Vec<usize>, Vec<u32>)> },
    GradientTape { capacity: usize, slots: Vec<GradientSlot>, gradients: Vec<u32>, backward_ready: bool },
    /// Incremental UTF-8 tokenizer with a one-scalar explicit vocabulary and bounded pending bytes/output.
    TokenizerState { vocabulary: Vec<String>, pending: Vec<u8>, tokens: Vec<i64>, max_pending: usize, max_tokens: usize, unknown_id: Option<i64> },
    /// Signed-int8 tensor payload and quantization metadata; scale is f32 bits.
    QuantizedTensor { shape: Vec<usize>, scale_bits: u32, zero_point: i8, values: Vec<i8> },
    /// Nominal domain label plus IEEE-754 f32 coordinate bits.
    Embedding { domain: String, values: Vec<u32> },
    /// Validated nominal attention configuration; owns metadata, not model weights.
    AttentionBlock { hidden: usize, heads: usize, head_dim: usize, causal: bool },
    /// Immutable, owned low-rank matrices. Applying returns a delta and never mutates base weights.
    LoRAAdapter { a_shape: Vec<usize>, a_values: Vec<u32>, b_shape: Vec<usize>, b_values: Vec<u32>, alpha_bits: u32, rank: usize },
    /// Nominal bounded expert-routing configuration, independent of model weight ownership.
    MoERouter { expert_count: usize, top_k: usize },
    /// Nominal domain-isolated vector index with finite f32 coordinate bits.
    VectorIndex { domain: String, dimensions: usize, entries: Vec<(u64, Vec<u32>)> },
    /// Bounded attention cache; float payloads are bit-encoded for deterministic equality.
    KvCache { capacity: usize, key_dim: Option<usize>, value_dim: Option<usize>, entries: Vec<(Vec<u32>, Vec<u32>)> },
    /// Bounded multimodal context with opaque byte payloads and explicit upstream token counts.
    ContextWindow { max_segments: usize, max_bytes: usize, max_tokens: usize, segments: Vec<(String, Vec<u8>, usize)>, bytes: usize, tokens: usize },
    String(String),
    List(Vec<NativeValue>),
    ResultOk(Box<NativeValue>),
    ResultErr(Box<NativeValue>),
    Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeError {
    Unsupported(String),
    Cancelled(String),
    InvalidProgram(String),
    Type(String),
    ResourceLimit(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub max_instructions: usize,
    pub max_call_depth: usize,
    pub max_tasks: usize,
    pub max_collection_items: usize,
    pub max_string_bytes: usize,
    /// Maximum total nodes in any single nested runtime value.
    pub max_value_nodes: usize,
    /// Maximum nesting depth for List and Result values (root depth is zero).
    pub max_value_depth: usize,
    /// Maximum values held on one VM frame's operand stack.
    pub max_stack_values: usize,
    /// Maximum local bindings in one VM frame.
    pub max_locals: usize,
    /// Aggregate nested value nodes across one frame's locals and operand stack.
    pub max_frame_value_nodes: usize,
    /// Aggregate string payload bytes across one frame's locals and operand stack.
    pub max_frame_string_bytes: usize,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_instructions: 1_000_000,
            max_call_depth: 128,
            max_tasks: 64,
            max_collection_items: 16_384,
            max_string_bytes: 1_048_576,
            max_value_nodes: 16_384,
            max_value_depth: 64,
            max_stack_values: 16_384,
            max_locals: 4_096,
            max_frame_value_nodes: 65_536,
            max_frame_string_bytes: 8_388_608,
        }
    }
}

struct ExecutionState {
    limits: ExecutionLimits,
    instructions: AtomicUsize,
    active_tasks: AtomicUsize,
}

struct ActiveTaskLease(Arc<ExecutionState>);

impl Drop for ActiveTaskLease {
    fn drop(&mut self) {
        self.0.active_tasks.fetch_sub(1, Ordering::AcqRel);
    }
}

fn reserve_task(state: &Arc<ExecutionState>) -> Result<ActiveTaskLease, NativeError> {
    let mut current = state.active_tasks.load(Ordering::Acquire);
    loop {
        if current >= state.limits.max_tasks {
            return Err(NativeError::ResourceLimit(format!(
                "active task limit exceeded (limit {})",
                state.limits.max_tasks
            )));
        }
        match state.active_tasks.compare_exchange_weak(
            current, current + 1, Ordering::AcqRel, Ordering::Acquire
        ) {
            Ok(_) => return Ok(ActiveTaskLease(Arc::clone(state))),
            Err(observed) => current = observed,
        }
    }
}

fn validate_value(value: &NativeValue, limits: ExecutionLimits) -> Result<(), NativeError> {
    let mut pending = vec![(value, 0usize)];
    let mut nodes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes = nodes.checked_add(1).ok_or_else(|| NativeError::ResourceLimit("value node count overflow".into()))?;
        if nodes > limits.max_value_nodes {
            return Err(NativeError::ResourceLimit(format!("nested value node limit exceeded (limit {})", limits.max_value_nodes)));
        }
        if depth > limits.max_value_depth {
            return Err(NativeError::ResourceLimit(format!("value nesting depth exceeded (limit {})", limits.max_value_depth)));
        }
        match value {
            NativeValue::String(s) if s.len() > limits.max_string_bytes => {
                return Err(NativeError::ResourceLimit("string exceeds byte limit".into()));
            }
            NativeValue::Float(bits) if !f32::from_bits(*bits).is_finite() => {
                return Err(NativeError::Type("Float value must be finite".into()));
            }
            NativeValue::Tensor { shape, values } => {
                validate_tensor_parts(shape, values, limits.max_collection_items)?;
            }
            NativeValue::AttentionBlock { hidden, heads, head_dim, .. } => {
                if *hidden == 0 || *heads == 0 || *head_dim == 0 || *heads > 256 || heads.checked_mul(*head_dim) != Some(*hidden) {
                    return Err(NativeError::Type("invalid AttentionBlock hidden/head configuration".into()));
                }
            }
            NativeValue::MoERouter { expert_count, top_k } => {
                if *expert_count == 0 || *expert_count > crate::ai_routing::MAX_EXPERTS
                    || *top_k == 0 || *top_k > crate::ai_routing::MAX_TOP_K || *top_k > *expert_count {
                    return Err(NativeError::Type("invalid MoERouter expert_count/top_k configuration".into()));
                }
            }
            NativeValue::VectorIndex { domain, dimensions, entries } => {
                if domain.trim().is_empty() || domain.len() > 1024 {
                    return Err(NativeError::Type("VectorIndex domain must contain 1..=1024 UTF-8 bytes".into()));
                }
                if *dimensions == 0 || *dimensions > crate::ai_routing::MAX_VECTOR_DIMENSIONS
                    || *dimensions > limits.max_collection_items {
                    return Err(NativeError::Type("invalid VectorIndex dimensions".into()));
                }
                if entries.len() > crate::ai_routing::MAX_INDEX_ENTRIES.min(limits.max_collection_items) {
                    return Err(NativeError::ResourceLimit("VectorIndex entry limit exceeded".into()));
                }
                let mut ids = std::collections::BTreeSet::new();
                for (id, coordinates) in entries {
                    if !ids.insert(*id) { return Err(NativeError::InvalidProgram("duplicate VectorIndex id".into())); }
                    if *id > i64::MAX as u64 { return Err(NativeError::InvalidProgram("VectorIndex id exceeds signed Int range".into())); }
                    if coordinates.len() != *dimensions {
                        return Err(NativeError::InvalidProgram("VectorIndex coordinate dimension mismatch".into()));
                    }
                    if coordinates.iter().any(|bits| !f32::from_bits(*bits).is_finite()) {
                        return Err(NativeError::InvalidProgram("VectorIndex contains non-finite coordinates".into()));
                    }
                }
                let nodes = entries.len().checked_mul(dimensions.checked_add(1).ok_or_else(|| NativeError::ResourceLimit("VectorIndex dimension accounting overflow".into()))?)
                    .and_then(|n| n.checked_add(1))
                    .ok_or_else(|| NativeError::ResourceLimit("VectorIndex footprint accounting overflow".into()))?;
                if nodes > limits.max_value_nodes {
                    return Err(NativeError::ResourceLimit("VectorIndex retained storage exceeds value-node budget".into()));
                }
            }
            NativeValue::LoRAAdapter { a_shape, a_values, b_shape, b_values, alpha_bits, rank } => {
                validate_lora_parts(a_shape, a_values, b_shape, b_values, *alpha_bits, *rank, limits.max_collection_items)?;
            }
            NativeValue::ActivationCache { capacity, checkpoint_stride, max_bytes, retained_bytes, entries } => {
                if *capacity==0 || *capacity>4096 || *checkpoint_stride==0 || *checkpoint_stride>4096 || *max_bytes==0 || *max_bytes>16_777_216 || entries.len()>*capacity { return Err(NativeError::InvalidProgram("invalid ActivationCache limits".into())); }
                let mut ids=std::collections::BTreeSet::new(); let mut bytes=0usize;
                for (step,shape,values) in entries { if *step<0 || (*step as usize)%*checkpoint_stride!=0 || !ids.insert(*step) { return Err(NativeError::InvalidProgram("invalid or duplicate ActivationCache checkpoint id".into())); } validate_tensor_parts(shape,values,MAX_TENSOR_ELEMENTS)?; if values.iter().any(|bits|!f32::from_bits(*bits).is_finite()) { return Err(NativeError::Type("ActivationCache values must be finite".into())); } bytes=bytes.checked_add(values.len().checked_mul(4).ok_or_else(||NativeError::ResourceLimit("ActivationCache byte count overflow".into()))?).and_then(|n|n.checked_add(shape.len().checked_mul(std::mem::size_of::<usize>())?)).ok_or_else(||NativeError::ResourceLimit("ActivationCache byte count overflow".into()))?; }
                if bytes!=*retained_bytes || bytes>*max_bytes { return Err(NativeError::InvalidProgram("ActivationCache retained-byte accounting mismatch".into())); }
            }
            NativeValue::TokenizerState { vocabulary, pending, tokens, max_pending, max_tokens, unknown_id } => {
                if vocabulary.is_empty() || vocabulary.len() > limits.max_collection_items || *max_pending == 0 || *max_pending > 4 || *max_tokens == 0 || *max_tokens > limits.max_collection_items || tokens.len() > *max_tokens || pending.len() > *max_pending {
                    return Err(NativeError::InvalidProgram("invalid TokenizerState capacity or buffer state".into()));
                }
                let mut seen = std::collections::BTreeSet::new();
                for piece in vocabulary { if piece.chars().count()!=1 || !seen.insert(piece) { return Err(NativeError::InvalidProgram("TokenizerState vocabulary must contain unique single Unicode scalars".into())); } }
                if unknown_id.is_some_and(|id| id < 0) { return Err(NativeError::InvalidProgram("TokenizerState unknown token id must be non-negative".into())); }
                if tokens.iter().any(|id| *id < 0 || (usize::try_from(*id).map_or(true, |index| index >= vocabulary.len()) && Some(*id) != *unknown_id)) { return Err(NativeError::InvalidProgram("TokenizerState contains a token id outside its vocabulary/unknown-id contract".into())); }
                if !pending.is_empty() {
                    match std::str::from_utf8(pending) {
                        Ok(_) => return Err(NativeError::InvalidProgram("TokenizerState pending bytes must be an incomplete UTF-8 suffix".into())),
                        Err(error) if error.error_len().is_some() || error.valid_up_to()!=0 => return Err(NativeError::InvalidProgram("TokenizerState pending bytes must be an incomplete UTF-8 suffix".into())),
                        Err(_) => {}
                    }
                }
            }
            NativeValue::GradientTape { capacity, slots, gradients, backward_ready } => {
                if *capacity == 0 || *capacity > 4096 || slots.len() > *capacity || (*backward_ready && gradients.len() != slots.len()) || (!*backward_ready && !gradients.is_empty()) { return Err(NativeError::InvalidProgram("invalid GradientTape capacity or gradient state".into())); }
                for (index, slot) in slots.iter().enumerate() { if !f32::from_bits(slot.value_bits).is_finite() { return Err(NativeError::Type("GradientTape values must be finite".into())); } let valid=match slot.op { GradientOp::Input=>true,GradientOp::Add{left,right}|GradientOp::Mul{left,right}=>left<index&&right<index }; if !valid{return Err(NativeError::InvalidProgram("GradientTape slot references must point to prior slots".into()));} }
                if gradients.iter().any(|bits|!f32::from_bits(*bits).is_finite()){return Err(NativeError::Type("GradientTape gradients must be finite".into()));}
            }
            NativeValue::Embedding { domain, values } => {
                validate_embedding_parts(domain, values, limits.max_collection_items)?;
            }
            NativeValue::QuantizedTensor { shape, scale_bits, values, .. } => {
                validate_quantized_tensor_parts(shape, *scale_bits, values, limits.max_collection_items)?;
            }
            NativeValue::KvCache { capacity, entries, .. } => {
                if *capacity == 0 || *capacity > crate::kv_cache::MAX_KV_CACHE_ENTRIES || entries.len() > *capacity || entries.len() > limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("invalid or oversized KVCache value".into()));
                }
                let mut scalar_count = 0usize;
                let mut expected_key_dim = None;
                let mut expected_value_dim = None;
                for (key, value) in entries {
                    if key.is_empty() || value.is_empty() || key.iter().chain(value).any(|bits| !f32::from_bits(*bits).is_finite()) {
                        return Err(NativeError::Type("KVCache entries require non-empty finite key/value vectors".into()));
                    }
                    if expected_key_dim.is_some_and(|expected| expected != key.len()) || expected_value_dim.is_some_and(|expected| expected != value.len()) {
                        return Err(NativeError::Type("KVCache entries have inconsistent dimensions".into()));
                    }
                    expected_key_dim = Some(key.len()); expected_value_dim = Some(value.len());
                    scalar_count = scalar_count.checked_add(key.len()).and_then(|n| n.checked_add(value.len())).ok_or_else(|| NativeError::ResourceLimit("KVCache scalar accounting overflow".into()))?;
                }
                if scalar_count > crate::kv_cache::MAX_KV_CACHE_SCALARS { return Err(NativeError::ResourceLimit("KVCache scalar budget exceeded".into())); }
                let NativeValue::KvCache { key_dim, value_dim, .. } = value else { unreachable!() };
                if *key_dim != expected_key_dim || *value_dim != expected_value_dim {
                    return Err(NativeError::InvalidProgram("KVCache dimension metadata does not match retained entries".into()));
                }
            }
            NativeValue::ContextWindow { max_segments, max_bytes, max_tokens, segments, bytes, tokens } => {
                if *max_segments == 0 || *max_bytes == 0 || *max_tokens == 0 || *max_segments > crate::context_window::MAX_CONTEXT_WINDOW_SEGMENTS || *max_bytes > crate::context_window::MAX_CONTEXT_WINDOW_BYTES || *max_tokens > crate::context_window::MAX_CONTEXT_WINDOW_TOKENS || segments.len() > *max_segments || *bytes > *max_bytes || *tokens > *max_tokens || segments.len() > limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("invalid or oversized ContextWindow value".into()));
                }
                let actual_bytes = segments.iter().try_fold(0usize, |sum, (modality, payload, _)| {
                    if payload.is_empty() || !matches!(modality.as_str(), "text" | "image" | "audio" | "video" | "structured") { return Err(NativeError::Type("invalid ContextWindow segment".into())); }
                    sum.checked_add(payload.len()).ok_or_else(|| NativeError::ResourceLimit("ContextWindow byte accounting overflow".into()))
                })?;
                let actual_tokens = segments.iter().try_fold(0usize, |sum, (_, _, count)| sum.checked_add(*count).ok_or_else(|| NativeError::ResourceLimit("ContextWindow token accounting overflow".into())))?;
                if actual_bytes != *bytes || actual_tokens != *tokens { return Err(NativeError::InvalidProgram("ContextWindow counters do not match retained segments".into())); }
            }
            NativeValue::List(items) => {
                if items.len() > limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("collection exceeds item limit".into()));
                }
                pending.extend(items.iter().map(|item| (item, depth + 1)));
            }
            NativeValue::ResultOk(inner) | NativeValue::ResultErr(inner) => {
                pending.push((inner, depth + 1));
            }
            _ => {}
        }
    }
    Ok(())
}

fn value_footprint(value: &NativeValue) -> Result<(usize, usize), NativeError> {
    let mut pending = vec![value];
    let mut nodes = 0usize;
    let mut string_bytes = 0usize;
    while let Some(value) = pending.pop() {
        nodes = nodes.checked_add(1)
            .ok_or_else(|| NativeError::ResourceLimit("value node count overflow".into()))?;
        match value {
            NativeValue::String(value) => {
                string_bytes = string_bytes.checked_add(value.len())
                    .ok_or_else(|| NativeError::ResourceLimit("string byte accounting overflow".into()))?;
            }
            NativeValue::ActivationCache { entries, .. } => { for (_,shape,values) in entries { nodes=nodes.checked_add(shape.len()).and_then(|n|n.checked_add(values.len())).ok_or_else(||NativeError::ResourceLimit("ActivationCache footprint accounting overflow".into()))?; } }
            NativeValue::GradientTape { slots, gradients, .. } => { nodes=nodes.checked_add(slots.len()).and_then(|n|n.checked_add(gradients.len())).ok_or_else(||NativeError::ResourceLimit("GradientTape footprint accounting overflow".into()))?; }
            NativeValue::TokenizerState { vocabulary, pending, tokens, .. } => { nodes=nodes.checked_add(vocabulary.len()).and_then(|n|n.checked_add(tokens.len())).and_then(|n|n.checked_add(pending.len())).ok_or_else(||NativeError::ResourceLimit("TokenizerState footprint accounting overflow".into()))?; for piece in vocabulary { string_bytes=string_bytes.checked_add(piece.len()).ok_or_else(||NativeError::ResourceLimit("TokenizerState vocabulary byte accounting overflow".into()))?; } }
            NativeValue::LoRAAdapter { a_values, b_values, .. } => {
                nodes = nodes.checked_add(a_values.len()).and_then(|n| n.checked_add(b_values.len()))
                    .ok_or_else(|| NativeError::ResourceLimit("LoRAAdapter footprint accounting overflow".into()))?;
            }
            NativeValue::KvCache { entries, .. } => {
                for (key, value) in entries {
                    nodes = nodes.checked_add(2)
                        .and_then(|n| n.checked_add(key.len()))
                        .and_then(|n| n.checked_add(value.len()))
                        .ok_or_else(|| NativeError::ResourceLimit("KVCache footprint accounting overflow".into()))?;
                }
            }
            NativeValue::ContextWindow { segments, .. } => {
                for (modality, payload, _) in segments {
                    nodes = nodes.checked_add(1)
                        .ok_or_else(|| NativeError::ResourceLimit("ContextWindow footprint node accounting overflow".into()))?;
                    string_bytes = string_bytes.checked_add(modality.len())
                        .and_then(|n| n.checked_add(payload.len()))
                        .ok_or_else(|| NativeError::ResourceLimit("ContextWindow footprint byte accounting overflow".into()))?;
                }
            }
            NativeValue::VectorIndex { domain, dimensions, entries } => {
                nodes = nodes.checked_add(entries.len().checked_mul(dimensions.checked_add(1).ok_or_else(|| NativeError::ResourceLimit("VectorIndex footprint accounting overflow".into()))?)
                    .ok_or_else(|| NativeError::ResourceLimit("VectorIndex footprint accounting overflow".into()))?)
                    .ok_or_else(|| NativeError::ResourceLimit("VectorIndex footprint accounting overflow".into()))?;
                string_bytes = string_bytes.checked_add(domain.len())
                    .ok_or_else(|| NativeError::ResourceLimit("VectorIndex domain byte accounting overflow".into()))?;
            }
            NativeValue::List(items) => pending.extend(items.iter()),
            NativeValue::ResultOk(inner) | NativeValue::ResultErr(inner) => pending.push(inner),
            _ => {}
        }
    }
    Ok((nodes, string_bytes))
}

fn check_frame_value_budget(
    locals: &HashMap<String, NativeValue>,
    stack: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<(), NativeError> {
    let mut nodes = 0usize;
    let mut string_bytes = 0usize;
    for value in locals.values().chain(stack.iter()) {
        validate_value(value, limits)?;
        let (value_nodes, value_string_bytes) = value_footprint(value)?;
        nodes = nodes.checked_add(value_nodes)
            .ok_or_else(|| NativeError::ResourceLimit("frame value node accounting overflow".into()))?;
        string_bytes = string_bytes.checked_add(value_string_bytes)
            .ok_or_else(|| NativeError::ResourceLimit("frame string accounting overflow".into()))?;
        if nodes > limits.max_frame_value_nodes {
            return Err(NativeError::ResourceLimit(format!(
                "aggregate frame value node limit exceeded (limit {})", limits.max_frame_value_nodes
            )));
        }
        if string_bytes > limits.max_frame_string_bytes {
            return Err(NativeError::ResourceLimit(format!(
                "aggregate frame string byte limit exceeded (limit {})", limits.max_frame_string_bytes
            )));
        }
    }
    Ok(())
}

fn validate_input_values(values: &[NativeValue], limits: ExecutionLimits) -> Result<(), NativeError> {
    for value in values {
        validate_value(value, limits)?;
    }
    Ok(())
}

/// Compile the first function for the legacy single-function API.
pub fn compile(module: &IrModule) -> Result<Vec<NativeInstr>, NativeError> {
    let function = module
        .functions
        .first()
        .ok_or_else(|| NativeError::InvalidProgram("module has no functions".into()))?;
    compile_function(function)
}

pub fn compile_program(module: &IrModule) -> Result<NativeProgram, NativeError> {
    let mut functions = BTreeMap::new();
    for function in &module.functions {
        functions.insert(
            function.name.clone(),
            NativeFunction {
                params: function
                    .params
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect(),
                code: compile_function(function)?,
            },
        );
    }
    Ok(NativeProgram { functions })
}

pub fn compile_function(function: &IrFunction) -> Result<Vec<NativeInstr>, NativeError> {
    if function.params.iter().any(|(_, ty)| !is_native_type(ty)) {
        return Err(NativeError::Unsupported(
            "native backend does not support this parameter type".into(),
        ));
    }
    let mut code = Vec::new();
    for (index, op) in function.ops.iter().enumerate() {
        let is_last_expression = index + 1 == function.ops.len()
            && matches!(op, IrOp::Expr(_))
            && function.return_type.is_some();
        if is_last_expression {
            if let IrOp::Expr(value) = op {
                emit_value(value, &mut code)?;
                code.push(NativeInstr::Return);
            }
        } else {
            emit_op(op, &mut code)?;
        }
    }
    if !matches!(code.last(), Some(NativeInstr::Return)) {
        code.push(NativeInstr::Return);
    }
    Ok(code)
}

fn is_native_type(ty: &TypeKind) -> bool {
    match ty {
        TypeKind::Int | TypeKind::Bool | TypeKind::String | TypeKind::Unit => true,
        TypeKind::List(element) => is_native_type(&element.kind),
        TypeKind::Result(ok, err) => is_native_type(&ok.kind) && is_native_type(&err.kind),
        TypeKind::Named(name) => matches!(name.as_str(), "KvCache" | "ContextWindow" | "MctsTree" | "AttentionBlock" | "LoRAAdapter" | "MoERouter" | "VectorIndex" | "GradientTape" | "ActivationCache" | "TokenizerState"),
        TypeKind::Generic(_, _) => false,
    }
}

fn emit_op(op: &IrOp, code: &mut Vec<NativeInstr>) -> Result<(), NativeError> {
    match op {
        IrOp::Let { name, value } => {
            emit_value(value, code)?;
            code.push(NativeInstr::Store(name.clone()));
        }
        IrOp::Set { name, value } => {
            if let IrValue::Binary {
                op: crate::BinaryOp::Add,
                left,
                right,
            } = value
            {
                if matches!(left.as_ref(), IrValue::Name(existing) if existing == name) {
                    emit_value(right, code)?;
                    code.push(NativeInstr::AddAssign(name.clone()));
                    return Ok(());
                }
            }
            emit_value(value, code)?;
            code.push(NativeInstr::Store(name.clone()));
        }
        IrOp::SetIndex {
            collection,
            index,
            value,
        } => {
            let IrValue::Name(name) = collection else {
                return Err(NativeError::Unsupported(
                    "indexed assignment currently requires a named list binding".into(),
                ));
            };
            code.push(NativeInstr::Load(name.clone()));
            emit_value(index, code)?;
            emit_value(value, code)?;
            code.push(NativeInstr::StoreIndex(name.clone()));
        }
        IrOp::Return(value) => {
            if let Some(value) = value {
                emit_value(value, code)?;
            } else {
                code.push(NativeInstr::PushInt(0));
            }
            code.push(NativeInstr::Return);
        }
        IrOp::Expr(value) => {
            emit_value(value, code)?;
            code.push(NativeInstr::Pop);
        }
        IrOp::While { condition, ops } => {
            let loop_start = code.len();
            emit_value(condition, code)?;
            let jump_if = code.len();
            code.push(NativeInstr::JumpIfFalse(usize::MAX));
            for op in ops {
                emit_op(op, code)?;
            }
            code.push(NativeInstr::Jump(loop_start));
            let end = code.len();
            code[jump_if] = NativeInstr::JumpIfFalse(end);
        }
        IrOp::Scope { ops } => {
            code.push(NativeInstr::ScopeStart);
            for op in ops {
                emit_op(op, code)?;
            }
            code.push(NativeInstr::ScopeEnd);
        }
        IrOp::Spawn { name, call } => {
            let IrValue::Call { callee, args } = call else {
                return Err(NativeError::Unsupported(
                    "spawn requires a function call".into(),
                ));
            };
            for arg in args {
                emit_value(arg, code)?;
            }
            code.push(NativeInstr::Spawn {
                name: name.clone(),
                callee: callee.clone(),
                argc: args.len(),
            });
        }
        IrOp::Join { name } => code.push(NativeInstr::Join { name: name.clone() }),
        IrOp::Cancel { name } => code.push(NativeInstr::Cancel { name: name.clone() }),
    }
    Ok(())
}

fn emit_branch(ops: &[IrOp], code: &mut Vec<NativeInstr>) -> Result<(), NativeError> {
    if ops.is_empty() {
        code.push(NativeInstr::PushUnit);
        return Ok(());
    }
    for (index, op) in ops.iter().enumerate() {
        let is_last = index + 1 == ops.len();
        if is_last {
            match op {
                IrOp::Expr(value) => emit_value(value, code)?,
                IrOp::Return(_) => emit_op(op, code)?,
                _ => {
                    emit_op(op, code)?;
                    code.push(NativeInstr::PushUnit);
                }
            }
        } else {
            emit_op(op, code)?;
        }
    }
    Ok(())
}

fn emit_value(value: &IrValue, code: &mut Vec<NativeInstr>) -> Result<(), NativeError> {
    match value {
        IrValue::Int(value) => code.push(NativeInstr::PushInt(*value)),
        IrValue::Float(bits) => code.push(NativeInstr::PushFloat(*bits)),
        IrValue::String(value) => code.push(NativeInstr::PushString(value.clone())),
        IrValue::List(values) => {
            for value in values {
                emit_value(value, code)?;
            }
            code.push(NativeInstr::PushList(values.len()));
        }
        IrValue::Index { collection, index } => {
            emit_value(collection, code)?;
            emit_value(index, code)?;
            code.push(NativeInstr::Index);
        }
        IrValue::Bool(value) => code.push(NativeInstr::PushBool(*value)),
        IrValue::Name(name) => code.push(NativeInstr::Load(name.clone())),
        IrValue::Binary { op, left, right } => {
            emit_value(left, code)?;
            emit_value(right, code)?;
            code.push(match op {
                crate::BinaryOp::Add => NativeInstr::Add,
                crate::BinaryOp::Sub => NativeInstr::Sub,
                crate::BinaryOp::Mul => NativeInstr::Mul,
                crate::BinaryOp::Div => NativeInstr::Div,
                crate::BinaryOp::Mod => NativeInstr::Mod,
                crate::BinaryOp::Equal => NativeInstr::Equal,
                crate::BinaryOp::NotEqual => NativeInstr::NotEqual,
                crate::BinaryOp::Less => NativeInstr::Less,
                crate::BinaryOp::LessEqual => NativeInstr::LessEqual,
                crate::BinaryOp::Greater => NativeInstr::Greater,
                crate::BinaryOp::GreaterEqual => NativeInstr::GreaterEqual,
            });
        }
        IrValue::If {
            condition,
            then_ops,
            else_ops,
        } => {
            emit_value(condition, code)?;
            let jump_if = code.len();
            code.push(NativeInstr::JumpIfFalse(usize::MAX));
            emit_branch(then_ops, code)?;
            let jump_end = code.len();
            code.push(NativeInstr::Jump(usize::MAX));
            let else_start = code.len();
            emit_branch(else_ops, code)?;
            let end = code.len();
            code[jump_if] = NativeInstr::JumpIfFalse(else_start);
            code[jump_end] = NativeInstr::Jump(end);
        }
        IrValue::Call { callee, args } => {
            if callee != "kv_append" && callee != "context_append" && callee != "vector_index_insert" && !matches!(callee.as_str(), "mcts_root" | "mcts_expand" | "mcts_update" | "mcts_backpropagate" | "activation_store" | "gradient_input" | "gradient_add" | "gradient_mul" | "gradient_backward") {
                for arg in args { emit_value(arg, code)?; }
            }
            if callee == "chr" && args.len() == 1 {
                code.push(NativeInstr::Chr);
            } else if callee == "ok" && args.len() == 1 {
                code.push(NativeInstr::MakeOk);
            } else if callee == "err" && args.len() == 1 {
                code.push(NativeInstr::MakeErr);
            } else if callee == "unwrap" && args.len() == 1 {
                code.push(NativeInstr::Unwrap);
            } else if callee == "tensor" && args.len() == 2 {
                code.push(NativeInstr::MakeTensor);
            } else if callee == "tensor_add" && args.len() == 2 {
                code.push(NativeInstr::TensorAdd);
            } else if callee == "tensor_matmul" && args.len() == 2 {
                code.push(NativeInstr::TensorMatMul);
            } else if callee == "activation_cache" && args.len() == 3 { code.push(NativeInstr::MakeActivationCache); }
            else if callee == "activation_store" && args.len() == 3 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("activation_store requires named ActivationCache".into()));};emit_value(&args[1],code)?;emit_value(&args[2],code)?;code.push(NativeInstr::ActivationStore{name:name.clone()}); }
            else if callee == "activation_restore" && args.len() == 2 { code.push(NativeInstr::ActivationRestore); }
            else if callee == "activation_cache_len" && args.len() == 1 { code.push(NativeInstr::ActivationCacheLen); }
            else if callee == "gradient_tape" && args.len() == 1 { code.push(NativeInstr::MakeGradientTape); }
            else if callee == "tokenizer_state" && args.len() == 4 { code.push(NativeInstr::MakeTokenizerState); }
            else if callee == "tokenizer_feed" && args.len() == 2 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("tokenizer_feed requires named TokenizerState".into()));}; emit_value(&args[1],code)?; code.push(NativeInstr::TokenizerFeed{name:name.clone()}); }
            else if callee == "tokenizer_flush" && args.len() == 1 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("tokenizer_flush requires named TokenizerState".into()));}; code.push(NativeInstr::TokenizerFlush{name:name.clone()}); }
            else if callee == "tokenizer_token_count" && args.len() == 1 { emit_value(&args[0],code)?; code.push(NativeInstr::TokenizerTokenCount); }
            else if callee == "tokenizer_token" && args.len() == 2 { emit_value(&args[0],code)?; emit_value(&args[1],code)?; code.push(NativeInstr::TokenizerToken); }
            else if callee == "gradient_input" && args.len() == 2 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("gradient_input requires named GradientTape".into()));}; emit_value(&args[1],code)?; code.push(NativeInstr::GradientInput{name:name.clone()}); }
            else if callee == "gradient_add" && args.len() == 3 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("gradient_add requires named GradientTape".into()));}; emit_value(&args[1],code)?;emit_value(&args[2],code)?;code.push(NativeInstr::GradientAdd{name:name.clone()}); }
            else if callee == "gradient_mul" && args.len() == 3 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("gradient_mul requires named GradientTape".into()));}; emit_value(&args[1],code)?;emit_value(&args[2],code)?;code.push(NativeInstr::GradientMul{name:name.clone()}); }
            else if callee == "gradient_backward" && args.len() == 3 { let IrValue::Name(name)=&args[0] else{return Err(NativeError::Unsupported("gradient_backward requires named GradientTape".into()));}; emit_value(&args[1],code)?;emit_value(&args[2],code)?;code.push(NativeInstr::GradientBackward{name:name.clone()}); }
            else if callee == "gradient_value" && args.len() == 2 { code.push(NativeInstr::GradientValue); }
            else if callee == "gradient_get" && args.len() == 2 { code.push(NativeInstr::GradientGet); }
            else if callee == "attention_block_new" && args.len() == 4 {
                code.push(NativeInstr::MakeAttentionBlock);
            } else if callee == "attention_block_apply" && args.len() == 4 {
                code.push(NativeInstr::AttentionBlockApply);
            } else if callee == "attention_block_cached" && args.len() == 3 {
                code.push(NativeInstr::AttentionBlockCached);
            } else if callee == "attention_block" && args.len() == 4 {
                code.push(NativeInstr::AttentionBlock);
            } else if callee == "attention_cached" && args.len() == 2 {
                code.push(NativeInstr::AttentionCached);
            } else if callee == "lora_adapter_new" && args.len() == 4 {
                code.push(NativeInstr::MakeLoRAAdapter);
            } else if callee == "lora_adapter_apply" && args.len() == 2 {
                code.push(NativeInstr::LoRAAdapterApply);
            } else if callee == "lora_apply" && args.len() == 5 {
                code.push(NativeInstr::LoRAApply);
            } else if callee == "embedding" && args.len() == 2 {
                code.push(NativeInstr::MakeEmbedding);
            } else if callee == "quantized_tensor" && args.len() == 4 {
                code.push(NativeInstr::MakeQuantizedTensor);
            } else if callee == "dequantize" && args.len() == 1 {
                code.push(NativeInstr::DequantizeTensor);
            } else if callee == "embedding_dot" && args.len() == 2 {
                code.push(NativeInstr::EmbeddingDot);
            } else if callee == "moe_router" && args.len() == 2 {
                code.push(NativeInstr::MakeMoERouter);
            } else if callee == "moe_route" && args.len() == 3 {
                code.push(NativeInstr::MoERoute);
            } else if callee == "route_top_k" && args.len() == 3 {
                code.push(NativeInstr::RouteTopK);
            } else if callee == "vector_index" && args.len() == 2 {
                code.push(NativeInstr::MakeVectorIndex);
            } else if callee == "vector_index_insert" && args.len() == 4 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("vector_index_insert requires a named VectorIndex binding".into())); };
                emit_value(&args[1], code)?;
                emit_value(&args[2], code)?;
                emit_value(&args[3], code)?;
                code.push(NativeInstr::VectorIndexInsert { name: name.clone() });
            } else if callee == "vector_index_search" && args.len() == 4 {
                code.push(NativeInstr::VectorIndexSearch);
            } else if callee == "mcts_tree" && args.len() == 2 {
                code.push(NativeInstr::MakeMctsTree);
            } else if callee == "mcts_root" && args.len() == 2 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("mcts_root requires a named MctsTree binding".into())); };
                emit_value(&args[1], code)?; code.push(NativeInstr::MctsRoot { name: name.clone() });
            } else if callee == "mcts_expand" && args.len() == 3 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("mcts_expand requires a named MctsTree binding".into())); };
                emit_value(&args[1], code)?; emit_value(&args[2], code)?; code.push(NativeInstr::MctsExpand { name: name.clone() });
            } else if callee == "mcts_child" && args.len() == 3 {
                for arg in args { emit_value(arg, code)?; } code.push(NativeInstr::MctsChild);
            } else if callee == "mcts_update" && args.len() == 3 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("mcts_update requires a named MctsTree binding".into())); };
                emit_value(&args[1], code)?; emit_value(&args[2], code)?; code.push(NativeInstr::MctsUpdate { name: name.clone() });
            } else if callee == "mcts_backpropagate" && args.len() == 3 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("mcts_backpropagate requires a named MctsTree binding".into())); };
                emit_value(&args[1], code)?; emit_value(&args[2], code)?; code.push(NativeInstr::MctsBackpropagate { name: name.clone() });
            } else if callee == "mcts_len" && args.len() == 1 {
                code.push(NativeInstr::MctsLen);
            } else if callee == "kv_cache" && args.len() == 1 {
                code.push(NativeInstr::MakeKvCache);
            } else if callee == "kv_len" && args.len() == 1 {
                code.push(NativeInstr::KvLen);
            } else if callee == "kv_capacity" && args.len() == 1 {
                code.push(NativeInstr::KvCapacity);
            } else if callee == "kv_append" && args.len() == 4 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("kv_append requires a named KvCache binding".into())); };
                let IrValue::String(policy) = &args[3] else { return Err(NativeError::Unsupported("kv_append policy must be a string literal".into())); };
                if policy != "reject" && policy != "evict_oldest" { return Err(NativeError::Unsupported("kv_append policy must be 'reject' or 'evict_oldest'".into())); }
                emit_value(&args[1], code)?;
                emit_value(&args[2], code)?;
                code.push(NativeInstr::KvAppend { name: name.clone(), evict_oldest: policy == "evict_oldest" });
            } else if callee == "context_window" && args.len() == 3 {
                code.push(NativeInstr::MakeContextWindow);
            } else if callee == "context_len" && args.len() == 1 {
                code.push(NativeInstr::ContextLen);
            } else if callee == "context_bytes" && args.len() == 1 {
                code.push(NativeInstr::ContextBytes);
            } else if callee == "context_tokens" && args.len() == 1 {
                code.push(NativeInstr::ContextTokens);
            } else if callee == "context_append" && args.len() == 4 {
                let IrValue::Name(name) = &args[0] else { return Err(NativeError::Unsupported("context_append requires a named ContextWindow binding".into())); };
                emit_value(&args[1], code)?;
                emit_value(&args[2], code)?;
                emit_value(&args[3], code)?;
                code.push(NativeInstr::ContextAppend { name: name.clone() });
            } else if callee == "len" && args.len() == 1 {
                code.push(NativeInstr::Len);
            } else if callee == "push" && args.len() == 2 {
                let crate::ir::IrValue::Name(name) = &args[0] else {
                    return Err(NativeError::Unsupported(
                        "push currently requires a named list binding".into(),
                    ));
                };
                emit_value(&args[1], code)?;
                code.push(NativeInstr::Append(name.clone()));
            } else {
                code.push(NativeInstr::Call {
                    callee: callee.clone(),
                    argc: args.len(),
                });
            }
        }
    }
    Ok(())
}

pub fn run(
    code: &[NativeInstr],
    args: &[(String, NativeValue)],
) -> Result<NativeValue, NativeError> {
    let params = args.iter().map(|(name, _)| name.clone()).collect();
    let values = args.iter().map(|(_, value)| value.clone()).collect::<Vec<_>>();
    let program = NativeProgram {
        functions: BTreeMap::from([(
            "main".to_string(),
            NativeFunction {
                params,
                code: code.to_vec(),
            },
        )]),
    };
    run_program_with_limits(&program, "main", &values, ExecutionLimits::default())
}

struct NativeTask {
    cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<Result<NativeValue, NativeError>>>,
}

impl NativeTask {
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

impl Drop for NativeTask {
    fn drop(&mut self) {
        self.cancel();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn run_program(
    program: &NativeProgram,
    entry: &str,
    args: &[NativeValue],
) -> Result<NativeValue, NativeError> {
    run_program_with_limits(program, entry, args, ExecutionLimits::default())
}

/// Execute with explicit instruction, recursion, task, collection, and string bounds.
/// Limits are shared by all calls and child tasks in this execution.
///
/// Program representation limits are independent of runtime value limits. They
/// bound function/code counts and embedded names/literals, but do not account for
/// allocator capacity, allocator metadata, host allocations, or concurrent programs.
pub fn run_program_with_limits(
    program: &NativeProgram,
    entry: &str,
    args: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    run_shared_program_with_limits(Arc::new(program.clone()), entry, args, limits)
}

/// Execute an immutable program shared by the caller and all child tasks.
///
/// This compatibility entry point validates on each call. Callers that can
/// retain a validated wrapper should use run_validated_program_with_limits to
/// avoid repeating program-level validation.
pub fn run_shared_program_with_limits(
    program: Arc<NativeProgram>,
    entry: &str,
    args: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    let validated = ValidatedNativeProgram::try_from_shared(program)?;
    run_validated_program_with_limits(&validated, entry, args, limits)
}

/// Execute a previously validated immutable program without revalidating its
/// code and metadata. Caller-specific inputs, execution limits, counters, and
/// task state are still fresh and checked for every invocation.
pub fn run_validated_program_with_limits(
    program: &ValidatedNativeProgram,
    entry: &str,
    args: &[NativeValue],
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    validate_input_values(args, limits)?;
    let function = program.program.functions.get(entry)
        .ok_or_else(|| NativeError::InvalidProgram(format!("unknown function '{entry}'")))?;
    let state = Arc::new(ExecutionState {
        limits,
        instructions: AtomicUsize::new(0),
        active_tasks: AtomicUsize::new(0),
    });
    run_function(&program.program, function, args, None, state, 0)
}

fn run_function(
    program: &Arc<NativeProgram>,
    function: &NativeFunction,
    args: &[NativeValue],
    cancellation: Option<Arc<AtomicBool>>,
    state: Arc<ExecutionState>,
    depth: usize,
) -> Result<NativeValue, NativeError> {
    if depth > state.limits.max_call_depth {
        return Err(NativeError::ResourceLimit(format!(
            "call depth limit exceeded (limit {})", state.limits.max_call_depth
        )));
    }
    if args.len() != function.params.len() {
        return Err(NativeError::InvalidProgram(format!(
            "function expects {} argument(s), got {}",
            function.params.len(),
            args.len()
        )));
    }
    let mut pc = 0usize;
    let mut stack = Vec::new();
    let mut locals = HashMap::new();
    let mut scopes: Vec<BTreeMap<String, NativeTask>> = Vec::new();
    for (name, value) in function.params.iter().zip(args.iter()) {
        if locals.len() >= state.limits.max_locals {
            return Err(NativeError::ResourceLimit("local binding limit exceeded".into()));
        }
        validate_value(value, state.limits)?;
        locals.insert(name.clone(), value.clone());
    }

    check_frame_value_budget(&locals, &stack, state.limits)?;
    while pc < function.code.len() {
        if stack.len() > state.limits.max_stack_values {
            return Err(NativeError::ResourceLimit("operand stack limit exceeded".into()));
        }
        if locals.len() > state.limits.max_locals {
            return Err(NativeError::ResourceLimit("local binding limit exceeded".into()));
        }
        let executed = state.instructions.fetch_add(1, Ordering::AcqRel);
        if executed >= state.limits.max_instructions {
            return Err(NativeError::ResourceLimit(format!(
                "instruction budget exceeded after {} instructions (limit {})", executed + 1, state.limits.max_instructions
            )));
        }
        if cancellation
            .as_ref()
            .is_some_and(|token| token.load(Ordering::Acquire))
        {
            return Err(NativeError::Cancelled("task cancelled".into()));
        }
        let instr = &function.code[pc];
        let check_values_after = matches!(
            instr,
            NativeInstr::PushInt(_)
                | NativeInstr::PushBool(_)
                | NativeInstr::PushProbability(_)
                | NativeInstr::ProbabilityAnd
                | NativeInstr::ProbabilityOr
                | NativeInstr::ProbabilityNot
                | NativeInstr::ProbabilityMeets
                | NativeInstr::PushTensor { .. }
                | NativeInstr::TensorAdd
                | NativeInstr::TensorMatMul
                | NativeInstr::MakeActivationCache | NativeInstr::ActivationStore { .. } | NativeInstr::ActivationRestore | NativeInstr::ActivationCacheLen
                | NativeInstr::MakeGradientTape | NativeInstr::MakeTokenizerState | NativeInstr::TokenizerFeed { .. } | NativeInstr::TokenizerFlush { .. } | NativeInstr::TokenizerTokenCount | NativeInstr::TokenizerToken | NativeInstr::GradientInput { .. } | NativeInstr::GradientAdd { .. } | NativeInstr::GradientMul { .. } | NativeInstr::GradientBackward { .. } | NativeInstr::GradientValue | NativeInstr::GradientGet
                | NativeInstr::MakeAttentionBlock
                | NativeInstr::AttentionBlockApply
                | NativeInstr::AttentionBlockCached
                | NativeInstr::AttentionBlock
                | NativeInstr::AttentionCached
                | NativeInstr::MakeLoRAAdapter
                | NativeInstr::LoRAAdapterApply
                | NativeInstr::LoRAApply
                | NativeInstr::MakeTensor
                | NativeInstr::MakeEmbedding
                | NativeInstr::PushQuantizedTensor { .. }
                | NativeInstr::DequantizeTensor
                | NativeInstr::PushEmbedding { .. }
                | NativeInstr::EmbeddingDot
                | NativeInstr::MakeMoERouter
                | NativeInstr::MoERoute
                | NativeInstr::RouteTopK
                | NativeInstr::MakeVectorIndex
                | NativeInstr::VectorIndexInsert { .. }
                | NativeInstr::VectorIndexSearch
                | NativeInstr::MakeKvCache
                | NativeInstr::KvLen
                | NativeInstr::KvCapacity
                | NativeInstr::KvAppend { .. }
                | NativeInstr::MakeContextWindow
                | NativeInstr::ContextLen
                | NativeInstr::ContextBytes
                | NativeInstr::ContextTokens
                | NativeInstr::ContextAppend { .. }
                | NativeInstr::PushUnit
                | NativeInstr::PushString(_)
                | NativeInstr::PushList(_)
                | NativeInstr::Index
                | NativeInstr::Len
                | NativeInstr::Append(_)
                | NativeInstr::MakeOk
                | NativeInstr::MakeErr
                | NativeInstr::Chr
                | NativeInstr::Unwrap
                | NativeInstr::Load(_)
                | NativeInstr::AddAssign(_)
                | NativeInstr::StoreIndex(_)
                | NativeInstr::Add
                | NativeInstr::Call { .. }
                | NativeInstr::Join { .. }
        );
        pc += 1;
        match instr {
            NativeInstr::ScopeStart => scopes.push(BTreeMap::new()),
            NativeInstr::ScopeEnd => {
                let mut tasks = scopes.pop().ok_or_else(|| {
                    NativeError::InvalidProgram("scope end without scope start".into())
                })?;
                let names = tasks.keys().cloned().collect::<Vec<_>>();
                for name in names {
                    let mut task = tasks.remove(&name).expect("task disappeared");
                    let result = task
                        .join
                        .take()
                        .expect("task already joined")
                        .join()
                        .map_err(|_| NativeError::Unsupported(format!("task '{name}' panicked")))?;
                    if let Err(error) = result {
                        if matches!(error, NativeError::Cancelled(_)) {
                            continue;
                        }
                        for sibling in tasks.values() {
                            sibling.cancel();
                        }
                        return Err(error);
                    }
                }
            }
            NativeInstr::Spawn { name, callee, argc } => {
                let scope = scopes.last_mut().ok_or_else(|| {
                    NativeError::InvalidProgram("spawn must occur inside a scope".into())
                })?;
                if scope.contains_key(name) {
                    return Err(NativeError::InvalidProgram(format!(
                        "duplicate task '{name}'"
                    )));
                }
                if stack.len() < *argc {
                    return Err(NativeError::InvalidProgram(
                        "spawn has fewer stack arguments than declared".into(),
                    ));
                }
                let start = stack.len() - *argc;
                let call_args = stack.split_off(start);
                let task_lease = reserve_task(&state)?;
                let child_program = Arc::clone(program);
                let child_state = Arc::clone(&state);
                let child_depth = depth + 1;
                let token = Arc::new(AtomicBool::new(false));
                let child_token = token.clone();
                let task_name = name.clone();
                let callee = callee.clone();
                let join = thread::Builder::new().name(format!("ardisa-{task_name}")).spawn(move || {
                    let _task_lease = task_lease;
                    let function = child_program.functions.get(&callee).ok_or_else(|| {
                        NativeError::InvalidProgram(format!("unknown function '{callee}'"))
                    })?;
                    run_function(&child_program, function, &call_args, Some(child_token), child_state, child_depth)
                }).map_err(|error| NativeError::ResourceLimit(format!("worker creation failed: {error}")))?;
                scope.insert(
                    name.clone(),
                    NativeTask {
                        cancel: token,
                        join: Some(join),
                    },
                );
            }
            NativeInstr::Join { name } => {
                let scope = scopes.last_mut().ok_or_else(|| {
                    NativeError::InvalidProgram("join must occur inside a scope".into())
                })?;
                let mut task = scope
                    .remove(name)
                    .ok_or_else(|| NativeError::InvalidProgram(format!("unknown task '{name}'")))?;
                let result = task
                    .join
                    .take()
                    .expect("task already joined")
                    .join()
                    .map_err(|_| NativeError::Unsupported(format!("task '{name}' panicked")))?;
                match result {
                    Ok(_) => stack.push(NativeValue::Unit),
                    Err(NativeError::Cancelled(_)) => stack.push(NativeValue::Unit),
                    Err(error) => {
                        for sibling in scope.values() {
                            sibling.cancel();
                        }
                        return Err(error);
                    }
                }
            }
            NativeInstr::Cancel { name } => {
                let scope = scopes.last_mut().ok_or_else(|| {
                    NativeError::InvalidProgram("cancel must occur inside a scope".into())
                })?;
                let task = scope
                    .get(name)
                    .ok_or_else(|| NativeError::InvalidProgram(format!("unknown task '{name}'")))?;
                task.cancel();
            }
            NativeInstr::Call { callee, argc } => {
                if stack.len() < *argc {
                    return Err(NativeError::InvalidProgram(
                        "call has fewer stack arguments than declared".into(),
                    ));
                }
                let start = stack.len() - *argc;
                let call_args = stack.split_off(start);
                let callee_fn = program.functions.get(callee).ok_or_else(|| {
                    NativeError::InvalidProgram(format!("unknown function '{callee}'"))
                })?;
                let value = run_function(program, callee_fn, &call_args, cancellation.clone(), Arc::clone(&state), depth + 1)?;
                stack.push(value);
            }
            NativeInstr::PushInt(value) => stack.push(NativeValue::Int(*value)),
            NativeInstr::PushFloat(bits) => {
                if !f32::from_bits(*bits).is_finite() { return Err(NativeError::Type("float literal must be finite".into())); }
                stack.push(NativeValue::Float(*bits));
            }
            NativeInstr::PushBool(value) => stack.push(NativeValue::Bool(*value)),
            NativeInstr::PushProbability(bits) => {
                validate_probability_bits(*bits)?;
                stack.push(NativeValue::Probability(*bits));
            }
            NativeInstr::ProbabilityAnd | NativeInstr::ProbabilityOr => {
                let right = pop_probability(&mut stack)?;
                let left = pop_probability(&mut stack)?;
                let value = if matches!(instr, NativeInstr::ProbabilityAnd) {
                    left * right
                } else {
                    left + right - left * right
                };
                let bits = value.to_bits();
                validate_probability_bits(bits)?;
                stack.push(NativeValue::Probability(bits));
            }
            NativeInstr::ProbabilityNot => {
                let value = pop_probability(&mut stack)?;
                let bits = (1.0f32 - value).to_bits();
                validate_probability_bits(bits)?;
                stack.push(NativeValue::Probability(bits));
            }
            NativeInstr::ProbabilityMeets => {
                let threshold = pop_probability(&mut stack)?;
                let value = pop_probability(&mut stack)?;
                stack.push(NativeValue::Bool(value >= threshold));
            }
            NativeInstr::MakeMoERouter => {
                let top_k = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("moe_router top_k must be positive".into()))?;
                let expert_count = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("moe_router expert_count must be positive".into()))?;
                let router = NativeValue::MoERouter { expert_count, top_k };
                validate_value(&router, state.limits)?;
                if top_k > expert_count { return Err(NativeError::Type("moe_router top_k cannot exceed expert_count".into())); }
                stack.push(router);
            }
            NativeInstr::MoERoute => {
                let masked = match stack.pop() { Some(NativeValue::List(values)) => values.into_iter().map(|v| match v { NativeValue::Bool(value) => Ok(value), _ => Err(NativeError::Type("moe_route mask must contain Bool values".into())) }).collect::<Result<Vec<_>,_>>()?, _ => return Err(NativeError::Type("moe_route mask must be a List<Bool>".into())) };
                let logits = match stack.pop() { Some(NativeValue::List(values)) => values.into_iter().map(|v| match v { NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(f32::from_bits(bits)), _ => Err(NativeError::Type("moe_route logits must contain finite F32 values".into())) }).collect::<Result<Vec<_>,_>>()?, _ => return Err(NativeError::Type("moe_route logits must be a List<F32>".into())) };
                let router = match stack.pop() { Some(NativeValue::MoERouter { expert_count, top_k }) => (expert_count, top_k), _ => return Err(NativeError::Type("moe_route requires a nominal MoERouter value".into())) };
                if logits.len() != router.0 || masked.len() != router.0 { return Err(NativeError::Type("moe_route logits and mask lengths must equal configured expert_count".into())); }
                let routed = crate::ai_routing::route_top_k(&logits, &masked, router.1).map_err(|e| NativeError::Type(format!("moe_route rejected input: {e:?}")))?;
                let rows = routed.into_iter().map(|v| NativeValue::List(vec![NativeValue::Int(v.expert as i64), NativeValue::Float(v.weight.to_bits())])).collect();
                let value = NativeValue::List(rows); validate_value(&value, state.limits)?; stack.push(value);
            }
            NativeInstr::RouteTopK => {
                let top_k = pop_int(&mut stack)?;
                let top_k = usize::try_from(top_k).map_err(|_| NativeError::Type("route_top_k top_k must be positive".into()))?;
                let masked = match stack.pop() { Some(NativeValue::List(values)) => values.into_iter().map(|v| match v { NativeValue::Bool(value) => Ok(value), _ => Err(NativeError::Type("route_top_k mask must contain Bool values".into())) }).collect::<Result<Vec<_>,_>>()?, _ => return Err(NativeError::Type("route_top_k mask must be a List<Bool>".into())) };
                let logits = match stack.pop() { Some(NativeValue::List(values)) => values.into_iter().map(|v| match v { NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(f32::from_bits(bits)), _ => Err(NativeError::Type("route_top_k logits must contain finite F32 values".into())) }).collect::<Result<Vec<_>,_>>()?, _ => return Err(NativeError::Type("route_top_k logits must be a List<F32>".into())) };
                let routed = crate::ai_routing::route_top_k(&logits, &masked, top_k).map_err(|e| NativeError::Type(format!("route_top_k rejected input: {e:?}")))?;
                let rows = routed.into_iter().map(|v| NativeValue::List(vec![NativeValue::Int(v.expert as i64), NativeValue::Float(v.weight.to_bits())])).collect();
                let value = NativeValue::List(rows); validate_value(&value, state.limits)?; stack.push(value);
            }
            NativeInstr::MakeVectorIndex => {
                let dimensions = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("vector_index dimensions must be positive".into()))?;
                let domain = match stack.pop() { Some(NativeValue::String(value)) => value, _ => return Err(NativeError::Type("vector_index domain must be String".into())) };
                if domain.trim().is_empty() || domain.len() > 1024 { return Err(NativeError::Type("vector_index domain must contain 1..=1024 bytes".into())); }
                if dimensions == 0 || dimensions > crate::ai_routing::MAX_VECTOR_DIMENSIONS || dimensions > state.limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("vector_index dimensions exceed supported bounds".into()));
                }
                let index = NativeValue::VectorIndex { domain, dimensions, entries: Vec::new() };
                validate_value(&index, state.limits)?;
                stack.push(index);
            }
            NativeInstr::VectorIndexInsert { name } => {
                let coordinates = match stack.pop() { Some(NativeValue::List(values)) => values.into_iter().map(|v| match v {
                    NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(bits),
                    _ => Err(NativeError::Type("vector coordinates must be finite F32 values".into())),
                }).collect::<Result<Vec<_>,_>>()?, _ => return Err(NativeError::Type("vector coordinates must be List<F32>".into())) };
                let supplied_domain = match stack.pop() { Some(NativeValue::String(value)) => value, _ => return Err(NativeError::Type("vector insert domain must be String".into())) };
                let id = u64::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("vector id must be non-negative".into()))?;
                if let Some(index) = locals.get(name) { validate_value(index, state.limits)?; }
                let Some(NativeValue::VectorIndex { domain, dimensions, entries }) = locals.get_mut(name) else {
                    return Err(NativeError::Type("vector_index_insert requires a nominal VectorIndex binding".into()));
                };
                if supplied_domain != *domain { return Err(NativeError::Type("vector index domain mismatch".into())); }
                if coordinates.len() != *dimensions { return Err(NativeError::Type("vector dimension mismatch".into())); }
                if entries.len() >= crate::ai_routing::MAX_INDEX_ENTRIES.min(state.limits.max_collection_items) {
                    return Err(NativeError::ResourceLimit("VectorIndex entry limit exceeded".into()));
                }
                if entries.iter().any(|(existing, _)| *existing == id) { return Err(NativeError::Type("duplicate VectorIndex id".into())); }
                let node_count = entries.len().checked_add(1)
                    .and_then(|count| dimensions.checked_add(1).and_then(|width| count.checked_mul(width)))
                    .and_then(|nodes| nodes.checked_add(1))
                    .ok_or_else(|| NativeError::ResourceLimit("VectorIndex retained storage accounting overflow".into()))?;
                if node_count > state.limits.max_value_nodes {
                    return Err(NativeError::ResourceLimit("VectorIndex retained storage exceeds value-node budget".into()));
                }
                entries.push((id, coordinates));
                stack.push(NativeValue::Unit);
            }
            NativeInstr::VectorIndexSearch => {
                let limit = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("vector_index_search limit must be non-negative".into()))?;
                if limit > crate::ai_routing::MAX_INDEX_ENTRIES { return Err(NativeError::ResourceLimit("VectorIndex search limit exceeded".into())); }
                let query = match stack.pop() { Some(NativeValue::List(values)) => values.into_iter().map(|v| match v {
                    NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(bits),
                    _ => Err(NativeError::Type("vector query must contain finite F32 values".into())),
                }).collect::<Result<Vec<_>,_>>()?, _ => return Err(NativeError::Type("vector query must be List<F32>".into())) };
                let supplied_domain = match stack.pop() { Some(NativeValue::String(value)) => value, _ => return Err(NativeError::Type("vector search domain must be String".into())) };
                let index = match stack.pop() { Some(value @ NativeValue::VectorIndex { .. }) => value, _ => return Err(NativeError::Type("vector_index_search requires a nominal VectorIndex".into())) };
                validate_value(&index, state.limits)?;
                let NativeValue::VectorIndex { domain, dimensions, entries } = index else { unreachable!() };
                if supplied_domain != domain { return Err(NativeError::Type("vector index domain mismatch".into())); }
                if query.len() != dimensions { return Err(NativeError::Type("vector query dimension mismatch".into())); }
                let work = entries.len().checked_mul(dimensions).ok_or_else(|| NativeError::ResourceLimit("VectorIndex search work overflow".into()))?;
                if work > 1_000_000 || work > state.limits.max_value_nodes.saturating_mul(16) {
                    return Err(NativeError::ResourceLimit("VectorIndex search work limit exceeded".into()));
                }
                let mut hits = Vec::with_capacity(entries.len());
                for (id, coordinates) in &entries {
                    let mut distance = 0.0f64;
                    for (bits, query_bits) in coordinates.iter().zip(&query) {
                        let delta = f64::from(f32::from_bits(*bits)) - f64::from(f32::from_bits(*query_bits));
                        distance += delta * delta;
                    }
                    if !distance.is_finite() { return Err(NativeError::Type("vector distance overflow".into())); }
                    let distance_f32 = distance as f32;
                    if !distance_f32.is_finite() { return Err(NativeError::Type("vector distance exceeds F32 output range".into())); }
                    hits.push((*id, distance_f32));
                }
                hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
                hits.truncate(limit);
                let value = NativeValue::List(hits.into_iter().map(|(id, distance)| NativeValue::List(vec![NativeValue::Int(id as i64), NativeValue::Float(distance.to_bits())])).collect());
                validate_value(&value, state.limits)?;
                stack.push(value);
            }
            NativeInstr::PushTensor { shape, values } => {
                validate_tensor_parts(shape, values, state.limits.max_collection_items)?;
                stack.push(NativeValue::Tensor { shape: shape.clone(), values: values.clone() });
            }
            NativeInstr::TensorAdd => {
                let right = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let left = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                if left.0 != right.0 {
                    return Err(NativeError::Type("tensor addition requires identical shapes".into()));
                }
                let values = left.1.iter().zip(&right.1).map(|(a, b)| {
                    let value = f32::from_bits(*a) + f32::from_bits(*b);
                    if !value.is_finite() { return Err(NativeError::Type("tensor operation produced non-finite value".into())); }
                    Ok(value.to_bits())
                }).collect::<Result<Vec<_>, _>>()?;
                stack.push(NativeValue::Tensor { shape: left.0, values });
            }
            NativeInstr::TensorMatMul => {
                let right = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let left = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                if left.0.len() != 2 || right.0.len() != 2 || left.0[1] != right.0[0] {
                    return Err(NativeError::Type("tensor matmul requires rank-2 tensors with compatible inner dimensions".into()));
                }
                let (m, k, n) = (left.0[0], left.0[1], right.0[1]);
                let out_count = m.checked_mul(n).ok_or_else(|| NativeError::ResourceLimit("tensor output size overflow".into()))?;
                if out_count > state.limits.max_collection_items || out_count > MAX_TENSOR_ELEMENTS {
                    return Err(NativeError::ResourceLimit("tensor output exceeds element limit".into()));
                }
                let mut values = Vec::with_capacity(out_count);
                for row in 0..m {
                    for col in 0..n {
                        let mut sum = 0.0f32;
                        for inner in 0..k {
                            sum += f32::from_bits(left.1[row * k + inner]) * f32::from_bits(right.1[inner * n + col]);
                        }
                        if !sum.is_finite() { return Err(NativeError::Type("tensor matmul produced non-finite value".into())); }
                        values.push(sum.to_bits());
                    }
                }
                stack.push(NativeValue::Tensor { shape: vec![m, n], values });
            }
            NativeInstr::MakeAttentionBlock => {
                let causal = pop_bool(&mut stack)?;
                let head_dim = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("AttentionBlock head_dim must be positive".into()))?;
                let heads = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("AttentionBlock heads must be positive".into()))?;
                let hidden = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("AttentionBlock hidden must be positive".into()))?;
                if hidden == 0 || heads == 0 || head_dim == 0 || heads > 256 || heads.checked_mul(head_dim) != Some(hidden) {
                    return Err(NativeError::Type("AttentionBlock requires hidden == heads * head_dim, positive dimensions, and heads <= 256".into()));
                }
                stack.push(NativeValue::AttentionBlock { hidden, heads, head_dim, causal });
            }
            NativeInstr::AttentionBlockApply => {
                let value = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let key = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let query = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let block = match stack.pop() { Some(NativeValue::AttentionBlock { hidden, heads, head_dim, causal }) => (hidden, heads, head_dim, causal), _ => return Err(NativeError::Type("attention_block_apply requires an AttentionBlock".into())) };
                let output = attention_multihead(&block, &query.0, &query.1, &key.0, &key.1, &value.0, &value.1, state.limits)?;
                stack.push(NativeValue::Tensor { shape: output.0, values: output.1 });
            }
            NativeInstr::AttentionBlockCached => {
                let cache = match stack.pop() { Some(NativeValue::KvCache { entries, .. }) => entries, _ => return Err(NativeError::Type("attention_block_cached requires a KvCache".into())) };
                let query = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let block = match stack.pop() { Some(NativeValue::AttentionBlock { hidden, heads, head_dim, causal }) => (hidden, heads, head_dim, causal), _ => return Err(NativeError::Type("attention_block_cached requires an AttentionBlock".into())) };
                if query.0.len() != 2 || query.0[1] != block.0 || cache.is_empty() || cache.iter().any(|(k,v)| k.len()!=block.0 || v.len()!=block.0) {
                    return Err(NativeError::Type("AttentionBlock/KVCache dimensions are incompatible or cache is empty".into()));
                }
                let (tq, hidden, heads, head_dim) = (query.0[0], block.0, block.1, block.2);
                let out_count = tq.checked_mul(hidden).ok_or_else(|| NativeError::ResourceLimit("cached attention output size overflow".into()))?;
                let work = tq.checked_mul(cache.len()).and_then(|n| n.checked_mul(hidden)).ok_or_else(|| NativeError::ResourceLimit("cached attention work overflow".into()))?;
                if out_count > state.limits.max_collection_items.min(MAX_TENSOR_ELEMENTS) || work > state.limits.max_value_nodes.saturating_mul(16) { return Err(NativeError::ResourceLimit("cached attention exceeds resource budget".into())); }
                let mut out = vec![0.0f32; out_count];
                for qi in 0..tq { for h in 0..heads {
                    let start = h*head_dim; let mut scores=Vec::with_capacity(cache.len());
                    for (k,_) in &cache { let mut dot=0.0f32; for d in 0..head_dim { dot += f32::from_bits(query.1[qi*hidden+start+d])*f32::from_bits(k[start+d]); } let score=dot/(head_dim as f32).sqrt(); if !score.is_finite(){return Err(NativeError::Type("cached attention score is non-finite".into()));} scores.push(score); }
                    let max=scores.iter().copied().fold(f32::NEG_INFINITY,f32::max); let mut weights=scores.iter().map(|x|(*x-max).exp()).collect::<Vec<_>>(); let total: f32=weights.iter().sum(); if !total.is_finite()||total<=0.0{return Err(NativeError::Type("cached attention softmax failed".into()));} for w in &mut weights{*w/=total;}
                    for d in 0..head_dim { let mut sum=0.0f32; for (i,(_,v)) in cache.iter().enumerate(){sum+=weights[i]*f32::from_bits(v[start+d]);} if !sum.is_finite(){return Err(NativeError::Type("cached attention output is non-finite".into()));} out[qi*hidden+start+d]=sum; }
                }}
                stack.push(NativeValue::Tensor { shape: vec![tq,hidden], values: out.into_iter().map(f32::to_bits).collect() });
            }
            NativeInstr::AttentionBlock => {
                let causal = pop_bool(&mut stack)?;
                let value = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let key = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let query = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                if query.0.len() != 2 || key.0.len() != 2 || value.0.len() != 2
                    || query.0[1] != key.0[1] || key.0[0] != value.0[0]
                    || query.0[0] == 0 || key.0[0] == 0 {
                    return Err(NativeError::Type("attention_block requires Q[Tq,D], K[Tk,D], V[Tk,Dv] with compatible non-empty shapes".into()));
                }
                let (tq, d, tk, dv) = (query.0[0], query.0[1], key.0[0], value.0[1]);
                let out_count = tq.checked_mul(dv).ok_or_else(|| NativeError::ResourceLimit("attention output size overflow".into()))?;
                let work = tq.checked_mul(tk).and_then(|x| x.checked_mul(d)).ok_or_else(|| NativeError::ResourceLimit("attention work size overflow".into()))?;
                if out_count > state.limits.max_collection_items.min(MAX_TENSOR_ELEMENTS)
                    || work > state.limits.max_value_nodes.saturating_mul(16) {
                    return Err(NativeError::ResourceLimit("attention computation exceeds resource budget".into()));
                }
                let scale = (d as f32).sqrt();
                if !scale.is_finite() || scale == 0.0 { return Err(NativeError::Type("attention head dimension must be positive".into())); }
                let mut output = vec![0.0f32; out_count];
                for qi in 0..tq {
                    let mut scores = Vec::with_capacity(tk);
                    for ki in 0..tk {
                        if causal && ki > qi { scores.push(f32::NEG_INFINITY); continue; }
                        let mut dot = 0.0f32;
                        for di in 0..d {
                            dot += f32::from_bits(query.1[qi*d+di]) * f32::from_bits(key.1[ki*d+di]);
                        }
                        let score = dot / scale;
                        if !score.is_finite() { return Err(NativeError::Type("attention score is non-finite".into())); }
                        scores.push(score);
                    }
                    let max_score = scores.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
                    if !max_score.is_finite() { return Err(NativeError::Type("attention row is fully masked".into())); }
                    let mut weights = scores.iter().map(|s| if s.is_finite() { (*s-max_score).exp() } else { 0.0 }).collect::<Vec<_>>();
                    let total: f32 = weights.iter().sum();
                    if !total.is_finite() || total <= 0.0 { return Err(NativeError::Type("attention softmax normalization failed".into())); }
                    for w in &mut weights { *w /= total; }
                    for vi in 0..dv {
                        let mut sum = 0.0f32;
                        for ki in 0..tk { sum += weights[ki] * f32::from_bits(value.1[ki*dv+vi]); }
                        if !sum.is_finite() { return Err(NativeError::Type("attention output is non-finite".into())); }
                        output[qi*dv+vi] = sum;
                    }
                }
                stack.push(NativeValue::Tensor { shape: vec![tq, dv], values: output.into_iter().map(f32::to_bits).collect() });
            }
            NativeInstr::AttentionCached => {
                let cache = match stack.pop() { Some(NativeValue::KvCache { entries, .. }) => entries, _ => return Err(NativeError::Type("attention_cached requires a KvCache".into())) };
                let query = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                if query.0.len() != 2 || query.0[1] == 0 || cache.is_empty() {
                    return Err(NativeError::Type("attention_cached requires a rank-2 query and non-empty KVCache".into()));
                }
                let (tq, d) = (query.0[0], query.0[1]);
                let dv = cache[0].1.len();
                if cache.iter().any(|(key, value)| key.len() != d || value.len() != dv) || dv == 0 {
                    return Err(NativeError::Type("attention_cached query/key/value dimensions are incompatible".into()));
                }
                let out_count = tq.checked_mul(dv).ok_or_else(|| NativeError::ResourceLimit("cached attention output size overflow".into()))?;
                let work = tq.checked_mul(cache.len()).and_then(|x| x.checked_mul(d)).ok_or_else(|| NativeError::ResourceLimit("cached attention work size overflow".into()))?;
                if out_count > state.limits.max_collection_items.min(MAX_TENSOR_ELEMENTS) || work > state.limits.max_value_nodes.saturating_mul(16) {
                    return Err(NativeError::ResourceLimit("cached attention exceeds resource budget".into()));
                }
                let scale = (d as f32).sqrt();
                if !scale.is_finite() || scale == 0.0 { return Err(NativeError::Type("attention head dimension must be positive".into())); }
                let mut output = vec![0.0f32; out_count];
                for qi in 0..tq {
                    let mut scores = Vec::with_capacity(cache.len());
                    for (key, _) in &cache {
                        let mut dot = 0.0f32;
                        for di in 0..d { dot += f32::from_bits(query.1[qi*d+di]) * f32::from_bits(key[di]); }
                        let score = dot / scale;
                        if !score.is_finite() { return Err(NativeError::Type("cached attention score is non-finite".into())); }
                        scores.push(score);
                    }
                    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut weights = scores.iter().map(|s| (*s-max_score).exp()).collect::<Vec<_>>();
                    let total: f32 = weights.iter().sum();
                    if !total.is_finite() || total <= 0.0 { return Err(NativeError::Type("cached attention softmax normalization failed".into())); }
                    for weight in &mut weights { *weight /= total; }
                    for vi in 0..dv {
                        let mut sum = 0.0f32;
                        for (ki, (_, value)) in cache.iter().enumerate() { sum += weights[ki] * f32::from_bits(value[vi]); }
                        if !sum.is_finite() { return Err(NativeError::Type("cached attention output is non-finite".into())); }
                        output[qi*dv+vi] = sum;
                    }
                }
                stack.push(NativeValue::Tensor { shape: vec![tq, dv], values: output.into_iter().map(f32::to_bits).collect() });
            }
            NativeInstr::MakeLoRAAdapter => {
                let rank = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("LoRA rank must be positive".into()))?;
                let alpha = pop_float(&mut stack)?;
                let b = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let a = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                validate_lora_parts(&a.0, &a.1, &b.0, &b.1, alpha.to_bits(), rank, state.limits.max_collection_items)?;
                stack.push(NativeValue::LoRAAdapter { a_shape: a.0, a_values: a.1, b_shape: b.0, b_values: b.1, alpha_bits: alpha.to_bits(), rank });
            }
            NativeInstr::LoRAAdapterApply => {
                let input = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let (a_shape,a_values,b_shape,b_values,alpha_bits,rank) = match stack.pop() { Some(NativeValue::LoRAAdapter { a_shape,a_values,b_shape,b_values,alpha_bits,rank }) => (a_shape,a_values,b_shape,b_values,alpha_bits,rank), _ => return Err(NativeError::Type("lora_adapter_apply requires a LoRAAdapter".into())) };
                validate_lora_parts(&a_shape,&a_values,&b_shape,&b_values,alpha_bits,rank,state.limits.max_collection_items)?;
                if input.0.len()!=2 || input.0[1]!=a_shape[1] { return Err(NativeError::Type("LoRAAdapter input dimension mismatch".into())); }
                let (batch,input_dim,output_dim)=(input.0[0],a_shape[1],b_shape[0]);
                let out_count=batch.checked_mul(output_dim).ok_or_else(||NativeError::ResourceLimit("LoRA adapter output overflow".into()))?;
                let work=batch.checked_mul(rank).and_then(|x|x.checked_mul(input_dim)).and_then(|x|x.checked_add(batch.checked_mul(rank)?.checked_mul(output_dim)?)).ok_or_else(||NativeError::ResourceLimit("LoRA adapter work overflow".into()))?;
                if out_count>state.limits.max_collection_items.min(MAX_TENSOR_ELEMENTS)||work>state.limits.max_value_nodes.saturating_mul(16){return Err(NativeError::ResourceLimit("LoRA adapter exceeds resource budget".into()));}
                let scale=f32::from_bits(alpha_bits)/(rank as f32); let mut out=vec![0.0f32;out_count];
                for row in 0..batch { for r in 0..rank { let mut sum=0.0f32; for col in 0..input_dim { sum+=f32::from_bits(input.1[row*input_dim+col])*f32::from_bits(a_values[r*input_dim+col]); } if !sum.is_finite(){return Err(NativeError::Type("LoRA adapter intermediate is non-finite".into()));} for col in 0..output_dim { out[row*output_dim+col]+=scale*sum*f32::from_bits(b_values[col*rank+r]); } }}
                if out.iter().any(|v|!v.is_finite()){return Err(NativeError::Type("LoRA adapter output is non-finite".into()));}
                stack.push(NativeValue::Tensor { shape: vec![batch,output_dim], values: out.into_iter().map(f32::to_bits).collect() });
            }
            NativeInstr::LoRAApply => {
                let rank = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("LoRA rank must be positive".into()))?;
                let alpha = pop_float(&mut stack)?;
                let b = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let a = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                let input = pop_tensor(&mut stack, state.limits.max_collection_items)?;
                if rank == 0 || rank > 256 || !alpha.is_finite()
                    || input.0.len()!=2 || a.0.len()!=2 || b.0.len()!=2
                    || a.0[0]!=rank || b.0[1]!=rank || a.0[1]!=input.0[1] {
                    return Err(NativeError::Type("lora_apply requires input[batch,in], A[rank,in], B[out,rank], finite alpha, and matching rank".into()));
                }
                let (batch, input_dim, output_dim)=(input.0[0],input.0[1],b.0[0]);
                let out_count=batch.checked_mul(output_dim).ok_or_else(||NativeError::ResourceLimit("LoRA output size overflow".into()))?;
                let work=batch.checked_mul(rank).and_then(|x|x.checked_mul(input_dim)).and_then(|x|x.checked_add(batch.saturating_mul(rank).saturating_mul(output_dim))).ok_or_else(||NativeError::ResourceLimit("LoRA work size overflow".into()))?;
                if out_count>state.limits.max_collection_items.min(MAX_TENSOR_ELEMENTS) || work>state.limits.max_value_nodes.saturating_mul(16) {
                    return Err(NativeError::ResourceLimit("LoRA computation exceeds resource budget".into()));
                }
                let scale=alpha/(rank as f32);
                if !scale.is_finite(){return Err(NativeError::Type("LoRA alpha/rank scale is non-finite".into()));}
                let mut output=vec![0.0f32;out_count];
                for row in 0..batch { for r in 0..rank {
                    let mut sum=0.0f32;
                    for col in 0..input_dim {sum+=f32::from_bits(input.1[row*input_dim+col])*f32::from_bits(a.1[r*input_dim+col]);}
                    if !sum.is_finite(){return Err(NativeError::Type("LoRA intermediate is non-finite".into()));}
                    for col in 0..output_dim {output[row*output_dim+col]+=scale*sum*f32::from_bits(b.1[col*rank+r]);}
                }}
                if output.iter().any(|v|!v.is_finite()){return Err(NativeError::Type("LoRA output is non-finite".into()));}
                stack.push(NativeValue::Tensor{shape:vec![batch,output_dim],values:output.into_iter().map(f32::to_bits).collect()});
            }
            NativeInstr::MakeActivationCache => { let max_bytes=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("ActivationCache byte budget must be positive".into()))?;let stride=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("ActivationCache stride must be positive".into()))?;let capacity=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("ActivationCache capacity must be positive".into()))?;if capacity==0||capacity>4096||stride==0||stride>4096||max_bytes==0||max_bytes>16_777_216{return Err(NativeError::ResourceLimit("ActivationCache limits exceed supported bounds".into()));}stack.push(NativeValue::ActivationCache{capacity,checkpoint_stride:stride,max_bytes,retained_bytes:0,entries:Vec::new()}); }
            NativeInstr::ActivationStore{name} => { let (shape,values)=pop_tensor(&mut stack,state.limits.max_collection_items)?;let step=pop_int(&mut stack)?;let cache=locals.get_mut(name).ok_or_else(||NativeError::Type("ActivationCache binding not found".into()))?;store_activation_checkpoint(cache,step,shape,values)?;stack.push(NativeValue::Unit); }
            NativeInstr::ActivationRestore => {let step=pop_int(&mut stack)?;let cache=stack.pop().ok_or_else(||NativeError::InvalidProgram("ActivationCache operand missing".into()))?;let NativeValue::ActivationCache{entries,..}=cache else{return Err(NativeError::Type("activation_restore requires ActivationCache".into()));};let (_,shape,values)=entries.iter().find(|(id,_,_)|*id==step).ok_or_else(||NativeError::Type("ActivationCache checkpoint id not found".into()))?;stack.push(NativeValue::Tensor{shape:shape.clone(),values:values.clone()});}
            NativeInstr::ActivationCacheLen => {let cache=stack.pop().ok_or_else(||NativeError::InvalidProgram("ActivationCache operand missing".into()))?;let NativeValue::ActivationCache{entries,..}=cache else{return Err(NativeError::Type("activation_cache_len requires ActivationCache".into()));};stack.push(NativeValue::Int(entries.len() as i64));}
            NativeInstr::MakeTokenizerState => {
                let unknown=pop_int(&mut stack)?; let unknown_id=if unknown == 0 { None } else if unknown > 0 { Some(unknown-1) } else { return Err(NativeError::Type("TokenizerState unknown_id encoding must be zero (reject) or a positive id-plus-one".into())); };
                let max_tokens=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("TokenizerState max_tokens must be positive".into()))?;
                let max_pending=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("TokenizerState max_pending must be positive".into()))?;
                let vocabulary=match stack.pop(){Some(NativeValue::List(values))=>values.into_iter().map(|v|match v{NativeValue::String(piece)=>Ok(piece),_=>Err(NativeError::Type("TokenizerState vocabulary must contain strings".into()))}).collect::<Result<Vec<_>,_>>()?,_=>return Err(NativeError::Type("TokenizerState vocabulary must be List<String>".into()))};
                if vocabulary.is_empty()||vocabulary.len()>state.limits.max_collection_items||max_pending==0||max_pending>4||max_tokens==0||max_tokens>state.limits.max_collection_items{return Err(NativeError::ResourceLimit("TokenizerState vocabulary and limits are outside supported bounds".into()));}
                let mut seen=std::collections::BTreeSet::new(); for piece in &vocabulary {if piece.chars().count()!=1||!seen.insert(piece.clone()){return Err(NativeError::Type("TokenizerState vocabulary must contain unique single Unicode scalars".into()));}}
                if unknown_id.is_some_and(|id|id<0){return Err(NativeError::Type("TokenizerState unknown id must be non-negative".into()));}
                stack.push(NativeValue::TokenizerState{vocabulary,pending:Vec::new(),tokens:Vec::new(),max_pending,max_tokens,unknown_id});
            }
            NativeInstr::TokenizerFeed{name} => {
                let bytes=match stack.pop(){Some(NativeValue::List(values))=>values.into_iter().map(|v|match v{NativeValue::Int(n) if (0..=255).contains(&n)=>Ok(n as u8),_=>Err(NativeError::Type("tokenizer_feed bytes must be Int values in 0..=255".into()))}).collect::<Result<Vec<_>,_>>()?,_=>return Err(NativeError::Type("tokenizer_feed expects List<Int> bytes".into()))};
                let value=locals.get_mut(name).ok_or_else(||NativeError::Type("TokenizerState binding not found".into()))?;
                let NativeValue::TokenizerState{vocabulary,pending,tokens,max_pending,max_tokens,unknown_id}=value else{return Err(NativeError::Type("tokenizer_feed requires TokenizerState".into()));};
                let mut candidate=pending.clone();candidate.extend(bytes);
                let decoded=match std::str::from_utf8(&candidate){Ok(text)=>(text.len(),text.to_string()),Err(error)=>{if error.error_len().is_some(){return Err(NativeError::Type("tokenizer_feed rejected invalid UTF-8".into()));}(error.valid_up_to(),std::str::from_utf8(&candidate[..error.valid_up_to()]).map_err(|_|NativeError::InvalidProgram("UTF-8 valid-prefix accounting failed".into()))?.to_string())}};
                let mut emitted=Vec::new();for ch in decoded.1.chars(){let piece=ch.to_string();let id=vocabulary.iter().position(|v|v==&piece).map(|v|v as i64).or(*unknown_id).ok_or_else(||NativeError::Type(format!("TokenizerState unknown vocabulary scalar {:?}",piece)))?;emitted.push(id);}
                let remain=candidate[decoded.0..].to_vec();if remain.len()>*max_pending{return Err(NativeError::ResourceLimit("TokenizerState pending UTF-8 bytes exceed configured limit".into()));}if tokens.len().checked_add(emitted.len()).is_none_or(|n|n>*max_tokens){return Err(NativeError::ResourceLimit("TokenizerState token output capacity exhausted".into()));}
                tokens.extend(emitted);*pending=remain;stack.push(NativeValue::Unit);
            }
            NativeInstr::TokenizerFlush{name} => {
                let value=locals.get_mut(name).ok_or_else(||NativeError::Type("TokenizerState binding not found".into()))?;
                let NativeValue::TokenizerState{pending,..}=value else{return Err(NativeError::Type("tokenizer_flush requires TokenizerState".into()));};
                if !pending.is_empty(){return Err(NativeError::Type("tokenizer_flush cannot finish an incomplete UTF-8 sequence".into()));}stack.push(NativeValue::Unit);
            }
            NativeInstr::TokenizerTokenCount => {let value=stack.pop().ok_or_else(||NativeError::InvalidProgram("TokenizerState operand missing".into()))?;let NativeValue::TokenizerState{tokens,..}=value else{return Err(NativeError::Type("tokenizer_token_count requires TokenizerState".into()));};stack.push(NativeValue::Int(tokens.len() as i64));}
            NativeInstr::TokenizerToken => {let index=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("tokenizer_token index must be non-negative".into()))?;let value=stack.pop().ok_or_else(||NativeError::InvalidProgram("TokenizerState operand missing".into()))?;let NativeValue::TokenizerState{tokens,..}=value else{return Err(NativeError::Type("tokenizer_token requires TokenizerState".into()));};stack.push(NativeValue::Int(*tokens.get(index).ok_or_else(||NativeError::Type("tokenizer_token index out of bounds".into()))?));}
            NativeInstr::MakeGradientTape => { let capacity=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("GradientTape capacity must be positive".into()))?; if capacity==0||capacity>4096{return Err(NativeError::ResourceLimit("GradientTape capacity must be in 1..=4096".into()));} stack.push(NativeValue::GradientTape{capacity,slots:Vec::new(),gradients:Vec::new(),backward_ready:false}); }
            NativeInstr::GradientInput{name} => { let value=pop_float(&mut stack)?; let tape=locals.get_mut(name).ok_or_else(||NativeError::Type("GradientTape binding not found".into()))?; let NativeValue::GradientTape{capacity,slots,gradients,backward_ready}=tape else{return Err(NativeError::Type("gradient_input requires GradientTape".into()));}; if slots.len()>=*capacity{return Err(NativeError::ResourceLimit("GradientTape slot capacity exhausted".into()));} slots.push(GradientSlot{value_bits:value.to_bits(),op:GradientOp::Input});gradients.clear();*backward_ready=false;stack.push(NativeValue::Int((slots.len()-1) as i64)); }
            NativeInstr::GradientAdd{name} | NativeInstr::GradientMul{name} => { let right=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("GradientTape slot id must be non-negative".into()))?; let left=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("GradientTape slot id must be non-negative".into()))?; let is_mul=matches!(instr,NativeInstr::GradientMul{..}); let tape=locals.get_mut(name).ok_or_else(||NativeError::Type("GradientTape binding not found".into()))?; let NativeValue::GradientTape{capacity,slots,gradients,backward_ready}=tape else{return Err(NativeError::Type("gradient operation requires GradientTape".into()));}; if left>=slots.len()||right>=slots.len(){return Err(NativeError::Type("GradientTape slot reference must exist before use".into()));} if slots.len()>=*capacity{return Err(NativeError::ResourceLimit("GradientTape slot capacity exhausted".into()));} let a=f32::from_bits(slots[left].value_bits);let b=f32::from_bits(slots[right].value_bits);let value=if is_mul{a*b}else{a+b};if !value.is_finite(){return Err(NativeError::Type("GradientTape intermediate must be finite".into()));} let op=if is_mul{GradientOp::Mul{left,right}}else{GradientOp::Add{left,right}};slots.push(GradientSlot{value_bits:value.to_bits(),op});gradients.clear();*backward_ready=false;stack.push(NativeValue::Int((slots.len()-1) as i64)); }
            NativeInstr::GradientBackward{name} => { let seed=pop_float(&mut stack)?;let output=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("GradientTape output slot must be non-negative".into()))?;let tape=locals.get_mut(name).ok_or_else(||NativeError::Type("GradientTape binding not found".into()))?;let NativeValue::GradientTape{slots,gradients,backward_ready,..}=tape else{return Err(NativeError::Type("gradient_backward requires GradientTape".into()));};if output>=slots.len(){return Err(NativeError::Type("GradientTape output slot does not exist".into()));}let mut grads=vec![0.0f32;slots.len()];grads[output]=seed;for index in (0..=output).rev(){let g=grads[index];match slots[index].op{GradientOp::Input=>{},GradientOp::Add{left,right}=>{grads[left]+=g;grads[right]+=g;},GradientOp::Mul{left,right}=>{grads[left]+=g*f32::from_bits(slots[right].value_bits);grads[right]+=g*f32::from_bits(slots[left].value_bits);}}if grads.iter().any(|v|!v.is_finite()){return Err(NativeError::Type("GradientTape gradient accumulation is non-finite".into()));}}*gradients=grads.into_iter().map(f32::to_bits).collect();*backward_ready=true;stack.push(NativeValue::Unit); }
            NativeInstr::GradientValue | NativeInstr::GradientGet => { let slot=usize::try_from(pop_int(&mut stack)?).map_err(|_|NativeError::Type("GradientTape slot id must be non-negative".into()))?;let tape=stack.pop().ok_or_else(||NativeError::InvalidProgram("GradientTape operand missing".into()))?;let NativeValue::GradientTape{slots,gradients,backward_ready,..}=tape else{return Err(NativeError::Type("gradient query requires GradientTape".into()));};if slot>=slots.len(){return Err(NativeError::Type("GradientTape slot does not exist".into()));}let bits=if matches!(instr,NativeInstr::GradientValue){slots[slot].value_bits}else{if !backward_ready{return Err(NativeError::Type("gradient_get requires gradient_backward first".into()));}*gradients.get(slot).ok_or_else(||NativeError::InvalidProgram("GradientTape gradient state malformed".into()))?};stack.push(NativeValue::Float(bits)); }
            NativeInstr::MakeMctsTree => {
                let branches = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS branch bound must be positive".into()))?;
                let nodes = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS node bound must be positive".into()))?;
                if nodes == 0 || nodes > crate::ai_mcts::MAX_MCTS_NODES || branches == 0 || branches > crate::ai_mcts::MAX_MCTS_BRANCHES { return Err(NativeError::ResourceLimit("MCTS bounds exceed supported limits".into())); }
                stack.push(NativeValue::List(vec![NativeValue::Int(nodes as i64), NativeValue::Int(branches as i64), NativeValue::List(Vec::new())]));
            }
            NativeInstr::MctsRoot { name } => {
                let state = u64::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS state id must be non-negative".into()))?;
                let tree = locals.get_mut(name).ok_or_else(|| NativeError::Type("MCTS tree binding not found".into()))?;
                mcts_root(tree, state)?; stack.push(NativeValue::Int(0));
            }
            NativeInstr::MctsExpand { name } => {
                let state = u64::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS state id must be non-negative".into()))?;
                let parent = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS parent index must be non-negative".into()))?;
                let tree = locals.get_mut(name).ok_or_else(|| NativeError::Type("MCTS tree binding not found".into()))?;
                let index = mcts_expand(tree, parent, state)?; stack.push(NativeValue::Int(index as i64));
            }
            NativeInstr::MctsChild => {
                let ordinal = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS child ordinal must be non-negative".into()))?;
                let parent = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS parent index must be non-negative".into()))?;
                let tree = stack.pop().ok_or_else(|| NativeError::Type("mcts_child missing tree".into()))?;
                stack.push(NativeValue::Int(mcts_child(&tree, parent, ordinal)? as i64));
            }
            NativeInstr::MctsUpdate { name } | NativeInstr::MctsBackpropagate { name } => {
                let reward = match stack.pop() { Some(NativeValue::Float(bits)) if f32::from_bits(bits).is_finite() && f32::from_bits(bits).abs() <= crate::ai_mcts::MAX_MCTS_REWARD_ABS as f32 => f64::from(f32::from_bits(bits)), _ => return Err(NativeError::Type("MCTS reward must be finite bounded F32".into())) };
                let node = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS node index must be non-negative".into()))?;
                let backprop = matches!(instr, NativeInstr::MctsBackpropagate { .. });
                let tree = locals.get_mut(name).ok_or_else(|| NativeError::Type("MCTS tree binding not found".into()))?;
                mcts_update(tree, node, reward, backprop)?; stack.push(NativeValue::Unit);
            }
            NativeInstr::MctsLen => {
                let tree = stack.pop().ok_or_else(|| NativeError::Type("mcts_len missing tree".into()))?;
                stack.push(NativeValue::Int(mcts_len(&tree)? as i64));
            }
            NativeInstr::MakeMctsTree => {
                let branches = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS branch bound must be positive".into()))?;
                let nodes = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS node bound must be positive".into()))?;
                if nodes == 0 || nodes > crate::ai_mcts::MAX_MCTS_NODES || branches == 0 || branches > crate::ai_mcts::MAX_MCTS_BRANCHES { return Err(NativeError::ResourceLimit("MCTS bounds exceed supported limits".into())); }
                stack.push(NativeValue::List(vec![NativeValue::Int(nodes as i64), NativeValue::Int(branches as i64), NativeValue::List(Vec::new())]));
            }
            NativeInstr::MctsRoot { name } => {
                let state = u64::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS state id must be non-negative".into()))?;
                let tree = locals.get_mut(name).ok_or_else(|| NativeError::Type("MCTS tree binding not found".into()))?;
                mcts_root(tree, state)?; stack.push(NativeValue::Int(0));
            }
            NativeInstr::MctsExpand { name } => {
                let state = u64::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS state id must be non-negative".into()))?;
                let parent = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS parent index must be non-negative".into()))?;
                let tree = locals.get_mut(name).ok_or_else(|| NativeError::Type("MCTS tree binding not found".into()))?;
                let index = mcts_expand(tree, parent, state)?; stack.push(NativeValue::Int(index as i64));
            }
            NativeInstr::MctsChild => {
                let ordinal = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS child ordinal must be non-negative".into()))?;
                let parent = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS parent index must be non-negative".into()))?;
                let tree = stack.pop().ok_or_else(|| NativeError::Type("mcts_child missing tree".into()))?;
                stack.push(NativeValue::Int(mcts_child(&tree, parent, ordinal)? as i64));
            }
            NativeInstr::MctsUpdate { name } | NativeInstr::MctsBackpropagate { name } => {
                let reward = match stack.pop() { Some(NativeValue::Float(bits)) if f32::from_bits(bits).is_finite() && f32::from_bits(bits).abs() <= crate::ai_mcts::MAX_MCTS_REWARD_ABS as f32 => f64::from(f32::from_bits(bits)), _ => return Err(NativeError::Type("MCTS reward must be finite bounded F32".into())) };
                let node = usize::try_from(pop_int(&mut stack)?).map_err(|_| NativeError::Type("MCTS node index must be non-negative".into()))?;
                let backprop = matches!(instr, NativeInstr::MctsBackpropagate { .. });
                let tree = locals.get_mut(name).ok_or_else(|| NativeError::Type("MCTS tree binding not found".into()))?;
                mcts_update(tree, node, reward, backprop)?; stack.push(NativeValue::Unit);
            }
            NativeInstr::MctsLen => {
                let tree = stack.pop().ok_or_else(|| NativeError::Type("mcts_len missing tree".into()))?;
                stack.push(NativeValue::Int(mcts_len(&tree)? as i64));
            }
            NativeInstr::MakeKvCache => {
                let capacity = match stack.pop() { Some(NativeValue::Int(value)) if value > 0 => usize::try_from(value).map_err(|_| NativeError::ResourceLimit("KVCache capacity overflow".into()))?, _ => return Err(NativeError::Type("kv_cache capacity must be a positive Int".into())) };
                let cache = KvCache::new(capacity).map_err(|error| NativeError::Type(format!("invalid KVCache capacity: {error:?}")))?;
                stack.push(NativeValue::KvCache { capacity: cache.capacity(), key_dim: None, value_dim: None, entries: Vec::new() });
            }
            NativeInstr::KvLen | NativeInstr::KvCapacity => {
                let value = stack.pop().ok_or_else(|| NativeError::InvalidProgram("KVCache query from empty stack".into()))?;
                let NativeValue::KvCache { capacity, entries, .. } = value else { return Err(NativeError::Type("KVCache query requires KvCache".into())); };
                stack.push(NativeValue::Int(if matches!(instr, NativeInstr::KvLen) { entries.len() as i64 } else { capacity as i64 }));
            }
            NativeInstr::KvAppend { name, evict_oldest } => {
                let value = stack.pop().ok_or_else(|| NativeError::InvalidProgram("KVCache value argument missing".into()))?;
                let key = stack.pop().ok_or_else(|| NativeError::InvalidProgram("KVCache key argument missing".into()))?;
                let to_bits = |value: NativeValue| -> Result<Vec<u32>, NativeError> {
                    let NativeValue::List(values) = value else { return Err(NativeError::Type("KVCache key and value must be numeric lists".into())); };
                    values.into_iter().map(|item| match item { NativeValue::Int(v) => Ok((v as f32).to_bits()), NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(bits), _ => Err(NativeError::Type("KVCache vectors must contain finite numeric values".into())) }).collect()
                };
                let key = to_bits(key)?; let value = to_bits(value)?;
                let Some(NativeValue::KvCache { capacity, key_dim, value_dim, entries }) = locals.get_mut(name) else { return Err(NativeError::Type("kv_append requires a KvCache local binding".into())); };
                let mut cache = KvCache::new(*capacity).map_err(|error| NativeError::Type(format!("invalid KVCache: {error:?}")))?;
                for (old_key, old_value) in entries.iter() { cache.append(old_key.iter().map(|bits| f32::from_bits(*bits)).collect(), old_value.iter().map(|bits| f32::from_bits(*bits)).collect(), CacheFullPolicy::Reject).map_err(|error| NativeError::Type(format!("invalid KVCache state: {error:?}")))?; }
                let policy = if *evict_oldest { CacheFullPolicy::EvictOldest } else { CacheFullPolicy::Reject };
                cache.append(key.iter().map(|bits| f32::from_bits(*bits)).collect(), value.iter().map(|bits| f32::from_bits(*bits)).collect(), policy).map_err(|error| NativeError::Type(format!("KVCache append rejected: {error:?}")))?;
                let retained = cache.window(0, cache.len()).map_err(|error| NativeError::Type(format!("KVCache window failed: {error:?}")))?;
                *key_dim = retained.first().map(|entry| entry.key.len()); *value_dim = retained.first().map(|entry| entry.value.len());
                *entries = retained.into_iter().map(|entry| (entry.key.into_iter().map(f32::to_bits).collect(), entry.value.into_iter().map(f32::to_bits).collect())).collect();
                stack.push(NativeValue::Unit);
            }
            NativeInstr::MakeContextWindow => {
                let tokens = match stack.pop() { Some(NativeValue::Int(v)) if v > 0 => usize::try_from(v).map_err(|_| NativeError::ResourceLimit("context token budget overflow".into()))?, _ => return Err(NativeError::Type("context token budget must be a positive Int".into())) };
                let bytes = match stack.pop() { Some(NativeValue::Int(v)) if v > 0 => usize::try_from(v).map_err(|_| NativeError::ResourceLimit("context byte budget overflow".into()))?, _ => return Err(NativeError::Type("context byte budget must be a positive Int".into())) };
                let segments = match stack.pop() { Some(NativeValue::Int(v)) if v > 0 => usize::try_from(v).map_err(|_| NativeError::ResourceLimit("context segment budget overflow".into()))?, _ => return Err(NativeError::Type("context segment budget must be a positive Int".into())) };
                ContextWindow::new(ContextBudget { max_segments: segments, max_bytes: bytes, max_tokens: tokens }).map_err(|error| NativeError::Type(format!("invalid ContextWindow budget: {error:?}")))?;
                stack.push(NativeValue::ContextWindow { max_segments: segments, max_bytes: bytes, max_tokens: tokens, segments: Vec::new(), bytes: 0, tokens: 0 });
            }
            NativeInstr::ContextLen | NativeInstr::ContextBytes | NativeInstr::ContextTokens => {
                let value = stack.pop().ok_or_else(|| NativeError::InvalidProgram("ContextWindow query from empty stack".into()))?;
                let NativeValue::ContextWindow { segments, bytes, tokens, .. } = value else { return Err(NativeError::Type("ContextWindow query requires ContextWindow".into())); };
                let result = match instr { NativeInstr::ContextLen => segments.len(), NativeInstr::ContextBytes => bytes, _ => tokens };
                stack.push(NativeValue::Int(result as i64));
            }
            NativeInstr::ContextAppend { name } => {
                let token_count = match stack.pop() { Some(NativeValue::Int(v)) if v >= 0 => usize::try_from(v).map_err(|_| NativeError::ResourceLimit("context token count overflow".into()))?, _ => return Err(NativeError::Type("context_append token count must be a non-negative Int".into())) };
                let payload = match stack.pop() { Some(NativeValue::String(v)) => v.into_bytes(), _ => return Err(NativeError::Type("context_append payload must be a String".into())) };
                let modality = match stack.pop() { Some(NativeValue::String(v)) => v, _ => return Err(NativeError::Type("context_append modality must be a String literal".into())) };
                let modality = match modality.as_str() { "text" => SegmentModality::Text, "image" => SegmentModality::Image, "audio" => SegmentModality::Audio, "video" => SegmentModality::Video, "structured" => SegmentModality::Structured, _ => return Err(NativeError::Type("context modality must be text, image, audio, video, or structured".into())) };
                let Some(NativeValue::ContextWindow { max_segments, max_bytes, max_tokens, segments, bytes, tokens }) = locals.get_mut(name) else { return Err(NativeError::Type("context_append requires a ContextWindow local binding".into())); };
                let mut window = ContextWindow::new(ContextBudget { max_segments: *max_segments, max_bytes: *max_bytes, max_tokens: *max_tokens }).map_err(|error| NativeError::Type(format!("invalid ContextWindow: {error:?}")))?;
                for (old_modality, old_payload, old_tokens) in segments.iter() {
                    let old_modality = match old_modality.as_str() { "text" => SegmentModality::Text, "image" => SegmentModality::Image, "audio" => SegmentModality::Audio, "video" => SegmentModality::Video, "structured" => SegmentModality::Structured, _ => return Err(NativeError::InvalidProgram("invalid stored context modality".into())) };
                    window.append(ContextSegment { modality: old_modality, payload: old_payload.clone(), token_count: *old_tokens }).map_err(|error| NativeError::Type(format!("invalid ContextWindow state: {error:?}")))?;
                }
                window.append(ContextSegment { modality, payload, token_count }).map_err(|error| NativeError::Type(format!("ContextWindow append rejected: {error:?}")))?;
                *segments = window.segments().iter().map(|segment| (match segment.modality { SegmentModality::Text => "text", SegmentModality::Image => "image", SegmentModality::Audio => "audio", SegmentModality::Video => "video", SegmentModality::Structured => "structured" }.to_string(), segment.payload.clone(), segment.token_count)).collect();
                *bytes = window.bytes(); *tokens = window.tokens();
                stack.push(NativeValue::Unit);
            }
            NativeInstr::MakeTensor => {
                let values = match stack.pop() { Some(NativeValue::List(values)) => values, _ => return Err(NativeError::Type("tensor expects a List<Int> value payload".into())) };
                let shape = match stack.pop() { Some(NativeValue::List(shape)) => shape, _ => return Err(NativeError::Type("tensor expects a List<Int> shape".into())) };
                let shape = shape.into_iter().map(|value| match value {
                    NativeValue::Int(dimension) if dimension > 0 => usize::try_from(dimension).map_err(|_| NativeError::ResourceLimit("tensor dimension overflow".into())),
                    _ => Err(NativeError::Type("tensor dimensions must be positive Int values".into())),
                }).collect::<Result<Vec<_>, _>>()?;
                let values = values.into_iter().map(|value| match value {
                    NativeValue::Int(number) => { let converted = number as f32; if converted.is_finite() { Ok(converted.to_bits()) } else { Err(NativeError::Type("tensor value is not finite in f32".into())) } },
                    NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(bits),
                    _ => Err(NativeError::Type("tensor values must be finite numeric values".into())),
                }).collect::<Result<Vec<_>, _>>()?;
                validate_tensor_parts(&shape, &values, state.limits.max_collection_items)?;
                stack.push(NativeValue::Tensor { shape, values });
            }
            NativeInstr::MakeEmbedding => {
                let values = match stack.pop() { Some(NativeValue::List(values)) => values, _ => return Err(NativeError::Type("embedding expects a List<Int> payload".into())) };
                let domain = match stack.pop() { Some(NativeValue::String(domain)) => domain, _ => return Err(NativeError::Type("embedding domain must be a String".into())) };
                let values = values.into_iter().map(|value| match value {
                    NativeValue::Int(number) => { let converted = number as f32; if converted.is_finite() { Ok(converted.to_bits()) } else { Err(NativeError::Type("embedding coordinate is not finite in f32".into())) } },
                    NativeValue::Float(bits) if f32::from_bits(bits).is_finite() => Ok(bits),
                    _ => Err(NativeError::Type("embedding coordinates must be finite numeric values".into())),
                }).collect::<Result<Vec<_>, _>>()?;
                validate_embedding_parts(&domain, &values, state.limits.max_collection_items)?;
                stack.push(NativeValue::Embedding { domain, values });
            }
            NativeInstr::PushQuantizedTensor { shape, scale_bits, zero_point, values } => {
                validate_quantized_tensor_parts(shape, *scale_bits, values, state.limits.max_collection_items)?;
                stack.push(NativeValue::QuantizedTensor { shape: shape.clone(), scale_bits: *scale_bits, zero_point: *zero_point, values: values.clone() });
            }
            NativeInstr::MakeQuantizedTensor => {
                let values = match stack.pop() { Some(NativeValue::List(values)) => values, _ => return Err(NativeError::Type("quantized_tensor values must be a List<Int>".into())) };
                let zero_point = match stack.pop() { Some(NativeValue::Int(value)) => i8::try_from(value).map_err(|_| NativeError::Type("quantized tensor zero point must be in [-128, 127]".into()))?, _ => return Err(NativeError::Type("quantized tensor zero point must be an Int".into())) };
                let scale_bits = match stack.pop() { Some(NativeValue::Float(bits)) if f32::from_bits(bits).is_finite() && f32::from_bits(bits) > 0.0 => bits, Some(NativeValue::Int(value)) if value > 0 && (value as f32).is_finite() => (value as f32).to_bits(), _ => return Err(NativeError::Type("quantization scale must be finite and positive".into())) };
                let shape = match stack.pop() { Some(NativeValue::List(shape)) => shape, _ => return Err(NativeError::Type("quantized_tensor shape must be a List<Int>".into())) };
                let shape = shape.into_iter().map(|value| match value { NativeValue::Int(dimension) if dimension > 0 => usize::try_from(dimension).map_err(|_| NativeError::ResourceLimit("quantized tensor dimension overflow".into())), _ => Err(NativeError::Type("quantized tensor dimensions must be positive Int values".into())) }).collect::<Result<Vec<_>, _>>()?;
                let values = values.into_iter().map(|value| match value { NativeValue::Int(number) => i8::try_from(number).map_err(|_| NativeError::Type("quantized tensor values must be in [-128, 127]".into())), _ => Err(NativeError::Type("quantized tensor values must be Int values".into())) }).collect::<Result<Vec<_>, _>>()?;
                validate_quantized_tensor_parts(&shape, scale_bits, &values, state.limits.max_collection_items)?;
                stack.push(NativeValue::QuantizedTensor { shape, scale_bits, zero_point, values });
            }
            NativeInstr::DequantizeTensor => {
                let value = stack.pop().ok_or_else(|| NativeError::Type("expected QuantizedTensor value".into()))?;
                match value {
                    NativeValue::QuantizedTensor { shape, scale_bits, zero_point, values } => {
                        validate_quantized_tensor_parts(&shape, scale_bits, &values, state.limits.max_collection_items)?;
                        let scale = f32::from_bits(scale_bits);
                        let values = values.iter().map(|q| ((*q as i16 - zero_point as i16) as f32 * scale).to_bits()).collect::<Vec<_>>();
                        validate_tensor_parts(&shape, &values, state.limits.max_collection_items)?;
                        stack.push(NativeValue::Tensor { shape, values });
                    }
                    _ => return Err(NativeError::Type("expected QuantizedTensor value".into())),
                }
            }
            NativeInstr::PushEmbedding { domain, values } => {
                validate_embedding_parts(domain, values, state.limits.max_collection_items)?;
                stack.push(NativeValue::Embedding { domain: domain.clone(), values: values.clone() });
            }
            NativeInstr::EmbeddingDot => {
                let right = pop_embedding(&mut stack, state.limits.max_collection_items)?;
                let left = pop_embedding(&mut stack, state.limits.max_collection_items)?;
                if left.0 != right.0 {
                    return Err(NativeError::Type("embedding dot product requires matching domains".into()));
                }
                if left.1.len() != right.1.len() {
                    return Err(NativeError::Type("embedding dot product requires matching dimensions".into()));
                }
                let mut sum = 0.0f32;
                for (a, b) in left.1.iter().zip(&right.1) {
                    sum += f32::from_bits(*a) * f32::from_bits(*b);
                }
                if !sum.is_finite() {
                    return Err(NativeError::Type("embedding dot product produced non-finite value".into()));
                }
                stack.push(NativeValue::Float(sum.to_bits()));
            }
            NativeInstr::PushUnit => stack.push(NativeValue::Unit),
            NativeInstr::PushString(value) => {
                if value.len() > state.limits.max_string_bytes {
                    return Err(NativeError::ResourceLimit("string exceeds byte limit".into()));
                }
                stack.push(NativeValue::String(value.clone()));
            }
            NativeInstr::PushList(len) => {
                if *len > state.limits.max_collection_items {
                    return Err(NativeError::ResourceLimit("collection exceeds item limit".into()));
                }
                if stack.len() < *len {
                    return Err(NativeError::InvalidProgram(
                        "list has insufficient stack values".into(),
                    ));
                }
                let start = stack.len() - *len;
                let value = NativeValue::List(stack.drain(start..).collect());
                validate_value(&value, state.limits)?;
                stack.push(value);
            }
            NativeInstr::Index => {
                let index = pop_int(&mut stack)?;
                let collection = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("index from empty stack".into()))?;
                let index = usize::try_from(index)
                    .map_err(|_| NativeError::Type("negative index".into()))?;
                match collection {
                    NativeValue::List(values) => stack.push(
                        values
                            .get(index)
                            .cloned()
                            .ok_or_else(|| NativeError::Type("list index out of bounds".into()))?,
                    ),
                    NativeValue::String(value) => stack.push(NativeValue::Int(i64::from(
                        value.as_bytes().get(index).copied().ok_or_else(|| {
                            NativeError::Type("string index out of bounds".into())
                        })?,
                    ))),
                    _ => {
                        return Err(NativeError::Type(
                            "indexing requires a list or String".into(),
                        ));
                    }
                }
            }
            NativeInstr::Len => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("len from empty stack".into()))?;
                let length = match value {
                    NativeValue::String(value) => value.len(),
                    NativeValue::List(values) => values.len(),
                    _ => return Err(NativeError::Type("len requires String or List".into())),
                };
                stack.push(NativeValue::Int(length as i64));
            }
            NativeInstr::Append(name) => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("push value missing".into()))?;
                {
                    let Some(NativeValue::List(items)) = locals.get_mut(name) else {
                        return Err(NativeError::Type("push requires a List binding".into()));
                    };
                    if items.len() >= state.limits.max_collection_items {
                        return Err(NativeError::ResourceLimit("collection exceeds item limit".into()));
                    }
                    items.push(value);
                }
                validate_value(locals.get(name).expect("list binding retained"), state.limits)?;
                stack.push(NativeValue::Unit);
            }
            NativeInstr::Chr => {
                let value = pop_int(&mut stack)?;
                let byte = u8::try_from(value)
                    .map_err(|_| NativeError::Type("chr requires a byte in 0..=255".into()))?;
                stack.push(NativeValue::String(char::from(byte).to_string()));
            }
            NativeInstr::MakeOk => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("ok value missing".into()))?;
                let result = NativeValue::ResultOk(Box::new(value));
                validate_value(&result, state.limits)?;
                stack.push(result);
            }
            NativeInstr::MakeErr => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("err value missing".into()))?;
                let result = NativeValue::ResultErr(Box::new(value));
                validate_value(&result, state.limits)?;
                stack.push(result);
            }
            NativeInstr::Unwrap => {
                match stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("unwrap value missing".into()))?
                {
                    NativeValue::ResultOk(value) => stack.push(*value),
                    NativeValue::ResultErr(_) => {
                        return Err(NativeError::Type("unwrap on Err".into()));
                    }
                    _ => return Err(NativeError::Type("unwrap requires Result".into())),
                }
            }
            NativeInstr::Load(name) => {
                stack.push(locals.get(name).cloned().ok_or_else(|| {
                    NativeError::InvalidProgram(format!("unknown local '{name}'"))
                })?);
            }
            NativeInstr::Store(name) => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("store from empty stack".into()))?;
                validate_value(&value, state.limits)?;
                if !locals.contains_key(name) && locals.len() >= state.limits.max_locals {
                    return Err(NativeError::ResourceLimit("local binding limit exceeded".into()));
                }
                locals.insert(name.clone(), value);
            }
            NativeInstr::AddAssign(name) => {
                let right = stack.pop().ok_or_else(|| NativeError::InvalidProgram("add assignment value missing".into()))?;
                let left = locals.remove(name).ok_or_else(|| NativeError::InvalidProgram(format!("unknown local '{name}'")))?;
                let value = add_values(left, right, state.limits)?;
                if matches!(&value, NativeValue::String(s) if s.len() > state.limits.max_string_bytes) {
                    return Err(NativeError::ResourceLimit("string exceeds byte limit".into()));
                }
                validate_value(&value, state.limits)?;
                locals.insert(name.clone(), value);
            }
            NativeInstr::StoreIndex(name) => {
                let value = stack.pop().ok_or_else(|| {
                    NativeError::InvalidProgram("indexed store value missing".into())
                })?;
                let index = pop_int(&mut stack)?;
                let collection = stack.pop().ok_or_else(|| {
                    NativeError::InvalidProgram("indexed store collection missing".into())
                })?;
                let NativeValue::List(mut items) = collection else {
                    return Err(NativeError::Type(
                        "indexed assignment requires a list".into(),
                    ));
                };
                let index = usize::try_from(index)
                    .map_err(|_| NativeError::Type("negative list index".into()))?;
                *items
                    .get_mut(index)
                    .ok_or_else(|| NativeError::Type("list index out of bounds".into()))? = value;
                let updated = NativeValue::List(items);
                validate_value(&updated, state.limits)?;
                locals.insert(name.clone(), updated);
            }
            NativeInstr::Add => {
                let right = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let left = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let value = add_values(left, right, state.limits)?;
                validate_value(&value, state.limits)?;
                stack.push(value);
            }
            NativeInstr::Sub | NativeInstr::Mul | NativeInstr::Div | NativeInstr::Mod => {
                let right = pop_int(&mut stack)?;
                let left = pop_int(&mut stack)?;
                let value = match instr {
                    NativeInstr::Sub => left.checked_sub(right)
                        .ok_or_else(|| NativeError::Type("integer overflow in subtraction".into()))?,
                    NativeInstr::Mul => left.checked_mul(right)
                        .ok_or_else(|| NativeError::Type("integer overflow in multiplication".into()))?,
                    NativeInstr::Div => {
                        if right == 0 {
                            return Err(NativeError::Type("division by zero".into()));
                        }
                        left.checked_div(right)
                            .ok_or_else(|| NativeError::Type("integer overflow in division".into()))?
                    }
                    NativeInstr::Mod => {
                        if right == 0 {
                            return Err(NativeError::Type("modulo by zero".into()));
                        }
                        left.checked_rem(right)
                            .ok_or_else(|| NativeError::Type("integer overflow in modulo".into()))?
                    }
                    _ => unreachable!(),
                };
                stack.push(NativeValue::Int(value));
            }
            NativeInstr::Equal
            | NativeInstr::NotEqual
            | NativeInstr::Less
            | NativeInstr::LessEqual
            | NativeInstr::Greater
            | NativeInstr::GreaterEqual => {
                let right = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let left = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty stack".into()))?;
                let result = match instr {
                    NativeInstr::Equal => left == right,
                    NativeInstr::NotEqual => left != right,
                    NativeInstr::Less => compare_ints(&left, &right, |a, b| a < b)?,
                    NativeInstr::LessEqual => compare_ints(&left, &right, |a, b| a <= b)?,
                    NativeInstr::Greater => compare_ints(&left, &right, |a, b| a > b)?,
                    NativeInstr::GreaterEqual => compare_ints(&left, &right, |a, b| a >= b)?,
                    _ => unreachable!(),
                };
                stack.push(NativeValue::Bool(result));
            }
            NativeInstr::JumpIfFalse(target) => {
                let value = stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("empty condition stack".into()))?;
                if value != NativeValue::Bool(true) {
                    pc = *target;
                }
            }
            NativeInstr::Jump(target) => pc = *target,
            NativeInstr::Return => return Ok(stack.pop().unwrap_or(NativeValue::Unit)),
            NativeInstr::Pop => {
                stack
                    .pop()
                    .ok_or_else(|| NativeError::InvalidProgram("pop from empty stack".into()))?;
            }
        }
        if check_values_after {
            check_frame_value_budget(&locals, &stack, state.limits)?;
        }
    }
    Err(NativeError::InvalidProgram(
        "program terminated without return".into(),
    ))
}

fn compare_ints(
    left: &NativeValue,
    right: &NativeValue,
    predicate: impl FnOnce(i64, i64) -> bool,
) -> Result<bool, NativeError> {
    let (NativeValue::Int(left), NativeValue::Int(right)) = (left, right) else {
        return Err(NativeError::Type(
            "ordering operators require Int operands".into(),
        ));
    };
    Ok(predicate(*left, *right))
}

fn add_values(
    left: NativeValue,
    right: NativeValue,
    limits: ExecutionLimits,
) -> Result<NativeValue, NativeError> {
    match (left, right) {
        (NativeValue::Int(left), NativeValue::Int(right)) => Ok(NativeValue::Int(
            left.checked_add(right)
                .ok_or_else(|| NativeError::Type("integer overflow in addition".into()))?,
        )),
        (NativeValue::String(left), NativeValue::String(right)) => {
            // Check the byte limit before allocating the combined buffer. String::len()
            // reports UTF-8 bytes, matching the VM's configured resource limits.
            let combined_len = left.len().checked_add(right.len()).ok_or_else(|| {
                NativeError::ResourceLimit("string concatenation length overflow".into())
            })?;
            if combined_len > limits.max_string_bytes {
                return Err(NativeError::ResourceLimit(format!(
                    "string concatenation exceeds byte limit ({} > {})",
                    combined_len, limits.max_string_bytes
                )));
            }
            let mut combined = String::with_capacity(combined_len);
            combined.push_str(&left);
            combined.push_str(&right);
            Ok(NativeValue::String(combined))
        }
        _ => Err(NativeError::Type(
            "String + String or Int + Int required".into(),
        )),
    }
}


/// Hard limits for the serialized/in-memory executable representation.
/// These are intentionally separate from ExecutionLimits, which bound runtime state.
pub const MAX_PROGRAM_FUNCTIONS: usize = 16_384;
pub const MAX_FUNCTION_INSTRUCTIONS: usize = 250_000;
pub const MAX_PROGRAM_INSTRUCTIONS: usize = 1_000_000;
pub const MAX_PROGRAM_EMBEDDED_STRING_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;

/// Validate an executable before cloning it into the shared runtime state.
/// The byte budget covers function names, parameter names, and string operands;
/// it is not a claim about total heap usage or allocator overhead.
fn validate_native_program(program: &NativeProgram) -> Result<(), NativeError> {
    if program.functions.len() > MAX_PROGRAM_FUNCTIONS {
        return Err(NativeError::ResourceLimit(format!(
            "program function limit exceeded (limit {MAX_PROGRAM_FUNCTIONS})"
        )));
    }
    let mut total_instructions = 0usize;
    let mut embedded_bytes = 0usize;
    let mut add_bytes = |value: &str| -> Result<(), NativeError> {
        embedded_bytes = embedded_bytes.checked_add(value.len()).ok_or_else(|| {
            NativeError::ResourceLimit("program embedded string byte count overflow".into())
        })?;
        if embedded_bytes > MAX_PROGRAM_EMBEDDED_STRING_BYTES {
            return Err(NativeError::ResourceLimit(format!(
                "program embedded string byte limit exceeded (limit {MAX_PROGRAM_EMBEDDED_STRING_BYTES})"
            )));
        }
        Ok(())
    };
    for (name, function) in &program.functions {
        add_bytes(name)?;
        if function.code.len() > MAX_FUNCTION_INSTRUCTIONS {
            return Err(NativeError::ResourceLimit(format!(
                "function instruction limit exceeded (limit {MAX_FUNCTION_INSTRUCTIONS})"
            )));
        }
        total_instructions = total_instructions.checked_add(function.code.len()).ok_or_else(|| {
            NativeError::ResourceLimit("program instruction count overflow".into())
        })?;
        if total_instructions > MAX_PROGRAM_INSTRUCTIONS {
            return Err(NativeError::ResourceLimit(format!(
                "program instruction limit exceeded (limit {MAX_PROGRAM_INSTRUCTIONS})"
            )));
        }
        for param in &function.params { add_bytes(param)?; }
        for instr in &function.code {
            match instr {
                NativeInstr::PushProbability(bits) => { validate_probability_bits(*bits)?; }
                NativeInstr::PushTensor { shape, values } => { validate_tensor_parts(shape, values, MAX_TENSOR_ELEMENTS)?; }
                NativeInstr::PushQuantizedTensor { shape, scale_bits, values, .. } => { validate_quantized_tensor_parts(shape, *scale_bits, values, MAX_TENSOR_ELEMENTS)?; }
                NativeInstr::PushEmbedding { domain, values } => {
                    validate_embedding_parts(domain, values, MAX_TENSOR_ELEMENTS)?;
                    add_bytes(domain)?;
                }
                NativeInstr::MctsRoot { name } | NativeInstr::MctsExpand { name } | NativeInstr::MctsUpdate { name } | NativeInstr::MctsBackpropagate { name } => { if name.trim().is_empty() { return Err(NativeError::InvalidProgram("MCTS binding name must not be empty".into())); } add_bytes(name)?; },
                NativeInstr::ActivationStore { name } => { if name.trim().is_empty(){return Err(NativeError::InvalidProgram("ActivationCache binding name must not be empty".into()));} add_bytes(name)?; },
                NativeInstr::GradientInput { name } | NativeInstr::GradientAdd { name } | NativeInstr::GradientMul { name } | NativeInstr::GradientBackward { name } => { if name.trim().is_empty() { return Err(NativeError::InvalidProgram("GradientTape binding name must not be empty".into())); } add_bytes(name)?; },
                NativeInstr::VectorIndexInsert { name } => {
                    if name.trim().is_empty() {
                        return Err(NativeError::InvalidProgram("VectorIndexInsert binding name must not be empty".into()));
                    }
                    add_bytes(name)?;
                },
                NativeInstr::PushString(s) | NativeInstr::Append(s) |
                NativeInstr::Load(s) | NativeInstr::Store(s) |
                NativeInstr::AddAssign(s) | NativeInstr::StoreIndex(s) |
                NativeInstr::Join { name: s } | NativeInstr::Cancel { name: s } => add_bytes(s)?,
                NativeInstr::Call { callee, .. } => add_bytes(callee)?,
                NativeInstr::KvAppend { name, .. } | NativeInstr::ContextAppend { name } => add_bytes(name)?,
                NativeInstr::Spawn { name, callee, .. } => { add_bytes(name)?; add_bytes(callee)?; }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Stable, dependency-free serialization for bootstrap artifacts.
///
/// The format is deliberately textual and line-oriented so an existing
/// Ardisa VM can consume a compiler artifact without invoking the Rust compiler.
pub const ARTIFACT_MAGIC: &str = "ARDISA-EXEC-V1";

pub fn encode_program(program: &NativeProgram) -> String {
    let mut out = String::from(ARTIFACT_MAGIC);
    out.push('\n');
    for (name, function) in &program.functions {
        out.push_str("FN|");
        out.push_str(&escape_artifact(name));
        out.push('|');
        out.push_str(&function.params.len().to_string());
        out.push('|');
        out.push_str(&function.params.iter().map(|p| escape_artifact(p)).collect::<Vec<_>>().join(","));
        out.push('\n');
        for instr in &function.code {
            out.push_str("I|");
            out.push_str(&encode_instr(instr));
            out.push('\n');
        }
        out.push_str("END\n");
    }
    out
}

pub fn decode_program(input: &str) -> Result<NativeProgram, NativeError> {
    if input.len() > MAX_ARTIFACT_BYTES {
        return Err(NativeError::ResourceLimit(format!(
            "executable artifact byte limit exceeded (limit {MAX_ARTIFACT_BYTES})"
        )));
    }
    let mut lines = input.lines();
    if lines.next() != Some(ARTIFACT_MAGIC) {
        return Err(NativeError::InvalidProgram("invalid Ardisa executable magic".into()));
    }
    let mut functions = BTreeMap::new();
    let mut current: Option<(String, Vec<String>, Vec<NativeInstr>)> = None;
    for line in lines {
        if let Some(rest) = line.strip_prefix("FN|") {
            if current.is_some() {
                return Err(NativeError::InvalidProgram("nested function in executable".into()));
            }
            let mut parts = rest.split('|');
            let name = unescape_artifact(parts.next().unwrap_or(""))?;
            let _count = parts.next().unwrap_or("0").parse::<usize>()
                .map_err(|_| NativeError::InvalidProgram("invalid parameter count".into()))?;
            let params = if let Some(raw) = parts.next() {
                if raw.is_empty() { Vec::new() } else {
                    raw.split(',').map(unescape_artifact).collect::<Result<Vec<_>, _>>()?
                }
            } else { Vec::new() };
            current = Some((name, params, Vec::new()));
        } else if line == "END" {
            let (name, params, code) = current.take()
                .ok_or_else(|| NativeError::InvalidProgram("function terminator without function".into()))?;
            functions.insert(name, NativeFunction { params, code });
        } else if let Some(rest) = line.strip_prefix("I|") {
            let (_, _, code) = current.as_mut()
                .ok_or_else(|| NativeError::InvalidProgram("instruction outside function".into()))?;
            code.push(decode_instr(rest)?);
        } else if !line.is_empty() {
            return Err(NativeError::InvalidProgram("unknown executable record".into()));
        }
    }
    if current.is_some() {
        return Err(NativeError::InvalidProgram("unterminated executable function".into()));
    }
    let program = NativeProgram { functions };
    validate_native_program(&program)?;
    Ok(program)
}

fn escape_artifact(value: &str) -> String {
    value.replace('\\', "\\\\").replace('|', r"\p").replace(',', r"\c").replace('\n', r"\n")
}

fn unescape_artifact(value: &str) -> Result<String, NativeError> {
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            out.push(match ch { 'p' => '|', 'c' => ',', 'n' => '\n', '\\' => '\\', other => other });
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else {
            out.push(ch);
        }
    }
    if escaped { return Err(NativeError::InvalidProgram("trailing artifact escape".into())); }
    Ok(out)
}

fn encode_instr(instr: &NativeInstr) -> String {
    match instr {
        NativeInstr::PushInt(v) => format!("PushInt:{v}"),
        NativeInstr::PushFloat(bits) => format!("PushFloat:{bits}"),
        NativeInstr::PushBool(v) => format!("PushBool:{v}"),
        NativeInstr::PushProbability(bits) => format!("PushProbability:{bits}"),
        NativeInstr::ProbabilityAnd => "ProbabilityAnd".into(),
        NativeInstr::ProbabilityOr => "ProbabilityOr".into(),
        NativeInstr::ProbabilityNot => "ProbabilityNot".into(),
        NativeInstr::ProbabilityMeets => "ProbabilityMeets".into(),
        NativeInstr::PushTensor { shape, values } => format!("PushTensor:{}/{}", shape.iter().map(usize::to_string).collect::<Vec<_>>().join(","), values.iter().map(u32::to_string).collect::<Vec<_>>().join(",")),
        NativeInstr::TensorAdd => "TensorAdd".into(),
        NativeInstr::TensorMatMul => "TensorMatMul".into(),
        NativeInstr::MakeActivationCache=>"MakeActivationCache".into(), NativeInstr::ActivationStore{name}=>format!("ActivationStore:{}",escape_artifact(name)), NativeInstr::ActivationRestore=>"ActivationRestore".into(), NativeInstr::ActivationCacheLen=>"ActivationCacheLen".into(),
        NativeInstr::MakeGradientTape => "MakeGradientTape".into(),
        NativeInstr::MakeTokenizerState => "MakeTokenizerState".into(), NativeInstr::TokenizerFeed{name}=>format!("TokenizerFeed:{}",escape_artifact(name)), NativeInstr::TokenizerFlush{name}=>format!("TokenizerFlush:{}",escape_artifact(name)), NativeInstr::TokenizerTokenCount=>"TokenizerTokenCount".into(), NativeInstr::TokenizerToken=>"TokenizerToken".into(), NativeInstr::GradientInput{name}=>format!("GradientInput:{}",escape_artifact(name)), NativeInstr::GradientAdd{name}=>format!("GradientAdd:{}",escape_artifact(name)), NativeInstr::GradientMul{name}=>format!("GradientMul:{}",escape_artifact(name)), NativeInstr::GradientBackward{name}=>format!("GradientBackward:{}",escape_artifact(name)), NativeInstr::GradientValue=>"GradientValue".into(), NativeInstr::GradientGet=>"GradientGet".into(),
        NativeInstr::MakeAttentionBlock => "MakeAttentionBlock".into(),
        NativeInstr::AttentionBlockApply => "AttentionBlockApply".into(),
        NativeInstr::AttentionBlockCached => "AttentionBlockCached".into(),
        NativeInstr::AttentionBlock => "AttentionBlock".into(),
        NativeInstr::AttentionCached => "AttentionCached".into(),
        NativeInstr::MakeLoRAAdapter => "MakeLoRAAdapter".into(),
        NativeInstr::LoRAAdapterApply => "LoRAAdapterApply".into(),
        NativeInstr::LoRAApply => "LoRAApply".into(),
        NativeInstr::MakeTensor => "MakeTensor".into(),
        NativeInstr::MakeEmbedding => "MakeEmbedding".into(),
        NativeInstr::MakeQuantizedTensor => "MakeQuantizedTensor".into(),
        NativeInstr::PushQuantizedTensor { shape, scale_bits, zero_point, values } => format!("PushQuantizedTensor:{}/{}/{}/{}", shape.iter().map(usize::to_string).collect::<Vec<_>>().join(","), scale_bits, zero_point, values.iter().map(i8::to_string).collect::<Vec<_>>().join(",")),
        NativeInstr::DequantizeTensor => "DequantizeTensor".into(),
        NativeInstr::PushEmbedding { domain, values } => format!("PushEmbedding:{}/{}", encode_hex(domain.as_bytes()), values.iter().map(u32::to_string).collect::<Vec<_>>().join(",")),
        NativeInstr::EmbeddingDot => "EmbeddingDot".into(),
        NativeInstr::MakeMoERouter => "MakeMoERouter".into(),
        NativeInstr::MoERoute => "MoERoute".into(),
        NativeInstr::RouteTopK => "RouteTopK".into(),
        NativeInstr::MakeVectorIndex => "MakeVectorIndex".into(),
        NativeInstr::VectorIndexInsert { name } => format!("VectorIndexInsert:{}", escape_artifact(name)),
        NativeInstr::VectorIndexSearch => "VectorIndexSearch".into(),
        NativeInstr::MakeMctsTree => "MakeMctsTree".into(),
        NativeInstr::MctsRoot { name } => format!("MctsRoot:{}", escape_artifact(name)),
        NativeInstr::MctsExpand { name } => format!("MctsExpand:{}", escape_artifact(name)),
        NativeInstr::MctsChild => "MctsChild".into(),
        NativeInstr::MctsUpdate { name } => format!("MctsUpdate:{}", escape_artifact(name)),
        NativeInstr::MctsBackpropagate { name } => format!("MctsBackpropagate:{}", escape_artifact(name)),
        NativeInstr::MctsLen => "MctsLen".into(),
        NativeInstr::MakeKvCache => "MakeKvCache".into(), NativeInstr::KvLen => "KvLen".into(), NativeInstr::KvCapacity => "KvCapacity".into(),
        NativeInstr::KvAppend { name, evict_oldest } => format!("KvAppend:{},{}", escape_artifact(name), evict_oldest),
        NativeInstr::MakeContextWindow => "MakeContextWindow".into(), NativeInstr::ContextLen => "ContextLen".into(), NativeInstr::ContextBytes => "ContextBytes".into(), NativeInstr::ContextTokens => "ContextTokens".into(),
        NativeInstr::ContextAppend { name } => format!("ContextAppend:{}", escape_artifact(name)),
        NativeInstr::PushUnit => "PushUnit".into(),
        NativeInstr::PushString(v) => format!("PushString:{}", escape_artifact(v)),
        NativeInstr::PushList(v) => format!("PushList:{v}"),
        NativeInstr::Index => "Index".into(), NativeInstr::Len => "Len".into(),
        NativeInstr::Append(v) => format!("Append:{}", escape_artifact(v)),
        NativeInstr::MakeOk => "MakeOk".into(), NativeInstr::Chr => "Chr".into(),
        NativeInstr::MakeErr => "MakeErr".into(), NativeInstr::Unwrap => "Unwrap".into(),
        NativeInstr::Load(v) => format!("Load:{}", escape_artifact(v)),
        NativeInstr::Store(v) => format!("Store:{}", escape_artifact(v)),
        NativeInstr::AddAssign(v) => format!("AddAssign:{}", escape_artifact(v)),
        NativeInstr::StoreIndex(v) => format!("StoreIndex:{}", escape_artifact(v)),
        NativeInstr::Add => "Add".into(), NativeInstr::Sub => "Sub".into(),
        NativeInstr::Mul => "Mul".into(), NativeInstr::Div => "Div".into(),
        NativeInstr::Mod => "Mod".into(), NativeInstr::Equal => "Equal".into(),
        NativeInstr::NotEqual => "NotEqual".into(), NativeInstr::Less => "Less".into(),
        NativeInstr::LessEqual => "LessEqual".into(), NativeInstr::Greater => "Greater".into(),
        NativeInstr::GreaterEqual => "GreaterEqual".into(),
        NativeInstr::JumpIfFalse(v) => format!("JumpIfFalse:{v}"),
        NativeInstr::Jump(v) => format!("Jump:{v}"), NativeInstr::Return => "Return".into(),
        NativeInstr::Pop => "Pop".into(),
        NativeInstr::Call { callee, argc } => format!("Call:{},{}", escape_artifact(callee), argc),
        NativeInstr::ScopeStart => "ScopeStart".into(), NativeInstr::ScopeEnd => "ScopeEnd".into(),
        NativeInstr::Spawn { name, callee, argc } => format!("Spawn:{},{},{}", escape_artifact(name), escape_artifact(callee), argc),
        NativeInstr::Join { name } => format!("Join:{}", escape_artifact(name)),
        NativeInstr::Cancel { name } => format!("Cancel:{}", escape_artifact(name)),
    }
}

fn decode_instr(s: &str) -> Result<NativeInstr, NativeError> {
    let mut p=s.splitn(2, ':'); let op=p.next().unwrap_or(""); let arg=p.next().unwrap_or("");
    let bad=||NativeError::InvalidProgram(format!("invalid instruction '{s}'"));
    let int=|v:&str|v.parse::<usize>().map_err(|_|bad());
    Ok(match op {
        "PushInt"=>NativeInstr::PushInt(arg.parse().map_err(|_|bad())?),
        "PushFloat"=>{let bits=arg.parse::<u32>().map_err(|_|bad())?;if !f32::from_bits(bits).is_finite(){return Err(bad());}NativeInstr::PushFloat(bits)},
        "PushBool"=>NativeInstr::PushBool(arg=="true"),
        "PushProbability"=>NativeInstr::PushProbability(arg.parse::<u32>().map_err(|_|bad())?),
        "ProbabilityAnd"=>NativeInstr::ProbabilityAnd,
        "ProbabilityOr"=>NativeInstr::ProbabilityOr,
        "ProbabilityNot"=>NativeInstr::ProbabilityNot,
        "ProbabilityMeets"=>NativeInstr::ProbabilityMeets,
        "PushTensor"=>{let (shape, values)=arg.split_once('/').ok_or_else(bad)?; let shape=shape.split(',').map(|v|v.parse::<usize>().map_err(|_|bad())).collect::<Result<Vec<_>,_>>()?; let values=values.split(',').filter(|v|!v.is_empty()).map(|v|v.parse::<u32>().map_err(|_|bad())).collect::<Result<Vec<_>,_>>()?; NativeInstr::PushTensor{shape,values}},
        "TensorAdd"=>NativeInstr::TensorAdd,
        "TensorMatMul"=>NativeInstr::TensorMatMul,
        "MakeActivationCache"=>NativeInstr::MakeActivationCache, "ActivationStore"=>NativeInstr::ActivationStore{name:unescape_artifact(arg)?}, "ActivationRestore"=>NativeInstr::ActivationRestore, "ActivationCacheLen"=>NativeInstr::ActivationCacheLen,
        "MakeGradientTape"=>NativeInstr::MakeGradientTape, "MakeTokenizerState"=>NativeInstr::MakeTokenizerState, "TokenizerFeed"=>NativeInstr::TokenizerFeed{name:unescape_artifact(arg)?}, "TokenizerFlush"=>NativeInstr::TokenizerFlush{name:unescape_artifact(arg)?}, "TokenizerTokenCount"=>NativeInstr::TokenizerTokenCount, "TokenizerToken"=>NativeInstr::TokenizerToken, "GradientInput"=>NativeInstr::GradientInput{name:unescape_artifact(arg)?}, "GradientAdd"=>NativeInstr::GradientAdd{name:unescape_artifact(arg)?}, "GradientMul"=>NativeInstr::GradientMul{name:unescape_artifact(arg)?}, "GradientBackward"=>NativeInstr::GradientBackward{name:unescape_artifact(arg)?}, "GradientValue"=>NativeInstr::GradientValue, "GradientGet"=>NativeInstr::GradientGet,
        "MakeAttentionBlock"=>NativeInstr::MakeAttentionBlock,
        "AttentionBlockApply"=>NativeInstr::AttentionBlockApply,
        "AttentionBlockCached"=>NativeInstr::AttentionBlockCached,
        "AttentionBlock"=>NativeInstr::AttentionBlock,
        "AttentionCached"=>NativeInstr::AttentionCached,
        "MakeLoRAAdapter"=>NativeInstr::MakeLoRAAdapter,
        "LoRAAdapterApply"=>NativeInstr::LoRAAdapterApply,
        "LoRAApply"=>NativeInstr::LoRAApply,
        "MakeTensor"=>NativeInstr::MakeTensor,
        "MakeEmbedding"=>NativeInstr::MakeEmbedding,
        "MakeQuantizedTensor"=>NativeInstr::MakeQuantizedTensor,
        "PushQuantizedTensor"=>{let mut parts=arg.split('/'); let shape=parts.next().ok_or_else(bad)?.split(',').map(|v|v.parse::<usize>().map_err(|_|bad())).collect::<Result<Vec<_>,_>>()?; let scale_bits=parts.next().ok_or_else(bad)?.parse::<u32>().map_err(|_|bad())?; let zero_point=parts.next().ok_or_else(bad)?.parse::<i8>().map_err(|_|bad())?; let values=parts.next().ok_or_else(bad)?.split(',').filter(|v|!v.is_empty()).map(|v|v.parse::<i8>().map_err(|_|bad())).collect::<Result<Vec<_>,_>>()?; if parts.next().is_some(){return Err(bad());} NativeInstr::PushQuantizedTensor{shape,scale_bits,zero_point,values}},
        "DequantizeTensor"=>NativeInstr::DequantizeTensor,
        "PushEmbedding"=>{let (domain, values)=arg.split_once('/').ok_or_else(bad)?; let domain=String::from_utf8(decode_hex(domain).map_err(|_|bad())?).map_err(|_|bad())?; let values=values.split(',').filter(|v|!v.is_empty()).map(|v|v.parse::<u32>().map_err(|_|bad())).collect::<Result<Vec<_>,_>>()?; NativeInstr::PushEmbedding{domain,values}},
        "EmbeddingDot"=>NativeInstr::EmbeddingDot,
        "MakeMoERouter"=>NativeInstr::MakeMoERouter,
        "MoERoute"=>NativeInstr::MoERoute,
        "RouteTopK"=>NativeInstr::RouteTopK,
        "MakeVectorIndex"=>NativeInstr::MakeVectorIndex,
        "VectorIndexInsert"=>NativeInstr::VectorIndexInsert{name:unescape_artifact(arg)?},
        "VectorIndexSearch"=>NativeInstr::VectorIndexSearch,
        "MakeMctsTree"=>NativeInstr::MakeMctsTree,
        "MctsRoot"=>NativeInstr::MctsRoot{name:unescape_artifact(arg)?},
        "MctsExpand"=>NativeInstr::MctsExpand{name:unescape_artifact(arg)?},
        "MctsChild"=>NativeInstr::MctsChild,
        "MctsUpdate"=>NativeInstr::MctsUpdate{name:unescape_artifact(arg)?},
        "MctsBackpropagate"=>NativeInstr::MctsBackpropagate{name:unescape_artifact(arg)?},
        "MctsLen"=>NativeInstr::MctsLen,
        "MakeKvCache"=>NativeInstr::MakeKvCache,"KvLen"=>NativeInstr::KvLen,"KvCapacity"=>NativeInstr::KvCapacity,
        "KvAppend"=>{let (name,policy)=arg.rsplit_once(',').ok_or_else(bad)?; let policy=policy.parse::<bool>().map_err(|_|bad())?; NativeInstr::KvAppend{name:unescape_artifact(name)?,evict_oldest:policy}},
        "MakeContextWindow"=>NativeInstr::MakeContextWindow,"ContextLen"=>NativeInstr::ContextLen,"ContextBytes"=>NativeInstr::ContextBytes,"ContextTokens"=>NativeInstr::ContextTokens,
        "ContextAppend"=>NativeInstr::ContextAppend{name:unescape_artifact(arg)?},
        "PushUnit"=>NativeInstr::PushUnit,"Index"=>NativeInstr::Index,"Len"=>NativeInstr::Len,
        "PushString"=>NativeInstr::PushString(unescape_artifact(arg)?),
        "PushList"=>NativeInstr::PushList(int(arg)?),"Append"=>NativeInstr::Append(unescape_artifact(arg)?),
        "MakeOk"=>NativeInstr::MakeOk,"Chr"=>NativeInstr::Chr,"MakeErr"=>NativeInstr::MakeErr,"Unwrap"=>NativeInstr::Unwrap,
        "Load"=>NativeInstr::Load(unescape_artifact(arg)?),"Store"=>NativeInstr::Store(unescape_artifact(arg)?),
        "AddAssign"=>NativeInstr::AddAssign(unescape_artifact(arg)?),
        "StoreIndex"=>NativeInstr::StoreIndex(unescape_artifact(arg)?),"Add"=>NativeInstr::Add,"Sub"=>NativeInstr::Sub,
        "Mul"=>NativeInstr::Mul,"Div"=>NativeInstr::Div,"Mod"=>NativeInstr::Mod,"Equal"=>NativeInstr::Equal,
        "NotEqual"=>NativeInstr::NotEqual,"Less"=>NativeInstr::Less,"LessEqual"=>NativeInstr::LessEqual,
        "Greater"=>NativeInstr::Greater,"GreaterEqual"=>NativeInstr::GreaterEqual,
        "JumpIfFalse"=>NativeInstr::JumpIfFalse(int(arg)?),"Jump"=>NativeInstr::Jump(int(arg)?),
        "Return"=>NativeInstr::Return,"Pop"=>NativeInstr::Pop,
        "Call"=>{let mut x=arg.split(','); NativeInstr::Call{callee:unescape_artifact(x.next().ok_or_else(bad)?)?,argc:int(x.next().ok_or_else(bad)?)?}},
        "ScopeStart"=>NativeInstr::ScopeStart,"ScopeEnd"=>NativeInstr::ScopeEnd,
        "Spawn"=>{let mut x=arg.split(','); NativeInstr::Spawn{name:unescape_artifact(x.next().ok_or_else(bad)?)?,callee:unescape_artifact(x.next().ok_or_else(bad)?)?,argc:int(x.next().ok_or_else(bad)?)?}},
        "Join"=>NativeInstr::Join{name:unescape_artifact(arg)?},"Cancel"=>NativeInstr::Cancel{name:unescape_artifact(arg)?},
        _=>return Err(bad()),
    })
}


fn store_activation_checkpoint(cache:&mut NativeValue,step:i64,shape:Vec<usize>,values:Vec<u32>)->Result<(),NativeError>{
    let step_usize=usize::try_from(step).map_err(|_|NativeError::Type("ActivationCache checkpoint id must be non-negative".into()))?;
    validate_tensor_parts(&shape,&values,MAX_TENSOR_ELEMENTS)?;
    if values.iter().any(|bits|!f32::from_bits(*bits).is_finite()){return Err(NativeError::Type("ActivationCache values must be finite".into()));}
    let NativeValue::ActivationCache{capacity,checkpoint_stride,max_bytes,retained_bytes,entries}=cache else{return Err(NativeError::Type("activation_store requires ActivationCache".into()));};
    if step_usize%*checkpoint_stride!=0{return Err(NativeError::Type("ActivationCache checkpoint id must align to checkpoint_stride".into()));}
    if entries.iter().any(|(id,_,_)|*id==step){return Err(NativeError::Type("duplicate ActivationCache checkpoint id".into()));}
    if entries.len()>=*capacity{return Err(NativeError::ResourceLimit("ActivationCache checkpoint capacity exhausted".into()));}
    let bytes=values.len().checked_mul(4).and_then(|n|n.checked_add(shape.len().checked_mul(std::mem::size_of::<usize>())?)).ok_or_else(||NativeError::ResourceLimit("ActivationCache byte accounting overflow".into()))?;
    let new_bytes=retained_bytes.checked_add(bytes).ok_or_else(||NativeError::ResourceLimit("ActivationCache byte accounting overflow".into()))?;
    if new_bytes>*max_bytes{return Err(NativeError::ResourceLimit("ActivationCache retained-byte budget exceeded".into()));}
    entries.push((step,shape,values));entries.sort_by_key(|(id,_,_)|*id);*retained_bytes=new_bytes;Ok(())
}

fn mcts_fields(tree: &NativeValue) -> Result<(usize, usize, &Vec<NativeValue>), NativeError> {
    let NativeValue::List(fields) = tree else { return Err(NativeError::Type("expected MctsTree".into())); };
    if fields.len() != 3 { return Err(NativeError::InvalidProgram("malformed MctsTree".into())); }
    let max_nodes = match fields.first() { Some(NativeValue::Int(v)) if *v > 0 && *v as usize <= crate::ai_mcts::MAX_MCTS_NODES => *v as usize, _ => return Err(NativeError::InvalidProgram("invalid MCTS node limit".into())) };
    let max_branches = match fields.get(1) { Some(NativeValue::Int(v)) if *v > 0 && *v as usize <= crate::ai_mcts::MAX_MCTS_BRANCHES => *v as usize, _ => return Err(NativeError::InvalidProgram("invalid MCTS branch limit".into())) };
    let nodes = match fields.get(2) { Some(NativeValue::List(v)) => v, _ => return Err(NativeError::InvalidProgram("invalid MCTS node arena".into())) };
    if nodes.len() > max_nodes { return Err(NativeError::ResourceLimit("MCTS node limit exceeded".into())); }
    validate_mcts_structure(nodes, max_branches)?;
    Ok((max_nodes, max_branches, nodes))
}
fn validate_mcts_structure(nodes: &[NativeValue], max_branches: usize) -> Result<(), NativeError> {
    for (index, node) in nodes.iter().enumerate() {
        let fields = match node { NativeValue::List(v) if v.len() == 5 => v, _ => return Err(NativeError::InvalidProgram("malformed MCTS node".into())) };
        if !matches!(fields.first(), Some(NativeValue::Int(id)) if *id >= 0)
            || !matches!(fields.get(1), Some(NativeValue::Int(parent)) if (*parent == -1 && index == 0) || (*parent >= 0 && (*parent as usize) < index))
            || !matches!(fields.get(2), Some(NativeValue::List(children)) if children.len() <= max_branches)
            || !matches!(fields.get(3), Some(NativeValue::Int(visits)) if *visits >= 0)
            || !matches!(fields.get(4), Some(NativeValue::Float(bits)) if f32::from_bits(*bits).is_finite()) {
            return Err(NativeError::InvalidProgram("invalid MCTS node fields".into()));
        }
        let children = match fields.get(2) { Some(NativeValue::List(v)) => v, _ => unreachable!() };
        let mut seen = std::collections::BTreeSet::new();
        for child in children {
            let child = match child { NativeValue::Int(v) if *v > index as i64 && (*v as usize) < nodes.len() => *v as usize, _ => return Err(NativeError::InvalidProgram("invalid MCTS child index".into())) };
            if !seen.insert(child) { return Err(NativeError::InvalidProgram("duplicate MCTS child edge".into())); }
            if !matches!(&nodes[child], NativeValue::List(child_fields) if matches!(child_fields.get(1), Some(NativeValue::Int(parent)) if *parent == index as i64)) {
                return Err(NativeError::InvalidProgram("MCTS parent-child relationship mismatch".into()));
            }
        }
        if index > 0 {
            let parent = match fields.get(1) { Some(NativeValue::Int(v)) if *v >= 0 => *v as usize, _ => return Err(NativeError::InvalidProgram("invalid MCTS parent".into())) };
            if !matches!(nodes.get(parent), Some(NativeValue::List(parent_fields)) if matches!(parent_fields.get(2), Some(NativeValue::List(children)) if children.iter().any(|child| matches!(child, NativeValue::Int(v) if *v == index as i64)))) {
                return Err(NativeError::InvalidProgram("MCTS parent is missing child edge".into()));
            }
            let mut cursor = index;
            let mut depth = 0usize;
            loop {
                let parent = match &nodes[cursor] { NativeValue::List(fields) => match fields.get(1) { Some(NativeValue::Int(-1)) if cursor == 0 => break, Some(NativeValue::Int(v)) if *v >= 0 => *v as usize,
                        _ => {
                            return Err(NativeError::InvalidProgram(
                                "invalid MCTS ancestry".into(),
                            ));
                        }
                    },
                    _ => unreachable!(),
                };
                cursor = parent;
                depth += 1;
                if depth >= crate::ai_mcts::MAX_MCTS_DEPTH { return Err(NativeError::ResourceLimit("MCTS depth limit exceeded".into())); }
            }
        }
    }
    Ok(())
}
fn mcts_node(state: u64, parent: Option<usize>) -> NativeValue {
    NativeValue::List(vec![NativeValue::Int(state as i64), NativeValue::Int(parent.map(|p| p as i64).unwrap_or(-1)), NativeValue::List(Vec::new()), NativeValue::Int(0), NativeValue::Float(0.0f32.to_bits())])
}
fn mcts_root(tree: &mut NativeValue, state: u64) -> Result<(), NativeError> {
    let (max_nodes, _, nodes) = mcts_fields(tree)?; if !nodes.is_empty() { return Err(NativeError::Type("MCTS root already exists".into())); } if max_nodes == 0 { return Err(NativeError::ResourceLimit("MCTS node limit exceeded".into())); }
    let NativeValue::List(fields) = tree else { unreachable!() }; let NativeValue::List(nodes) = &mut fields[2] else { unreachable!() }; nodes.push(mcts_node(state, None)); Ok(())
}
fn mcts_expand(tree: &mut NativeValue, parent: usize, state: u64) -> Result<usize, NativeError> {
    let (max_nodes, max_branches, nodes) = mcts_fields(tree)?;
    if parent >= nodes.len() { return Err(NativeError::Type("MCTS parent index out of bounds".into())); }
    let fields = match &nodes[parent] { NativeValue::List(v) if v.len() == 5 => v, _ => return Err(NativeError::InvalidProgram("malformed MCTS node".into())) };
    let children = match fields.get(2) { Some(NativeValue::List(v)) => v, _ => return Err(NativeError::InvalidProgram("malformed MCTS children".into())) };
    if children.len() >= max_branches { return Err(NativeError::ResourceLimit("MCTS branch limit exceeded".into())); } if nodes.len() >= max_nodes { return Err(NativeError::ResourceLimit("MCTS node limit exceeded".into())); }
    if children.iter().any(|v| matches!(v, NativeValue::Int(i) if *i >= 0 && matches!(nodes.get(*i as usize), Some(NativeValue::List(n)) if matches!(n.first(), Some(NativeValue::Int(id)) if *id == state as i64)))) { return Err(NativeError::Type("duplicate MCTS child state".into())); }
    let mut depth = 0; let mut cursor = parent; loop { let f = match nodes.get(cursor) { Some(NativeValue::List(v)) => v, _ => return Err(NativeError::InvalidProgram("invalid MCTS parent".into())) }; match f.get(1) { Some(NativeValue::Int(-1)) if cursor == 0 => break, Some(NativeValue::Int(p)) if *p >= 0 && (*p as usize) < cursor => { cursor = *p as usize; depth += 1; if depth >= crate::ai_mcts::MAX_MCTS_DEPTH { return Err(NativeError::ResourceLimit("MCTS depth limit exceeded".into())); } }, _ => return Err(NativeError::InvalidProgram("invalid MCTS parent chain".into())) } }
    if depth + 1 >= crate::ai_mcts::MAX_MCTS_DEPTH { return Err(NativeError::ResourceLimit("MCTS depth limit exceeded".into())); }
    let index = nodes.len(); let NativeValue::List(fields) = tree else { unreachable!() }; let NativeValue::List(nodes) = &mut fields[2] else { unreachable!() };
    nodes.push(mcts_node(state, Some(parent))); let NativeValue::List(parent_fields) = &mut nodes[parent] else { unreachable!() }; let NativeValue::List(children) = &mut parent_fields[2] else { unreachable!() }; children.push(NativeValue::Int(index as i64)); Ok(index)
}
fn mcts_child(tree: &NativeValue, parent: usize, ordinal: usize) -> Result<usize, NativeError> {
    let (_, _, nodes) = mcts_fields(tree)?; let fields = match nodes.get(parent) { Some(NativeValue::List(v)) => v, _ => return Err(NativeError::Type("MCTS parent index out of bounds".into())) };
    match fields.get(2) { Some(NativeValue::List(children)) => match children.get(ordinal) { Some(NativeValue::Int(v)) if *v >= 0 => Ok(*v as usize), _ => Err(NativeError::Type("MCTS child ordinal out of bounds".into())) }, _ => Err(NativeError::InvalidProgram("malformed MCTS child list".into())) }
}
fn mcts_len(tree: &NativeValue) -> Result<usize, NativeError> { Ok(mcts_fields(tree)?.2.len()) }
fn mcts_update(tree: &mut NativeValue, node: usize, reward: f64, backprop: bool) -> Result<(), NativeError> {
    if !reward.is_finite() || reward.abs() > crate::ai_mcts::MAX_MCTS_REWARD_ABS { return Err(NativeError::Type("MCTS reward is outside finite bounds".into())); }
    let (_, _, nodes) = mcts_fields(tree)?; if node >= nodes.len() { return Err(NativeError::Type("MCTS node index out of bounds".into())); }
    let mut path = vec![node]; if backprop { let mut cursor = node; loop { let f = match nodes.get(cursor) { Some(NativeValue::List(v)) => v, _ => return Err(NativeError::InvalidProgram("malformed MCTS node".into())) }; match f.get(1) { Some(NativeValue::Int(-1)) if cursor == 0 => break, Some(NativeValue::Int(p)) if *p >= 0 && (*p as usize) < cursor => { let p = *p as usize; if !matches!(nodes.get(p), Some(NativeValue::List(pf)) if matches!(pf.get(2), Some(NativeValue::List(ch)) if ch.iter().any(|x| matches!(x, NativeValue::Int(v) if *v == cursor as i64)))) { return Err(NativeError::InvalidProgram("MCTS parent-child mismatch".into())); } cursor = p; path.push(cursor); if path.len() >= crate::ai_mcts::MAX_MCTS_DEPTH { return Err(NativeError::ResourceLimit("MCTS depth limit exceeded".into())); } }, _ => return Err(NativeError::InvalidProgram("invalid MCTS parent chain".into())) } } }
    for &idx in &path { let f = match nodes.get(idx) { Some(NativeValue::List(v)) if v.len() == 5 => v, _ => return Err(NativeError::Type("MCTS node index out of bounds".into())) }; let visits = match f.get(3) { Some(NativeValue::Int(v)) if *v >= 0 && *v < i64::MAX => *v, _ => return Err(NativeError::ResourceLimit("MCTS visit counter overflow".into())) }; let old = match f.get(4) { Some(NativeValue::Float(bits)) if f32::from_bits(*bits).is_finite() => f64::from(f32::from_bits(*bits)), _ => return Err(NativeError::InvalidProgram("invalid MCTS value sum".into())) }; let next = old + reward; if !next.is_finite() || next.abs() > crate::ai_mcts::MAX_MCTS_REWARD_ABS * (visits + 1) as f64 || !(next as f32).is_finite() { return Err(NativeError::Type("MCTS accumulated reward exceeds supported bounds".into())); } }
    let NativeValue::List(fields) = tree else { unreachable!() }; let NativeValue::List(nodes) = &mut fields[2] else { unreachable!() };
    for idx in path { let NativeValue::List(f) = &mut nodes[idx] else { unreachable!() }; let NativeValue::Int(visits) = &mut f[3] else { unreachable!() }; *visits += 1; let old = match f[4] { NativeValue::Float(bits) => f32::from_bits(bits), _ => unreachable!() }; f[4] = NativeValue::Float(((f64::from(old) + reward) as f32).to_bits()); } Ok(())
}
fn validate_tensor_parts(shape: &[usize], values: &[u32], max_elements: usize) -> Result<(), NativeError> {
    if shape.is_empty() || shape.len() > 2 || shape.iter().any(|dimension| *dimension == 0) {
        return Err(NativeError::InvalidProgram("tensor shape must have rank 1 or 2 and positive dimensions".into()));
    }
    let expected = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d).ok_or_else(|| NativeError::ResourceLimit("tensor shape overflow".into())))?;
    let limit = max_elements.min(MAX_TENSOR_ELEMENTS);
    if expected > limit || values.len() > limit {
        return Err(NativeError::ResourceLimit(format!("tensor element limit exceeded (limit {limit})")));
    }
    if expected != values.len() {
        return Err(NativeError::InvalidProgram(format!("tensor shape expects {expected} values but received {}", values.len())));
    }
    if values.iter().any(|bits| !f32::from_bits(*bits).is_finite()) {
        return Err(NativeError::InvalidProgram("tensor values must be finite".into()));
    }
    Ok(())
}

fn validate_quantized_tensor_parts(shape: &[usize], scale_bits: u32, values: &[i8], max_elements: usize) -> Result<(), NativeError> {
    if shape.is_empty() || shape.len() > 2 || shape.iter().any(|dimension| *dimension == 0) {
        return Err(NativeError::InvalidProgram("quantized tensor shape must have rank 1 or 2 and positive dimensions".into()));
    }
    let expected = shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d).ok_or_else(|| NativeError::ResourceLimit("quantized tensor shape overflow".into())))?;
    let limit = max_elements.min(MAX_TENSOR_ELEMENTS);
    if expected > limit || values.len() > limit { return Err(NativeError::ResourceLimit(format!("quantized tensor element limit exceeded (limit {limit})"))); }
    if expected != values.len() { return Err(NativeError::InvalidProgram(format!("quantized tensor shape expects {expected} values but received {}", values.len()))); }
    let scale = f32::from_bits(scale_bits);
    if !scale.is_finite() || scale <= 0.0 { return Err(NativeError::InvalidProgram("quantization scale must be finite and positive".into())); }
    Ok(())
}

fn validate_lora_parts(a_shape: &[usize], a_values: &[u32], b_shape: &[usize], b_values: &[u32], alpha_bits: u32, rank: usize, limit: usize) -> Result<(), NativeError> {
    validate_tensor_parts(a_shape, a_values, limit)?;
    validate_tensor_parts(b_shape, b_values, limit)?;
    let adapter_elements = a_values.len().checked_add(b_values.len())
        .ok_or_else(|| NativeError::ResourceLimit("LoRAAdapter storage accounting overflow".into()))?;
    if adapter_elements > limit.min(MAX_TENSOR_ELEMENTS) {
        return Err(NativeError::ResourceLimit("LoRAAdapter aggregate matrix storage exceeds element budget".into()));
    }
    let alpha = f32::from_bits(alpha_bits);
    if rank == 0 || rank > 256 || !alpha.is_finite() || a_shape.len() != 2 || b_shape.len() != 2
        || a_shape[0] != rank || b_shape[1] != rank || a_shape[1] == 0 || b_shape[0] == 0
        || a_values.iter().chain(b_values).any(|bits| !f32::from_bits(*bits).is_finite()) {
        return Err(NativeError::Type("LoRAAdapter requires finite A[rank,in], B[out,rank], finite alpha, and rank in 1..=256".into()));
    }
    Ok(())
}

fn attention_multihead(
    config: &(usize, usize, usize, bool),
    q_shape: &[usize], q: &[u32],
    k_shape: &[usize], k: &[u32],
    v_shape: &[usize], v: &[u32],
    limits: ExecutionLimits,
) -> Result<(Vec<usize>, Vec<u32>), NativeError> {
    let (hidden, heads, head_dim, causal) = *config;
    validate_tensor_parts(q_shape, q, limits.max_collection_items)?;
    validate_tensor_parts(k_shape, k, limits.max_collection_items)?;
    validate_tensor_parts(v_shape, v, limits.max_collection_items)?;
    if hidden == 0 || heads == 0 || head_dim == 0 || heads.checked_mul(head_dim) != Some(hidden)
        || q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2
        || q_shape[1] != hidden || k_shape[1] != hidden || v_shape[1] != hidden
        || k_shape[0] != v_shape[0] || q_shape[0] == 0 || k_shape[0] == 0 {
        return Err(NativeError::Type("AttentionBlock requires Q[Tq,hidden], K[Tk,hidden], V[Tk,hidden] and compatible configuration".into()));
    }
    let (tq, tk) = (q_shape[0], k_shape[0]);
    let out_count = tq.checked_mul(hidden).ok_or_else(|| NativeError::ResourceLimit("attention output size overflow".into()))?;
    let work = tq.checked_mul(tk).and_then(|n| n.checked_mul(hidden)).ok_or_else(|| NativeError::ResourceLimit("attention work overflow".into()))?;
    if out_count > limits.max_collection_items.min(MAX_TENSOR_ELEMENTS) || work > limits.max_value_nodes.saturating_mul(16) {
        return Err(NativeError::ResourceLimit("AttentionBlock computation exceeds resource budget".into()));
    }
    let mut out = vec![0.0f32; out_count];
    for qi in 0..tq {
        for head in 0..heads {
            let offset = head * head_dim;
            let mut scores = Vec::with_capacity(tk);
            for ki in 0..tk {
                if causal && ki > qi { scores.push(f32::NEG_INFINITY); continue; }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += f32::from_bits(q[qi*hidden+offset+d]) * f32::from_bits(k[ki*hidden+offset+d]);
                }
                let score = dot / (head_dim as f32).sqrt();
                if !score.is_finite() { return Err(NativeError::Type("AttentionBlock score is non-finite".into())); }
                scores.push(score);
            }
            let max = scores.iter().copied().filter(|x| x.is_finite()).fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() { return Err(NativeError::Type("AttentionBlock row is fully masked".into())); }
            let mut weights = scores.iter().map(|x| if x.is_finite() { (*x-max).exp() } else { 0.0 }).collect::<Vec<_>>();
            let total: f32 = weights.iter().sum();
            if !total.is_finite() || total <= 0.0 { return Err(NativeError::Type("AttentionBlock softmax normalization failed".into())); }
            for w in &mut weights { *w /= total; }
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..tk { sum += weights[ki] * f32::from_bits(v[ki*hidden+offset+d]); }
                if !sum.is_finite() { return Err(NativeError::Type("AttentionBlock output is non-finite".into())); }
                out[qi*hidden+offset+d] = sum;
            }
        }
    }
    Ok((vec![tq, hidden], out.into_iter().map(f32::to_bits).collect()))
}

fn pop_tensor(stack: &mut Vec<NativeValue>, limit: usize) -> Result<(Vec<usize>, Vec<u32>), NativeError> {
    match stack.pop() {
        Some(NativeValue::Tensor { shape, values }) => { validate_tensor_parts(&shape, &values, limit)?; Ok((shape, values)) }
        _ => Err(NativeError::Type("expected Tensor value".into())),
    }
}

fn validate_embedding_parts(domain: &str, values: &[u32], max_elements: usize) -> Result<(), NativeError> {
    if domain.trim().is_empty() || domain.len() > 1024 {
        return Err(NativeError::InvalidProgram("embedding domain must be non-empty and at most 1024 bytes".into()));
    }
    if values.is_empty() || values.len() > max_elements.min(MAX_TENSOR_ELEMENTS) {
        return Err(NativeError::ResourceLimit("embedding dimension is empty or exceeds the element limit".into()));
    }
    if values.iter().any(|bits| !f32::from_bits(*bits).is_finite()) {
        return Err(NativeError::InvalidProgram("embedding coordinates must be finite".into()));
    }
    Ok(())
}

fn pop_embedding(stack: &mut Vec<NativeValue>, limit: usize) -> Result<(String, Vec<u32>), NativeError> {
    match stack.pop() {
        Some(NativeValue::Embedding { domain, values }) => {
            validate_embedding_parts(&domain, &values, limit)?;
            Ok((domain, values))
        }
        _ => Err(NativeError::Type("expected Embedding value".into())),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes { out.push(HEX[(byte >> 4) as usize] as char); out.push(HEX[(byte & 0x0f) as usize] as char); }
    out
}

fn decode_hex(value: &str) -> Result<Vec<u8>, ()> {
    if value.len() % 2 != 0 { return Err(()); }
    let mut out = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let digit = |b: u8| -> Result<u8, ()> { match b { b'0'..=b'9' => Ok(b-b'0'), b'a'..=b'f' => Ok(b-b'a'+10), b'A'..=b'F' => Ok(b-b'A'+10), _ => Err(()) } };
        out.push((digit(pair[0])? << 4) | digit(pair[1])?);
    }
    Ok(out)
}

fn validate_probability_bits(bits: u32) -> Result<f32, NativeError> {
    let value = f32::from_bits(bits);
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(NativeError::InvalidProgram("probability constant must be finite and in [0, 1]".into()));
    }
    Ok(value)
}

fn pop_probability(stack: &mut Vec<NativeValue>) -> Result<f32, NativeError> {
    match stack.pop() {
        Some(NativeValue::Probability(bits)) => validate_probability_bits(bits),
        _ => Err(NativeError::Type("expected Probability value".into())),
    }
}

fn pop_bool(stack: &mut Vec<NativeValue>) -> Result<bool, NativeError> {
    match stack.pop() { Some(NativeValue::Bool(value)) => Ok(value), _ => Err(NativeError::Type("expected Bool value".into())) }
}
fn pop_float(stack: &mut Vec<NativeValue>) -> Result<f32, NativeError> {
    match stack.pop() {
        Some(NativeValue::Float(bits)) if f32::from_bits(bits).is_finite() => Ok(f32::from_bits(bits)),
        Some(NativeValue::Int(value)) => { let value = value as f32; if value.is_finite() { Ok(value) } else { Err(NativeError::Type("expected finite F32 value".into())) } },
        _ => Err(NativeError::Type("expected finite F32 value".into())),
    }
}
fn pop_int(stack: &mut Vec<NativeValue>) -> Result<i64, NativeError> {
    match stack.pop() {
        Some(NativeValue::Int(value)) => Ok(value),
        _ => Err(NativeError::Type("expected Int value".into())),
    }
}

#[cfg(test)]
mod ai_routing_language_tests {
    use super::*;

    fn compile(source: &str) -> NativeProgram {
        let module = crate::parse(source).expect("source parses");
        crate::sema::check(&module).expect("source passes semantic checks");
        compile_program(&crate::ir::lower(&module)).expect("source lowers to native program")
    }

    #[test]
    fn source_route_top_k_runs_through_artifact_roundtrip() {
        let source = r#"module route_fixture
fn main() -> List<ExpertWeight>
  route_top_k([1.0, 3.0, 2.0], [false, false, false], 2)
"#;
        let program = compile(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("well-formed route artifact");
        assert_eq!(encode_program(&decoded), artifact);
        let NativeValue::List(rows) = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected route rows"); };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0], NativeValue::List(vec![NativeValue::Int(1), NativeValue::Float(0.7310586f32.to_bits())]));
        assert_eq!(rows[1], NativeValue::List(vec![NativeValue::Int(2), NativeValue::Float(0.26894143f32.to_bits())]));
    }

    #[test]
    fn source_vector_index_is_domain_isolated_and_deterministic() {
        let source = r#"module vector_fixture
fn main() -> List<VectorHit>
  let index = vector_index("text", 2)
  vector_index_insert(index, 9, "text", [1.0, 0.0])
  vector_index_insert(index, 3, "text", [0.0, 1.0])
  return vector_index_search(index, "text", [0.0, 0.0], 2)
"#;
        let program = compile(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("well-formed vector artifact");
        assert_eq!(encode_program(&decoded), artifact);
        let expected = NativeValue::List(vec![
            NativeValue::List(vec![NativeValue::Int(3), NativeValue::Float(1.0f32.to_bits())]),
            NativeValue::List(vec![NativeValue::Int(9), NativeValue::Float(1.0f32.to_bits())]),
        ]);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), expected);
    }

    #[test]
    fn routing_and_vector_builtins_reject_bad_runtime_inputs() {
        let bad_route = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushList(0), NativeInstr::PushList(0), NativeInstr::PushInt(1), NativeInstr::RouteTopK, NativeInstr::Return,
        ] })])};
        assert!(matches!(run_program(&bad_route, "main", &[]), Err(NativeError::Type(_))));

        let bad_insert = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushString("text".into()), NativeInstr::PushInt(2), NativeInstr::MakeVectorIndex,
            NativeInstr::Store("index".into()),
            NativeInstr::PushInt(1), NativeInstr::PushString("image".into()), NativeInstr::PushList(0),
            NativeInstr::VectorIndexInsert { name: "index".into() }, NativeInstr::Return,
        ] })])};
        assert!(matches!(run_program(&bad_insert, "main", &[]), Err(NativeError::Type(message)) if message.contains("domain mismatch")));
    }

    #[test]
    fn malformed_ai_artifacts_fail_closed() {
        assert!(matches!(decode_program("ARDISA-EXEC-V1\nI|RouteTopK\n"), Err(NativeError::InvalidProgram(_))));
        let malformed_binding = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![
            NativeInstr::VectorIndexInsert { name: " ".into() }, NativeInstr::Return,
        ] })])};
        assert!(matches!(decode_program(&encode_program(&malformed_binding)), Err(NativeError::InvalidProgram(message)) if message.contains("binding name")));
        let malformed_index = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushString("".into()), NativeInstr::PushInt(1), NativeInstr::MakeVectorIndex, NativeInstr::Return,
        ] })])};
        assert!(matches!(run_program(&malformed_index, "main", &[]), Err(NativeError::Type(_))));
    }
}

#[cfg(test)]
mod artifact_tests {
    use super::*;

    #[test]
    fn executable_artifact_round_trips_deterministically() {
        let module = crate::parse(
            r#"module artifact
fn main(a: String) -> String
  a + "!"
"#,
        ).unwrap();
        crate::sema::check(&module).unwrap();
        crate::ownership::infer(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let encoded = encode_program(&program);
        assert!(encoded.starts_with(ARTIFACT_MAGIC));
        let decoded = decode_program(&encoded).unwrap();
        assert_eq!(decoded, program);
        assert_eq!(encode_program(&decoded), encoded);
        assert_eq!(
            run_program(&decoded, "main", &[NativeValue::String("Ardisa".into())]).unwrap(),
            NativeValue::String("Ardisa!".into())
        );
    }

    #[test]
    fn probability_instructions_execute_and_round_trip_as_ardisa_exec_v1() {
        let half = 0.5f32.to_bits();
        let quarter = 0.25f32.to_bits();
        let program = NativeProgram { functions: BTreeMap::from([(
            "main".to_string(), NativeFunction { params: vec![], code: vec![
                NativeInstr::PushProbability(half),
                NativeInstr::PushProbability(quarter),
                NativeInstr::ProbabilityAnd,
                NativeInstr::PushProbability(0.1f32.to_bits()),
                NativeInstr::ProbabilityMeets,
                NativeInstr::Return,
            ] }
        )])};
        let encoded = encode_program(&program);
        let decoded = decode_program(&encoded).expect("probability bytecode decodes");
        assert_eq!(encode_program(&decoded), encoded);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Bool(true));

        let or_program = NativeProgram { functions: BTreeMap::from([(
            "main".to_string(), NativeFunction { params: vec![], code: vec![
                NativeInstr::PushProbability(half),
                NativeInstr::PushProbability(quarter),
                NativeInstr::ProbabilityOr,
                NativeInstr::PushProbability(0.6f32.to_bits()),
                NativeInstr::ProbabilityMeets,
                NativeInstr::Return,
            ] }
        )])};
        assert_eq!(run_program(&or_program, "main", &[]).unwrap(), NativeValue::Bool(true));

        let low = 0.25f32.to_bits();
        let expected_not = 0.75f32.to_bits();
        let not_program = NativeProgram { functions: BTreeMap::from([(
            "main".to_string(), NativeFunction { params: vec![], code: vec![
                NativeInstr::PushProbability(low), NativeInstr::ProbabilityNot, NativeInstr::Return,
            ] }
        )])};
        assert_eq!(run_program(&not_program, "main", &[]).unwrap(), NativeValue::Probability(expected_not));
    }

    #[test]
    fn invalid_probability_constants_and_operands_fail_closed() {
        let invalid = NativeProgram { functions: BTreeMap::from([(
            "main".to_string(), NativeFunction { params: vec![], code: vec![
                NativeInstr::PushProbability(f32::NAN.to_bits()), NativeInstr::Return,
            ] }
        )])};
        assert!(matches!(decode_program(&encode_program(&invalid)), Err(NativeError::InvalidProgram(_))));

        let wrong_type = NativeProgram { functions: BTreeMap::from([(
            "main".to_string(), NativeFunction { params: vec![], code: vec![
                NativeInstr::PushInt(1), NativeInstr::ProbabilityNot, NativeInstr::Return,
            ] }
        )])};
        assert!(matches!(run_program(&wrong_type, "main", &[]), Err(NativeError::Type(_))));
    }

    #[test]
    fn tensor_instructions_execute_and_round_trip_deterministically() {
        let tensor = |shape: Vec<usize>, values: &[f32]| NativeInstr::PushTensor { shape, values: values.iter().map(|v| v.to_bits()).collect() };
        let program = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]),
            tensor(vec![2, 2], &[10.0, 20.0, 30.0, 40.0]),
            NativeInstr::TensorAdd, NativeInstr::Return,
        ] })])};
        let encoded = encode_program(&program);
        let decoded = decode_program(&encoded).unwrap();
        assert_eq!(encode_program(&decoded), encoded);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Tensor { shape: vec![2,2], values: [11.0f32,22.0,33.0,44.0].iter().map(|v|v.to_bits()).collect() });

        let matmul = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]), tensor(vec![2, 1], &[5.0, 6.0]), NativeInstr::TensorMatMul, NativeInstr::Return,
        ] })])};
        assert_eq!(run_program(&matmul, "main", &[]).unwrap(), NativeValue::Tensor { shape: vec![2,1], values: [17.0f32,39.0].iter().map(|v|v.to_bits()).collect() });
    }

    #[test]
    fn domain_embeddings_execute_and_round_trip_with_domain_safety() {
        let embedding = |domain: &str, values: &[f32]| NativeInstr::PushEmbedding { domain: domain.to_owned(), values: values.iter().map(|v| v.to_bits()).collect() };
        let program = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            embedding("text/domain", &[1.0, 2.0]), embedding("text/domain", &[3.0, 4.0]), NativeInstr::EmbeddingDot, NativeInstr::Return,
        ] })])};
        let encoded = encode_program(&program);
        let decoded = decode_program(&encoded).unwrap();
        assert_eq!(encode_program(&decoded), encoded);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Float(11.0f32.to_bits()));

        let mismatched = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            embedding("text", &[1.0]), embedding("image", &[1.0]), NativeInstr::EmbeddingDot, NativeInstr::Return,
        ] })])};
        assert!(matches!(run_program(&mismatched, "main", &[]), Err(NativeError::Type(_))));
    }

    #[test]
    fn quantized_tensor_runtime_round_trips_and_dequantizes_safely() {
        let program = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushQuantizedTensor { shape: vec![3], scale_bits: 0.5f32.to_bits(), zero_point: 0, values: vec![-2, 0, 4] },
            NativeInstr::DequantizeTensor, NativeInstr::Return,
        ] })])};
        let encoded = encode_program(&program);
        let decoded = decode_program(&encoded).unwrap();
        assert_eq!(encode_program(&decoded), encoded);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Tensor { shape: vec![3], values: [-1.0f32, 0.0, 2.0].iter().map(|v| v.to_bits()).collect() });

        let invalid_scale = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushQuantizedTensor { shape: vec![1], scale_bits: 0.0f32.to_bits(), zero_point: 0, values: vec![1] }, NativeInstr::Return,
        ] })])};
        assert!(matches!(decode_program(&encode_program(&invalid_scale)), Err(NativeError::InvalidProgram(_))));

        let malformed_shape = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushQuantizedTensor { shape: vec![2], scale_bits: 1.0f32.to_bits(), zero_point: 0, values: vec![1] }, NativeInstr::Return,
        ] })])};
        assert!(matches!(decode_program(&encode_program(&malformed_shape)), Err(NativeError::InvalidProgram(_))));
    }

    #[test]
    fn malformed_tensors_and_tensor_shapes_fail_closed() {
        let bad = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushTensor { shape: vec![2,2], values: vec![1.0f32.to_bits()] }, NativeInstr::Return,
        ] })])};
        assert!(matches!(decode_program(&encode_program(&bad)), Err(NativeError::InvalidProgram(_))));
        let mismatched = NativeProgram { functions: BTreeMap::from([("main".to_string(), NativeFunction { params: vec![], code: vec![
            NativeInstr::PushTensor { shape: vec![2], values: vec![1.0f32.to_bits(),2.0f32.to_bits()] },
            NativeInstr::PushTensor { shape: vec![1], values: vec![3.0f32.to_bits()] }, NativeInstr::TensorAdd, NativeInstr::Return,
        ] })])};
        assert!(matches!(run_program(&mismatched, "main", &[]), Err(NativeError::Type(_))));
    }

    #[test]
    fn malformed_executable_is_rejected() {
        assert!(matches!(
            decode_program("ARDISA-EXEC-V1\nI|Return\n"),
            Err(NativeError::InvalidProgram(_))
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_type_coverage_includes_aggregate_parameters() {
        assert!(is_native_type(&TypeKind::Int));
        assert!(is_native_type(&TypeKind::List(Box::new(crate::ast::Type {
            id: crate::NodeId(1),
            span: crate::source::Span::new(0, 0),
            kind: TypeKind::Int,
        }))));
        assert!(is_native_type(&TypeKind::Result(
            Box::new(crate::ast::Type {
                id: crate::NodeId(2),
                span: crate::source::Span::new(0, 0),
                kind: TypeKind::Int,
            }),
            Box::new(crate::ast::Type {
                id: crate::NodeId(3),
                span: crate::source::Span::new(0, 0),
                kind: TypeKind::String,
            }),
        )));
        assert!(!is_native_type(&TypeKind::Named("Custom".into())));
    }

    #[test]
    fn compiles_and_runs_aggregate_values_without_rust() {
        let module = crate::parse(
            "module x\nfn main(values: List<Int>) -> Int\n  len(values)\n",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(
            &program,
            "main",
            &[NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)])],
        )
        .unwrap();
        assert_eq!(result, NativeValue::Int(2));
    }

    #[test]
    fn compiles_and_runs_result_values_without_rust() {
        let module = crate::parse(
            "module x\nfn main(value: Result<Int, String>) -> Int\n  unwrap(value)\n",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(
            &program,
            "main",
            &[NativeValue::ResultOk(Box::new(NativeValue::Int(9)))],
        )
        .unwrap();
        assert_eq!(result, NativeValue::Int(9));
    }

    #[test]
    fn compiles_and_runs_arithmetic_without_rust() {
        let module = crate::parse(
            "module x
fn main(a: Int, b: Int) -> Int
  a + b * 2
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let code = compile(&ir).unwrap();
        let result = run(
            &code,
            &[
                (String::from("a"), NativeValue::Int(3)),
                (String::from("b"), NativeValue::Int(4)),
            ],
        )
        .unwrap();
        assert_eq!(result, NativeValue::Int(11));
    }

    #[test]
    fn compiles_and_runs_conditionals_without_rust() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  if a == 0
    return 1
  else
    return 2
",
        )
        .unwrap();
        let ir = crate::ir::lower(&module);
        let code = compile(&ir).unwrap();
        let result = run(&code, &[(String::from("a"), NativeValue::Int(0))]).unwrap();
        assert_eq!(result, NativeValue::Int(1));
    }

    #[test]
    fn compiles_and_runs_statement_style_if() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  let value = 0
  if a == 0
    set value = 1
  else
    set value = 2
  value
",
        )
        .unwrap();
        let ir = crate::ir::lower(&module);
        let code = compile(&ir).unwrap();
        let result = run(&code, &[(String::from("a"), NativeValue::Int(0))]).unwrap();
        assert_eq!(result, NativeValue::Int(1));
    }

    #[test]
    fn compiles_and_runs_mutable_assignments() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  let value = a
  set value = value + 2
  value
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[NativeValue::Int(5)]).unwrap();
        assert_eq!(result, NativeValue::Int(7));
    }

    #[test]
    fn compiles_and_runs_while_loops() {
        let module = crate::parse(
            "module x
fn main(a: Int) -> Int
  while a == 0
    return 7
  return 9
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[NativeValue::Int(0)]).unwrap();
        assert_eq!(result, NativeValue::Int(7));
    }

    #[test]
    fn child_runtime_error_cancels_and_joins_running_sibling() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("fail".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushBool(true),
                        NativeInstr::PushInt(1),
                        NativeInstr::Add,
                        NativeInstr::Return,
                    ],
                }),
                ("worker".into(), NativeFunction {
                    params: vec![],
                    code: vec![NativeInstr::Jump(0)],
                }),
                ("main".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::ScopeStart,
                        NativeInstr::Spawn { name: "a_fail".into(), callee: "fail".into(), argc: 0 },
                        NativeInstr::Spawn { name: "z_worker".into(), callee: "worker".into(), argc: 0 },
                        NativeInstr::ScopeEnd,
                        NativeInstr::PushInt(1),
                        NativeInstr::Return,
                    ],
                }),
            ]),
        };
        let error = run_program_with_limits(
            &program,
            "main",
            &[],
            ExecutionLimits { max_instructions: usize::MAX, ..ExecutionLimits::default() },
        ).unwrap_err();
        assert!(matches!(error, NativeError::Type(_)), "unexpected result: {error:?}");
    }

    #[test]
    fn cancelled_child_does_not_mask_sibling_runtime_error() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("fail".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushBool(true),
                        NativeInstr::PushInt(1),
                        NativeInstr::Add,
                        NativeInstr::Return,
                    ],
                }),
                ("worker".into(), NativeFunction {
                    params: vec![],
                    code: vec![NativeInstr::Jump(0)],
                }),
                ("main".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::ScopeStart,
                        NativeInstr::Spawn { name: "a_cancel".into(), callee: "worker".into(), argc: 0 },
                        NativeInstr::Spawn { name: "z_fail".into(), callee: "fail".into(), argc: 0 },
                        NativeInstr::Cancel { name: "a_cancel".into() },
                        NativeInstr::ScopeEnd,
                        NativeInstr::PushInt(1),
                        NativeInstr::Return,
                    ],
                }),
            ]),
        };
        let error = run_program_with_limits(
            &program,
            "main",
            &[],
            ExecutionLimits { max_instructions: usize::MAX, ..ExecutionLimits::default() },
        ).unwrap_err();
        assert!(matches!(error, NativeError::Type(_)), "unexpected result: {error:?}");
    }

    #[test]
    fn string_concatenation_checks_limit_before_allocating() {
        let limits = ExecutionLimits {
            max_string_bytes: 5,
            ..ExecutionLimits::default()
        };
        assert_eq!(
            add_values(
                NativeValue::String("ab".into()),
                NativeValue::String("cde".into()),
                limits,
            ).unwrap(),
            NativeValue::String("abcde".into())
        );
        assert!(matches!(
            add_values(
                NativeValue::String("abc".into()),
                NativeValue::String("def".into()),
                limits,
            ),
            Err(NativeError::ResourceLimit(message)) if message.contains("concatenation")
        ));
    }

    #[test]
    fn string_concatenation_limit_uses_utf8_byte_length() {
        let limits = ExecutionLimits {
            max_string_bytes: 4,
            ..ExecutionLimits::default()
        };
        assert_eq!(
            add_values(
                NativeValue::String("é".into()),
                NativeValue::String("ab".into()),
                limits,
            ).unwrap(),
            NativeValue::String("éab".into())
        );
        assert!(matches!(
            add_values(
                NativeValue::String("é".into()),
                NativeValue::String("abc".into()),
                limits,
            ),
            Err(NativeError::ResourceLimit(_))
        ));
    }

    #[test]
    fn compiles_and_runs_string_concatenation() {
        let module = crate::parse(
            r#"module x
fn main() -> String
  "hello " + "ardisa"
"#,
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[]).unwrap();
        assert_eq!(result, NativeValue::String("hello ardisa".into()));
    }

    #[test]
    fn compiles_and_runs_function_calls() {
        let module = crate::parse(
            "module x
fn double(a: Int) -> Int
  a * 2
fn main(a: Int) -> Int
  double(a) + 1
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "main", &[NativeValue::Int(3)]).unwrap();
        assert_eq!(result, NativeValue::Int(7));
    }

    #[test]
    fn executes_structured_scope_with_joined_child() {
        let module = crate::parse(
            "module x
fn worker(a: Int) -> Int
  a + 1
fn main() -> Int
  scope
    spawn worker_task = worker(4)
    join worker_task
  7
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        crate::concurrency::analyze(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::Int(7)
        );
    }

    #[test]
    fn executes_structured_scope_with_cancelled_child() {
        let module = crate::parse(
            "module x
fn worker() -> Int
  while true
    1
  return 0
fn main() -> Int
  scope
    spawn worker_task = worker()
    cancel worker_task
  9
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        crate::concurrency::analyze(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::Int(9)
        );
    }

    #[test]
    fn cancellation_is_a_normal_scope_terminal_state() {
        let module = crate::parse(
            "module x
fn worker() -> Int
  while true
    1
  return 0
fn main() -> Int
  scope
    spawn worker_task = worker()
    cancel worker_task
  9
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        crate::concurrency::analyze(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::Int(9)
        );
    }

    #[test]
    fn compiles_and_runs_recursive_calls() {
        let module = crate::parse(
            "module x
fn fact(n: Int) -> Int
  if n == 0
    return 1
  else
    return n * fact(n - 1)
",
        )
        .unwrap();
        crate::sema::check(&module).unwrap();
        let ir = crate::ir::lower(&module);
        let program = compile_program(&ir).unwrap();
        let result = run_program(&program, "fact", &[NativeValue::Int(5)]).unwrap();
        assert_eq!(result, NativeValue::Int(120));
    }

    #[test]
    fn instruction_budget_stops_infinite_loops() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![NativeInstr::Jump(0)] },
            )]),
        };
        let error = run_program_with_limits(&program, "main", &[], ExecutionLimits {
            max_instructions: 12,
            ..ExecutionLimits::default()
        }).unwrap_err();
        assert!(matches!(error, NativeError::ResourceLimit(message) if message.contains("instruction budget")));
    }

    #[test]
    fn call_depth_limit_stops_unbounded_recursion() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![
                    NativeInstr::Call { callee: "main".into(), argc: 0 },
                    NativeInstr::Return,
                ] },
            )]),
        };
        let error = run_program_with_limits(&program, "main", &[], ExecutionLimits {
            max_instructions: 100,
            max_call_depth: 3,
            ..ExecutionLimits::default()
        }).unwrap_err();
        assert!(matches!(error, NativeError::ResourceLimit(message) if message.contains("call depth")));
    }

    #[test]
    fn input_collection_and_string_limits_fail_closed() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec!["value".into()], code: vec![
                    NativeInstr::Load("value".into()),
                    NativeInstr::Return,
                ] },
            )]),
        };
        let limits = ExecutionLimits {
            max_string_bytes: 3,
            max_collection_items: 1,
            ..ExecutionLimits::default()
        };
        assert!(matches!(
            run_program_with_limits(&program, "main", &[NativeValue::String("long".into())], limits),
            Err(NativeError::ResourceLimit(_))
        ));
        assert!(matches!(
            run_program_with_limits(&program, "main", &[NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)])], limits),
            Err(NativeError::ResourceLimit(_))
        ));
    }

    #[test]
    fn task_limit_stops_excess_workers_and_cleans_up() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("worker".into(), NativeFunction { params: vec![], code: vec![NativeInstr::Jump(0)] }),
                ("main".into(), NativeFunction { params: vec![], code: vec![
                    NativeInstr::ScopeStart,
                    NativeInstr::Spawn { name: "one".into(), callee: "worker".into(), argc: 0 },
                    NativeInstr::Spawn { name: "two".into(), callee: "worker".into(), argc: 0 },
                    NativeInstr::ScopeEnd,
                    NativeInstr::PushInt(1),
                    NativeInstr::Return,
                ] }),
            ]),
        };
        let error = run_program_with_limits(&program, "main", &[], ExecutionLimits {
            max_instructions: usize::MAX,
            max_tasks: 1,
            ..ExecutionLimits::default()
        }).unwrap_err();
        assert!(matches!(error, NativeError::ResourceLimit(message) if message.contains("active task limit")));
    }

    #[test]
    fn self_appending_string_assignment_uses_in_place_add_assign() {
        let module = crate::parse(
            "module x\nfn main() -> String\n  let output = \"\"\n  set output = output + \"hello\"\n  output\n",
        ).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let main = &program.functions["main"];
        assert!(main.code.iter().any(|instr| matches!(instr, NativeInstr::AddAssign(name) if name == "output")));
        assert_eq!(run_program(&program, "main", &[]).unwrap(), NativeValue::String("hello".into()));
        let decoded = decode_program(&encode_program(&program)).unwrap();
        assert_eq!(decoded, program);
    }

    #[test]
    fn nested_value_depth_and_node_budgets_reject_adversarial_inputs() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec!["value".into()], code: vec![
                    NativeInstr::Load("value".into()),
                    NativeInstr::Return,
                ] },
            )]),
        };
        let depth_limited = ExecutionLimits {
            max_value_depth: 2,
            ..ExecutionLimits::default()
        };
        let nested = NativeValue::List(vec![NativeValue::List(vec![
            NativeValue::List(vec![NativeValue::Int(1)])
        ])]);
        assert!(matches!(
            run_program_with_limits(&program, "main", &[nested], depth_limited),
            Err(NativeError::ResourceLimit(message)) if message.contains("nesting depth")
        ));

        let node_limited = ExecutionLimits {
            max_value_nodes: 2,
            ..ExecutionLimits::default()
        };
        let broad = NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)]);
        assert!(matches!(
            run_program_with_limits(&program, "main", &[broad], node_limited),
            Err(NativeError::ResourceLimit(message)) if message.contains("node limit")
        ));
    }

    #[test]
    fn vm_frame_stack_and_local_limits_fail_closed() {
        let stack_program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![
                    NativeInstr::PushInt(1),
                    NativeInstr::PushInt(2),
                    NativeInstr::PushInt(3),
                    NativeInstr::Return,
                ] },
            )]),
        };
        assert!(matches!(
            run_program_with_limits(&stack_program, "main", &[], ExecutionLimits {
                max_stack_values: 1,
                ..ExecutionLimits::default()
            }),
            Err(NativeError::ResourceLimit(message)) if message.contains("operand stack")
        ));

        let local_program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![
                    NativeInstr::PushInt(1), NativeInstr::Store("a".into()),
                    NativeInstr::PushInt(2), NativeInstr::Store("b".into()),
                    NativeInstr::Return,
                ] },
            )]),
        };
        assert!(matches!(
            run_program_with_limits(&local_program, "main", &[], ExecutionLimits {
                max_locals: 1,
                ..ExecutionLimits::default()
            }),
            Err(NativeError::ResourceLimit(message)) if message.contains("local binding")
        ));
    }

    #[test]
    fn integer_overflow_returns_errors_instead_of_panicking() {
        let cases = [
            (NativeInstr::Add, i64::MAX, 1, "addition"),
            (NativeInstr::Sub, i64::MIN, 1, "subtraction"),
            (NativeInstr::Mul, i64::MAX, 2, "multiplication"),
            (NativeInstr::Div, i64::MIN, -1, "division"),
            (NativeInstr::Mod, i64::MIN, -1, "modulo"),
        ];
        for (operation, left, right, label) in cases {
            let program = NativeProgram {
                functions: BTreeMap::from([(
                    "main".into(),
                    NativeFunction {
                        params: vec![],
                        code: vec![
                            NativeInstr::PushInt(left),
                            NativeInstr::PushInt(right),
                            operation,
                            NativeInstr::Return,
                        ],
                    },
                )]),
            };
            let result = run_program(&program, "main", &[]);
            assert!(
                matches!(&result, Err(NativeError::Type(message)) if message.contains("overflow")),
                "expected explicit {label} overflow error, got {result:?}"
            );
        }
    }

    #[test]
    fn self_appending_integer_assignment_checks_overflow() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushInt(i64::MAX),
                        NativeInstr::Store("value".into()),
                        NativeInstr::PushInt(1),
                        NativeInstr::AddAssign("value".into()),
                        NativeInstr::Load("value".into()),
                        NativeInstr::Return,
                    ],
                },
            )]),
        };
        assert!(matches!(
            run_program(&program, "main", &[]),
            Err(NativeError::Type(message)) if message.contains("overflow")
        ));
    }

    #[test]
    fn shared_program_execution_reuses_immutable_program_and_returns_result() {
        let program = Arc::new(NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![NativeInstr::PushInt(7), NativeInstr::PushInt(8), NativeInstr::Add, NativeInstr::Return],
                },
            )]),
        });
        assert_eq!(
            run_shared_program_with_limits(
                Arc::clone(&program), "main", &[], ExecutionLimits::default()
            ),
            Ok(NativeValue::Int(15))
        );
        assert_eq!(
            run_shared_program_with_limits(
                program, "main", &[], ExecutionLimits::default()
            ),
            Ok(NativeValue::Int(15))
        );
    }

    #[test]
    fn validated_program_rejects_invalid_program_at_construction() {
        let invalid = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushString("x".repeat(MAX_PROGRAM_EMBEDDED_STRING_BYTES + 1)),
                        NativeInstr::Return,
                    ],
                },
            )]),
        };
        assert!(matches!(
            ValidatedNativeProgram::try_new(invalid),
            Err(NativeError::ResourceLimit(message)) if message.contains("embedded string byte")
        ));
    }

    #[test]
    fn validated_program_reuses_code_but_resets_budgets_and_checks_each_call() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec!["input".into()],
                    code: vec![NativeInstr::Load("input".into()), NativeInstr::Return],
                },
            )]),
        };
        let validated = ValidatedNativeProgram::try_new(program).unwrap();
        let limits = ExecutionLimits {
            max_instructions: 2,
            max_string_bytes: 4,
            ..ExecutionLimits::default()
        };

        assert_eq!(
            validated.run_with_limits("main", &[NativeValue::String("ok".into())], limits),
            Ok(NativeValue::String("ok".into()))
        );
        // A fresh execution budget is created for each call.
        assert_eq!(
            validated.run_with_limits("main", &[NativeValue::String("yes".into())], limits),
            Ok(NativeValue::String("yes".into()))
        );
        // Caller-specific arguments are still validated even though the program is trusted.
        assert!(matches!(
            validated.run_with_limits("main", &[NativeValue::String("too-long".into())], limits),
            Err(NativeError::ResourceLimit(_))
        ));
    }

    #[test]
    fn validated_program_rejects_unknown_entry_per_invocation() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction { params: vec![], code: vec![NativeInstr::PushUnit, NativeInstr::Return] },
            )]),
        };
        let validated = ValidatedNativeProgram::try_new(program).unwrap();
        assert!(matches!(
            validated.run_with_limits("missing", &[], ExecutionLimits::default()),
            Err(NativeError::InvalidProgram(message)) if message.contains("unknown function")
        ));
    }

    #[test]
    fn native_program_rejects_oversized_instruction_payload_before_execution() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushString("x".repeat(MAX_PROGRAM_EMBEDDED_STRING_BYTES + 1)),
                        NativeInstr::Return,
                    ],
                },
            )]),
        };
        assert!(matches!(
            run_program(&program, "main", &[]),
            Err(NativeError::ResourceLimit(message)) if message.contains("embedded string byte")
        ));
    }

    #[test]
    fn decoder_rejects_oversized_artifact_before_parsing_records() {
        let input = format!("{}\\n{}", ARTIFACT_MAGIC, "x".repeat(MAX_ARTIFACT_BYTES));
        assert!(matches!(
            decode_program(&input),
            Err(NativeError::ResourceLimit(message)) if message.contains("artifact byte")
        ));
    }

    #[test]
    fn native_program_rejects_excessive_function_count() {
        let functions = (0..=MAX_PROGRAM_FUNCTIONS)
            .map(|i| (format!("f{i}"), NativeFunction { params: vec![], code: vec![] }))
            .collect();
        let program = NativeProgram { functions };
        assert!(matches!(
            run_program(&program, "f0", &[]),
            Err(NativeError::ResourceLimit(message)) if message.contains("function limit")
        ));
    }


    #[test]
    fn borrowed_dispatch_preserves_string_and_nested_task_semantics() {
        let program = NativeProgram {
            functions: BTreeMap::from([
                ("worker".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::PushString("child payload".into()),
                        NativeInstr::Return,
                    ],
                }),
                ("main".into(), NativeFunction {
                    params: vec![],
                    code: vec![
                        NativeInstr::ScopeStart,
                        NativeInstr::PushString("local payload".into()),
                        NativeInstr::Store("message".into()),
                        NativeInstr::Load("message".into()),
                        NativeInstr::Pop,
                        NativeInstr::Spawn {
                            name: "child".into(),
                            callee: "worker".into(),
                            argc: 0,
                        },
                        NativeInstr::Join { name: "child".into() },
                        NativeInstr::ScopeEnd,
                        NativeInstr::PushString("dispatch-ok".into()),
                        NativeInstr::Return,
                    ],
                }),
            ]),
        };
        assert_eq!(
            run_program(&program, "main", &[]).unwrap(),
            NativeValue::String("dispatch-ok".into())
        );
    }

    #[test]
    fn aggregate_frame_value_budget_bounds_combined_locals() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec!["left".into(), "right".into()],
                    code: vec![NativeInstr::PushUnit, NativeInstr::Return],
                },
            )]),
        };
        let list = || NativeValue::List(vec![NativeValue::Int(1), NativeValue::Int(2)]);
        let limits = ExecutionLimits {
            max_value_nodes: 3,
            max_frame_value_nodes: 5,
            ..ExecutionLimits::default()
        };
        assert!(matches!(
            run_program_with_limits(&program, "main", &[list(), list()], limits),
            Err(NativeError::ResourceLimit(message)) if message.contains("aggregate frame value node")
        ));
    }

    #[test]
    fn aggregate_frame_string_budget_bounds_combined_strings() {
        let program = NativeProgram {
            functions: BTreeMap::from([(
                "main".into(),
                NativeFunction {
                    params: vec!["left".into(), "right".into()],
                    code: vec![NativeInstr::PushUnit, NativeInstr::Return],
                },
            )]),
        };
        let limits = ExecutionLimits {
            max_string_bytes: 8,
            max_frame_string_bytes: 10,
            ..ExecutionLimits::default()
        };
        assert!(matches!(
            run_program_with_limits(
                &program, "main",
                &[NativeValue::String("123456".into()), NativeValue::String("abcdef".into())],
                limits
            ),
            Err(NativeError::ResourceLimit(message)) if message.contains("aggregate frame string byte")
        ));
    }

    #[test]
    fn source_level_kv_cache_runs_through_native_artifact_roundtrip() {
        let source = "module kv_native\nfn main() -> Int\n  let cache = kv_cache(2)\n  kv_append(cache, [1, 2], [10], \"reject\")\n  kv_append(cache, [2, 3], [20], \"reject\")\n  kv_append(cache, [3, 4], [30], \"evict_oldest\")\n  kv_len(cache)\n";
        let module = crate::parse(source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).unwrap();
        assert_eq!(encode_program(&decoded), artifact);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Int(2));
    }

    #[test]
    fn source_level_context_window_evicts_by_byte_and_token_budgets() {
        let source = "module context_native\nfn main() -> Int\n  let window = context_window(3, 8, 5)\n  context_append(window, \"text\", \"hello\", 2)\n  context_append(window, \"image\", \"img\", 3)\n  context_append(window, \"audio\", \"abc\", 2)\n  context_tokens(window)\n";
        let module = crate::parse(source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let decoded = decode_program(&encode_program(&program)).unwrap();
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Int(5));
    }

    #[test]
    fn kv_cache_rejects_shape_mismatch_without_committing_mutation() {
        let source = "module kv_bad\nfn main() -> Int\n  let cache = kv_cache(2)\n  kv_append(cache, [1, 2], [10], \"reject\")\n  kv_append(cache, [3], [20], \"reject\")\n  kv_len(cache)\n";
        let module = crate::parse(source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("KeyDimension")));
    }

    #[test]
    fn native_runtime_rejects_invalid_kv_cache_capacity_and_context_modality() {
        let cache = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![NativeInstr::PushInt(0), NativeInstr::MakeKvCache, NativeInstr::Return] })]) };
        assert!(matches!(run_program(&cache, "main", &[]), Err(NativeError::Type(_))));
        let context = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![NativeInstr::PushInt(2), NativeInstr::PushInt(8), NativeInstr::PushInt(5), NativeInstr::MakeContextWindow, NativeInstr::Store("window".into()), NativeInstr::PushString("bogus".into()), NativeInstr::PushString("payload".into()), NativeInstr::PushInt(1), NativeInstr::ContextAppend { name: "window".into() }, NativeInstr::Return] })]) };
        assert!(matches!(run_program(&context, "main", &[]), Err(NativeError::Type(message)) if message.contains("modality")));
    }

}
    #[test]
    fn ai_value_footprints_include_retained_cache_and_context_payloads() {
        let cache = NativeValue::KvCache {
            capacity: 2,
            key_dim: Some(2),
            value_dim: Some(1),
            entries: vec![(vec![1.0f32.to_bits(), 2.0f32.to_bits()], vec![3.0f32.to_bits()])],
        };
        let (cache_nodes, cache_bytes) = value_footprint(&cache).unwrap();
        assert!(cache_nodes >= 6, "cache entries and scalars count toward frame nodes");
        assert_eq!(cache_bytes, 0);

        let context = NativeValue::ContextWindow {
            max_segments: 2,
            max_bytes: 64,
            max_tokens: 8,
            segments: vec![(String::from("image"), b"payload-bytes".to_vec(), 3)],
            bytes: 13,
            tokens: 3,
        };
        let (context_nodes, context_bytes) = value_footprint(&context).unwrap();
        assert!(context_nodes >= 2, "context segments count toward frame nodes");
        assert_eq!(context_bytes, "image".len() + "payload-bytes".len());
    }

    #[test]
    fn kv_cache_dimension_metadata_must_match_entries() {
        let invalid = NativeValue::KvCache {
            capacity: 2,
            key_dim: Some(3),
            value_dim: Some(1),
            entries: vec![(vec![1.0f32.to_bits(), 2.0f32.to_bits()], vec![3.0f32.to_bits()])],
        };
        assert!(matches!(validate_value(&invalid, ExecutionLimits::default()), Err(NativeError::InvalidProgram(_))));
    }


#[cfg(test)]
mod mcts_native_integration_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn source_mcts_compiles_runs_and_roundtrips_ardisa_exec_v1() {
        let source = "module mcts_native\nfn main() -> Int\n  let tree = mcts_tree(8, 3)\n  let root = mcts_root(tree, 10)\n  let child = mcts_expand(tree, root, 11)\n  mcts_backpropagate(tree, child, 0.75)\n  mcts_child(tree, root, 0)\n";
        let module = crate::parse(source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).unwrap();
        assert_eq!(encode_program(&decoded), artifact);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Int(1));
    }

    #[test]
    fn mcts_runtime_rejects_invalid_indices_bounds_nonfinite_rewards_and_bad_artifacts() {
        let source = "module mcts_bad\nfn main() -> Int\n  let tree = mcts_tree(2, 1)\n  let root = mcts_root(tree, 1)\n  mcts_expand(tree, 99, 2)\n  mcts_len(tree)\n";
        let module = crate::parse(source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("parent index")));

        let invalid = NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![NativeInstr::MakeMctsTree, NativeInstr::Return] })]) };
        let artifact = encode_program(&invalid).replace("MakeMctsTree", "MctsBogus");
        assert!(decode_program(&artifact).is_err());
    }

    #[test]
    fn mcts_native_mutation_is_atomic_on_reward_rejection() {
        let source = "module mcts_atomic\nfn main() -> Int\n  let tree = mcts_tree(4, 2)\n  let root = mcts_root(tree, 1)\n  mcts_update(tree, root, 1000001.0)\n  mcts_len(tree)\n";
        let module = crate::parse(source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("reward")));
    }
    #[test]
    fn mcts_native_enforces_node_and_depth_budgets() {
        let capacity_source = "module mcts_capacity\nfn main() -> Int\n  let tree = mcts_tree(2, 2)\n  let root = mcts_root(tree, 1)\n  let child = mcts_expand(tree, root, 2)\n  mcts_expand(tree, child, 3)\n  mcts_len(tree)\n";
        let capacity_module = crate::parse(capacity_source).unwrap();
        crate::sema::check(&capacity_module).unwrap();
        let capacity_program = compile_program(&crate::ir::lower(&capacity_module)).unwrap();
        assert!(matches!(run_program(&capacity_program, "main", &[]), Err(NativeError::ResourceLimit(message)) if message.contains("node limit")));

        let mut source = String::from("module mcts_depth\nfn main() -> Int\n  let tree = mcts_tree(300, 1)\n  let n0 = mcts_root(tree, 0)\n");
        for index in 1..=256 {
            source.push_str(&format!("  let n{index} = mcts_expand(tree, n{}, {index})\n", index - 1));
        }
        source.push_str("  mcts_len(tree)\n");
        let module = crate::parse(&source).unwrap();
        crate::sema::check(&module).unwrap();
        let program = compile_program(&crate::ir::lower(&module)).unwrap();
        let result = run_program(&program, "main", &[]); assert!(matches!(&result, Err(NativeError::ResourceLimit(message)) if message.contains("depth limit")), "unexpected MCTS depth result: {result:?}");
    }

}


#[cfg(test)]
mod attention_lora_source_tests {
    use super::*;

    fn compile_source(source: &str) -> NativeProgram {
        let module = crate::parse(source).expect("source parses");
        crate::sema::check(&module).expect("source passes semantic validation");
        compile_program(&crate::ir::lower(&module)).expect("source lowers to native code")
    }

    #[test]
    fn attention_block_is_deterministic_through_ardisa_exec_v1() {
        let source = r#"module attention_fixture
fn main() -> Tensor<F32>
  let q = tensor([1, 2], [1.0, 0.0])
  let k = tensor([2, 2], [1.0, 0.0, 0.0, 1.0])
  let v = tensor([2, 1], [2.0, 4.0])
  attention_block(q, k, v, false)
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("valid deterministic artifact");
        assert_eq!(encode_program(&decoded), artifact);
        let NativeValue::Tensor { shape, values } = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected tensor output"); };
        assert_eq!(shape, vec![1, 1]);
        let result = f32::from_bits(values[0]);
        assert!((result - 2.660476f32).abs() < 0.001, "unexpected attention result {result}");
    }

    #[test]
    fn attention_block_causal_mask_changes_early_token_context() {
        let causal = r#"module attention_causal
fn main() -> Tensor<F32>
  let q = tensor([2, 1], [1.0, 1.0])
  let k = tensor([2, 1], [0.0, 0.0])
  let v = tensor([2, 1], [2.0, 10.0])
  attention_block(q, k, v, true)
"#;
        let noncausal = causal.replace("attention_causal", "attention_noncausal").replace("attention_block(q, k, v, true)", "attention_block(q, k, v, false)");
        let causal_program = compile_source(causal);
        let noncausal_program = compile_source(&noncausal);
        let causal_artifact = encode_program(&causal_program);
        let decoded = decode_program(&causal_artifact).expect("causal artifact round-trips");
        let NativeValue::Tensor { shape, values } = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected causal tensor"); };
        assert_eq!(shape, vec![2, 1]);
        let causal_values = values.iter().map(|bits| f32::from_bits(*bits)).collect::<Vec<_>>();
        assert!((causal_values[0] - 2.0).abs() < 0.001, "first token must not see future values: {causal_values:?}");
        assert!((causal_values[1] - 6.0).abs() < 0.001, "second token sees both values: {causal_values:?}");
        let NativeValue::Tensor { values, .. } = run_program(&noncausal_program, "main", &[]).unwrap() else { panic!("expected noncausal tensor"); };
        let noncausal_values = values.iter().map(|bits| f32::from_bits(*bits)).collect::<Vec<_>>();
        assert!(noncausal_values.iter().all(|value| (*value - 6.0).abs() < 0.001), "noncausal rows should see both values: {noncausal_values:?}");
    }

    #[test]
    fn attention_cached_consumes_validated_kv_cache_entries() {
        let source = r#"module attention_cached_fixture
fn main() -> Tensor<F32>
  let cache = kv_cache(2)
  kv_append(cache, [1.0, 0.0], [2.0], "reject")
  kv_append(cache, [0.0, 1.0], [4.0], "reject")
  let q = tensor([1, 2], [1.0, 0.0])
  attention_cached(q, cache)
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).unwrap();
        assert_eq!(encode_program(&decoded), artifact);
        let NativeValue::Tensor { shape, values } = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected cached attention tensor"); };
        assert_eq!(shape, vec![1, 1]);
        assert!((f32::from_bits(values[0]) - 2.660476f32).abs() < 0.001);
    }

    #[test]
    fn lora_apply_computes_low_rank_delta_without_materializing_full_delta() {
        let source = r#"module lora_fixture
fn main() -> Tensor<F32>
  let input = tensor([1, 2], [1.0, 2.0])
  let a = tensor([1, 2], [1.0, 1.0])
  let b = tensor([2, 1], [2.0, 3.0])
  lora_apply(input, a, b, 2.0, 1)
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).unwrap();
        assert_eq!(encode_program(&decoded), artifact);
        let NativeValue::Tensor { shape, values } = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected tensor output"); };
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(values.iter().map(|bits| f32::from_bits(*bits)).collect::<Vec<_>>(), vec![12.0, 18.0]);
    }

    #[test]
    fn attention_and_lora_reject_shape_mismatches_and_nonfinite_results() {
        let bad_attention = r#"module bad_attention
fn main() -> Tensor<F32>
  let q = tensor([1, 2], [1.0, 0.0])
  let k = tensor([1, 3], [1.0, 0.0, 0.0])
  let v = tensor([1, 1], [1.0])
  attention_block(q, k, v, false)
"#;
        let program = compile_source(bad_attention);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("attention_block")));
        let bad_lora = r#"module bad_lora
fn main() -> Tensor<F32>
  let input = tensor([1, 2], [1.0, 2.0])
  let a = tensor([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  let b = tensor([1, 2], [1.0, 1.0])
  lora_apply(input, a, b, 1.0, 2)
"#;
        let program = compile_source(bad_lora);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(_))));
        let malformed = encode_program(&NativeProgram { functions: BTreeMap::from([("main".into(), NativeFunction { params: vec![], code: vec![NativeInstr::AttentionBlock, NativeInstr::Return] })]) }).replace("AttentionBlock", "AttentionBlockBogus");
        assert!(decode_program(&malformed).is_err());
    }

    #[test]
    fn nominal_attention_block_roundtrips_and_executes_multhead_source_to_runtime() {
        let source = r#"module nominal_attention
fn main() -> Tensor<F32>
  let block = attention_block_new(2, 2, 1, true)
  let q = tensor([2, 2], [0.0, 0.0, 0.0, 0.0])
  let k = tensor([2, 2], [0.0, 0.0, 0.0, 0.0])
  let v = tensor([2, 2], [2.0, 4.0, 10.0, 20.0])
  attention_block_apply(block, q, k, v)
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("nominal attention artifact decodes");
        assert_eq!(encode_program(&decoded), artifact);
        let NativeValue::Tensor { shape, values } = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected Tensor output"); };
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(values.iter().map(|bits| f32::from_bits(*bits)).collect::<Vec<_>>(), vec![2.0, 4.0, 6.0, 12.0]);
    }

    #[test]
    fn nominal_attention_cache_path_checks_configuration_and_cache_dimensions() {
        let source = r#"module nominal_attention_cache
fn main() -> Tensor<F32>
  let block = attention_block_new(2, 2, 1, true)
  let cache = kv_cache(2)
  kv_append(cache, [1.0, 0.0], [2.0, 4.0], "reject")
  let q = tensor([1, 2], [1.0, 0.0])
  attention_block_cached(block, q, cache)
"#;
        let program = compile_source(source);
        let NativeValue::Tensor { shape, values } = run_program(&decode_program(&encode_program(&program)).unwrap(), "main", &[]).unwrap() else { panic!("expected cached Tensor"); };
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(values.iter().map(|bits| f32::from_bits(*bits)).collect::<Vec<_>>(), vec![2.0, 4.0]);
        let bad = r#"module nominal_attention_bad
fn main(hidden: Int) -> AttentionBlock
  attention_block_new(hidden, 2, 1, false)
"#;
        let program = compile_source(bad);
        assert!(matches!(run_program(&program, "main", &[NativeValue::Int(3)]), Err(NativeError::Type(message)) if message.contains("hidden")));
    }

    #[test]
    fn nominal_lora_adapter_owns_validated_matrices_and_returns_delta_without_base_mutation() {
        let source = r#"module nominal_lora
fn main() -> Tensor<F32>
  let a = tensor([1, 2], [1.0, 1.0])
  let b = tensor([2, 1], [2.0, 3.0])
  let adapter = lora_adapter_new(a, b, 2.0, 1)
  let input = tensor([1, 2], [1.0, 2.0])
  lora_adapter_apply(adapter, input)
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("LoRA artifact decodes");
        assert_eq!(encode_program(&decoded), artifact);
        let NativeValue::Tensor { shape, values } = run_program(&decoded, "main", &[]).unwrap() else { panic!("expected Tensor output"); };
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(values.iter().map(|bits| f32::from_bits(*bits)).collect::<Vec<_>>(), vec![12.0, 18.0]);
    }

    #[test]
    fn lora_adapter_rejects_nonfinite_alpha_and_aggregate_storage_over_budget() {
        let small_a = vec![0.0f32.to_bits(); 2];
        let small_b = vec![0.0f32.to_bits(); 2];
        assert!(matches!(
            validate_lora_parts(&[1, 2], &small_a, &[2, 1], &small_b, f32::NAN.to_bits(), 1, MAX_TENSOR_ELEMENTS),
            Err(NativeError::Type(_))
        ));
        let a = vec![0.0f32.to_bits(); 10_000];
        let b = vec![0.0f32.to_bits(); 10_000];
        assert!(matches!(
            validate_lora_parts(&[100, 100], &a, &[100, 100], &b, 1.0f32.to_bits(), 100, MAX_TENSOR_ELEMENTS),
            Err(NativeError::ResourceLimit(message)) if message.contains("aggregate")
        ));
    }

    #[test]
    fn nominal_constructors_fail_closed_on_invalid_static_shapes_and_generic_forms() {
        let bad_attention = r#"module nominal_bad_attention
fn main() -> AttentionBlock
  attention_block_new(3, 2, 1, false)
"#;
        let module = crate::parse(bad_attention).unwrap();
        assert!(crate::sema::check(&module).is_err(), "hidden/head dimensions must be checked statically when literal");
        let bad_lora = r#"module nominal_bad_lora
fn main() -> LoRAAdapter
  let a = tensor([2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  let b = tensor([1, 2], [1.0, 1.0])
  lora_adapter_new(a, b, 1.0, 1)
"#;
        let program = compile_source(bad_lora);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(_))));
        let unsupported = crate::parse("module unsupported_nominal\nfn bad(x: AttentionBlock<Int, Int, Int>) -> Int\n  1\n").unwrap();
        assert!(crate::sema::check(&unsupported).is_err(), "unsupported compile-time shape forms must fail closed");
    }
}

#[cfg(test)]
mod moe_router_source_tests {
    use super::*;

    fn compile_source(source: &str) -> NativeProgram {
        let module = crate::parse(source).expect("source parses");
        crate::sema::check(&module).expect("source passes semantic validation");
        compile_program(&crate::ir::lower(&module)).expect("source lowers to native code")
    }

    #[test]
    fn nominal_moe_router_roundtrips_and_routes_ties_deterministically() {
        let source = r#"module nominal_moe_router
fn main() -> Int
  let router = moe_router(4, 2)
  let logits = [0.5, 2.0, 2.0, 0.0]
  let masked = [false, false, false, false]
  let routed = moe_route(router, logits, masked)
  routed[0][0]
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("MoERouter artifact decodes");
        assert_eq!(encode_program(&decoded), artifact);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Int(1));
    }

    #[test]
    fn nominal_moe_router_fails_closed_for_all_masked_and_mismatched_lengths() {
        let all_masked = r#"module moe_all_masked
fn main() -> Int
  let router = moe_router(3, 2)
  let logits = [1.0, 2.0, 3.0]
  let masked = [true, true, true]
  let routed = moe_route(router, logits, masked)
  routed[0][0]
"#;
        let program = compile_source(all_masked);
        assert!(matches!(run_program(&decode_program(&encode_program(&program)).unwrap(), "main", &[]), Err(NativeError::Type(message)) if message.contains("AllExpertsMasked")));

        let mismatched = r#"module moe_length_mismatch
fn main() -> Int
  let router = moe_router(3, 2)
  let logits = [1.0, 2.0]
  let masked = [false, false]
  let routed = moe_route(router, logits, masked)
  routed[0][0]
"#;
        let program = compile_source(mismatched);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("lengths")));
    }

    #[test]
    fn nominal_moe_router_rejects_invalid_literal_bounds_and_nonfinite_logits() {
        assert!(matches!(
            validate_value(&NativeValue::MoERouter { expert_count: 2, top_k: 3 }, ExecutionLimits::default()),
            Err(NativeError::Type(message)) if message.contains("configuration")
        ), "runtime validation must reject invalid nominal router arguments");
        let invalid = crate::parse(r#"module invalid_moe_router
fn main() -> MoERouter
  moe_router(2, 3)
"#).unwrap();
        assert!(crate::sema::check(&invalid).is_err(), "top_k cannot exceed expert_count");

        let source = r#"module moe_nonfinite
fn main() -> Int
  let router = moe_router(2, 1)
  let logits = [1.0, 2.0]
  let masked = [false, false]
  let routed = moe_route(router, logits, masked)
  routed[0][0]
"#;
        let mut program = compile_source(source);
        let function = program.functions.get_mut("main").unwrap();
        let index = function.code.iter().position(|instruction| matches!(instruction, NativeInstr::PushFloat(_))).unwrap();
        function.code[index] = NativeInstr::PushFloat(f32::INFINITY.to_bits());
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("finite")));
    }
}

#[cfg(test)]
mod vector_index_source_tests {
    use super::*;

    fn compile_source(source: &str) -> NativeProgram {
        let module = crate::parse(source).expect("source parses");
        crate::sema::check(&module).expect("source passes semantic validation");
        compile_program(&crate::ir::lower(&module)).expect("source lowers to native code")
    }

    #[test]
    fn nominal_vector_index_roundtrips_searches_and_breaks_distance_ties_by_id() {
        let source = r#"module nominal_vector_index
fn main() -> Int
  let index = vector_index("semantic", 2)
  vector_index_insert(index, 9, "semantic", [1.0, 0.0])
  vector_index_insert(index, 3, "semantic", [1.0, 0.0])
  vector_index_insert(index, 1, "semantic", [4.0, 0.0])
  let hits = vector_index_search(index, "semantic", [1.0, 0.0], 2)
  hits[0][0] * 10 + hits[1][0]
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("VectorIndex artifact decodes");
        assert_eq!(encode_program(&decoded), artifact);
        let malformed_artifact = artifact.replacen("MakeVectorIndex", "MakeVectorIndexBogus", 1);
        assert!(matches!(decode_program(&malformed_artifact), Err(NativeError::InvalidProgram(_))));
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Int(39));
    }

    #[test]
    fn nominal_vector_index_rejects_domain_dimension_and_duplicate_id_errors() {
        let domain_mismatch = r#"module vector_domain_mismatch
fn main() -> Int
  let index = vector_index("domain-a", 2)
  vector_index_insert(index, 1, "domain-a", [0.0, 1.0])
  let hits = vector_index_search(index, "domain-b", [0.0, 1.0], 1)
  hits[0][0]
"#;
        let program = compile_source(domain_mismatch);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("domain mismatch")));

        let dimension_mismatch = r#"module vector_dimension_mismatch
fn main() -> Int
  let index = vector_index("domain-a", 2)
  vector_index_insert(index, 1, "domain-a", [0.0])
  0
"#;
        let program = compile_source(dimension_mismatch);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("dimension mismatch")));

        let duplicate_id = r#"module vector_duplicate_id
fn main() -> Int
  let index = vector_index("domain-a", 1)
  vector_index_insert(index, 7, "domain-a", [0.0])
  vector_index_insert(index, 7, "domain-a", [1.0])
  0
"#;
        let program = compile_source(duplicate_id);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("duplicate VectorIndex id")));
    }

    #[test]
    fn nominal_vector_index_rejects_nonfinite_values_distance_overflow_and_malformed_state() {
        let source = r#"module vector_nonfinite
fn main() -> Int
  let index = vector_index("domain-a", 1)
  vector_index_insert(index, 1, "domain-a", [1.0])
  let hits = vector_index_search(index, "domain-a", [0.0], 1)
  hits[0][0]
"#;
        let mut program = compile_source(source);
        let function = program.functions.get_mut("main").unwrap();
        let index = function.code.iter().position(|instruction| matches!(instruction, NativeInstr::PushFloat(_))).unwrap();
        function.code[index] = NativeInstr::PushFloat(f32::INFINITY.to_bits());
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("finite")));

        let overflow = r#"module vector_distance_overflow
fn main() -> Int
  let index = vector_index("domain-a", 1)
  vector_index_insert(index, 1, "domain-a", [300000000000000000000000000000000000000.0])
  let hits = vector_index_search(index, "domain-a", [0.0], 1)
  hits[0][0]
"#;
        let program = compile_source(overflow);
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("F32 output range")));

        let malformed = NativeValue::VectorIndex { domain: "domain-a".into(), dimensions: 2, entries: vec![(1, vec![1.0f32.to_bits()])] };
        assert!(matches!(validate_value(&malformed, ExecutionLimits::default()), Err(NativeError::InvalidProgram(message)) if message.contains("dimension")));
        let duplicate = NativeValue::VectorIndex { domain: "domain-a".into(), dimensions: 1, entries: vec![(1, vec![0.0f32.to_bits()]), (1, vec![1.0f32.to_bits()])] };
        assert!(matches!(validate_value(&duplicate, ExecutionLimits::default()), Err(NativeError::InvalidProgram(message)) if message.contains("duplicate")));
        let oversized_id = NativeValue::VectorIndex { domain: "domain-a".into(), dimensions: 1, entries: vec![(u64::MAX, vec![0.0f32.to_bits()])] };
        assert!(matches!(validate_value(&oversized_id, ExecutionLimits::default()), Err(NativeError::InvalidProgram(message)) if message.contains("signed Int")));
        let limited = NativeValue::VectorIndex { domain: "domain-a".into(), dimensions: 2, entries: vec![(1, vec![0.0f32.to_bits(), 0.0f32.to_bits()]), (2, vec![1.0f32.to_bits(), 1.0f32.to_bits()])] };
        let limits = ExecutionLimits { max_value_nodes: 4, ..ExecutionLimits::default() };
        assert!(matches!(validate_value(&limited, limits), Err(NativeError::ResourceLimit(_))));
    }

    #[test]
    fn nominal_vector_index_empty_search_and_zero_limit_return_empty_results() {
        let source = r#"module vector_index_empty_results
fn main() -> Int
  let index = vector_index("empty-domain", 2)
  let empty_hits = vector_index_search(index, "empty-domain", [0.0, 0.0], 10)
  vector_index_insert(index, 5, "empty-domain", [0.0, 0.0])
  let zero_hits = vector_index_search(index, "empty-domain", [0.0, 0.0], 0)
  len(empty_hits) + len(zero_hits)
"#;
        let program = compile_source(source);
        assert_eq!(run_program(&decode_program(&encode_program(&program)).unwrap(), "main", &[]).unwrap(), NativeValue::Int(0));
    }

    #[test]
    fn nominal_moe_router_rejects_empty_logits_and_masks_at_runtime() {
        let source = r#"module moe_empty_logits
fn main() -> Int
  let router = moe_router(1, 1)
  let logits = [1.0]
  let masked = [false]
  let routed = moe_route(router, logits, masked)
  len(routed)
"#;
        let mut program = compile_source(source);
        let function = program.functions.get_mut("main").unwrap();
        let mut changed = 0;
        for instruction in &mut function.code {
            if matches!(instruction, NativeInstr::PushList(1)) && changed < 2 {
                *instruction = NativeInstr::PushList(0);
                changed += 1;
            }
        }
        assert_eq!(changed, 2, "fixture should contain the logits and mask list constructors");
        assert!(matches!(run_program(&program, "main", &[]), Err(NativeError::Type(message)) if message.contains("lengths")));
    }
}

#[cfg(test)]
mod gradient_tape_source_tests {
    use super::*;
    fn compile_source(source: &str) -> NativeProgram {
        let module=crate::parse(source).expect("source parses");
        crate::sema::check(&module).expect("GradientTape source passes semantic checks");
        compile_program(&crate::ir::lower(&module)).expect("GradientTape source lowers")
    }
    #[test]
    fn gradient_tape_forward_backward_and_artifact_roundtrip_are_deterministic() {
        let source=r#"module gradient_tape_chain
fn main() -> F32
  let tape = gradient_tape(8)
  let x = gradient_input(tape, 3.0)
  let y = gradient_input(tape, 4.0)
  let product = gradient_mul(tape, x, y)
  let output = gradient_add(tape, product, x)
  gradient_backward(tape, output, 1.0)
  gradient_get(tape, x)
"#;
        let program=compile_source(source);
        let artifact=encode_program(&program);
        let decoded=decode_program(&artifact).expect("artifact decodes");
        assert_eq!(encode_program(&decoded),artifact);
        assert_eq!(run_program(&decoded,"main",&[]).unwrap(),NativeValue::Float(5.0f32.to_bits()));
    }
    #[test]
    fn gradient_tape_repeated_operands_accumulate_and_values_are_observable() {
        let source=r#"module gradient_tape_square
fn main() -> F32
  let tape = gradient_tape(4)
  let x = gradient_input(tape, 3.0)
  let square = gradient_mul(tape, x, x)
  gradient_backward(tape, square, 1.0)
  gradient_get(tape, x)
"#;
        let program=compile_source(source);
        assert_eq!(run_program(&decode_program(&encode_program(&program)).unwrap(),"main",&[]).unwrap(),NativeValue::Float(6.0f32.to_bits()));
    }
    #[test]
    fn gradient_tape_rejects_forward_slot_refs_capacity_overflow_and_missing_backward() {
        let bad_ref=r#"module gradient_tape_bad_ref
fn main() -> Int
  let tape = gradient_tape(4)
  let x = gradient_input(tape, 2.0)
  gradient_add(tape, x, 9)
"#;
        let p=compile_source(bad_ref);
        assert!(matches!(run_program(&p,"main",&[]),Err(NativeError::Type(message)) if message.contains("exist before use")));
        let full=r#"module gradient_tape_full
fn main() -> Int
  let tape = gradient_tape(1)
  let x = gradient_input(tape, 2.0)
  gradient_input(tape, 3.0)
"#;
        let p=compile_source(full);
        assert!(matches!(run_program(&p,"main",&[]),Err(NativeError::ResourceLimit(message)) if message.contains("capacity")));
        let no_backward=r#"module gradient_tape_no_backward
fn main() -> F32
  let tape = gradient_tape(2)
  let x = gradient_input(tape, 2.0)
  gradient_get(tape, x)
"#;
        let p=compile_source(no_backward);
        assert!(matches!(run_program(&p,"main",&[]),Err(NativeError::Type(message)) if message.contains("backward")));
        let invalid=crate::parse("module invalid_gradient_tape\nfn main() -> GradientTape\n  gradient_tape(0)\n").unwrap();
        assert!(crate::sema::check(&invalid).is_err());
    }
    #[test]
    fn gradient_tape_rejects_malformed_artifacts_and_overflowing_intermediates() {
        let source=r#"module gradient_tape_overflow
fn main() -> Int
  let tape = gradient_tape(4)
  let x = gradient_input(tape, 2.0)
  let y = gradient_input(tape, 3.0)
  gradient_mul(tape, x, y)
"#;
        let mut program=compile_source(source);
        let code=&mut program.functions.get_mut("main").unwrap().code;
        let mut replaced=0;
        for op in code.iter_mut() {
            if matches!(op,NativeInstr::PushFloat(_)) {
                *op=NativeInstr::PushFloat(f32::MAX.to_bits()); replaced+=1;
            }
        }
        assert_eq!(replaced,2);
        assert!(matches!(run_program(&program,"main",&[]),Err(NativeError::Type(message)) if message.contains("intermediate")));

        let valid=compile_source(r#"module gradient_tape_malformed
fn main() -> Int
  let tape = gradient_tape(2)
  let x = gradient_input(tape, 1.0)
  x
"#);
        let malformed=encode_program(&valid).replace("GradientInput:", "GradientInputBogus:");
        assert!(matches!(decode_program(&malformed),Err(NativeError::InvalidProgram(_))));
        let invalid=NativeValue::GradientTape { capacity: 2, slots: vec![GradientSlot { value_bits: 1.0f32.to_bits(), op: GradientOp::Add { left: 0, right: 0 } }], gradients: vec![], backward_ready: false };
        assert!(matches!(validate_value(&invalid,ExecutionLimits::default()),Err(NativeError::InvalidProgram(message)) if message.contains("prior slots")));
    }
    #[test]
    fn gradient_tape_rejects_nonfinite_inputs_and_gradient_overflow() {
        let source=r#"module gradient_tape_nonfinite
fn main() -> F32
  let tape = gradient_tape(3)
  let x = gradient_input(tape, 2.0)
  gradient_backward(tape, x, 1.0)
  gradient_get(tape, x)
"#;
        let mut p=compile_source(source);
        let code=&mut p.functions.get_mut("main").unwrap().code;
        let index=code.iter().position(|op|matches!(op,NativeInstr::PushFloat(_))).unwrap();
        code[index]=NativeInstr::PushFloat(f32::INFINITY.to_bits());
        assert!(matches!(run_program(&p,"main",&[]),Err(NativeError::Type(message)) if message.contains("finite")));
    }
}

#[cfg(test)]
mod activation_cache_source_tests {
    use super::*;
    fn compile_source(source:&str)->NativeProgram {
        let module=crate::parse(source).expect("source parses");
        crate::sema::check(&module).expect("ActivationCache source passes semantic checks");
        compile_program(&crate::ir::lower(&module)).expect("source lowers to native code")
    }
    #[test]
    fn activation_cache_roundtrips_and_restores_exact_checkpoint_shape_and_values() {
        let source=r#"module activation_cache_roundtrip
fn main() -> Tensor<F32>
  let cache = activation_cache(3, 2, 1024)
  activation_store(cache, 0, tensor([1, 2], [1.0, 2.0]))
  activation_store(cache, 2, tensor([2, 1], [3.0, 4.0]))
  activation_restore(cache, 2)
"#;
        let program=compile_source(source);let artifact=encode_program(&program);let decoded=decode_program(&artifact).expect("artifact decodes");
        assert_eq!(encode_program(&decoded),artifact);
        assert_eq!(run_program(&decoded,"main",&[]).unwrap(),NativeValue::Tensor{shape:vec![2,1],values:vec![3.0f32.to_bits(),4.0f32.to_bits()]});
    }
    #[test]
    fn activation_cache_rejects_duplicate_missing_non_stride_and_capacity_overflow() {
        let duplicate=r#"module activation_duplicate
fn main() -> Int
  let cache = activation_cache(3, 2, 1024)
  activation_store(cache, 0, tensor([1], [1.0]))
  activation_store(cache, 0, tensor([1], [2.0]))
  activation_cache_len(cache)
"#;
        assert!(matches!(run_program(&compile_source(duplicate),"main",&[]),Err(NativeError::Type(message)) if message.contains("duplicate")));
        let missing=r#"module activation_missing
fn main() -> Tensor<F32>
  let cache = activation_cache(2, 1, 1024)
  activation_restore(cache, 7)
"#;
        assert!(matches!(run_program(&compile_source(missing),"main",&[]),Err(NativeError::Type(message)) if message.contains("not found")));
        let stride=r#"module activation_stride
fn main() -> Int
  let cache = activation_cache(2, 2, 1024)
  activation_store(cache, 1, tensor([1], [1.0]))
  0
"#;
        assert!(matches!(run_program(&compile_source(stride),"main",&[]),Err(NativeError::Type(message)) if message.contains("checkpoint_stride")));
        let full=r#"module activation_full
fn main() -> Int
  let cache = activation_cache(1, 2, 1024)
  activation_store(cache, 0, tensor([1], [1.0]))
  activation_store(cache, 2, tensor([1], [2.0]))
  0
"#;
        assert!(matches!(run_program(&compile_source(full),"main",&[]),Err(NativeError::ResourceLimit(message)) if message.contains("capacity")));
    }
    #[test]
    fn activation_cache_enforces_retained_byte_budget_and_static_limits() {
        let budget=r#"module activation_budget
fn main() -> Int
  let cache = activation_cache(2, 1, 8)
  activation_store(cache, 0, tensor([1, 2], [1.0, 2.0]))
  0
"#;
        assert!(matches!(run_program(&compile_source(budget),"main",&[]),Err(NativeError::ResourceLimit(message)) if message.contains("byte budget")));
        let invalid=crate::parse("module invalid_activation_cache\nfn main() -> ActivationCache\n  activation_cache(0, 1, 100)\n").unwrap();
        assert!(crate::sema::check(&invalid).is_err());
        let malformed=NativeValue::ActivationCache{capacity:2,checkpoint_stride:1,max_bytes:1024,retained_bytes:0,entries:vec![(0,vec![1],vec![1.0f32.to_bits()])]};
        assert!(matches!(validate_value(&malformed,ExecutionLimits::default()),Err(NativeError::InvalidProgram(message)) if message.contains("accounting")));
    }

    #[test]
    fn tokenizer_state_handles_split_utf8_incrementally_and_roundtrips_deterministically() {
        let source = r#"module tokenizer_split_utf8
fn main() -> Int
  let state = tokenizer_state(["a", "é"], 4, 8, 0)
  tokenizer_feed(state, [195])
  let before = tokenizer_token_count(state)
  tokenizer_feed(state, [169])
  tokenizer_flush(state)
  tokenizer_token(state, 0)
"#;
        let program = compile_source(source);
        let artifact = encode_program(&program);
        let decoded = decode_program(&artifact).expect("TokenizerState artifact decodes");
        assert_eq!(encode_program(&decoded), artifact);
        assert_eq!(run_program(&decoded, "main", &[]).unwrap(), NativeValue::Int(1));
    }

    #[test]
    fn tokenizer_state_rejects_invalid_utf8_unknown_scalars_and_incomplete_flush() {
        let invalid = compile_source(r#"module tokenizer_invalid
fn main() -> Int
  let state = tokenizer_state(["a"], 4, 8, 0)
  tokenizer_feed(state, [255])
  tokenizer_token_count(state)
"#);
        assert!(matches!(run_program(&invalid, "main", &[]), Err(NativeError::Type(message)) if message.contains("invalid UTF-8")));
        let unknown = compile_source(r#"module tokenizer_unknown
fn main() -> Int
  let state = tokenizer_state(["a"], 4, 8, 0)
  tokenizer_feed(state, [98])
  tokenizer_token_count(state)
"#);
        assert!(matches!(run_program(&unknown, "main", &[]), Err(NativeError::Type(message)) if message.contains("unknown vocabulary")));
        let incomplete = compile_source(r#"module tokenizer_incomplete
fn main() -> Int
  let state = tokenizer_state(["a", "é"], 4, 8, 0)
  tokenizer_feed(state, [195])
  tokenizer_flush(state)
  tokenizer_token_count(state)
"#);
        assert!(matches!(run_program(&incomplete, "main", &[]), Err(NativeError::Type(message)) if message.contains("incomplete UTF-8")));
    }

    #[test]
    fn tokenizer_state_unknown_fallback_and_token_budget_are_deterministic() {
        let fallback = compile_source(r#"module tokenizer_fallback
fn main() -> Int
  let state = tokenizer_state(["a"], 4, 8, 100)
  tokenizer_feed(state, [98])
  tokenizer_token(state, 0)
"#);
        assert_eq!(run_program(&decode_program(&encode_program(&fallback)).unwrap(), "main", &[]).unwrap(), NativeValue::Int(99));
        let full = compile_source(r#"module tokenizer_full
fn main() -> Int
  let state = tokenizer_state(["a"], 4, 1, 0)
  tokenizer_feed(state, [97, 97])
  tokenizer_token_count(state)
"#);
        assert!(matches!(run_program(&full, "main", &[]), Err(NativeError::ResourceLimit(message)) if message.contains("capacity exhausted")));
    }

    #[test]
    fn tokenizer_state_rejects_duplicate_or_multiscalar_vocabulary_entries() {
        let duplicate = crate::parse(r#"module tokenizer_duplicate
fn main() -> TokenizerState
  tokenizer_state(["a", "a"], 4, 8, 0)
"#).unwrap();
        assert!(crate::sema::check(&duplicate).is_err());
        let multi = crate::parse(r#"module tokenizer_multiscalar
fn main() -> TokenizerState
  tokenizer_state(["ab"], 4, 8, 0)
"#).unwrap();
        assert!(crate::sema::check(&multi).is_err());
    }
}

#[cfg(test)]
mod activation_cache_adversarial_hardening_tests {
    use super::*;
    #[test]
    fn activation_cache_mutations_are_atomic_for_duplicate_budget_and_nonfinite_failures() {
        let initial=NativeValue::ActivationCache{capacity:2,checkpoint_stride:1,max_bytes:12,retained_bytes:12,entries:vec![(0,vec![1],vec![1.0f32.to_bits()])]};
        let mut cache=initial.clone();
        assert!(matches!(store_activation_checkpoint(&mut cache,0,vec![1],vec![2.0f32.to_bits()]),Err(NativeError::Type(message)) if message.contains("duplicate")));
        assert_eq!(cache,initial,"duplicate rejection must leave checkpoint state unchanged");
        assert!(matches!(store_activation_checkpoint(&mut cache,1,vec![1],vec![2.0f32.to_bits()]),Err(NativeError::ResourceLimit(message)) if message.contains("byte budget")));
        assert_eq!(cache,initial,"budget rejection must leave checkpoint state unchanged");
        assert!(matches!(store_activation_checkpoint(&mut cache,1,vec![1],vec![f32::INFINITY.to_bits()]),Err(NativeError::InvalidProgram(message)) if message.contains("finite")));
        assert_eq!(cache,initial,"non-finite rejection must leave checkpoint state unchanged");
        assert!(matches!(store_activation_checkpoint(&mut cache,1,vec![2],vec![2.0f32.to_bits()]),Err(NativeError::InvalidProgram(_))));
        assert_eq!(cache,initial,"shape rejection must leave checkpoint state unchanged");
    }
    #[test]
    fn activation_cache_validator_rejects_duplicate_ids_and_nonfinite_retained_payloads() {
        let duplicate=NativeValue::ActivationCache{capacity:2,checkpoint_stride:1,max_bytes:1024,retained_bytes:24,entries:vec![(0,vec![1],vec![1.0f32.to_bits()]),(0,vec![1],vec![2.0f32.to_bits()])]};
        assert!(matches!(validate_value(&duplicate,ExecutionLimits::default()),Err(NativeError::InvalidProgram(message)) if message.contains("duplicate")));
        let nonfinite=NativeValue::ActivationCache{capacity:2,checkpoint_stride:1,max_bytes:1024,retained_bytes:12,entries:vec![(0,vec![1],vec![f32::NAN.to_bits()])]};
        assert!(matches!(validate_value(&nonfinite,ExecutionLimits::default()),Err(NativeError::InvalidProgram(message)) if message.contains("finite")));
        let shape_mismatch=NativeValue::ActivationCache{capacity:2,checkpoint_stride:1,max_bytes:1024,retained_bytes:12,entries:vec![(0,vec![2],vec![1.0f32.to_bits()])]};
        assert!(matches!(validate_value(&shape_mismatch,ExecutionLimits::default()),Err(NativeError::InvalidProgram(_))));
        let program=NativeProgram{functions:BTreeMap::from([("main".into(),NativeFunction{params:vec![],code:vec![NativeInstr::PushInt(2),NativeInstr::PushInt(1),NativeInstr::PushInt(1024),NativeInstr::MakeActivationCache,NativeInstr::Return]})])};
        let malformed=encode_program(&program).replace("MakeActivationCache","MakeActivationCacheBogus");
        assert!(matches!(decode_program(&malformed),Err(NativeError::InvalidProgram(_))));
    }

    #[test]
    fn source_numeric_boundary_contract_survives_artifact_round_trip() {
        let cases = [
            (
                "module numeric_add_overflow\\nfn main() -> Int\\n  9223372036854775807 + 1\\n",
                "overflow",
            ),
            (
                "module numeric_div_zero\\nfn main() -> Int\\n  5 / 0\\n",
                "division by zero",
            ),
            (
                "module numeric_mod_zero\\nfn main() -> Int\\n  5 % 0\\n",
                "modulo by zero",
            ),
        ];

        for (source, expected_error) in cases {
            let program = compile_source(source);
            let artifact = encode_program(&program);
            let decoded = decode_program(&artifact).expect("numeric fixture artifact decodes");
            assert_eq!(encode_program(&decoded), artifact, "artifact must round-trip deterministically");
            let result = run_program(&decoded, "main", &[]);
            assert!(
                matches!(result, Err(NativeError::Type(message)) if message.contains(expected_error)),
                "expected {expected_error:?} for source fixture, got {result:?}"
            );
        }
    }

}
