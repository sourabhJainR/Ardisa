//! Runtime foundations for uncertainty-aware AI values.
//! These are runtime APIs, not yet first-class Ardisa source-language types.
//! They avoid hidden model calls, implicit repair, and unsubstantiated GPU guarantees.

use std::collections::{BTreeMap, BTreeSet};
use crate::ai::{Probability, RuntimeError, Tensor};

#[derive(Debug, Clone, PartialEq)]
pub struct Alternative<T> {
    pub value: T,
    pub probability: Probability,
}

/// A selected value plus explicit uncertainty metadata. Callers must choose
/// the confidence threshold appropriate to the operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Probabilistic<T> {
    value: T,
    confidence: Probability,
    alternatives: Vec<Alternative<T>>,
}

impl<T> Probabilistic<T> {
    pub fn new(value: T, confidence: Probability, alternatives: Vec<Alternative<T>>) -> Self {
        Self { value, confidence, alternatives }
    }
    pub fn value(&self) -> &T { &self.value }
    pub fn confidence(&self) -> Probability { self.confidence }
    pub fn alternatives(&self) -> &[Alternative<T>] { &self.alternatives }
    pub fn meets_threshold(&self, threshold: Probability) -> bool {
        self.confidence.value() >= threshold.value()
    }
    pub fn into_value(self) -> T { self.value }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphTensorNode {
    pub id: String,
    pub label: String,
    pub tensor_row: usize,
    pub confidence: Option<Probability>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphTensorEdge {
    pub from: String,
    pub relation: String,
    pub to: String,
}

/// Tensor storage plus referentially validated semantic graph metadata.
/// This does not itself provide a vector database, GPU allocator, or model.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphTensor {
    tensor: Tensor,
    nodes: Vec<GraphTensorNode>,
    edges: Vec<GraphTensorEdge>,
}

impl GraphTensor {
    pub fn new(tensor: Tensor, nodes: Vec<GraphTensorNode>, edges: Vec<GraphTensorEdge>) -> Result<Self, RuntimeError> {
        let row_count = tensor.shape().first().copied().ok_or_else(|| RuntimeError::new("AIT001", "graph-tensor requires a leading tensor axis"))?;
        let mut ids = BTreeSet::new();
        for node in &nodes {
            if !ids.insert(node.id.as_str()) {
                return Err(RuntimeError::new("AIT002", format!("duplicate graph node id '{}'", node.id)));
            }
            if node.tensor_row >= row_count {
                return Err(RuntimeError::new("AIT003", format!("tensor row {} for node '{}' is out of bounds", node.tensor_row, node.id)));
            }
        }
        for edge in &edges {
            if !ids.contains(edge.from.as_str()) || !ids.contains(edge.to.as_str()) {
                return Err(RuntimeError::new("AIT004", format!("edge '{} -> {}' references an unknown node", edge.from, edge.to)));
            }
        }
        Ok(Self { tensor, nodes, edges })
    }
    pub fn tensor(&self) -> &Tensor { &self.tensor }
    pub fn nodes(&self) -> &[GraphTensorNode] { &self.nodes }
    pub fn edges(&self) -> &[GraphTensorEdge] { &self.edges }

    /// Merge metadata only; tensor layout/arithmetic remains explicit because
    /// the two tensor shapes may differ. Conflicts require a chosen policy.
    pub fn merge_metadata(&self, other: &Self, policy: GraphConflictPolicy) -> Result<(Vec<GraphTensorNode>, Vec<GraphTensorEdge>), RuntimeError> {
        let mut nodes: BTreeMap<String, GraphTensorNode> = self.nodes.iter().cloned().map(|n| (n.id.clone(), n)).collect();
        for incoming in &other.nodes {
            match nodes.get(&incoming.id) {
                None => { nodes.insert(incoming.id.clone(), incoming.clone()); }
                Some(existing) if existing == incoming => {}
                Some(_) => match policy {
                    GraphConflictPolicy::Reject => return Err(RuntimeError::new("AIT005", format!("conflicting graph node '{}'", incoming.id))),
                    GraphConflictPolicy::PreferLeft => {}
                    GraphConflictPolicy::PreferHigherConfidence => {
                        let old_conf = nodes.get(&incoming.id).and_then(|n| n.confidence).map(Probability::value).unwrap_or(0.0);
                        let new_conf = incoming.confidence.map(Probability::value).unwrap_or(0.0);
                        if new_conf > old_conf { nodes.insert(incoming.id.clone(), incoming.clone()); }
                    }
                }
            }
        }
        let mut edges: BTreeSet<GraphTensorEdge> = self.edges.iter().cloned().collect();
        edges.extend(other.edges.iter().cloned());
        for edge in &edges {
            if !nodes.contains_key(&edge.from) || !nodes.contains_key(&edge.to) {
                return Err(RuntimeError::new("AIT004", "merged edge references an unknown node"));
            }
        }
        Ok((nodes.into_values().collect(), edges.into_iter().collect()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphConflictPolicy { Reject, PreferLeft, PreferHigherConfidence }

/// A value admitted only after caller-supplied structural validation.
/// Repair is opt-in, bounded, and every candidate is validated before return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guaranteed<T> { value: T }

impl<T> Guaranteed<T> {
    pub fn try_new(value: T, validate: impl FnOnce(&T) -> Result<(), RuntimeError>) -> Result<Self, RuntimeError> {
        validate(&value)?;
        Ok(Self { value })
    }
    pub fn as_ref(&self) -> &T { &self.value }
    pub fn into_inner(self) -> T { self.value }

    /// Try the initial value, then at most max_attempts repaired candidates.
    /// The repair callback is supplied by the application; no AI is invoked here.
    pub fn repair_with(mut candidate: T, max_attempts: usize, mut validate: impl FnMut(&T) -> Result<(), RuntimeError>, mut repair: impl FnMut(&T, usize) -> Result<T, RuntimeError>) -> Result<Self, RuntimeError> {
        if validate(&candidate).is_ok() { return Ok(Self { value: candidate }); }
        for attempt in 1..=max_attempts {
            candidate = repair(&candidate, attempt)?;
            if validate(&candidate).is_ok() { return Ok(Self { value: candidate }); }
        }
        Err(RuntimeError::new("AIT006", format!("payload did not satisfy its schema after {} repair attempt(s)", max_attempts)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{Quantization, Tensor};
    fn p(v: f32) -> Probability { Probability::new(v).expect("valid probability") }

    #[test]
    fn probabilistic_requires_explicit_threshold_check() {
        let value = Probabilistic::new(true, p(0.7), vec![]);
        assert!(!value.meets_threshold(p(0.85)));
        assert!(value.meets_threshold(p(0.6)));
    }

    #[test]
    fn graph_tensor_checks_rows_and_edge_references() {
        let tensor = Tensor::new(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4], Quantization::Fp32).unwrap();
        let nodes = vec![GraphTensorNode { id: "intent".into(), label: "User Intent".into(), tensor_row: 0, confidence: Some(p(0.9)) }];
        let bad_edge = GraphTensorEdge { from: "intent".into(), relation: "relates_to".into(), to: "missing".into() };
        assert_eq!(GraphTensor::new(tensor.clone(), nodes.clone(), vec![bad_edge]).unwrap_err().code, "AIT004");
        let bad_node = GraphTensorNode { id: "overflow".into(), label: "Overflow".into(), tensor_row: 2, confidence: None };
        assert_eq!(GraphTensor::new(tensor, vec![bad_node], vec![]).unwrap_err().code, "AIT003");
    }

    #[test]
    fn graph_merge_requires_explicit_conflict_policy() {
        let tensor = Tensor::new(vec![1, 1], vec![1.0], Quantization::Fp32).unwrap();
        let left = GraphTensor::new(tensor.clone(), vec![GraphTensorNode { id: "x".into(), label: "left".into(), tensor_row: 0, confidence: Some(p(0.6)) }], vec![]).unwrap();
        let right = GraphTensor::new(tensor, vec![GraphTensorNode { id: "x".into(), label: "right".into(), tensor_row: 0, confidence: Some(p(0.9)) }], vec![]).unwrap();
        assert!(left.merge_metadata(&right, GraphConflictPolicy::Reject).is_err());
        let (nodes, _) = left.merge_metadata(&right, GraphConflictPolicy::PreferHigherConfidence).unwrap();
        assert_eq!(nodes[0].label, "right");
    }

    #[test]
    fn guaranteed_repairs_are_bounded_and_revalidated() {
        let valid = |v: &i32| if *v >= 0 { Ok(()) } else { Err(RuntimeError::new("SCHEMA", "must be nonnegative")) };
        let value = Guaranteed::repair_with(-1, 2, valid, |_, _| Ok(7)).unwrap();
        assert_eq!(*value.as_ref(), 7);
        let failed = Guaranteed::repair_with(-1, 1, valid, |_, _| Ok(-2));
        assert_eq!(failed.unwrap_err().code, "AIT006");
    }
}


/// Hard ceiling for one tape so callers cannot request unbounded operation storage.
const MAX_GRADIENT_TAPE_NODES: usize = 1_000_000;
/// Hard ceiling for retained activation values across all checkpoints.
const MAX_ACTIVATION_CACHE_SCALARS: usize = 1_048_576;
const MAX_ACTIVATION_CHECKPOINTS: usize = 16_384;

#[derive(Debug, Clone, PartialEq)]
enum GradientOp {
    Input,
    Constant,
    Add(usize, usize),
    Multiply(usize, usize),
    Negate(usize),
}

#[derive(Debug, Clone, PartialEq)]
struct GradientNode {
    value: f32,
    gradient: f32,
    op: GradientOp,
}

/// A bounded reverse-mode tape for deterministic scalar arithmetic.
///
/// This is a runtime helper, not a first-class Ardisa source type. Only input,
/// constant, addition, multiplication and negation are supported. The tape
/// rejects non-finite values and never calls a model or an external backend.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientTape {
    nodes: Vec<GradientNode>,
    max_nodes: usize,
}

impl GradientTape {
    pub fn new(max_nodes: usize) -> Result<Self, RuntimeError> {
        if max_nodes == 0 || max_nodes > MAX_GRADIENT_TAPE_NODES {
            return Err(RuntimeError::new("AIT010", format!(
                "gradient tape capacity must be between 1 and {}", MAX_GRADIENT_TAPE_NODES
            )));
        }
        Ok(Self { nodes: Vec::new(), max_nodes })
    }

    pub fn input(&mut self, value: f32) -> Result<usize, RuntimeError> {
        self.push(value, GradientOp::Input)
    }

    pub fn constant(&mut self, value: f32) -> Result<usize, RuntimeError> {
        self.push(value, GradientOp::Constant)
    }

    pub fn add(&mut self, left: usize, right: usize) -> Result<usize, RuntimeError> {
        let value = self.prior_value(left)? + self.prior_value(right)?;
        self.push(value, GradientOp::Add(left, right))
    }

    pub fn multiply(&mut self, left: usize, right: usize) -> Result<usize, RuntimeError> {
        let value = self.prior_value(left)? * self.prior_value(right)?;
        self.push(value, GradientOp::Multiply(left, right))
    }

    pub fn negate(&mut self, input: usize) -> Result<usize, RuntimeError> {
        let value = -self.prior_value(input)?;
        self.push(value, GradientOp::Negate(input))
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    pub fn value(&self, slot: usize) -> Result<f32, RuntimeError> {
        self.nodes.get(slot).map(|node| node.value)
            .ok_or_else(|| RuntimeError::new("AIT011", format!("gradient tape slot {} is out of bounds", slot)))
    }

    pub fn gradient(&self, slot: usize) -> Result<f32, RuntimeError> {
        self.nodes.get(slot).map(|node| node.gradient)
            .ok_or_else(|| RuntimeError::new("AIT011", format!("gradient tape slot {} is out of bounds", slot)))
    }

    /// Compute gradients for a scalar output. Gradients are committed only if
    /// every intermediate result remains finite.
    pub fn backward(&mut self, output: usize) -> Result<(), RuntimeError> {
        if output >= self.nodes.len() {
            return Err(RuntimeError::new("AIT011", format!("gradient tape slot {} is out of bounds", output)));
        }
        let mut gradients = vec![0.0f32; self.nodes.len()];
        gradients[output] = 1.0;
        for index in (0..self.nodes.len()).rev() {
            let gradient = gradients[index];
            let node = &self.nodes[index];
            match &node.op {
                GradientOp::Input | GradientOp::Constant => {}
                GradientOp::Add(left, right) => {
                    gradients[*left] += gradient;
                    gradients[*right] += gradient;
                }
                GradientOp::Multiply(left, right) => {
                    gradients[*left] += gradient * self.nodes[*right].value;
                    gradients[*right] += gradient * self.nodes[*left].value;
                }
                GradientOp::Negate(input) => gradients[*input] -= gradient,
            }
        }
        if gradients.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::new("AIT012", "gradient computation produced a non-finite value"));
        }
        for (node, gradient) in self.nodes.iter_mut().zip(gradients) {
            node.gradient = gradient;
        }
        Ok(())
    }

    fn prior_value(&self, slot: usize) -> Result<f32, RuntimeError> {
        if slot >= self.nodes.len() {
            return Err(RuntimeError::new("AIT011", format!("gradient tape slot {} must reference an existing operation", slot)));
        }
        Ok(self.nodes[slot].value)
    }

    fn push(&mut self, value: f32, op: GradientOp) -> Result<usize, RuntimeError> {
        if self.nodes.len() >= self.max_nodes {
            return Err(RuntimeError::new("AIT013", "gradient tape capacity exceeded"));
        }
        if !value.is_finite() {
            return Err(RuntimeError::new("AIT012", "gradient tape operation produced a non-finite value"));
        }
        let slot = self.nodes.len();
        self.nodes.push(GradientNode { value, gradient: 0.0, op });
        Ok(slot)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ActivationCheckpoint {
    step: u64,
    activations: Vec<f32>,
}

/// A bounded checkpoint store for deterministic activation recomputation.
///
/// The cache stores checkpoints only at the configured stride. It does not
/// implement model execution or recomputation; callers supply activations and
/// remain responsible for proving recompute equivalence for their workload.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivationCache {
    stride: u64,
    max_checkpoints: usize,
    max_retained_scalars: usize,
    retained_scalars: usize,
    shape: Option<usize>,
    last_observed_step: Option<u64>,
    checkpoints: Vec<ActivationCheckpoint>,
}

impl ActivationCache {
    pub fn new(stride: u64, max_checkpoints: usize, max_retained_scalars: usize) -> Result<Self, RuntimeError> {
        if stride == 0 {
            return Err(RuntimeError::new("AIT020", "activation checkpoint stride must be positive"));
        }
        if max_checkpoints == 0 || max_checkpoints > MAX_ACTIVATION_CHECKPOINTS {
            return Err(RuntimeError::new("AIT020", format!(
                "activation checkpoint capacity must be between 1 and {}", MAX_ACTIVATION_CHECKPOINTS
            )));
        }
        if max_retained_scalars == 0 || max_retained_scalars > MAX_ACTIVATION_CACHE_SCALARS {
            return Err(RuntimeError::new("AIT020", format!(
                "activation scalar budget must be between 1 and {}", MAX_ACTIVATION_CACHE_SCALARS
            )));
        }
        Ok(Self {
            stride, max_checkpoints, max_retained_scalars, retained_scalars: 0,
            shape: None, last_observed_step: None, checkpoints: Vec::new(),
        })
    }

    /// Record an observation. Returns true when a checkpoint was retained and
    /// false when this step falls between checkpoint boundaries.
    pub fn record(&mut self, step: u64, activations: &[f32]) -> Result<bool, RuntimeError> {
        if self.last_observed_step.map(|last| step <= last).unwrap_or(false) {
            return Err(RuntimeError::new("AIT021", "activation steps must be strictly increasing"));
        }
        if activations.is_empty() || activations.iter().any(|value| !value.is_finite()) {
            return Err(RuntimeError::new("AIT022", "activation values must be non-empty and finite"));
        }
        if let Some(shape) = self.shape {
            if activations.len() != shape {
                return Err(RuntimeError::new("AIT023", format!(
                    "activation shape changed from {} to {}", shape, activations.len()
                )));
            }
        }
        let checkpoint = step % self.stride == 0;
        if checkpoint {
            if self.checkpoints.len() >= self.max_checkpoints {
                return Err(RuntimeError::new("AIT024", "activation checkpoint capacity exceeded"));
            }
            let retained = self.retained_scalars.checked_add(activations.len())
                .ok_or_else(|| RuntimeError::new("AIT024", "activation scalar budget overflow"))?;
            if retained > self.max_retained_scalars {
                return Err(RuntimeError::new("AIT024", "activation scalar budget exceeded"));
            }
            self.retained_scalars = retained;
            self.checkpoints.push(ActivationCheckpoint { step, activations: activations.to_vec() });
        }
        self.shape.get_or_insert(activations.len());
        self.last_observed_step = Some(step);
        Ok(checkpoint)
    }

    pub fn get(&self, step: u64) -> Option<&[f32]> {
        self.checkpoints.iter().find(|checkpoint| checkpoint.step == step)
            .map(|checkpoint| checkpoint.activations.as_slice())
    }

    pub fn checkpoint_steps(&self) -> Vec<u64> {
        self.checkpoints.iter().map(|checkpoint| checkpoint.step).collect()
    }

    pub fn retained_scalars(&self) -> usize { self.retained_scalars }
    pub fn len(&self) -> usize { self.checkpoints.len() }
    pub fn is_empty(&self) -> bool { self.checkpoints.is_empty() }
}

#[cfg(test)]
mod bounded_ai_tape_tests {
    use super::*;

    #[test]
    fn gradient_tape_computes_deterministic_scalar_gradients() {
        let mut tape = GradientTape::new(16).unwrap();
        let x = tape.input(3.0).unwrap();
        let two = tape.constant(2.0).unwrap();
        let square = tape.multiply(x, x).unwrap();
        let output = tape.add(square, two).unwrap();
        assert_eq!(tape.value(output).unwrap(), 11.0);
        tape.backward(output).unwrap();
        assert_eq!(tape.gradient(x).unwrap(), 6.0);
        assert_eq!(tape.gradient(two).unwrap(), 1.0);
    }

    #[test]
    fn gradient_tape_rejects_bad_slots_capacity_and_non_finite_values_atomically() {
        let mut tape = GradientTape::new(2).unwrap();
        assert_eq!(tape.add(0, 0).unwrap_err().code, "AIT011");
        tape.input(1.0).unwrap();
        assert_eq!(tape.input(f32::INFINITY).unwrap_err().code, "AIT012");
        tape.constant(2.0).unwrap();
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.input(3.0).unwrap_err().code, "AIT013");
        assert_eq!(tape.len(), 2);
    }

    #[test]
    fn failed_backward_does_not_overwrite_previous_gradients() {
        let mut tape = GradientTape::new(8).unwrap();
        let x = tape.input(2.0).unwrap();
        let y = tape.multiply(x, x).unwrap();
        tape.backward(y).unwrap();
        assert_eq!(tape.gradient(x).unwrap(), 4.0);
        assert_eq!(tape.backward(99).unwrap_err().code, "AIT011");
        assert_eq!(tape.gradient(x).unwrap(), 4.0);
    }

    #[test]
    fn activation_cache_retains_only_stride_checkpoints_with_fixed_shape() {
        let mut cache = ActivationCache::new(2, 4, 8).unwrap();
        assert!(!cache.record(1, &[1.0, 2.0]).unwrap());
        assert!(cache.record(2, &[3.0, 4.0]).unwrap());
        assert!(!cache.record(3, &[5.0, 6.0]).unwrap());
        assert!(cache.record(4, &[7.0, 8.0]).unwrap());
        assert_eq!(cache.checkpoint_steps(), vec![2, 4]);
        assert_eq!(cache.get(2), Some(&[3.0, 4.0][..]));
        assert_eq!(cache.retained_scalars(), 4);
        assert_eq!(cache.record(6, &[1.0, 2.0, 3.0]).unwrap_err().code, "AIT023");
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.retained_scalars(), 4);
    }

    #[test]
    fn activation_cache_rejects_shape_changes_between_checkpoint_steps_atomically() {
        let mut cache = ActivationCache::new(3, 4, 12).unwrap();
        assert!(!cache.record(1, &[1.0, 2.0]).unwrap());
        assert_eq!(cache.record(2, &[3.0, 4.0, 5.0]).unwrap_err().code, "AIT023");
        assert_eq!(cache.checkpoint_steps(), Vec::<u64>::new());
        assert_eq!(cache.retained_scalars(), 0);
        assert!(!cache.record(2, &[3.0, 4.0]).unwrap());
        assert!(cache.record(3, &[5.0, 6.0]).unwrap());
    }

    #[test]
    fn activation_cache_rejects_budget_and_out_of_order_steps_without_partial_write() {
        let mut cache = ActivationCache::new(1, 3, 2).unwrap();
        cache.record(1, &[1.0, 2.0]).unwrap();
        assert_eq!(cache.record(2, &[3.0, 4.0]).unwrap_err().code, "AIT024");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.retained_scalars(), 2);
        assert_eq!(cache.record(1, &[5.0, 6.0]).unwrap_err().code, "AIT021");
        assert_eq!(cache.len(), 1);
    }
}
