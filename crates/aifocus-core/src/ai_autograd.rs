//! Bounded scalar reverse-mode automatic differentiation for Ardisa's native VM.
//!
//! The supported graph is deliberately small: finite F32 inputs, addition and
//! multiplication. Slots are append-only and operands must reference prior slots.
pub const MAX_GRAD_TAPE_SLOTS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientError {
    ResourceLimit,
    InvalidSlot,
    NonFinite,
    BackwardNotRun,
    InvalidTape,
}

#[derive(Debug, Clone, PartialEq)]
enum Op {
    Input,
    Add { left: usize, right: usize },
    Mul { left: usize, right: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    op: Op,
    value_bits: u32,
    gradient_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradientTape {
    max_slots: usize,
    nodes: Vec<Node>,
    backward_done: bool,
}

impl GradientTape {
    pub fn new(max_slots: usize) -> Result<Self, GradientError> {
        if max_slots == 0 || max_slots > MAX_GRAD_TAPE_SLOTS {
            return Err(GradientError::ResourceLimit);
        }
        Ok(Self { max_slots, nodes: Vec::new(), backward_done: false })
    }

    fn append(&mut self, op: Op, value: f32) -> Result<usize, GradientError> {
        if !value.is_finite() { return Err(GradientError::NonFinite); }
        if self.nodes.len() >= self.max_slots { return Err(GradientError::ResourceLimit); }
        let slot = self.nodes.len();
        self.nodes.push(Node { op, value_bits: value.to_bits(), gradient_bits: 0.0f32.to_bits() });
        self.backward_done = false;
        Ok(slot)
    }

    pub fn input(&mut self, value: f32) -> Result<usize, GradientError> {
        self.append(Op::Input, value)
    }

    pub fn add(&mut self, left: usize, right: usize) -> Result<usize, GradientError> {
        let a = f32::from_bits(self.nodes.get(left).ok_or(GradientError::InvalidSlot)?.value_bits);
        let b = f32::from_bits(self.nodes.get(right).ok_or(GradientError::InvalidSlot)?.value_bits);
        self.append(Op::Add { left, right }, a + b)
    }

    pub fn mul(&mut self, left: usize, right: usize) -> Result<usize, GradientError> {
        let a = self.nodes.get(left).ok_or(GradientError::InvalidSlot)?.value;
        let b = self.nodes.get(right).ok_or(GradientError::InvalidSlot)?.value;
        self.append(Op::Mul { left, right }, a * b)
    }

    /// Reverse-mode backpropagation with deterministic left-then-right accumulation.
    /// Repeated operands accumulate twice, as required by the chain rule.
    pub fn backward(&mut self, output: usize, seed: f32) -> Result<(), GradientError> {
        if !seed.is_finite() { return Err(GradientError::NonFinite); }
        if output >= self.nodes.len() { return Err(GradientError::InvalidSlot); }
        let mut gradients = vec![0.0f32; self.nodes.len()];
        gradients[output] = seed;
        for slot in (0..self.nodes.len()).rev() {
            let gradient = gradients[slot];
            let node = &self.nodes[slot];
            match node.op {
                Op::Input => {}
                Op::Add { left, right } => {
                    gradients[left] += gradient;
                    gradients[right] += gradient;
                }
                Op::Mul { left, right } => {
                    gradients[left] += gradient * f32::from_bits(self.nodes[right].value_bits);
                    gradients[right] += gradient * f32::from_bits(self.nodes[left].value_bits);
                }
            }
            if gradients.iter().any(|value| !value.is_finite()) {
                return Err(GradientError::NonFinite);
            }
        }
        for (node, gradient) in self.nodes.iter_mut().zip(gradients) {
            node.gradient_bits = gradient.to_bits();
        }
        self.backward_done = true;
        Ok(())
    }

    pub fn value(&self, slot: usize) -> Result<f32, GradientError> {
        self.nodes.get(slot).map(|node| f32::from_bits(node.value_bits)).ok_or(GradientError::InvalidSlot)
    }

    pub fn derivative(&self, slot: usize) -> Result<f32, GradientError> {
        if !self.backward_done { return Err(GradientError::BackwardNotRun); }
        self.nodes.get(slot).map(|node| f32::from_bits(node.gradient_bits)).ok_or(GradientError::InvalidSlot)
    }

    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    pub fn validate(&self) -> Result<(), GradientError> {
        if self.max_slots == 0 || self.max_slots > MAX_GRAD_TAPE_SLOTS || self.nodes.len() > self.max_slots {
            return Err(GradientError::ResourceLimit);
        }
        for (slot, node) in self.nodes.iter().enumerate() {
            if !f32::from_bits(node.value_bits).is_finite() || !f32::from_bits(node.gradient_bits).is_finite() { return Err(GradientError::NonFinite); }
            match node.op {
                Op::Input => {}
                Op::Add { left, right } | Op::Mul { left, right } => {
                    if left >= slot || right >= slot { return Err(GradientError::InvalidSlot); }
                }
            }
        }
        Ok(())
    }

    pub fn footprint_nodes(&self) -> Result<usize, GradientError> {
        self.nodes.len().checked_mul(3).and_then(|count| count.checked_add(2))
            .ok_or(GradientError::ResourceLimit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_operand_gradient_accumulates_deterministically() {
        let mut tape = GradientTape::new(8).unwrap();
        let x = tape.input(3.0).unwrap();
        let square = tape.mul(x, x).unwrap();
        tape.backward(square, 1.0).unwrap();
        assert_eq!(tape.value(square).unwrap(), 9.0);
        assert_eq!(tape.derivative(x).unwrap(), 6.0);
        assert!(tape.validate().is_ok());
    }

    #[test]
    fn rejects_forward_slots_nonfinite_values_and_slot_overflow() {
        let mut tape = GradientTape::new(1).unwrap();
        assert_eq!(tape.add(0, 0), Err(GradientError::InvalidSlot));
        assert_eq!(tape.input(f32::NAN), Err(GradientError::NonFinite));
        assert_eq!(tape.input(1.0), Ok(0));
        assert_eq!(tape.input(2.0), Err(GradientError::ResourceLimit));
    }

    #[test]
    fn backward_is_atomic_on_nonfinite_intermediate() {
        let mut tape = GradientTape::new(4).unwrap();
        let x = tape.input(f32::MAX).unwrap();
        let y = tape.input(2.0).unwrap();
        let product = tape.mul(x, y).unwrap();
        assert_eq!(tape.mul(x, y), Err(GradientError::NonFinite));
        assert!(!tape.backward_done);
    }
}
