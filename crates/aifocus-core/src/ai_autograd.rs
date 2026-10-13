//! Bounded deterministic scalar automatic-differentiation tape.
//!
//! This module defines the runtime contract used by the native Ardisa built-ins.
//! It supports only finite F32 input, add, and multiply nodes. References must
//! point backward to existing slots; unsupported operators are not represented.

pub const MAX_GRADIENT_TAPE_SLOTS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientError {
    InvalidCapacity,
    SlotLimit,
    InvalidSlot,
    NonFiniteValue,
    BackwardNotRun,
    CorruptTape,
    ResourceLimit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientOp {
    Input,
    Add { left: usize, right: usize },
    Mul { left: usize, right: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradientNode {
    pub value_bits: u32,
    pub op: GradientOp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradientTape {
    max_slots: usize,
    nodes: Vec<GradientNode>,
    gradients: Option<Vec<u32>>,
}

impl GradientTape {
    pub fn new(max_slots: usize) -> Result<Self, GradientError> {
        if max_slots == 0 || max_slots > MAX_GRADIENT_TAPE_SLOTS {
            return Err(GradientError::InvalidCapacity);
        }
        Ok(Self { max_slots, nodes: Vec::new(), gradients: None })
    }

    pub fn max_slots(&self) -> usize { self.max_slots }
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
    pub fn has_gradients(&self) -> bool { self.gradients.is_some() }
    pub fn nodes(&self) -> &[GradientNode] { &self.nodes }

    pub fn input(&mut self, value: f32) -> Result<usize, GradientError> {
        if !value.is_finite() { return Err(GradientError::NonFiniteValue); }
        self.push(value, GradientOp::Input)
    }

    pub fn add(&mut self, left: usize, right: usize) -> Result<usize, GradientError> {
        let a = self.value(left)?;
        let b = self.value(right)?;
        let value = a + b;
        if !value.is_finite() { return Err(GradientError::NonFiniteValue); }
        self.push(value, GradientOp::Add { left, right })
    }

    pub fn mul(&mut self, left: usize, right: usize) -> Result<usize, GradientError> {
        let a = self.value(left)?;
        let b = self.value(right)?;
        let value = a * b;
        if !value.is_finite() { return Err(GradientError::NonFiniteValue); }
        self.push(value, GradientOp::Mul { left, right })
    }

    fn push(&mut self, value: f32, op: GradientOp) -> Result<usize, GradientError> {
        if self.nodes.len() >= self.max_slots { return Err(GradientError::SlotLimit); }
        if !value.is_finite() { return Err(GradientError::NonFiniteValue); }
        let index = self.nodes.len();
        match op {
            GradientOp::Input => {}
            GradientOp::Add { left, right } | GradientOp::Mul { left, right } => {
                if left >= index || right >= index { return Err(GradientError::InvalidSlot); }
            }
        }
        self.nodes.push(GradientNode { value_bits: value.to_bits(), op });
        self.gradients = None;
        Ok(index)
    }

    pub fn value(&self, slot: usize) -> Result<f32, GradientError> {
        self.nodes.get(slot).map(|node| f32::from_bits(node.value_bits)).ok_or(GradientError::InvalidSlot)
    }

    pub fn derivative(&self, slot: usize) -> Result<f32, GradientError> {
        let gradients = self.gradients.as_ref().ok_or(GradientError::BackwardNotRun)?;
        gradients.get(slot).map(|bits| f32::from_bits(*bits)).ok_or(GradientError::InvalidSlot)
    }

    pub fn backward(&mut self, output: usize, seed: f32) -> Result<(), GradientError> {
        if !seed.is_finite() { return Err(GradientError::NonFiniteValue); }
        if output >= self.nodes.len() { return Err(GradientError::InvalidSlot); }
        let mut gradients = vec![0.0f32; self.nodes.len()];
        gradients[output] = seed;
        for slot in (0..=output).rev() {
            let upstream = gradients[slot];
            if !upstream.is_finite() { return Err(GradientError::NonFiniteValue); }
            match self.nodes[slot].op {
                GradientOp::Input => {}
                GradientOp::Add { left, right } => {
                    Self::accumulate(&mut gradients, left, upstream)?;
                    Self::accumulate(&mut gradients, right, upstream)?;
                }
                GradientOp::Mul { left, right } => {
                    let left_value = f32::from_bits(self.nodes[left].value_bits);
                    let right_value = f32::from_bits(self.nodes[right].value_bits);
                    Self::accumulate(&mut gradients, left, upstream * right_value)?;
                    Self::accumulate(&mut gradients, right, upstream * left_value)?;
                }
            }
        }
        if gradients.iter().any(|value| !value.is_finite()) { return Err(GradientError::NonFiniteValue); }
        self.gradients = Some(gradients.into_iter().map(f32::to_bits).collect());
        Ok(())
    }

    fn accumulate(gradients: &mut [f32], slot: usize, contribution: f32) -> Result<(), GradientError> {
        if slot >= gradients.len() || !contribution.is_finite() { return Err(GradientError::NonFiniteValue); }
        let next = gradients[slot] + contribution;
        if !next.is_finite() { return Err(GradientError::NonFiniteValue); }
        gradients[slot] = next;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), GradientError> {
        if self.max_slots == 0 || self.max_slots > MAX_GRADIENT_TAPE_SLOTS || self.nodes.len() > self.max_slots {
            return Err(GradientError::CorruptTape);
        }
        for (slot, node) in self.nodes.iter().enumerate() {
            if !f32::from_bits(node.value_bits).is_finite() { return Err(GradientError::CorruptTape); }
            match node.op {
                GradientOp::Input => {}
                GradientOp::Add { left, right } | GradientOp::Mul { left, right } => {
                    if left >= slot || right >= slot { return Err(GradientError::CorruptTape); }
                }
            }
        }
        if let Some(gradients) = &self.gradients {
            if gradients.len() != self.nodes.len() || gradients.iter().any(|bits| !f32::from_bits(*bits).is_finite()) {
                return Err(GradientError::CorruptTape);
            }
        }
        Ok(())
    }

    pub fn footprint_nodes(&self) -> Result<usize, GradientError> {
        self.nodes.len().checked_mul(2)
            .and_then(|n| n.checked_add(self.gradients.as_ref().map_or(0, Vec::len)))
            .ok_or(GradientError::ResourceLimit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_multiply_backward_is_deterministic() {
        let mut tape = GradientTape::new(8).unwrap();
        let x = tape.input(3.0).unwrap();
        let y = tape.input(4.0).unwrap();
        let z = tape.mul(x, y).unwrap();
        assert_eq!(tape.value(z), Ok(12.0));
        assert_eq!(tape.derivative(x), Err(GradientError::BackwardNotRun));
        tape.backward(z, 1.0).unwrap();
        assert_eq!(tape.derivative(x), Ok(4.0));
        assert_eq!(tape.derivative(y), Ok(3.0));
        assert_eq!(tape.derivative(z), Ok(1.0));
    }

    #[test]
    fn repeated_operand_accumulates_both_paths_and_failed_append_is_atomic() {
        let mut tape = GradientTape::new(8).unwrap();
        let x = tape.input(3.0).unwrap();
        let twice = tape.add(x, x).unwrap();
        tape.backward(twice, 2.0).unwrap();
        assert_eq!(tape.derivative(x), Ok(4.0));
        let before = tape.clone();
        let huge_a = tape.input(f32::MAX).unwrap();
        let huge_b = tape.input(2.0).unwrap();
        let before_bad = tape.clone();
        assert_eq!(tape.mul(huge_a, huge_b), Err(GradientError::NonFiniteValue));
        assert_eq!(tape, before_bad);
        assert_ne!(tape, before);
    }

    #[test]
    fn rejects_invalid_slots_nonfinite_values_and_slot_exhaustion() {
        let mut tape = GradientTape::new(2).unwrap();
        assert_eq!(tape.add(0, 0), Err(GradientError::InvalidSlot));
        assert_eq!(tape.input(f32::NAN), Err(GradientError::NonFiniteValue));
        tape.input(1.0).unwrap();
        tape.input(2.0).unwrap();
        let before = tape.clone();
        assert_eq!(tape.input(3.0), Err(GradientError::SlotLimit));
        assert_eq!(tape, before);
        assert_eq!(tape.backward(8, 1.0), Err(GradientError::InvalidSlot));
        assert_eq!(tape.backward(0, f32::INFINITY), Err(GradientError::NonFiniteValue));
    }
}
