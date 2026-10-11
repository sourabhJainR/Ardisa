//! First-class AI primitive semantic/runtime foundations.
//!
//! A bounded vertical slice of checked shape, domain, probability,
//! quantization and canonical descriptor encoding contracts. These descriptors
//! are not yet accepted as .ardisa source type syntax or emitted by ARDISA-EXEC-V1.

use std::convert::TryFrom;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AiTypeError {
    ShapeOverflow,
    ShapeElementCount { expected: usize, actual: usize },
    RankMismatch { left: usize, right: usize },
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    InvalidDimension,
    DomainMismatch { left: String, right: String },
    EmbeddingDimension { expected: usize, actual: usize },
    InvalidProbability,
    InvalidScale,
    QuantizedOutOfRange,
    ResourceLimit { requested: usize, limit: usize },
}
impl std::fmt::Display for AiTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for AiTypeError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape(Vec<usize>);
impl TensorShape {
    pub fn new(dimensions: Vec<usize>, max_elements: usize) -> Result<Self, AiTypeError> {
        let mut count = 1usize;
        for dimension in &dimensions {
            if *dimension == 0 { return Err(AiTypeError::InvalidDimension); }
            count = count.checked_mul(*dimension).ok_or(AiTypeError::ShapeOverflow)?;
        }
        if count > max_elements { return Err(AiTypeError::ResourceLimit { requested: count, limit: max_elements }); }
        Ok(Self(dimensions))
    }
    pub fn dimensions(&self) -> &[usize] { &self.0 }
    pub fn rank(&self) -> usize { self.0.len() }
    pub fn elements(&self) -> Result<usize, AiTypeError> {
        self.0.iter().try_fold(1usize, |n, d| n.checked_mul(*d).ok_or(AiTypeError::ShapeOverflow))
    }
    pub fn broadcast(&self, other: &Self, max_elements: usize) -> Result<Self, AiTypeError> {
        let rank = self.rank().max(other.rank());
        let mut dims = Vec::with_capacity(rank);
        for offset in 0..rank {
            let a = self.0.iter().rev().nth(offset).copied().unwrap_or(1);
            let b = other.0.iter().rev().nth(offset).copied().unwrap_or(1);
            if a != b && a != 1 && b != 1 {
                return Err(AiTypeError::ShapeMismatch { left: self.0.clone(), right: other.0.clone() });
            }
            dims.push(a.max(b));
        }
        dims.reverse();
        Self::new(dims, max_elements)
    }
    pub fn matmul(&self, other: &Self, max_elements: usize) -> Result<Self, AiTypeError> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(AiTypeError::RankMismatch { left: self.rank(), right: other.rank() });
        }
        let (m, k) = (self.0[self.rank()-2], self.0[self.rank()-1]);
        let (k2, n) = (other.0[other.rank()-2], other.0[other.rank()-1]);
        if k != k2 { return Err(AiTypeError::ShapeMismatch { left: self.0.clone(), right: other.0.clone() }); }
        let batch = Self(self.0[..self.rank()-2].to_vec()).broadcast(&Self(other.0[..other.rank()-2].to_vec()), max_elements)?;
        let mut out = batch.0;
        out.extend([m, n]);
        Self::new(out, max_elements)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AiNumericType { F32, F16, BF16, I8, I4, Quant4 }

#[derive(Debug, Clone, PartialEq)]
pub struct TensorValue { pub shape: TensorShape, pub element_type: AiNumericType, pub values: Vec<f32> }
impl TensorValue {
    pub fn new(shape: TensorShape, element_type: AiNumericType, values: Vec<f32>) -> Result<Self, AiTypeError> {
        let expected = shape.elements()?;
        if expected != values.len() { return Err(AiTypeError::ShapeElementCount { expected, actual: values.len() }); }
        if values.iter().any(|x| !x.is_finite()) { return Err(AiTypeError::InvalidScale); }
        Ok(Self { shape, element_type, values })
    }
    pub fn add(&self, other: &Self, max_elements: usize) -> Result<Self, AiTypeError> {
        if self.element_type != other.element_type || self.shape != other.shape {
            return Err(AiTypeError::ShapeMismatch { left: self.shape.0.clone(), right: other.shape.0.clone() });
        }
        let shape = self.shape.broadcast(&other.shape, max_elements)?;
        let values = self.values.iter().zip(&other.values).map(|(a,b)| a+b).collect();
        Self::new(shape, self.element_type, values)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DomainEmbedding<const N: usize> { domain: String, values: [f32; N] }
impl<const N: usize> DomainEmbedding<N> {
    pub fn new(domain: impl Into<String>, values: [f32; N]) -> Result<Self, AiTypeError> {
        let domain = domain.into();
        if domain.is_empty() || values.iter().any(|x| !x.is_finite()) {
            return Err(AiTypeError::EmbeddingDimension { expected: N, actual: values.len() });
        }
        Ok(Self { domain, values })
    }
    pub fn domain(&self) -> &str { &self.domain }
    pub fn values(&self) -> &[f32; N] { &self.values }
    pub fn dot(&self, other: &Self) -> Result<f32, AiTypeError> {
        if self.domain != other.domain {
            return Err(AiTypeError::DomainMismatch { left: self.domain.clone(), right: other.domain.clone() });
        }
        Ok(self.values.iter().zip(other.values.iter()).map(|(a,b)| a*b).sum())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f32);
impl Probability {
    pub fn new(value: f32) -> Result<Self, AiTypeError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) { return Err(AiTypeError::InvalidProbability); }
        Ok(Self(value))
    }
    pub fn value(self) -> f32 { self.0 }
    pub fn and(self, other: Self) -> Self { Self(self.0 * other.0) }
    pub fn or(self, other: Self) -> Self { Self(self.0 + other.0 - self.0 * other.0) }
    pub fn not(self) -> Self { Self(1.0 - self.0) }
    pub fn meets(self, threshold: Self) -> bool { self.0 >= threshold.0 }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationScale(f32);
impl QuantizationScale {
    pub fn new(value: f32) -> Result<Self, AiTypeError> {
        if !value.is_finite() || value <= 0.0 { return Err(AiTypeError::InvalidScale); }
        Ok(Self(value))
    }
    pub fn value(self) -> f32 { self.0 }
    pub fn quantize_i8(self, value: f32) -> Result<i8, AiTypeError> {
        if !value.is_finite() { return Err(AiTypeError::InvalidScale); }
        let q = (value / self.0).round();
        if q < i8::MIN as f32 || q > i8::MAX as f32 { return Err(AiTypeError::QuantizedOutOfRange); }
        Ok(q as i8)
    }
    pub fn dequantize_i8(self, value: i8) -> f32 { value as f32 * self.0 }
}
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor { pub shape: TensorShape, pub scale: QuantizationScale, pub zero_point: i8, pub values: Vec<i8> }
impl QuantizedTensor {
    pub fn new(shape: TensorShape, scale: QuantizationScale, zero_point: i8, values: Vec<i8>) -> Result<Self, AiTypeError> {
        let expected = shape.elements()?;
        if values.len() != expected { return Err(AiTypeError::ShapeElementCount { expected, actual: values.len() }); }
        Ok(Self { shape, scale, zero_point, values })
    }
    pub fn add(&self, other: &Self) -> Result<Self, AiTypeError> {
        if self.shape != other.shape || self.scale != other.scale || self.zero_point != other.zero_point {
            return Err(AiTypeError::ShapeMismatch { left: self.shape.0.clone(), right: other.shape.0.clone() });
        }
        let values = self.values.iter().zip(&other.values).map(|(a,b)| {
            let v = (*a as i16) + (*b as i16) - self.zero_point as i16;
            i8::try_from(v).map_err(|_| AiTypeError::QuantizedOutOfRange)
        }).collect::<Result<Vec<_>, _>>()?;
        Self::new(self.shape.clone(), self.scale, self.zero_point, values)
    }
}

/// Canonical descriptor format (not executable ARDISA-EXEC-V1 bytecode).
pub fn encode_tensor_descriptor(value: &TensorValue) -> Result<Vec<u8>, AiTypeError> {
    let rank = u8::try_from(value.shape.rank()).map_err(|_| AiTypeError::ResourceLimit { requested: value.shape.rank(), limit: u8::MAX as usize })?;
    let elements = u32::try_from(value.values.len()).map_err(|_| AiTypeError::ResourceLimit { requested: value.values.len(), limit: u32::MAX as usize })?;
    let mut out = b"ARDISA-AIT\x01".to_vec();
    out.push(match value.element_type { AiNumericType::F32=>1, AiNumericType::F16=>2, AiNumericType::BF16=>3, AiNumericType::I8=>4, AiNumericType::I4=>5, AiNumericType::Quant4=>6 });
    out.push(rank);
    for d in value.shape.dimensions() {
        out.extend_from_slice(&u64::try_from(*d).map_err(|_| AiTypeError::ShapeOverflow)?.to_le_bytes());
    }
    out.extend_from_slice(&elements.to_le_bytes());
    for item in &value.values { out.extend_from_slice(&item.to_bits().to_le_bytes()); }
    Ok(out)
}


/// Canonical encoding for a domain-tagged embedding. Domain strings are length
/// prefixed UTF-8 and coordinates are serialized as little-endian IEEE-754 f32.
pub fn encode_embedding<const N: usize>(value: &DomainEmbedding<N>) -> Result<Vec<u8>, AiTypeError> {
    let domain = value.domain.as_bytes();
    let domain_len = u32::try_from(domain.len()).map_err(|_| AiTypeError::ResourceLimit { requested: domain.len(), limit: u32::MAX as usize })?;
    let dimension = u32::try_from(N).map_err(|_| AiTypeError::ResourceLimit { requested: N, limit: u32::MAX as usize })?;
    let mut out = b"ARDISA-AIE\x01".to_vec();
    out.extend_from_slice(&domain_len.to_le_bytes());
    out.extend_from_slice(domain);
    out.extend_from_slice(&dimension.to_le_bytes());
    for item in &value.values { out.extend_from_slice(&item.to_bits().to_le_bytes()); }
    Ok(out)
}

/// Canonical encoding of a validated probability value.
pub fn encode_probability(value: Probability) -> Vec<u8> {
    let mut out = b"ARDISA-AIP\x01".to_vec();
    out.extend_from_slice(&value.0.to_bits().to_le_bytes());
    out
}

/// Canonical encoding of signed-int8 quantized tensor storage and quantization metadata.
pub fn encode_quantized_tensor(value: &QuantizedTensor) -> Result<Vec<u8>, AiTypeError> {
    let rank = u8::try_from(value.shape.rank()).map_err(|_| AiTypeError::ResourceLimit { requested: value.shape.rank(), limit: u8::MAX as usize })?;
    let elements = u32::try_from(value.values.len()).map_err(|_| AiTypeError::ResourceLimit { requested: value.values.len(), limit: u32::MAX as usize })?;
    let mut out = b"ARDISA-AIQ\x01".to_vec();
    out.push(rank);
    for d in value.shape.dimensions() { out.extend_from_slice(&u64::try_from(*d).map_err(|_| AiTypeError::ShapeOverflow)?.to_le_bytes()); }
    out.extend_from_slice(&value.scale.0.to_bits().to_le_bytes());
    out.push(value.zero_point as u8);
    out.extend_from_slice(&elements.to_le_bytes());
    out.extend(value.values.iter().map(|v| *v as u8));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape(d: &[usize]) -> TensorShape { TensorShape::new(d.to_vec(), 4096).unwrap() }
    #[test]
    fn shape_broadcast_and_matmul_are_checked() {
        assert_eq!(shape(&[2,1,3]).broadcast(&shape(&[1,4,3]), 4096).unwrap().dimensions(), &[2,4,3]);
        assert!(shape(&[2,3]).matmul(&shape(&[4,2]), 4096).is_err());
        assert!(TensorShape::new(vec![usize::MAX, 3], 4096).is_err());
    }
    #[test]
    fn tensor_requires_exact_element_count_and_bounded_allocation() {
        assert!(TensorValue::new(shape(&[2,2]), AiNumericType::F32, vec![1.0,2.0]).is_err());
        assert!(TensorShape::new(vec![100,100], 10).is_err());
    }
    #[test]
    fn embeddings_are_nominally_domain_safe() {
        let a = DomainEmbedding::<2>::new("text", [1.0,0.0]).unwrap();
        let b = DomainEmbedding::<2>::new("image", [1.0,0.0]).unwrap();
        assert!(matches!(a.dot(&b), Err(AiTypeError::DomainMismatch { .. })));
    }
    #[test]
    fn probability_rejects_nan_infinity_and_out_of_range() {
        for value in [f32::NAN, f32::INFINITY, -0.01, 1.01] { assert!(Probability::new(value).is_err()); }
        let a = Probability::new(0.5).unwrap();
        assert_eq!(a.and(a).value(), 0.25);
        assert_eq!(a.or(a).value(), 0.75);
    }
    #[test]
    fn quantization_rejects_bad_scales_and_overflow() {
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] { assert!(QuantizationScale::new(scale).is_err()); }
        let scale = QuantizationScale::new(0.5).unwrap();
        assert!(scale.quantize_i8(1000.0).is_err());
        let a = QuantizedTensor::new(shape(&[2]), scale, 0, vec![1,2]).unwrap();
        let b = QuantizedTensor::new(shape(&[2]), QuantizationScale::new(0.25).unwrap(), 0, vec![1,2]).unwrap();
        assert!(a.add(&b).is_err());
    }
    #[test]
    fn embedding_probability_and_quantized_encodings_are_deterministic() {
        let embedding = DomainEmbedding::<2>::new("text", [1.0, 2.0]).unwrap();
        assert_eq!(encode_embedding(&embedding).unwrap(), encode_embedding(&embedding).unwrap());
        let probability = Probability::new(0.75).unwrap();
        assert_eq!(encode_probability(probability), encode_probability(probability));
        let quantized = QuantizedTensor::new(shape(&[2]), QuantizationScale::new(0.5).unwrap(), 0, vec![1, 2]).unwrap();
        assert_eq!(encode_quantized_tensor(&quantized).unwrap(), encode_quantized_tensor(&quantized).unwrap());
    }

    #[test]
    fn canonical_encoding_is_stable() {
        let value = TensorValue::new(shape(&[2]), AiNumericType::F32, vec![1.0,2.0]).unwrap();
        assert_eq!(encode_tensor_descriptor(&value).unwrap(), encode_tensor_descriptor(&value).unwrap());
        assert!(encode_tensor_descriptor(&value).unwrap().starts_with(b"ARDISA-AIT\x01"));
    }
}
