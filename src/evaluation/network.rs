use std::ops::{Deref, DerefMut};
use crate::evaluation::NUM_BUCKETS;

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 1280;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../hobbes.nnue")) };

pub type FeatureWeights = [i16; FEATURES * HIDDEN];
pub type Block = [i16; HIDDEN];

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [FeatureWeights; NUM_BUCKETS],
    pub feature_bias: Block,
    pub output_weights: [Block; 2],
    pub output_bias: i16,
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub struct Align64<T>(pub T);

impl<T, const N: usize> Deref for Align64<[T; N]> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Align64<[T; N]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
