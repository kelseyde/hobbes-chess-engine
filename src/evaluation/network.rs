use crate::evaluation::NUM_BUCKETS;

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 1024;
pub const SCALE: i32 = 400 * 2 / 3;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../hobbes.nnue")) };

pub type FeatureWeights = [i16; FEATURES * HIDDEN];

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [FeatureWeights; NUM_BUCKETS],
    pub feature_bias: [i16; HIDDEN],
    pub output_weights: [[i16; HIDDEN]; 2],
    pub output_bias: i16,
}