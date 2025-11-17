use crate::evaluation::{NUM_INPUT_BUCKETS, NUM_OUTPUT_BUCKETS};

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 1280;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../hobbes.nnue")) };

pub type FeatureWeights = [i16; FEATURES * HIDDEN];

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [FeatureWeights; NUM_INPUT_BUCKETS],
    pub feature_bias: [i16; HIDDEN],
    pub output_weights: [[[i16; HIDDEN]; 2]; NUM_OUTPUT_BUCKETS],
    pub output_bias: [i16; NUM_OUTPUT_BUCKETS],
}
