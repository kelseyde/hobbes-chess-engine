use crate::evaluation::NUM_BUCKETS;

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 1280;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../hobbes.nnue")) };

#[repr(align(64))]
pub struct Align64<T>(pub T);

pub type FeatureWeights = Align64<[i16; FEATURES * HIDDEN]>;
pub type FeatureBiases = Align64<[i16; HIDDEN]>;
pub type InputFeatures = Align64<[i16; HIDDEN]>;
pub type OutputWeights = Align64<[[i16; HIDDEN]; 2]>;

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: [FeatureWeights; NUM_BUCKETS],
    pub feature_bias: FeatureBiases,
    pub output_weights: OutputWeights,
    pub output_bias: i16,
}
