/// Represents the architecture of Hobbes' neural network, defining constants for the size of each
/// layer, the input bucket layout, output bucket count, quantisation factors, and the eval scale.

#[rustfmt::skip]
pub const BUCKETS: [usize; 64] = [
     0,  1,  2,  3, 3, 2,  1,  0,
     4,  5,  6,  7, 7, 6,  5,  4,
     8,  8,  9,  9, 9, 9,  8,  8,
    10, 10, 11, 11, 11, 11, 10, 10,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
];

pub const L0_SIZE: usize = 768;
pub const L0_QUANT: usize = 255;
pub const L0_SHIFT: usize = 9;

pub const INPUT_BUCKET_COUNT: usize = get_num_buckets(&BUCKETS);
pub const OUTPUT_BUCKET_COUNT: usize = 8;

pub const L1_SIZE: usize = 1280;
pub const L1_SHIFT: usize = 8;
pub const L1_QUANT: usize = 128;

pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 32;

pub const Q: i64 = 64;
pub const Q_BITS: usize = 6;
pub const SCALE: i64 = 650;

pub type FeatureWeights = [i16; L0_SIZE * L1_SIZE];

/// The `UntransposedNetwork` represents the net outputted by Bullet, with weights and biases in the
/// original [input][bucket][output] format.
#[repr(C, align(64))]
pub struct UntransposedNetwork {
    pub l0_weights: [FeatureWeights; INPUT_BUCKET_COUNT],
    pub l0_biases:  [i16; L1_SIZE],
    pub l1_weights: [[[i8; L2_SIZE]; OUTPUT_BUCKET_COUNT]; L1_SIZE],
    pub l1_biases:  [[i32; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l2_weights: [[[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT]; L2_SIZE * 2],
    pub l2_biases:  [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_weights: [[[i32; 1]; OUTPUT_BUCKET_COUNT]; L3_SIZE],
    pub l3_biases:  [i32; OUTPUT_BUCKET_COUNT],
}

/// The `Network` represents the net in the optimal format for inference, with weights and biases
/// permuted and transposed into the [bucket][output][input] format.
#[repr(C, align(64))]
pub struct Network {
    pub l0_weights: [FeatureWeights; INPUT_BUCKET_COUNT],
    pub l0_biases:  [i16; L1_SIZE],
    pub l1_weights: [[[i8; L1_SIZE]; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l1_biases:  [[i32; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l2_weights: [[[i32; L3_SIZE]; L2_SIZE * 2]; OUTPUT_BUCKET_COUNT],
    pub l2_biases:  [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_weights: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_biases:  [i32; OUTPUT_BUCKET_COUNT],
}

pub const fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;
    while i < N {
        if arr[i] > max {
            max = arr[i];
        }
        i += 1;
    }
    max + 1
}

