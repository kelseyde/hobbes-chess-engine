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
pub const NUM_BUCKETS: usize = get_num_buckets(&BUCKETS);

pub const L0_SIZE: usize = 768;
pub const L0_QUANT: usize = 255;
pub const L0_SHIFT: usize = 9;
pub const L0_BUCKET_COUNT: usize = 16;

pub const OUTPUT_BUCKET_COUNT: usize = 8;

pub const L1_SIZE: usize = 2048;

// ceil(log2(127 * 128 / 64)) = 8
pub const L1_SHIFT: usize = 8;
pub const L1_QUANT: usize = 128;

pub const L2_SIZE: usize = 16;
pub const L3_SIZE: usize = 32;

pub const Q: i64 = 64;
pub const SCALE: i64 = 400;

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../hobbes.nnue")) };

pub type FeatureWeights = [i16; L0_SIZE * L1_SIZE];

pub struct Arch {
    pub input_buckets: [usize; 64],
    pub output_bucket_count: usize,
    pub l0_size: usize,
    pub l0_quant: usize,
    pub l0_shift: usize,
    pub l1_size: usize,
    pub l1_quant: usize,
    pub l1_shift: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub scale: i64,
}

#[repr(C, align(64))]
pub struct Network {
    pub l0_weights: [FeatureWeights; L0_BUCKET_COUNT],
    pub l0_biases: [i16; L1_SIZE],
    pub l1_weights: [[i8; L1_SIZE * L2_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l1_biases: [[i32; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l2_weights: [[i32; L2_SIZE * L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l2_biases: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_weights: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_biases: [i32; OUTPUT_BUCKET_COUNT],
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
