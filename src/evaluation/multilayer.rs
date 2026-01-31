use crate::board::side::Side::{Black, White};
use crate::board::Board;
use crate::evaluation::accumulator::Accumulator;

const L0_SIZE: usize = 768;
const L0_QUANT: usize = 255;
const L0_SHIFT: usize = 9;
const L0_BUCKET_COUNT: usize = 16;

const L1_SIZE: usize = 1280;
const L1_SHIFT: usize = 8;
const L1_QUANT: usize = 128;

const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;

const SCALE: i32 = 400;

pub type FeatureWeights = [i16; L0_SIZE * L1_SIZE];

#[repr(C, align(64))]
pub struct Network {
    pub l0_weights: [FeatureWeights; L0_BUCKET_COUNT],
    pub l0_biases: [i16; L1_SIZE],
    pub l1_weights: [i8; L1_SIZE * L2_SIZE],
    pub l1_biases: [i32; L2_SIZE],
    pub l2_weights: [i32; L2_SIZE * L3_SIZE],
    pub l2_biases: [i32; L3_SIZE],
    pub l3_weights: [i32; L3_SIZE],
    pub l3_bias: i32,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            l0_weights: [[0; L0_SIZE * L1_SIZE]; L0_BUCKET_COUNT],
            l0_biases: [0; L1_SIZE],
            l1_weights: [0; L1_SIZE * L2_SIZE],
            l1_biases: [0; L2_SIZE],
            l2_weights: [0; L2_SIZE * L3_SIZE],
            l2_biases: [0; L3_SIZE],
            l3_weights: [0; L3_SIZE],
            l3_bias: 0,
        }
    }
}

pub fn forward(acc: &Accumulator, board: &Board) -> i32 {
    // TODO: load real net lol
    let net = Network::default();
    let us = match board.stm {
        White => &acc.white_features,
        Black => &acc.black_features,
    };
    let them = match board.stm {
        White => &acc.black_features,
        Black => &acc.white_features,
    };

    let l0_outputs = activate_l0(us, them);
    let l1_outputs = propagate_l1(&l0_outputs, &net.l1_weights, &net.l1_biases);
    let l2_outputs = propagate_l2(&l1_outputs, &net.l2_weights, &net.l2_biases);
    let l3_output = propagate_l3(&l2_outputs, &net.l3_weights, net.l3_bias);

    l3_output * SCALE
}

/// L0 ('feature transformer') activation
/// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
pub fn activate_l0(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE]) -> [u8; L1_SIZE] {
    let mut output = [0; L1_SIZE];

    for (side, feats) in [us, them].into_iter().enumerate() {
        let base = side * (L1_SIZE / 2);
        for i in 0..(L1_SIZE / 2) {
            // Load the pair of inputs to be multiplied.
            let left: i16 = feats[i];
            let right: i16 = feats[i + (L1_SIZE / 2)];

            // Clipped ReLU activation
            let l_clamped: u8 = left.clamp(0, L0_QUANT as i16) as u8;
            let r_clamped: u8 = right.clamp(0, L0_QUANT as i16) as u8;

            // Pairwise multiplication of left and right input.
            let multiplied: i32 = l_clamped as i32 * r_clamped as i32;

            // Downshift back into ~[0, 127].
            // Note: this is equivalent to the << 7 >> 16 that mulhi does.
            let result: u8 = (multiplied >> L0_SHIFT).clamp(0, 255) as u8;
            output[base + i] = result;
        }
    }
    output
}

/// L1 propagation
fn propagate_l1(
    input: &[u8; L1_SIZE],
    weights: &[i8; L1_SIZE * L2_SIZE],
    biases: &[i32; L2_SIZE],
) -> [i32; L2_SIZE] {
    let mut intermediate: [i32; L2_SIZE] = [0; L2_SIZE];

    // L1 matrix multiplication
    for input_idx in 0..L1_SIZE {
        let input: i32 = input[input_idx] as i32;
        for output_idx in 0..L2_SIZE {
            let weight: i32 = weights[input_idx * output_idx] as i32;
            intermediate[output_idx] += input * weight;
        }
    }

    // Re-quantise, add biases and activate L1 outputs
    let mut output: [i32; L2_SIZE] = [0; L2_SIZE];
    for i in 0..L2_SIZE {
        let bias: i32 = biases[i];
        let mut out: i32 = intermediate[i];

        // Down-shift into L1 Q space
        out >>= L1_SHIFT;

        // Add the bias
        out += bias;

        // Clipped ReLU activation
        let clamped: i32 = out.clamp(0, L1_QUANT as i32);

        output[i] = clamped;
    }


    output
}

/// L2 propagation
fn propagate_l2(input: &[i32; L2_SIZE], weights: &[i32; L2_SIZE * L3_SIZE], biases: &[i32; L3_SIZE]) -> [i32; L3_SIZE] {
    let mut out = [0; L3_SIZE];
    for input_idx in 0..L2_SIZE {
        let input = input[input_idx] as i32;
        for output_idx in 0..L3_SIZE {
            let w_idx = input_idx * L3_SIZE + output_idx;
            let weight = weights[w_idx];
            out[output_idx] += input * weight;
        }
    }
    out
}

/// L3 propagation
fn propagate_l3(input: &[i32; L3_SIZE], weights: &[i32; L3_SIZE], bias: i32) -> i32 {
    let mut output: i32 = bias;
    for (&input, &weight) in input.iter().zip(weights.iter()) {
        output += input * weight;
    }
    output
}
