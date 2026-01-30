use crate::board::Board;
use crate::board::side::Side::{Black, White};
use crate::evaluation::accumulator::Accumulator;
use crate::evaluation::network::HIDDEN;

const FT_SIZE: usize = 768;
const FT_QUANT: usize = 255;
const FT_SHIFT: usize = 9;
const INPUT_BUCKETS: usize = 16;

const L1_SIZE: usize = 1536;
const L1_QUANT: usize = 128;

const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;

const SCALE: i32 = 400;

#[repr(C, align(64))]
pub struct Network {
    pub ft_weights: [i16; INPUT_BUCKETS * FT_SIZE * L1_SIZE],
    pub ft_biases: [i16; L1_SIZE],
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
            ft_weights: [0; INPUT_BUCKETS * FT_SIZE * L1_SIZE],
            ft_biases: [0; L1_SIZE],
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
    // TODO: load real net
    let net = Network::default();
    let us = match board.stm {
        White => &acc.white_features,
        Black => &acc.black_features,
    };
    let them = match board.stm {
        White => &acc.black_features,
        Black => &acc.white_features,
    };

    let activated_ft = activate_ft(us, them);
    let l1_outputs = propagate_l1(&activated_ft, &net.l1_weights, &net.l1_biases);
    let l2_outputs = propagate_l2(&l1_outputs, &net.l2_weights, &net.l2_biases);
    let l3_output = propagate_l3(&l2_outputs, &net.l3_weights, net.l3_bias);

    l3_output * SCALE
}

/// Feature transformer activation
/// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
pub fn activate_ft(us: &[i16; HIDDEN], them: &[i16; HIDDEN]) -> [u8; L1_SIZE] {
    let mut output = [0; L1_SIZE];

    for (side, feats) in [us, them].into_iter().enumerate() {
        let base = side * (L1_SIZE / 2);
        for i in 0..(L1_SIZE / 2) {
            // Load the pair of inputs to be multiplied.
            let left = feats[i];
            let right = feats[i + (L1_SIZE / 2)];

            // Clamp inputs to [0, 255] space.
            let l_clamped = left.clamp(0, FT_QUANT as i16) as u8;
            let r_clamped = right.clamp(0, FT_QUANT as i16) as u8;

            // Pairwise multiplication of left and right input.
            let multiplied: i32 = l_clamped as i32 * r_clamped as i32;

            // Downshift back into ~[0, 127].
            // Note: this is equivalent to the << 7 >> 16 that mulhi does.
            // where the fuck do I shift by 8 though
            let result: u8 = ((multiplied >> FT_SHIFT)).clamp(0, 255) as u8;
            output[base + i] = result;
        }
    }
    output
}

/// L1 propagation
fn propagate_l1(input: &[u8; L1_SIZE], weights: &[i8; L1_SIZE * L2_SIZE], biases: &[i32; L2_SIZE]) -> [i16; L2_SIZE] {
    // Unactivated L1 outputs in FT_QUANT*L1_QUANT space
    let mut out = [0; L2_SIZE];

    for o in 0..L1_SIZE {
        // do some dpbusd shit here
    }

    out
}

/// L2 propagation
fn propagate_l2(input: &[i16; L2_SIZE], weights: &[i32; L2_SIZE * L3_SIZE], biases: &[i32; L3_SIZE]) -> [i16; L3_SIZE] {
    let mut out = [0i16; L3_SIZE];

    for o in 0..L3_SIZE {
        // do some shit here too
    }

    out
}

/// L3 propagation
fn propagate_l3(input: &[i16; L3_SIZE], weights: &[i32; L3_SIZE], bias: i32) -> i32 {
    let mut sum: i32 = bias;
    for i in 0..L3_SIZE {
        // finally, also do some shit here
    }
    sum
}