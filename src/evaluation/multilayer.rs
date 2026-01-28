use crate::board::Board;
use crate::board::side::Side::{Black, White};
use crate::evaluation::accumulator::Accumulator;
use crate::evaluation::network::HIDDEN;

const kFtQBits: i32 = 8;
const kL1QBits: i32 = 7;
const kQBits: i32 = 6;
const kFtScaleBits: i32 = 7;

const FT_SIZE: u32 = 768;
const FT_QUANT: usize = 255;
const FT_SHIFT: u32 = 9;

const L1_SIZE: usize = 1536;
const L1_QUANT: usize = 64;

const L2_SIZE: u32 = 16;
const L3_SIZE: u32 = 32;

const SCALE: i32 = 400;

pub fn forward(acc: &Accumulator, board: &Board) -> i32 {
    let us = match board.stm {
        White => &acc.white_features,
        Black => &acc.black_features,
    };
    let them = match board.stm {
        White => &acc.black_features,
        Black => &acc.white_features,
    };
    let activated_ft = activate_ft(us, them);

    0
}

/// We are in [0, 255] space, we want to end up in [0, 64] space for int multilayer.
/// Our L1 weights are in [-128, 127] space.
pub fn activate_ft(us: &[i16; HIDDEN], them: &[i16; HIDDEN]) -> [u8; L1_SIZE] {
    let mut output = [0; L1_SIZE];
    for (side, feats) in [us, them].into_iter().enumerate() {
        for i in 0..L1_SIZE / 2 {
            // Load the pair of inputs to be multiplied
            let left = feats[i];
            let right = feats[i + L1_SIZE / 2];

            // Clamp inputs to [0, 255] space
            let l_clamped = left.clamp(0, FT_QUANT as i16) as u8;
            let r_clamped = right.clamp(0, FT_QUANT as i16) as u8;

            // Pairwise multiplication of left and right input
            let multiplied: i32 = (l_clamped as i32) * (r_clamped as i32);

            // Culminating in a right shift by 9 back down into [0, 127] space
            // Note: this is equivalent to the << 7 >> 16 that mulhi does.
            let result: u8 = (multiplied >> FT_SHIFT) as u8;
            output[i + side * L1_SIZE / 2] = result;
        }
    }
    output
}