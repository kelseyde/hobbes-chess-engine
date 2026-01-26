use crate::board::side::Side::{Black, White};
use crate::evaluation::accumulator::Accumulator;

const L1_SIZE: usize = 1024;
const FT_QUANT: usize = 255;
const FT_SHIFT: u32 = 9;

/// We are in [0, 255] space, we want to end up in [0, 64] space for int multilayer.
/// Our L1 weights are in [-128, 127] space.
pub fn activate_ft(acc: &Accumulator) -> [u8; L1_SIZE] {
    let mut output = [0; L1_SIZE];
    for side in [White, Black] {
        let input = acc.features(side);
        for i in 0..L1_SIZE / 2 {
            // Load the pair of inputs to be multiplied
            let left: i16 = input[i];
            let right: i16 = input[i + L1_SIZE / 2];

            // Clamp inputs to [0, 255] space
            let left_clamped: u8 = left.clamp(0, FT_QUANT as i16) as u8;
            let right_clamped: u8 = right.clamp(0, FT_QUANT as i16) as u8;

            // Pairwise multiplication of left and right input
            let multiplied: i32 = (left_clamped as i32) * (right_clamped as i32);
            // Culminating in a right shift by 9 back down into [0, 127] space
            // Note: this is equivalent to the << 7 >> 16 that mulhi does.
            let output_val: u8 = (multiplied >> FT_SHIFT) as u8;
            output[i + side as usize * L1_SIZE / 2] = output_val;
        }
    }
    output
}