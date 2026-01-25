use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::evaluation::accumulator::Accumulator;

const L1_SIZE: usize = 1024;
const FT_QUANT: usize = 255;
const FT_SHIFT: u32 = 9;



const fn mulhi(a: i16, b: i16) -> i16 {
    let product = (a as i32) * (b as i32);
    (product >> 16) as i16
}

pub fn activate_ft(acc: &Accumulator) -> [u8; L1_SIZE] {
    let mut output = [0; L1_SIZE];
    for side in [White, Black] {
        let input = acc.features(side);
        for i in 0..L1_SIZE / 2 {
            let left_clamped = input[i].clamp(0, FT_QUANT as i16);
            let right_clamped = input[i + L1_SIZE / 2].clamp(0, FT_QUANT as i16);
            let multiplied = (left_clamped as i32) * (right_clamped as i32);
            let output_val = (multiplied >> FT_SHIFT) as u8;
            output[i + side as usize * L1_SIZE / 2] = output_val;
        }
    }
    output
}

pub fn activate_ft_2(acc: &Accumulator, side: Side) -> [u8; L1_SIZE] {
    let mut activated_ft = [0u8; L1_SIZE];

    let stm_acc = acc.features(side);
    let ntm_acc = acc.features(side);

    for i in 0..L1_SIZE / 2 {
        // clamp to [0, 255] and cast to u8
        let s1 = stm_acc[i].clamp(0, FT_QUANT) as u8;
        let s2 = stm_acc[i + L1_SIZE / 2].clamp(0, FT_QUANT) as u8;

        let n1 = ntm_acc[i].clamp(0, FT_QUANT) as u8;
        let n2 = ntm_acc[i + L1_SIZE / 2].clamp(0, FT_QUANT) as u8;

        // same as: (@as(i32, s1) * s2) << 7 >> 16 in Zig
        let stm_val = (((s1 as i32) * (s2 as i32)) << 7) >> 16;
        let ntm_val = (((n1 as i32) * (n2 as i32)) << 7) >> 16;

        // clamp to [0, 255] before narrowing to u8
        let stm_clamped = stm_val.clamp(0, 255) as u8;
        let ntm_clamped = ntm_val.clamp(0, 255) as u8;

        activated_ft[i] = stm_clamped;
        activated_ft[i + L1_SIZE / 2] = ntm_clamped;
    }

    activated_ft
}

pub fn activate_ft_simd(acc: &Accumulator) -> [u8; L1_SIZE] {
    0
}

mod simd {
    pub fn dpbusd() {
        // Placeholder for SIMD implementation
    }
}