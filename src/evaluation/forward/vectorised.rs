use crate::evaluation::arch::{L0_QUANT, L0_SHIFT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, NETWORK, Q};
use crate::evaluation::simd;

/// L0 ('feature transformer') activation
/// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
pub unsafe fn activate_l0(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE]) -> [u8; L1_SIZE] {
    let mut output = [0; L1_SIZE];

    let lo = simd::splat_i16(0);
    let hi = simd::splat_i16(L0_QUANT as i16);

    for (side, feats) in [us, them].into_iter().enumerate() {
        let base = side * (L1_SIZE / 2);

        for i in (0..L1_SIZE / 2).step_by(2 * simd::I16_LANES) {
            let left1 = *feats.as_ptr().add(i).cast();
            let left2 = *feats.as_ptr().add(i + simd::I16_LANES).cast();

            let right1 = *feats.as_ptr().add(i + L1_SIZE / 2).cast();
            let right2 = *feats.as_ptr().add(i + L1_SIZE / 2 + simd::I16_LANES).cast();

            let left1_clipped = simd::clamp_i16(left1, lo, hi);
            let left2_clipped = simd::clamp_i16(left2, lo, hi);

            let right1_clipped = simd::clamp_i16(right1, lo, hi);
            let right2_clipped = simd::clamp_i16(right2, lo, hi);

            let shifted1 = simd::shift_left_i16::<{ 16 - L0_SHIFT as i32 }>(left1_clipped);
            let shifted2 = simd::shift_left_i16::<{ 16 - L0_SHIFT as i32 }>(left2_clipped);

            let product1 = simd::mul_high_i16(shifted1, right1_clipped);
            let product2 = simd::mul_high_i16(shifted2, right2_clipped);

            let packed = simd::packus(product1, product2);
            let unpacked = simd::permute(packed);

            *output.as_mut_ptr().add(i + base).cast() = unpacked;
        }

    }

    output
}

/// L1 propagation
pub fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE] {
    let weights = &NETWORK.l1_weights[output_bucket];
    let biases = &NETWORK.l1_biases[output_bucket];

    // Unactivated L1 outputs in the quantized space L0_QUANT * L1_QUANT
    let mut intermediate: [i32; L2_SIZE] = [0; L2_SIZE];

    // L1 matrix multiplication
    for input_idx in 0..L1_SIZE {
        let input: i32 = input[input_idx] as i32;
        for output_idx in 0..L2_SIZE {
            let w_idx = input_idx * L2_SIZE + output_idx;
            let weight: i32 = weights[w_idx] as i32;
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

        // Squared Clipped ReLU activation
        // Clamp to [0, Q]
        let clamped: i32 = out.clamp(0, Q as i32);
        // Square the clamped value, moving to [0, Q*Q]
        let activated = clamped * clamped;

        output[i] = activated;
    }


    output
}

/// L2 propagation
pub fn propagate_l2(input: &[i32; L2_SIZE], output_bucket: usize) -> [i32; L3_SIZE] {
    let weights = &NETWORK.l2_weights[output_bucket];

    let mut out = NETWORK.l2_biases[output_bucket];
    for input_idx in 0..L2_SIZE {
        let input = input[input_idx];
        for output_idx in 0..L3_SIZE {
            let w_idx = input_idx * L3_SIZE + output_idx;
            let weight = weights[w_idx];
            // This multiplication moves us into [0, Q^3] space
            out[output_idx] += input * weight;
        }
    }
    out
}

/// L3 propagation
pub unsafe fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32 {
    const LANES: usize = L3_SIZE / simd::I32_LANES;

    let input_ptr = input.as_ptr();
    let weights = &NETWORK.l3_weights[output_bucket].as_ptr();
    let bias = NETWORK.l3_biases[output_bucket];

    let lo = simd::splat_i32(0);
    let hi = simd::splat_i32((Q * Q * Q) as i32);

    let mut output = [simd::splat_i32(0); LANES];
    for (lane, result) in output.iter_mut().enumerate() {
        for i in (0..L3_SIZE).step_by(LANES * simd::I32_LANES) {
            let w = *weights.add(i + lane * simd::I32_LANES).cast();
            let b = *input_ptr.add(i + lane * simd::I32_LANES).cast();
            let b_clamped = simd::clamp_i32(b, lo, hi);

            *result = simd::mul_add_i32(w, b_clamped, *result);
        }
    }
    simd::horizontal_sum_i32(output) + bias

}

