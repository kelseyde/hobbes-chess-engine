use hobbes_nnue_arch::{L0_QUANT, L0_SHIFT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q};
use crate::evaluation::{simd, NETWORK};

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

            *output.as_mut_ptr().add(i + base).cast() = packed;
        }

    }

    output
}

/// L1 propagation using dpbusd (dot-product of u8 inputs × i8 weights → i32)
///
/// The L1 weights are stored in input-major order: `weights[input_idx * L2_SIZE + output_idx]`.
/// We group 4 consecutive inputs (one dpbusd's worth) and process `L2_SIZE / I32_LANES`
/// accumulator registers in parallel, each holding I32_LANES output neurons.
///
/// For 4 consecutive inputs i..i+3 and I32_LANES consecutive outputs o..o+I32_LANES-1,
/// we broadcast the 4 input u8s into every 32-bit lane, and pack the corresponding
/// `I32_LANES × 4` weights so that each 32-bit lane holds the 4 weights for one output.
/// dpbusd then computes the partial dot-product for I32_LANES outputs in one instruction.
pub unsafe fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE] {
    let weights = &NETWORK.l1_weights[output_bucket];
    let biases = &NETWORK.l1_biases[output_bucket];

    const NUM_ACC: usize = L2_SIZE / simd::I32_LANES;
    let mut acc = [simd::splat_i32(0); NUM_ACC];

    // Process 4 inputs at a time (one dpbusd group)
    for i in (0..L1_SIZE).step_by(4) {
        // Broadcast the 4 input bytes into every 32-bit lane:
        // each lane = [input[i], input[i+1], input[i+2], input[i+3]]
        let inp_bytes = u32::from_le_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]);
        let inp = simd::splat_i32(inp_bytes as i32);

        // For each group of I32_LANES output neurons, pack the weights and accumulate via dpbusd
        for a in 0..NUM_ACC {
            let o_base = a * simd::I32_LANES;
            // Pack weights: lane j gets [w[i,o], w[i+1,o], w[i+2,o], w[i+3,o]]
            let mut weight_bytes = [0i8; 64]; // max I32_LANES * 4 across platforms
            for j in 0..simd::I32_LANES {
                let o = o_base + j;
                weight_bytes[j * 4]     = weights[(i)     * L2_SIZE + o];
                weight_bytes[j * 4 + 1] = weights[(i + 1) * L2_SIZE + o];
                weight_bytes[j * 4 + 2] = weights[(i + 2) * L2_SIZE + o];
                weight_bytes[j * 4 + 3] = weights[(i + 3) * L2_SIZE + o];
            }
            let w = *(weight_bytes.as_ptr() as *const _);
            acc[a] = simd::dpbusd(acc[a], inp, w);
        }
    }

    // Store accumulators to output
    let mut intermediate = [0i32; L2_SIZE];
    for a in 0..NUM_ACC {
        simd::store_i32(intermediate.as_mut_ptr().add(a * simd::I32_LANES), acc[a]);
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
pub unsafe fn propagate_l2(input: &[i32; L2_SIZE], output_bucket: usize) -> [i32; L3_SIZE] {
    const LANES: usize = L3_SIZE / simd::I32_LANES;
    let weights = &NETWORK.l2_weights[output_bucket];
    let mut out = NETWORK.l2_biases[output_bucket];

    let mut acc = [simd::splat_i32(0); LANES];
    for (v, vec) in acc.iter_mut().enumerate() {
        *vec = simd::load_i32(out.as_ptr().add(v * simd::I32_LANES));
    }

    for input_idx in 0..L2_SIZE {
        let x = simd::splat_i32(input[input_idx]);
        let base = input_idx * L3_SIZE;

        for v in 0..acc.len() {
            let w = simd::load_i32(weights.as_ptr().add(base + v * simd::I32_LANES));
            acc[v] = simd::mul_add_i32(w, x, acc[v]);
        }
    }

    for (v, vec) in acc.into_iter().enumerate() {
        simd::store_i32(out.as_mut_ptr().add(v * simd::I32_LANES), vec);
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
