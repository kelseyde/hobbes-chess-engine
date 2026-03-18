use crate::evaluation::{simd, NETWORK};
use hobbes_nnue_arch::{L0_QUANT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q};

/// L0 ('feature transformer') activation
/// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
pub unsafe fn activate_l0(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE]) -> [u8; L1_SIZE] {
    let mut output = [0u8; L1_SIZE];

    let lo = simd::splat_i16(0);
    let hi = simd::splat_i16(L0_QUANT as i16);

    for (side, feats) in [us, them].into_iter().enumerate() {
        let base = side * (L1_SIZE / 2);

        for i in (0..L1_SIZE / 2).step_by(2 * simd::I16_LANES) {
            let left1 = simd::load_i16(feats.as_ptr().add(i));
            let left2 = simd::load_i16(feats.as_ptr().add(i + simd::I16_LANES));
            let right1 = simd::load_i16(feats.as_ptr().add(i + L1_SIZE / 2));
            let right2 = simd::load_i16(feats.as_ptr().add(i + L1_SIZE / 2 + simd::I16_LANES));

            let left1_clipped = simd::clamp_i16(left1, lo, hi);
            let left2_clipped = simd::clamp_i16(left2, lo, hi);
            let right1_clipped = simd::clamp_i16(right1, lo, hi);
            let right2_clipped = simd::clamp_i16(right2, lo, hi);

            let product1 = simd::shift_left_mul_high_i16(left1_clipped, right1_clipped);
            let product2 = simd::shift_left_mul_high_i16(left2_clipped, right2_clipped);

            let packed = simd::packus(product1, product2);
            simd::store_u8(output.as_mut_ptr().add(i + base), packed);
        }
    }

    output
}

/// L1 propagation
pub unsafe fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE] {
    let biases = &NETWORK.l1_biases[output_bucket];

    let mut output = [0i32; L2_SIZE];

    const STRIDE: usize = simd::I32_LANES * 4;

    for output_neuron in 0..L2_SIZE {
        let weights = &NETWORK.l1_weights[output_bucket][output_neuron];

        let mut acc0 = simd::splat_i32(0);
        let mut acc1 = simd::splat_i32(0);
        let mut acc2 = simd::splat_i32(0);
        let mut acc3 = simd::splat_i32(0);

        let mut i = 0;
        while i + 4 * STRIDE <= L1_SIZE {
            let input0 = simd::load_u8(input.as_ptr().add(i));
            let input1 = simd::load_u8(input.as_ptr().add(i + STRIDE));
            let input2 = simd::load_u8(input.as_ptr().add(i + 2 * STRIDE));
            let input3 = simd::load_u8(input.as_ptr().add(i + 3 * STRIDE));

            let weight0 = simd::load_i8(weights.as_ptr().add(i));
            let weight1 = simd::load_i8(weights.as_ptr().add(i + STRIDE));
            let weight2 = simd::load_i8(weights.as_ptr().add(i + 2 * STRIDE));
            let weight3 = simd::load_i8(weights.as_ptr().add(i + 3 * STRIDE));

            acc0 = simd::dpbusd(acc0, input0, weight0);
            acc1 = simd::dpbusd(acc1, input1, weight1);
            acc2 = simd::dpbusd(acc2, input2, weight2);
            acc3 = simd::dpbusd(acc3, input3, weight3);

            i += 4 * STRIDE;
        }
        while i < L1_SIZE {
            let input_chunk = simd::load_u8(input.as_ptr().add(i));
            let weight_chunk = simd::load_i8(weights.as_ptr().add(i));
            acc0 = simd::dpbusd(acc0, input_chunk, weight_chunk);
            i += STRIDE;
        }

        let combined = simd::add_i32(simd::add_i32(acc0, acc1), simd::add_i32(acc2, acc3));
        let raw = simd::horizontal_sum_i32_single(combined);
        let shifted = (raw >> L1_SHIFT) + biases[output_neuron];
        let clamped = shifted.clamp(0, Q as i32);
        output[output_neuron] = clamped * clamped;
    }

    output
}

/// L2 propagation
pub unsafe fn propagate_l2(input: &[i32; L2_SIZE], output_bucket: usize) -> [i32; L3_SIZE] {
    const LANES: usize = L3_SIZE / simd::I32_LANES;
    let weights = &NETWORK.l2_weights[output_bucket];
    let biases = &NETWORK.l2_biases[output_bucket];

    let mut acc = [simd::splat_i32(0); LANES];
    for lane in 0..LANES {
        acc[lane] = simd::load_i32(biases.as_ptr().add(lane * simd::I32_LANES));
    }

    for input_neuron in 0..L2_SIZE {
        let input_val = simd::splat_i32(input[input_neuron]);
        let weight_row = weights.as_ptr().add(input_neuron * L3_SIZE);

        let mut lane = 0;
        while lane + 4 <= LANES {
            let off = lane * simd::I32_LANES;
            let weight0 = simd::load_i32(weight_row.add(off));
            let weight1 = simd::load_i32(weight_row.add(off + simd::I32_LANES));
            let weight2 = simd::load_i32(weight_row.add(off + 2 * simd::I32_LANES));
            let weight3 = simd::load_i32(weight_row.add(off + 3 * simd::I32_LANES));
            acc[lane] = simd::mul_add_i32(weight0, input_val, acc[lane]);
            acc[lane + 1] = simd::mul_add_i32(weight1, input_val, acc[lane + 1]);
            acc[lane + 2] = simd::mul_add_i32(weight2, input_val, acc[lane + 2]);
            acc[lane + 3] = simd::mul_add_i32(weight3, input_val, acc[lane + 3]);
            lane += 4;
        }
        while lane < LANES {
            let weight = simd::load_i32(weight_row.add(lane * simd::I32_LANES));
            acc[lane] = simd::mul_add_i32(weight, input_val, acc[lane]);
            lane += 1;
        }
    }

    let mut output = [0i32; L3_SIZE];
    for lane in 0..LANES {
        simd::store_i32(output.as_mut_ptr().add(lane * simd::I32_LANES), acc[lane]);
    }
    output
}

/// L3 propagation
pub unsafe fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32 {
    const LANES: usize = L3_SIZE / simd::I32_LANES;

    let weights = NETWORK.l3_weights[output_bucket].as_ptr();
    let bias = NETWORK.l3_biases[output_bucket];
    let lo = simd::splat_i32(0);
    let hi = simd::splat_i32((Q * Q * Q) as i32);

    let mut acc = [simd::splat_i32(0); LANES];
    for lane in 0..LANES {
        let off = lane * simd::I32_LANES;
        let input_chunk = simd::load_i32(input.as_ptr().add(off));
        let weight_chunk = simd::load_i32(weights.add(off));
        let clamped = simd::clamp_i32(input_chunk, lo, hi);
        acc[lane] = simd::mul_i32(weight_chunk, clamped);
    }

    simd::horizontal_sum_i32(acc) + bias
}
