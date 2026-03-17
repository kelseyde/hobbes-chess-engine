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

/// L1 propagation using dpbusd (dot-product of u8 inputs × i8 weights -> i32)
pub unsafe fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE] {
    let biases = &NETWORK.l1_biases[output_bucket];

    let mut output = [0i32; L2_SIZE];

    const STRIDE: usize = simd::I32_LANES * 4; // bytes per dpbusd

    for output_idx in 0..L2_SIZE {
        let weight_row = &NETWORK.l1_weights[output_bucket][output_idx];

        let mut acc0 = simd::splat_i32(0);
        let mut acc1 = simd::splat_i32(0);
        let mut acc2 = simd::splat_i32(0);
        let mut acc3 = simd::splat_i32(0);

        let mut i = 0;
        while i + 4 * STRIDE <= L1_SIZE {
            let inp0 = *(input.as_ptr().add(i) as *const _);
            let w0 = *(weight_row.as_ptr().add(i) as *const _);
            acc0 = simd::dpbusd(acc0, inp0, w0);

            let inp1 = *(input.as_ptr().add(i + STRIDE) as *const _);
            let w1 = *(weight_row.as_ptr().add(i + STRIDE) as *const _);
            acc1 = simd::dpbusd(acc1, inp1, w1);

            let inp2 = *(input.as_ptr().add(i + 2 * STRIDE) as *const _);
            let w2 = *(weight_row.as_ptr().add(i + 2 * STRIDE) as *const _);
            acc2 = simd::dpbusd(acc2, inp2, w2);

            let inp3 = *(input.as_ptr().add(i + 3 * STRIDE) as *const _);
            let w3 = *(weight_row.as_ptr().add(i + 3 * STRIDE) as *const _);
            acc3 = simd::dpbusd(acc3, inp3, w3);

            i += 4 * STRIDE;
        }
        while i < L1_SIZE {
            let inp = *(input.as_ptr().add(i) as *const _);
            let w = *(weight_row.as_ptr().add(i) as *const _);
            acc0 = simd::dpbusd(acc0, inp, w);
            i += STRIDE;
        }

        let combined = simd::add_i32(simd::add_i32(acc0, acc1), simd::add_i32(acc2, acc3));
        let raw = simd::horizontal_sum_i32_single(combined);

        let out = (raw >> L1_SHIFT) + biases[output_idx];
        let clamped = out.clamp(0, Q as i32);
        output[output_idx] = clamped * clamped;
    }


    output
}

/// L2 propagation
pub unsafe fn propagate_l2(input: &[i32; L2_SIZE], output_bucket: usize) -> [i32; L3_SIZE] {
    const LANES: usize = L3_SIZE / simd::I32_LANES;
    let weights = &NETWORK.l2_weights[output_bucket];
    let biases = &NETWORK.l2_biases[output_bucket];

    let mut acc = [simd::splat_i32(0); LANES];
    for (v, vec) in acc.iter_mut().enumerate() {
        *vec = simd::load_i32(biases.as_ptr().add(v * simd::I32_LANES));
    }

    for input_idx in 0..L2_SIZE {
        let x = simd::splat_i32(input[input_idx]);
        let base = input_idx * L3_SIZE;

        for v in 0..LANES {
            let w = simd::load_i32(weights.as_ptr().add(base + v * simd::I32_LANES));
            acc[v] = simd::mul_add_i32(w, x, acc[v]);
        }
    }

    let mut out = [0i32; L3_SIZE];
    for (v, vec) in acc.into_iter().enumerate() {
        simd::store_i32(out.as_mut_ptr().add(v * simd::I32_LANES), vec);
    }

    out
}

/// L3 propagation
pub unsafe fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32 {
    const LANES: usize = L3_SIZE / simd::I32_LANES;

    let weights = NETWORK.l3_weights[output_bucket].as_ptr();
    let bias = NETWORK.l3_biases[output_bucket];

    let lo = simd::splat_i32(0);
    let hi = simd::splat_i32((Q * Q * Q) as i32);

    let mut acc = [simd::splat_i32(0); LANES];
    for v in 0..LANES {
        let off = v * simd::I32_LANES;
        let w = simd::load_i32(weights.add(off));
        let b = simd::load_i32(input.as_ptr().add(off));
        let b_clamped = simd::clamp_i32(b, lo, hi);
        acc[v] = simd::mul_add_i32(w, b_clamped, acc[v]);
    }

    simd::horizontal_sum_i32(acc) + bias
}
