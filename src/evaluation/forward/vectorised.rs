use crate::evaluation::{simd, sparse, NETWORK};
use hobbes_nnue_arch::{L0_QUANT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q, Q_BITS};

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
/// Abandon hope, all ye who enter here.
pub unsafe fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE * 2] {
    const ACC_LANES: usize = L2_SIZE / simd::I32_LANES;
    const WEIGHT_STRIDE: usize = simd::I32_LANES * 4;
    const UNROLL: usize = 4;

    let zero = simd::splat_i32(0);
    let mut acc = [[zero; UNROLL]; ACC_LANES];

    let (nonzero_indices, num_nonzero_indices) = sparse::find_nonzero_indices(input);
    let input_i32 = input.as_ptr() as *const i32;
    let w_base = NETWORK.l1_weights[output_bucket].as_ptr() as *const i8;

    let mut nnz = 0;
    while nnz + 2 * UNROLL <= num_nonzero_indices {
        for lane in 0..ACC_LANES {
            let off = lane * WEIGHT_STRIDE;
            for sub in 0..UNROLL {
                let i1 = nonzero_indices[nnz + 2 * sub] as usize;
                let i2 = nonzero_indices[nnz + 2 * sub + 1] as usize;
                let ft1 = simd::splat_i32_as_u8(*input_i32.add(i1));
                let ft2 = simd::splat_i32_as_u8(*input_i32.add(i2));
                let w1 = simd::load_i8(w_base.add(i1 * L2_SIZE * 4 + off));
                let w2 = simd::load_i8(w_base.add(i2 * L2_SIZE * 4 + off));
                acc[lane][sub] = simd::dpbusdx2(acc[lane][sub], ft1, w1, ft2, w2);
            }
        }
        nnz += 2 * UNROLL;
    }

    while nnz < num_nonzero_indices {
        let b = nonzero_indices[nnz] as usize;
        let ft_val = simd::splat_i32_as_u8(*input_i32.add(b));
        for lane in 0..ACC_LANES {
            let off = lane * WEIGHT_STRIDE;
            let wv = simd::load_i8(w_base.add(b * L2_SIZE * 4 + off));
            acc[lane][0] = simd::dpbusd(acc[lane][0], ft_val, wv);
        }
        nnz += 1;
    }

    let bias_ptr = NETWORK.l1_biases[output_bucket].as_ptr() as *const simd::VecI32;
    let lo = simd::splat_i32(0);
    let hi = simd::splat_i32(Q as i32);
    let hi2 = simd::splat_i32((Q * Q) as i32);

    let mut output = [0i32; L2_SIZE * 2];
    let out_ptr = output.as_mut_ptr() as *mut simd::VecI32;

    for lane in 0..ACC_LANES {
        let mut sum = acc[lane][0];
        for sub in 1..UNROLL {
            sum = simd::add_i32(sum, acc[lane][sub]);
        }

        let bias = simd::load_i32(bias_ptr.add(lane) as *const i32);
        let shifted = simd::add_i32(simd::shift_right_i32::<{ L1_SHIFT as _ }>(sum), bias);

        let crelu = simd::shift_left_i32::<{ Q_BITS as _ }>(simd::clamp_i32(shifted, lo, hi));
        let csrelu = simd::clamp_i32(simd::mul_i32(shifted, shifted), lo, hi2);

        simd::store_i32(out_ptr.add(lane) as *mut i32, crelu);
        simd::store_i32(out_ptr.add(lane + ACC_LANES) as *mut i32, csrelu);
    }

    output
}

/// L2 propagation
pub unsafe fn propagate_l2(input: &[i32; L2_SIZE * 2], output_bucket: usize) -> [i32; L3_SIZE] {
    const LANES: usize = L3_SIZE / simd::I32_LANES;
    let weights = &NETWORK.l2_weights[output_bucket];
    let biases = &NETWORK.l2_biases[output_bucket];

    let mut acc = [simd::splat_i32(0); LANES];
    for (lane, acc_lane) in acc.iter_mut().enumerate() {
        *acc_lane = simd::load_i32(biases.as_ptr().add(lane * simd::I32_LANES));
    }

    for (&input_scalar, weight_row_arr) in input.iter().zip(weights.iter()) {
        let input_val = simd::splat_i32(input_scalar);
        let weight_row = weight_row_arr.as_ptr();

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
    for (lane, acc_lane) in acc.iter().enumerate() {
        simd::store_i32(output.as_mut_ptr().add(lane * simd::I32_LANES), *acc_lane);
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
    for (lane, acc_lane) in acc.iter_mut().enumerate() {
        let off = lane * simd::I32_LANES;
        let input_chunk = simd::load_i32(input.as_ptr().add(off));
        let weight_chunk = simd::load_i32(weights.add(off));
        let clamped = simd::clamp_i32(input_chunk, lo, hi);
        *acc_lane = simd::mul_i32(weight_chunk, clamped);
    }

    simd::horizontal_sum_i32(acc) + bias
}
