use crate::evaluation::{simd, NETWORK};
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
    let biases = &NETWORK.l1_biases[output_bucket];

    let mut output = [0i32; L2_SIZE * 2];

    const STRIDE: usize = simd::I32_LANES * 4;
    const OUT_UNROLL: usize = 8;

    let mut out_idx = 0;
    while out_idx + OUT_UNROLL <= L2_SIZE {
        let mut w0 = NETWORK.l1_weights[output_bucket][out_idx].as_ptr();
        let mut w1 = NETWORK.l1_weights[output_bucket][out_idx + 1].as_ptr();
        let mut w2 = NETWORK.l1_weights[output_bucket][out_idx + 2].as_ptr();
        let mut w3 = NETWORK.l1_weights[output_bucket][out_idx + 3].as_ptr();
        let mut w4 = NETWORK.l1_weights[output_bucket][out_idx + 4].as_ptr();
        let mut w5 = NETWORK.l1_weights[output_bucket][out_idx + 5].as_ptr();
        let mut w6 = NETWORK.l1_weights[output_bucket][out_idx + 6].as_ptr();
        let mut w7 = NETWORK.l1_weights[output_bucket][out_idx + 7].as_ptr();

        let (mut acc00, mut acc01, mut acc02, mut acc03) = simd::splat_i32_x4(0);
        let (mut acc10, mut acc11, mut acc12, mut acc13) = simd::splat_i32_x4(0);
        let (mut acc20, mut acc21, mut acc22, mut acc23) = simd::splat_i32_x4(0);
        let (mut acc30, mut acc31, mut acc32, mut acc33) = simd::splat_i32_x4(0);
        let (mut acc40, mut acc41, mut acc42, mut acc43) = simd::splat_i32_x4(0);
        let (mut acc50, mut acc51, mut acc52, mut acc53) = simd::splat_i32_x4(0);
        let (mut acc60, mut acc61, mut acc62, mut acc63) = simd::splat_i32_x4(0);
        let (mut acc70, mut acc71, mut acc72, mut acc73) = simd::splat_i32_x4(0);

        let mut in_ptr = input.as_ptr();
        let end_ptr = input.as_ptr().add(L1_SIZE).sub(4 * STRIDE);

        while in_ptr <= end_ptr {
            let in0 = simd::load_u8(in_ptr);
            let in1 = simd::load_u8(in_ptr.add(STRIDE));
            let in2 = simd::load_u8(in_ptr.add(2 * STRIDE));
            let in3 = simd::load_u8(in_ptr.add(3 * STRIDE));

            let (w0_0, w0_1, w0_2, w0_3) = simd::load_i8x4(w0, STRIDE);
            (acc00, acc01, acc02, acc03) = simd::dpbusd_x4(
                acc00, acc01, acc02, acc03, in0, in1, in2, in3, w0_0, w0_1, w0_2, w0_3,
            );

            let (w1_0, w1_1, w1_2, w1_3) = simd::load_i8x4(w1, STRIDE);
            (acc10, acc11, acc12, acc13) = simd::dpbusd_x4(
                acc10, acc11, acc12, acc13, in0, in1, in2, in3, w1_0, w1_1, w1_2, w1_3,
            );

            let (w2_0, w2_1, w2_2, w2_3) = simd::load_i8x4(w2, STRIDE);
            (acc20, acc21, acc22, acc23) = simd::dpbusd_x4(
                acc20, acc21, acc22, acc23, in0, in1, in2, in3, w2_0, w2_1, w2_2, w2_3,
            );

            let (w3_0, w3_1, w3_2, w3_3) = simd::load_i8x4(w3, STRIDE);
            (acc30, acc31, acc32, acc33) = simd::dpbusd_x4(
                acc30, acc31, acc32, acc33, in0, in1, in2, in3, w3_0, w3_1, w3_2, w3_3,
            );

            let (w4_0, w4_1, w4_2, w4_3) = simd::load_i8x4(w4, STRIDE);
            (acc40, acc41, acc42, acc43) = simd::dpbusd_x4(
                acc40, acc41, acc42, acc43, in0, in1, in2, in3, w4_0, w4_1, w4_2, w4_3,
            );

            let (w5_0, w5_1, w5_2, w5_3) = simd::load_i8x4(w5, STRIDE);
            (acc50, acc51, acc52, acc53) = simd::dpbusd_x4(
                acc50, acc51, acc52, acc53, in0, in1, in2, in3, w5_0, w5_1, w5_2, w5_3,
            );

            let (w6_0, w6_1, w6_2, w6_3) = simd::load_i8x4(w6, STRIDE);
            (acc60, acc61, acc62, acc63) = simd::dpbusd_x4(
                acc60, acc61, acc62, acc63, in0, in1, in2, in3, w6_0, w6_1, w6_2, w6_3,
            );

            let (w7_0, w7_1, w7_2, w7_3) = simd::load_i8x4(w7, STRIDE);
            (acc70, acc71, acc72, acc73) = simd::dpbusd_x4(
                acc70, acc71, acc72, acc73, in0, in1, in2, in3, w7_0, w7_1, w7_2, w7_3,
            );

            in_ptr = in_ptr.add(4 * STRIDE);
            w0 = w0.add(4 * STRIDE);
            w1 = w1.add(4 * STRIDE);
            w2 = w2.add(4 * STRIDE);
            w3 = w3.add(4 * STRIDE);
            w4 = w4.add(4 * STRIDE);
            w5 = w5.add(4 * STRIDE);
            w6 = w6.add(4 * STRIDE);
            w7 = w7.add(4 * STRIDE);
        }

        let combined0 = simd::add_i32(simd::add_i32(acc00, acc01), simd::add_i32(acc02, acc03));
        let raw0 = simd::horizontal_sum_i32_single(combined0);
        let shifted0 = (raw0 >> L1_SHIFT) + biases[out_idx];
        let crelu0: i32 = shifted0.clamp(0, Q as i32) << Q_BITS;
        let csrelu0: i32 = (shifted0 * shifted0).clamp(0, (Q * Q) as i32);
        output[out_idx] = crelu0;
        output[out_idx + L2_SIZE] = csrelu0;

        let combined1 = simd::add_i32(simd::add_i32(acc10, acc11), simd::add_i32(acc12, acc13));
        let raw1 = simd::horizontal_sum_i32_single(combined1);
        let shifted1 = (raw1 >> L1_SHIFT) + biases[out_idx + 1];
        let crelu1: i32 = shifted1.clamp(0, Q as i32) << Q_BITS;
        let csrelu1: i32 = (shifted1 * shifted1).clamp(0, (Q * Q) as i32);
        output[out_idx + 1] = crelu1;
        output[out_idx + 1 + L2_SIZE] = csrelu1;

        let combined2 = simd::add_i32(simd::add_i32(acc20, acc21), simd::add_i32(acc22, acc23));
        let raw2 = simd::horizontal_sum_i32_single(combined2);
        let shifted2 = (raw2 >> L1_SHIFT) + biases[out_idx + 2];
        let crelu2: i32 = shifted2.clamp(0, Q as i32) << Q_BITS;
        let csrelu2: i32 = (shifted2 * shifted2).clamp(0, (Q * Q) as i32);
        output[out_idx + 2] = crelu2;
        output[out_idx + 2 + L2_SIZE] = csrelu2;

        let combined3 = simd::add_i32(simd::add_i32(acc30, acc31), simd::add_i32(acc32, acc33));
        let raw3 = simd::horizontal_sum_i32_single(combined3);
        let shifted3 = (raw3 >> L1_SHIFT) + biases[out_idx + 3];
        let crelu3: i32 = shifted3.clamp(0, Q as i32) << Q_BITS;
        let csrelu3: i32 = (shifted3 * shifted3).clamp(0, (Q * Q) as i32);
        output[out_idx + 3] = crelu3;
        output[out_idx + 3 + L2_SIZE] = csrelu3;

        let combined4 = simd::add_i32(simd::add_i32(acc40, acc41), simd::add_i32(acc42, acc43));
        let raw4 = simd::horizontal_sum_i32_single(combined4);
        let shifted4 = (raw4 >> L1_SHIFT) + biases[out_idx + 4];
        let crelu4: i32 = shifted4.clamp(0, Q as i32) << Q_BITS;
        let csrelu4: i32 = (shifted4 * shifted4).clamp(0, (Q * Q) as i32);
        output[out_idx + 4] = crelu4;
        output[out_idx + 4 + L2_SIZE] = csrelu4;

        let combined5 = simd::add_i32(simd::add_i32(acc50, acc51), simd::add_i32(acc52, acc53));
        let raw5 = simd::horizontal_sum_i32_single(combined5);
        let shifted5 = (raw5 >> L1_SHIFT) + biases[out_idx + 5];
        let crelu5: i32 = shifted5.clamp(0, Q as i32) << Q_BITS;
        let csrelu5: i32 = (shifted5 * shifted5).clamp(0, (Q * Q) as i32);
        output[out_idx + 5] = crelu5;
        output[out_idx + 5 + L2_SIZE] = csrelu5;

        let combined6 = simd::add_i32(simd::add_i32(acc60, acc61), simd::add_i32(acc62, acc63));
        let raw6 = simd::horizontal_sum_i32_single(combined6);
        let shifted6 = (raw6 >> L1_SHIFT) + biases[out_idx + 6];
        let crelu6: i32 = shifted6.clamp(0, Q as i32) << Q_BITS;
        let csrelu6: i32 = (shifted6 * shifted6).clamp(0, (Q * Q) as i32);
        output[out_idx + 6] = crelu6;
        output[out_idx + 6 + L2_SIZE] = csrelu6;

        let combined7 = simd::add_i32(simd::add_i32(acc70, acc71), simd::add_i32(acc72, acc73));
        let raw7 = simd::horizontal_sum_i32_single(combined7);
        let shifted7 = (raw7 >> L1_SHIFT) + biases[out_idx + 7];
        let crelu7: i32 = shifted7.clamp(0, Q as i32) << Q_BITS;
        let csrelu7: i32 = (shifted7 * shifted7).clamp(0, (Q * Q) as i32);
        output[out_idx + 7] = crelu7;
        output[out_idx + 7 + L2_SIZE] = csrelu7;

        out_idx += OUT_UNROLL;
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
