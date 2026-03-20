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

    const OUT_UNROLL: usize = L2_SIZE;
    const STRIDE: usize = simd::I32_LANES * 4;

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
        let mut w8 = NETWORK.l1_weights[output_bucket][out_idx + 8].as_ptr();
        let mut w9 = NETWORK.l1_weights[output_bucket][out_idx + 9].as_ptr();
        let mut w10 = NETWORK.l1_weights[output_bucket][out_idx + 10].as_ptr();
        let mut w11 = NETWORK.l1_weights[output_bucket][out_idx + 11].as_ptr();
        let mut w12 = NETWORK.l1_weights[output_bucket][out_idx + 12].as_ptr();
        let mut w13 = NETWORK.l1_weights[output_bucket][out_idx + 13].as_ptr();
        let mut w14 = NETWORK.l1_weights[output_bucket][out_idx + 14].as_ptr();
        let mut w15 = NETWORK.l1_weights[output_bucket][out_idx + 15].as_ptr();

        let (mut acc00, mut acc01, mut acc02, mut acc03) = simd::splat_i32_x4(0);
        let (mut acc10, mut acc11, mut acc12, mut acc13) = simd::splat_i32_x4(0);
        let (mut acc20, mut acc21, mut acc22, mut acc23) = simd::splat_i32_x4(0);
        let (mut acc30, mut acc31, mut acc32, mut acc33) = simd::splat_i32_x4(0);
        let (mut acc40, mut acc41, mut acc42, mut acc43) = simd::splat_i32_x4(0);
        let (mut acc50, mut acc51, mut acc52, mut acc53) = simd::splat_i32_x4(0);
        let (mut acc60, mut acc61, mut acc62, mut acc63) = simd::splat_i32_x4(0);
        let (mut acc70, mut acc71, mut acc72, mut acc73) = simd::splat_i32_x4(0);
        let (mut acc80, mut acc81, mut acc82, mut acc83) = simd::splat_i32_x4(0);
        let (mut acc90, mut acc91, mut acc92, mut acc93) = simd::splat_i32_x4(0);
        let (mut acc100, mut acc101, mut acc102, mut acc103) = simd::splat_i32_x4(0);
        let (mut acc110, mut acc111, mut acc112, mut acc113) = simd::splat_i32_x4(0);
        let (mut acc120, mut acc121, mut acc122, mut acc123) = simd::splat_i32_x4(0);
        let (mut acc130, mut acc131, mut acc132, mut acc133) = simd::splat_i32_x4(0);
        let (mut acc140, mut acc141, mut acc142, mut acc143) = simd::splat_i32_x4(0);
        let (mut acc150, mut acc151, mut acc152, mut acc153) = simd::splat_i32_x4(0);

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

            let (w8_0, w8_1, w8_2, w8_3) = simd::load_i8x4(w8, STRIDE);
            (acc80, acc81, acc82, acc83) = simd::dpbusd_x4(
                acc80, acc81, acc82, acc83, in0, in1, in2, in3, w8_0, w8_1, w8_2, w8_3,
            );

            let (w9_0, w9_1, w9_2, w9_3) = simd::load_i8x4(w9, STRIDE);
            (acc90, acc91, acc92, acc93) = simd::dpbusd_x4(
                acc90, acc91, acc92, acc93, in0, in1, in2, in3, w9_0, w9_1, w9_2, w9_3,
            );

            let (w10_0, w10_1, w10_2, w10_3) = simd::load_i8x4(w10, STRIDE);
            (acc100, acc101, acc102, acc103) = simd::dpbusd_x4(
                acc100, acc101, acc102, acc103, in0, in1, in2, in3, w10_0, w10_1, w10_2, w10_3,
            );

            let (w11_0, w11_1, w11_2, w11_3) = simd::load_i8x4(w11, STRIDE);
            (acc110, acc111, acc112, acc113) = simd::dpbusd_x4(
                acc110, acc111, acc112, acc113, in0, in1, in2, in3, w11_0, w11_1, w11_2, w11_3,
            );

            let (w12_0, w12_1, w12_2, w12_3) = simd::load_i8x4(w12, STRIDE);
            (acc120, acc121, acc122, acc123) = simd::dpbusd_x4(
                acc120, acc121, acc122, acc123, in0, in1, in2, in3, w12_0, w12_1, w12_2, w12_3,
            );

            let (w13_0, w13_1, w13_2, w13_3) = simd::load_i8x4(w13, STRIDE);
            (acc130, acc131, acc132, acc133) = simd::dpbusd_x4(
                acc130, acc131, acc132, acc133, in0, in1, in2, in3, w13_0, w13_1, w13_2, w13_3,
            );

            let (w14_0, w14_1, w14_2, w14_3) = simd::load_i8x4(w14, STRIDE);
            (acc140, acc141, acc142, acc143) = simd::dpbusd_x4(
                acc140, acc141, acc142, acc143, in0, in1, in2, in3, w14_0, w14_1, w14_2, w14_3,
            );

            let (w15_0, w15_1, w15_2, w15_3) = simd::load_i8x4(w15, STRIDE);
            (acc150, acc151, acc152, acc153) = simd::dpbusd_x4(
                acc150, acc151, acc152, acc153, in0, in1, in2, in3, w15_0, w15_1, w15_2, w15_3,
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
            w8 = w8.add(4 * STRIDE);
            w9 = w9.add(4 * STRIDE);
            w10 = w10.add(4 * STRIDE);
            w11 = w11.add(4 * STRIDE);
            w12 = w12.add(4 * STRIDE);
            w13 = w13.add(4 * STRIDE);
            w14 = w14.add(4 * STRIDE);
            w15 = w15.add(4 * STRIDE);
        }

        let mut raws = [0i32; OUT_UNROLL];
        raws[0] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc00, acc01),
            simd::add_i32(acc02, acc03),
        ));
        raws[1] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc10, acc11),
            simd::add_i32(acc12, acc13),
        ));
        raws[2] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc20, acc21),
            simd::add_i32(acc22, acc23),
        ));
        raws[3] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc30, acc31),
            simd::add_i32(acc32, acc33),
        ));
        raws[4] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc40, acc41),
            simd::add_i32(acc42, acc43),
        ));
        raws[5] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc50, acc51),
            simd::add_i32(acc52, acc53),
        ));
        raws[6] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc60, acc61),
            simd::add_i32(acc62, acc63),
        ));
        raws[7] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc70, acc71),
            simd::add_i32(acc72, acc73),
        ));
        raws[8] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc80, acc81),
            simd::add_i32(acc82, acc83),
        ));
        raws[9] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc90, acc91),
            simd::add_i32(acc92, acc93),
        ));
        raws[10] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc100, acc101),
            simd::add_i32(acc102, acc103),
        ));
        raws[11] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc110, acc111),
            simd::add_i32(acc112, acc113),
        ));
        raws[12] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc120, acc121),
            simd::add_i32(acc122, acc123),
        ));
        raws[13] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc130, acc131),
            simd::add_i32(acc132, acc133),
        ));
        raws[14] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc140, acc141),
            simd::add_i32(acc142, acc143),
        ));
        raws[15] = simd::horizontal_sum_i32_single(simd::add_i32(
            simd::add_i32(acc150, acc151),
            simd::add_i32(acc152, acc153),
        ));

        // SIMD activation: apply bias + shift + dual crelu + csrelu activation
        let lo = simd::splat_i32(0);
        let hi = simd::splat_i32(Q as i32);
        let hi2 = simd::splat_i32((Q * Q) as i32);

        let mut chunk = 0;
        while chunk * simd::I32_LANES < OUT_UNROLL {
            let offset = chunk * simd::I32_LANES;
            let raw_vec = simd::load_i32(raws.as_ptr().add(offset));
            let bias_vec = simd::load_i32(biases.as_ptr().add(out_idx + offset));
            let shifted = simd::add_i32(simd::shr_i32::<{ L1_SHIFT as i32 }>(raw_vec), bias_vec);
            let crelu = simd::shl_i32::<{ Q_BITS as i32 }>(simd::clamp_i32(shifted, lo, hi));
            let csrelu = simd::clamp_i32(simd::mul_i32(shifted, shifted), lo, hi2);
            simd::store_i32(output.as_mut_ptr().add(out_idx + offset), crelu);
            simd::store_i32(output.as_mut_ptr().add(out_idx + offset + L2_SIZE), csrelu);
            chunk += 1;
        }

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
