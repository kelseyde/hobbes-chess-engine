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
    const OUT_UNROLL: usize = 8;

    let mut out_idx = 0;
    while out_idx + OUT_UNROLL <= L2_SIZE {
        let mut w0 = NETWORK.l1_weights[output_bucket][out_idx + 0].as_ptr();
        let mut w1 = NETWORK.l1_weights[output_bucket][out_idx + 1].as_ptr();
        let mut w2 = NETWORK.l1_weights[output_bucket][out_idx + 2].as_ptr();
        let mut w3 = NETWORK.l1_weights[output_bucket][out_idx + 3].as_ptr();
        let mut w4 = NETWORK.l1_weights[output_bucket][out_idx + 4].as_ptr();
        let mut w5 = NETWORK.l1_weights[output_bucket][out_idx + 5].as_ptr();
        let mut w6 = NETWORK.l1_weights[output_bucket][out_idx + 6].as_ptr();
        let mut w7 = NETWORK.l1_weights[output_bucket][out_idx + 7].as_ptr();

        let mut acc00 = simd::splat_i32(0);
        let mut acc01 = simd::splat_i32(0);
        let mut acc02 = simd::splat_i32(0);
        let mut acc03 = simd::splat_i32(0);

        let mut acc10 = simd::splat_i32(0);
        let mut acc11 = simd::splat_i32(0);
        let mut acc12 = simd::splat_i32(0);
        let mut acc13 = simd::splat_i32(0);

        let mut acc20 = simd::splat_i32(0);
        let mut acc21 = simd::splat_i32(0);
        let mut acc22 = simd::splat_i32(0);
        let mut acc23 = simd::splat_i32(0);

        let mut acc30 = simd::splat_i32(0);
        let mut acc31 = simd::splat_i32(0);
        let mut acc32 = simd::splat_i32(0);
        let mut acc33 = simd::splat_i32(0);

        let mut acc40 = simd::splat_i32(0);
        let mut acc41 = simd::splat_i32(0);
        let mut acc42 = simd::splat_i32(0);
        let mut acc43 = simd::splat_i32(0);

        let mut acc50 = simd::splat_i32(0);
        let mut acc51 = simd::splat_i32(0);
        let mut acc52 = simd::splat_i32(0);
        let mut acc53 = simd::splat_i32(0);

        let mut acc60 = simd::splat_i32(0);
        let mut acc61 = simd::splat_i32(0);
        let mut acc62 = simd::splat_i32(0);
        let mut acc63 = simd::splat_i32(0);

        let mut acc70 = simd::splat_i32(0);
        let mut acc71 = simd::splat_i32(0);
        let mut acc72 = simd::splat_i32(0);
        let mut acc73 = simd::splat_i32(0);

        let mut in_ptr = input.as_ptr();
        let end_ptr = input.as_ptr().add(L1_SIZE).sub(4 * STRIDE);

        while in_ptr <= end_ptr {
            let in0 = simd::load_u8(in_ptr);
            let in1 = simd::load_u8(unsafe { in_ptr.add(STRIDE) });
            let in2 = simd::load_u8(unsafe { in_ptr.add(2 * STRIDE) });
            let in3 = simd::load_u8(unsafe { in_ptr.add(3 * STRIDE) });

            let (w0_0, w0_1, w0_2, w0_3) = simd::load_i8x4(w0, STRIDE);
            acc00 = simd::dpbusd(acc00, in0, w0_0);
            acc01 = simd::dpbusd(acc01, in1, w0_1);
            acc02 = simd::dpbusd(acc02, in2, w0_2);
            acc03 = simd::dpbusd(acc03, in3, w0_3);

            let (w1_0, w1_1, w1_2, w1_3) = simd::load_i8x4(w1, STRIDE);
            acc10 = simd::dpbusd(acc10, in0, w1_0);
            acc11 = simd::dpbusd(acc11, in1, w1_1);
            acc12 = simd::dpbusd(acc12, in2, w1_2);
            acc13 = simd::dpbusd(acc13, in3, w1_3);

            let (w2_0, w2_1, w2_2, w2_3) = simd::load_i8x4(w2, STRIDE);
            acc20 = simd::dpbusd(acc20, in0, w2_0);
            acc21 = simd::dpbusd(acc21, in1, w2_1);
            acc22 = simd::dpbusd(acc22, in2, w2_2);
            acc23 = simd::dpbusd(acc23, in3, w2_3);

            let (w3_0, w3_1, w3_2, w3_3) = simd::load_i8x4(w3, STRIDE);
            acc30 = simd::dpbusd(acc30, in0, w3_0);
            acc31 = simd::dpbusd(acc31, in1, w3_1);
            acc32 = simd::dpbusd(acc32, in2, w3_2);
            acc33 = simd::dpbusd(acc33, in3, w3_3);

            let (w4_0, w4_1, w4_2, w4_3) = simd::load_i8x4(w4, STRIDE);
            acc40 = simd::dpbusd(acc40, in0, w4_0);
            acc41 = simd::dpbusd(acc41, in1, w4_1);
            acc42 = simd::dpbusd(acc42, in2, w4_2);
            acc43 = simd::dpbusd(acc43, in3, w4_3);

            let (w5_0, w5_1, w5_2, w5_3) = simd::load_i8x4(w5, STRIDE);
            acc50 = simd::dpbusd(acc50, in0, w5_0);
            acc51 = simd::dpbusd(acc51, in1, w5_1);
            acc52 = simd::dpbusd(acc52, in2, w5_2);
            acc53 = simd::dpbusd(acc53, in3, w5_3);

            let (w6_0, w6_1, w6_2, w6_3) = simd::load_i8x4(w6, STRIDE);
            acc60 = simd::dpbusd(acc60, in0, w6_0);
            acc61 = simd::dpbusd(acc61, in1, w6_1);
            acc62 = simd::dpbusd(acc62, in2, w6_2);
            acc63 = simd::dpbusd(acc63, in3, w6_3);

            let (w7_0, w7_1, w7_2, w7_3) = simd::load_i8x4(w7, STRIDE);
            acc70 = simd::dpbusd(acc70, in0, w7_0);
            acc71 = simd::dpbusd(acc71, in1, w7_1);
            acc72 = simd::dpbusd(acc72, in2, w7_2);
            acc73 = simd::dpbusd(acc73, in3, w7_3);

            in_ptr = unsafe { in_ptr.add(4 * STRIDE) };
            w0 = unsafe { w0.add(4 * STRIDE) };
            w1 = unsafe { w1.add(4 * STRIDE) };
            w2 = unsafe { w2.add(4 * STRIDE) };
            w3 = unsafe { w3.add(4 * STRIDE) };
            w4 = unsafe { w4.add(4 * STRIDE) };
            w5 = unsafe { w5.add(4 * STRIDE) };
            w6 = unsafe { w6.add(4 * STRIDE) };
            w7 = unsafe { w7.add(4 * STRIDE) };
        }

        while in_ptr < end_ptr {
            let in_chunk = simd::load_u8(in_ptr);

            let ww0 = simd::load_i8(w0);
            acc00 = simd::dpbusd(acc00, in_chunk, ww0);
            w0 = unsafe { w0.add(STRIDE) };

            let ww1 = simd::load_i8(w1);
            acc10 = simd::dpbusd(acc10, in_chunk, ww1);
            w1 = unsafe { w1.add(STRIDE) };

            let ww2 = simd::load_i8(w2);
            acc20 = simd::dpbusd(acc20, in_chunk, ww2);
            w2 = unsafe { w2.add(STRIDE) };

            let ww3 = simd::load_i8(w3);
            acc30 = simd::dpbusd(acc30, in_chunk, ww3);
            w3 = unsafe { w3.add(STRIDE) };

            let ww4 = simd::load_i8(w4);
            acc40 = simd::dpbusd(acc40, in_chunk, ww4);
            w4 = unsafe { w4.add(STRIDE) };

            let ww5 = simd::load_i8(w5);
            acc50 = simd::dpbusd(acc50, in_chunk, ww5);
            w5 = unsafe { w5.add(STRIDE) };

            let ww6 = simd::load_i8(w6);
            acc60 = simd::dpbusd(acc60, in_chunk, ww6);
            w6 = unsafe { w6.add(STRIDE) };

            let ww7 = simd::load_i8(w7);
            acc70 = simd::dpbusd(acc70, in_chunk, ww7);
            w7 = unsafe { w7.add(STRIDE) };

            in_ptr = unsafe { in_ptr.add(STRIDE) };
        }

        let combined0 = simd::add_i32(simd::add_i32(acc00, acc01), simd::add_i32(acc02, acc03));
        let raw0 = simd::horizontal_sum_i32_single(combined0);
        let shifted0 = (raw0 >> L1_SHIFT) + biases[out_idx + 0];
        let clamped0 = shifted0.clamp(0, Q as i32);
        output[out_idx + 0] = clamped0 * clamped0;

        let combined1 = simd::add_i32(simd::add_i32(acc10, acc11), simd::add_i32(acc12, acc13));
        let raw1 = simd::horizontal_sum_i32_single(combined1);
        let shifted1 = (raw1 >> L1_SHIFT) + biases[out_idx + 1];
        let clamped1 = shifted1.clamp(0, Q as i32);
        output[out_idx + 1] = clamped1 * clamped1;

        let combined2 = simd::add_i32(simd::add_i32(acc20, acc21), simd::add_i32(acc22, acc23));
        let raw2 = simd::horizontal_sum_i32_single(combined2);
        let shifted2 = (raw2 >> L1_SHIFT) + biases[out_idx + 2];
        let clamped2 = shifted2.clamp(0, Q as i32);
        output[out_idx + 2] = clamped2 * clamped2;

        let combined3 = simd::add_i32(simd::add_i32(acc30, acc31), simd::add_i32(acc32, acc33));
        let raw3 = simd::horizontal_sum_i32_single(combined3);
        let shifted3 = (raw3 >> L1_SHIFT) + biases[out_idx + 3];
        let clamped3 = shifted3.clamp(0, Q as i32);
        output[out_idx + 3] = clamped3 * clamped3;

        let combined4 = simd::add_i32(simd::add_i32(acc40, acc41), simd::add_i32(acc42, acc43));
        let raw4 = simd::horizontal_sum_i32_single(combined4);
        let shifted4 = (raw4 >> L1_SHIFT) + biases[out_idx + 4];
        let clamped4 = shifted4.clamp(0, Q as i32);
        output[out_idx + 4] = clamped4 * clamped4;

        let combined5 = simd::add_i32(simd::add_i32(acc50, acc51), simd::add_i32(acc52, acc53));
        let raw5 = simd::horizontal_sum_i32_single(combined5);
        let shifted5 = (raw5 >> L1_SHIFT) + biases[out_idx + 5];
        let clamped5 = shifted5.clamp(0, Q as i32);
        output[out_idx + 5] = clamped5 * clamped5;

        let combined6 = simd::add_i32(simd::add_i32(acc60, acc61), simd::add_i32(acc62, acc63));
        let raw6 = simd::horizontal_sum_i32_single(combined6);
        let shifted6 = (raw6 >> L1_SHIFT) + biases[out_idx + 6];
        let clamped6 = shifted6.clamp(0, Q as i32);
        output[out_idx + 6] = clamped6 * clamped6;

        let combined7 = simd::add_i32(simd::add_i32(acc70, acc71), simd::add_i32(acc72, acc73));
        let raw7 = simd::horizontal_sum_i32_single(combined7);
        let shifted7 = (raw7 >> L1_SHIFT) + biases[out_idx + 7];
        let clamped7 = shifted7.clamp(0, Q as i32);
        output[out_idx + 7] = clamped7 * clamped7;

        out_idx += OUT_UNROLL;
    }

    // handle remainder outputs
    while out_idx < L2_SIZE {
        let weights = &NETWORK.l1_weights[output_bucket][out_idx];

        let mut acc0 = simd::splat_i32(0);
        let mut acc1 = simd::splat_i32(0);
        let mut acc2 = simd::splat_i32(0);
        let mut acc3 = simd::splat_i32(0);

        let mut in_ptr = input.as_ptr();
        let mut w_ptr = weights.as_ptr();
        let end_ptr = unsafe { input.as_ptr().add(L1_SIZE) };

        while unsafe { in_ptr.add(4 * STRIDE) } <= end_ptr {
            let input0 = simd::load_u8(in_ptr);
            let input1 = simd::load_u8(unsafe { in_ptr.add(STRIDE) });
            let input2 = simd::load_u8(unsafe { in_ptr.add(2 * STRIDE) });
            let input3 = simd::load_u8(unsafe { in_ptr.add(3 * STRIDE) });

            let weight0 = simd::load_i8(w_ptr);
            let weight1 = simd::load_i8(unsafe { w_ptr.add(STRIDE) });
            let weight2 = simd::load_i8(unsafe { w_ptr.add(2 * STRIDE) });
            let weight3 = simd::load_i8(unsafe { w_ptr.add(3 * STRIDE) });

            acc0 = simd::dpbusd(acc0, input0, weight0);
            acc1 = simd::dpbusd(acc1, input1, weight1);
            acc2 = simd::dpbusd(acc2, input2, weight2);
            acc3 = simd::dpbusd(acc3, input3, weight3);

            in_ptr = unsafe { in_ptr.add(4 * STRIDE) };
            w_ptr = unsafe { w_ptr.add(4 * STRIDE) };
        }
        while in_ptr < end_ptr {
            let input_chunk = simd::load_u8(in_ptr);
            let weight_chunk = simd::load_i8(w_ptr);
            acc0 = simd::dpbusd(acc0, input_chunk, weight_chunk);
            in_ptr = unsafe { in_ptr.add(STRIDE) };
            w_ptr = unsafe { w_ptr.add(STRIDE) };
        }

        let combined = simd::add_i32(simd::add_i32(acc0, acc1), simd::add_i32(acc2, acc3));
        let raw = simd::horizontal_sum_i32_single(combined);
        let shifted = (raw >> L1_SHIFT) + biases[out_idx];
        let clamped = shifted.clamp(0, Q as i32);
        output[out_idx] = clamped * clamped;

        out_idx += 1;
    }

    output
}

/// L2 propagation
pub unsafe fn propagate_l2(input: &[i32; L2_SIZE], output_bucket: usize) -> [i32; L3_SIZE] {
    const LANES: usize = L3_SIZE / simd::I32_LANES;
    let weights = &NETWORK.l2_weights[output_bucket];
    let biases = &NETWORK.l2_biases[output_bucket];

    let mut acc = [simd::splat_i32(0); LANES];
    for (lane, acc_lane) in acc.iter_mut().enumerate() {
        *acc_lane = simd::load_i32(biases.as_ptr().add(lane * simd::I32_LANES));
    }

    for (input_neuron, &input_val_scalar) in input.iter().enumerate() {
        let input_val = simd::splat_i32(input_val_scalar);
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
