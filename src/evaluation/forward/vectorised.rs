use crate::evaluation::forward::Forward;
use crate::evaluation::{simd, sparse, NETWORK};
use hobbes_nnue_arch::{L0_QUANT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q, Q_BITS};

pub struct Vectorised;

impl Forward for Vectorised {
    /// L0 ('feature transformer') activation
    /// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
    unsafe fn activate_l0(
        us_psq: &[i16; L1_SIZE],
        us_threat: &[i16; L1_SIZE],
        them_psq: &[i16; L1_SIZE],
        them_threat: &[i16; L1_SIZE],
        output: &mut [u8; L1_SIZE],
    ) {
        let lo = simd::i16::splat(0);
        let hi = simd::i16::splat(L0_QUANT as i16);

        for (side, (psq_feats, threat_feats)) in [(us_psq, us_threat), (them_psq, them_threat)]
            .into_iter()
            .enumerate()
        {
            let base = side * (L1_SIZE / 2);

            for i in (0..L1_SIZE / 2).step_by(2 * simd::i16::LANES) {
                let psq_left1 = simd::i16::load(psq_feats.as_ptr().add(i));
                let psq_left2 = simd::i16::load(psq_feats.as_ptr().add(i + simd::i16::LANES));
                let psq_right1 = simd::i16::load(psq_feats.as_ptr().add(i + L1_SIZE / 2));
                let psq_right2 =
                    simd::i16::load(psq_feats.as_ptr().add(i + L1_SIZE / 2 + simd::i16::LANES));

                let threat_left1 = simd::i16::load(threat_feats.as_ptr().add(i));
                let threat_left2 = simd::i16::load(threat_feats.as_ptr().add(i + simd::i16::LANES));
                let threat_right1 = simd::i16::load(threat_feats.as_ptr().add(i + L1_SIZE / 2));
                let threat_right2 = simd::i16::load(
                    threat_feats
                        .as_ptr()
                        .add(i + L1_SIZE / 2 + simd::i16::LANES),
                );

                let left1_clipped =
                    simd::i16::clamp(simd::i16::add(psq_left1, threat_left1), lo, hi);
                let left2_clipped =
                    simd::i16::clamp(simd::i16::add(psq_left2, threat_left2), lo, hi);
                let right1_clipped =
                    simd::i16::clamp(simd::i16::add(psq_right1, threat_right1), lo, hi);
                let right2_clipped =
                    simd::i16::clamp(simd::i16::add(psq_right2, threat_right2), lo, hi);

                let product1 = simd::i16::shift_left_mul_high(left1_clipped, right1_clipped);
                let product2 = simd::i16::shift_left_mul_high(left2_clipped, right2_clipped);

                let packed = simd::packus(product1, product2);
                simd::u8::store(output.as_mut_ptr().add(i + base), packed);
            }
        }
    }

    /// L1 propagation
    unsafe fn propagate_l1(
        input: &[u8; L1_SIZE],
        output_bucket: usize,
        output: &mut [i32; L2_SIZE * 2],
    ) {
        const ACC_LANES: usize = L2_SIZE / simd::I32_LANES;
        const WEIGHT_STRIDE: usize = simd::I32_LANES * 4;
        const UNROLL: usize = 4;

        let zero = simd::i32::zero();
        let mut acc = [[zero; UNROLL]; ACC_LANES];

        let (nonzero_indices, num_nonzero_indices) = sparse::find_nonzero_indices(input);
        let input_i32 = input.as_ptr() as *const i32;
        let weights = NETWORK.l1_weights[output_bucket].as_ptr() as *const i8;

        let mut nnz = 0;
        while nnz + 2 * UNROLL <= num_nonzero_indices {
            for (lane, acc_lane) in acc.iter_mut().enumerate() {
                let offset = lane * WEIGHT_STRIDE;
                for sub in 0..UNROLL {
                    let idx_1 = nonzero_indices[nnz + 2 * sub] as usize;
                    let idx_2 = nonzero_indices[nnz + 2 * sub + 1] as usize;
                    let feat_1 = simd::i32::splat_as_u8(*input_i32.add(idx_1));
                    let feat_2 = simd::i32::splat_as_u8(*input_i32.add(idx_2));
                    let weight_1 = simd::i8::load(weights.add(idx_1 * L2_SIZE * 4 + offset));
                    let weight_2 = simd::i8::load(weights.add(idx_2 * L2_SIZE * 4 + offset));
                    acc_lane[sub] =
                        simd::dpbusdx2(acc_lane[sub], feat_1, weight_1, feat_2, weight_2);
                }
            }
            nnz += 2 * UNROLL;
        }

        while nnz < num_nonzero_indices {
            let idx = nonzero_indices[nnz] as usize;
            let feat = simd::i32::splat_as_u8(*input_i32.add(idx));
            for (lane, acc_lane) in acc.iter_mut().enumerate() {
                let offset = lane * WEIGHT_STRIDE;
                let weight = simd::i8::load(weights.add(idx * L2_SIZE * 4 + offset));
                acc_lane[0] = simd::dpbusd(acc_lane[0], feat, weight);
            }
            nnz += 1;
        }

        let bias_ptr = NETWORK.l1_biases[output_bucket].as_ptr() as *const simd::VecI32;
        let lo = simd::i32::zero();
        let hi = simd::i32::splat(Q as i32);
        let hi2 = simd::i32::splat((Q * Q) as i32);

        let out_ptr = output.as_mut_ptr() as *mut simd::VecI32;

        for (lane, acc_lane) in acc.iter().enumerate() {
            let half_0 = simd::i32::add(acc_lane[0], acc_lane[1]);
            let half_1 = simd::i32::add(acc_lane[2], acc_lane[3]);
            let sum = simd::i32::add(half_0, half_1);

            let bias = simd::i32::load(bias_ptr.add(lane) as *const i32);
            let shifted = simd::i32::shift_right::<{ L1_SHIFT as _ }>(simd::i32::add(sum, bias));

            let crelu = simd::i32::shift_left::<{ Q_BITS as _ }>(simd::i32::clamp(shifted, lo, hi));
            let csrelu = simd::i32::clamp(simd::i32::mul(shifted, shifted), lo, hi2);

            simd::i32::store(out_ptr.add(lane) as *mut i32, crelu);
            simd::i32::store(out_ptr.add(lane + ACC_LANES) as *mut i32, csrelu);
        }
    }

    /// L2 propagation
    unsafe fn propagate_l2(
        input: &[i32; L2_SIZE * 2],
        output_bucket: usize,
        output: &mut [i32; L3_SIZE],
    ) {
        const LANES: usize = L3_SIZE / simd::I32_LANES;
        let weights = &NETWORK.l2_weights[output_bucket];
        let biases = &NETWORK.l2_biases[output_bucket];

        let mut acc = [simd::i32::zero(); LANES];
        for (lane, acc_lane) in acc.iter_mut().enumerate() {
            *acc_lane = simd::i32::load(biases.as_ptr().add(lane * simd::I32_LANES));
        }

        for (&input_scalar, weight_row_arr) in input.iter().zip(weights.iter()) {
            let input_val = simd::i32::splat(input_scalar);
            let weight_row = weight_row_arr.as_ptr();

            let mut lane = 0;
            while lane + 4 <= LANES {
                let offset = lane * simd::I32_LANES;
                let weight_0 = simd::i32::load(weight_row.add(offset));
                let weight_1 = simd::i32::load(weight_row.add(offset + simd::I32_LANES));
                let weight_2 = simd::i32::load(weight_row.add(offset + 2 * simd::I32_LANES));
                let weight_3 = simd::i32::load(weight_row.add(offset + 3 * simd::I32_LANES));
                acc[lane] = simd::i32::mul_add(weight_0, input_val, acc[lane]);
                acc[lane + 1] = simd::i32::mul_add(weight_1, input_val, acc[lane + 1]);
                acc[lane + 2] = simd::i32::mul_add(weight_2, input_val, acc[lane + 2]);
                acc[lane + 3] = simd::i32::mul_add(weight_3, input_val, acc[lane + 3]);
                lane += 4;
            }
            while lane < LANES {
                let weight = simd::i32::load(weight_row.add(lane * simd::I32_LANES));
                acc[lane] = simd::i32::mul_add(weight, input_val, acc[lane]);
                lane += 1;
            }
        }

        for (lane, acc_lane) in acc.iter().enumerate() {
            simd::i32::store(output.as_mut_ptr().add(lane * simd::I32_LANES), *acc_lane);
        }
    }

    /// L3 propagation
    unsafe fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32 {
        const LANES: usize = L3_SIZE / simd::I32_LANES;

        let weights = NETWORK.l3_weights[output_bucket].as_ptr();
        let bias = NETWORK.l3_biases[output_bucket];
        let lo = simd::i32::zero();
        let hi = simd::i32::splat((Q * Q * Q) as i32);

        let mut acc = [simd::i32::zero(); LANES];
        for (lane, acc_lane) in acc.iter_mut().enumerate() {
            let offset = lane * simd::I32_LANES;
            let input_chunk = simd::i32::load(input.as_ptr().add(offset));
            let weight_chunk = simd::i32::load(weights.add(offset));
            let clamped = simd::i32::clamp(input_chunk, lo, hi);
            *acc_lane = simd::i32::mul(weight_chunk, clamped);
        }

        simd::i32::horizontal_sum(acc) + bias
    }
}
