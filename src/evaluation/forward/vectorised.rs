use crate::evaluation::{simd, NETWORK};
use hobbes_nnue_arch::{L0_QUANT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q, Q_BITS};

/// L0 ('feature transformer') activation with NNZ tracking.
/// Produces u8 activations and a list of non-zero i32-block indices.
pub unsafe fn activate_l0(
    us: &[i16; L1_SIZE],
    them: &[i16; L1_SIZE],
) -> ([u8; L1_SIZE], [u16; L1_SIZE / 4], usize) {
    let mut output = [0u8; L1_SIZE];
    let mut nnz = [0u16; L1_SIZE / 4];

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

    let nnz_count = simd::find_nnz(&output, &mut nnz);
    (output, nnz, nnz_count)
}

/// L1 propagation — sparse matrix multiplication.
/// Instead of iterating over all L1_SIZE inputs, we only process non-zero i32 blocks
/// (groups of 4 u8 activations). This skips the many zero activations produced by
/// the pairwise CReLU activation.
///
/// Weight layout is sparse-friendly: [input_block][output * 4 + byte_within_block].
/// For a given input block `idx`, the L2_SIZE * 4 = 64 weight bytes are contiguous,
/// with each group of 4 bytes belonging to one output neuron. This allows dpbusd
/// with splatted activations: each SIMD lane dot-products the same 4 activation bytes
/// against 4 different weight bytes for a different output neuron.
pub unsafe fn propagate_l1(
    input: &[u8; L1_SIZE],
    nnz: &[u16],
    output_bucket: usize,
) -> [i32; L2_SIZE * 2] {
    let biases = &NETWORK.l1_biases[output_bucket];
    let weights = NETWORK.l1_weights[output_bucket].as_ptr();

    const L1_CHUNK_PER_32: usize = 4;
    let input32 = input.as_ptr() as *const i32;

    // Number of SIMD vectors needed to cover L2_SIZE output neurons.
    // Each i32 SIMD lane computes the dot product for one output neuron.
    // L2_SIZE * 4 bytes of weights per input block.
    //   NEON  (I32_LANES=4,  I8_BYTES=16):  64/16 = 4 vectors
    //   AVX2  (I32_LANES=8,  I8_BYTES=32):  64/32 = 2 vectors
    //   AVX512(I32_LANES=16, I8_BYTES=64):  64/64 = 1 vector
    const WEIGHT_BYTES_PER_BLOCK: usize = L2_SIZE * L1_CHUNK_PER_32;
    const LANES: usize = WEIGHT_BYTES_PER_BLOCK / (simd::I32_LANES * L1_CHUNK_PER_32);

    // 4 accumulator sets for instruction-level parallelism
    let mut acc = [[simd::splat_i32(0); LANES]; 4];

    let nnz_count = nnz.len();
    let mut nnz_idx = 0;

    // Main loop: process 4 NNZ entries at a time
    while nnz_idx + 4 <= nnz_count {
        let idx0 = *nnz.get_unchecked(nnz_idx) as usize;
        let idx1 = *nnz.get_unchecked(nnz_idx + 1) as usize;
        let idx2 = *nnz.get_unchecked(nnz_idx + 2) as usize;
        let idx3 = *nnz.get_unchecked(nnz_idx + 3) as usize;

        // Splat each 4-byte activation block across all SIMD lanes
        let in0 = simd::reinterpret_i32_as_i8(simd::splat_i32(*input32.add(idx0)));
        let in1 = simd::reinterpret_i32_as_i8(simd::splat_i32(*input32.add(idx1)));
        let in2 = simd::reinterpret_i32_as_i8(simd::splat_i32(*input32.add(idx2)));
        let in3 = simd::reinterpret_i32_as_i8(simd::splat_i32(*input32.add(idx3)));

        // Weight pointers: for input block `idx`, weights start at idx * WEIGHT_BYTES_PER_BLOCK
        let w_base0 = weights.add(idx0 * WEIGHT_BYTES_PER_BLOCK);
        let w_base1 = weights.add(idx1 * WEIGHT_BYTES_PER_BLOCK);
        let w_base2 = weights.add(idx2 * WEIGHT_BYTES_PER_BLOCK);
        let w_base3 = weights.add(idx3 * WEIGHT_BYTES_PER_BLOCK);

        for lane in 0..LANES {
            let off = lane * simd::I32_LANES * L1_CHUNK_PER_32;
            let w0 = simd::load_i8(w_base0.add(off));
            let w1 = simd::load_i8(w_base1.add(off));
            let w2 = simd::load_i8(w_base2.add(off));
            let w3 = simd::load_i8(w_base3.add(off));

            acc[0][lane] = simd::dpbusd(acc[0][lane], in0, w0);
            acc[1][lane] = simd::dpbusd(acc[1][lane], in1, w1);
            acc[2][lane] = simd::dpbusd(acc[2][lane], in2, w2);
            acc[3][lane] = simd::dpbusd(acc[3][lane], in3, w3);
        }

        nnz_idx += 4;
    }

    // Tail: remaining NNZ entries
    while nnz_idx < nnz_count {
        let idx = *nnz.get_unchecked(nnz_idx) as usize;
        let in_vec = simd::reinterpret_i32_as_i8(simd::splat_i32(*input32.add(idx)));
        let w_base = weights.add(idx * WEIGHT_BYTES_PER_BLOCK);

        for lane in 0..LANES {
            let off = lane * simd::I32_LANES * L1_CHUNK_PER_32;
            let w = simd::load_i8(w_base.add(off));
            acc[0][lane] = simd::dpbusd(acc[0][lane], in_vec, w);
        }

        nnz_idx += 1;
    }

    // Sum accumulator sets and apply shift, bias, and dual activation
    let mut output = [0i32; L2_SIZE * 2];

    for lane in 0..LANES {
        let combined = simd::add_i32(
            simd::add_i32(acc[0][lane], acc[1][lane]),
            simd::add_i32(acc[2][lane], acc[3][lane]),
        );
        simd::store_i32(output.as_mut_ptr().add(lane * simd::I32_LANES), combined);
    }

    // Apply shift, bias, and dual activation (crelu + csrelu)
    for i in 0..L2_SIZE {
        let shifted = (output[i] >> L1_SHIFT) + biases[i];
        let crelu = shifted.clamp(0, Q as i32) << Q_BITS;
        let csrelu = (shifted * shifted).clamp(0, (Q * Q) as i32);
        output[i] = crelu;
        output[i + L2_SIZE] = csrelu;
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


