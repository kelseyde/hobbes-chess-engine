#[cfg(target_feature = "avx512f")]
pub(crate) mod avx512 {
    use crate::evaluation::arch::{HIDDEN, QA};
    use std::arch::x86_64::*;

    const CHUNK_SIZE: usize = 32; // 32 i16 elements per 512-bit vector
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    #[inline]
    pub unsafe fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        debug_assert_eq!(HIDDEN % CHUNK_SIZE, 0);
        let mut sum = _mm512_setzero_si512();
        let mut i = 0;
        while i < LOOP_LENGTH {
            let f = _mm512_load_si512(features.as_ptr().add(i * CHUNK_SIZE) as *const __m512i);
            let w = _mm512_load_si512(weights.as_ptr().add(i * CHUNK_SIZE) as *const __m512i);
            let clipped = clipped_relu(f);
            let prod = _mm512_mullo_epi16(clipped, w);
            // madd pairs 16-bit into 32-bit: (prod_lo * clipped_lo) + (prod_hi * clipped_hi)
            let pair = _mm512_madd_epi16(prod, clipped);
            sum = _mm512_add_epi32(sum, pair);
            i += 1;
        }
        horizontal_add(sum)
    }

    #[inline]
    unsafe fn clipped_relu(v: __m512i) -> __m512i {
        let zero = _mm512_set1_epi16(0);
        let qmax = _mm512_set1_epi16(QA as i16);
        _mm512_min_epi16(_mm512_max_epi16(v, zero), qmax)
    }

    #[inline]
    unsafe fn horizontal_add(v: __m512i) -> i32 {
        // Reduce 16 lanes of i32 in the 512-bit vector by splitting into 2x256, then 2x128, etc.
        let lo256 = _mm512_castsi512_si256(v);
        let hi256 = _mm512_extracti64x4_epi64::<1>(v);
        let sum256 = _mm256_add_epi32(lo256, hi256);

        let hi128 = _mm256_extracti128_si256::<1>(sum256);
        let lo128 = _mm256_castsi256_si128(sum256);
        let sum128 = _mm_add_epi32(lo128, hi128);

        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(sum128, hi64);

        let hi32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum64);
        let sum32 = _mm_add_epi32(sum64, hi32);
        _mm_cvtsi128_si32(sum32)
    }
}

// AVX2 path is compiled only if AVX512F not enabled (so AVX512 takes precedence)
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub(crate) mod avx2 {
    use crate::evaluation::arch::{HIDDEN, QA};
    use std::arch::x86_64::*;

    const CHUNK_SIZE: usize = 16;
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    pub unsafe fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        let mut sum = _mm256_setzero_si256();
        let mut i = 0;
        while i < LOOP_LENGTH {
            let f = _mm256_load_si256(features.as_ptr().add(i * CHUNK_SIZE).cast());
            let w = _mm256_load_si256(weights.as_ptr().add(i * CHUNK_SIZE).cast());
            let clipped = clipped_relu(f);
            let v = _mm256_mullo_epi16(clipped, w);
            let mul = _mm256_madd_epi16(v, clipped);
            sum = _mm256_add_epi32(sum, mul);
            i += 1;
        }
        horizontal_add(sum)
    }

    #[inline]
    unsafe fn horizontal_add(sum: __m256i) -> i32 {
        let upper_128 = _mm256_extracti128_si256::<1>(sum);
        let lower_128 = _mm256_castsi256_si128(sum);
        let sum_128 = _mm_add_epi32(upper_128, lower_128);
        let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
        let sum_64 = _mm_add_epi32(upper_64, sum_128);
        let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
        let sum_32 = _mm_add_epi32(upper_32, sum_64);
        _mm_cvtsi128_si32(sum_32)
    }

    #[inline]
    unsafe fn clipped_relu(i: __m256i) -> __m256i {
        let min = _mm256_set1_epi16(0);
        let max = _mm256_set1_epi16(QA as i16);
        _mm256_min_epi16(_mm256_max_epi16(i, min), max)
    }
}

// Scalar fallback if neither AVX512F nor AVX2 is enabled
#[cfg(all(not(target_feature = "avx2"), not(target_feature = "avx512f")))]
pub(crate) mod scalar {
    use crate::board::Board;
    use crate::evaluation::arch::{L0_QUANT, L0_SHIFT, L1_QUANT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, NETWORK, OUTPUT_BUCKET_COUNT, Q, SCALE};

    pub fn forward(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE], board: &Board) -> i32 {
        let output_bucket = get_output_bucket(board);
        let l0_outputs = activate_l0(us, them);
        let l1_outputs = propagate_l1(&l0_outputs, output_bucket);
        let l2_outputs = propagate_l2(&l1_outputs, output_bucket);
        let l3_output = propagate_l3(&l2_outputs, output_bucket);
        let mut output = l3_output;
        output /= Q;
        output *= SCALE;
        output /= Q * Q * Q;
        output
    }

    /// L0 ('feature transformer') activation
    /// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
    pub fn activate_l0(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE]) -> [u8; L1_SIZE] {
        let mut output = [0; L1_SIZE];

        for (side, feats) in [us, them].into_iter().enumerate() {
            let base = side * (L1_SIZE / 2);
            for i in 0..(L1_SIZE / 2) {
                // Load the pair of inputs to be multiplied.
                let left: i16 = feats[i];
                let right: i16 = feats[i + (L1_SIZE / 2)];

                // Clipped ReLU activation
                let l_clamped: u8 = left.clamp(0, L0_QUANT as i16) as u8;
                let r_clamped: u8 = right.clamp(0, L0_QUANT as i16) as u8;

                // Pairwise multiplication of left and right input.
                let multiplied: i32 = l_clamped as i32 * r_clamped as i32;

                // Downshift back into [0, 127].
                // Note: this is equivalent to the << 7 >> 16 that mulhi does.
                let result: u8 = (multiplied >> L0_SHIFT).clamp(0, 255) as u8;
                output[base + i] = result;
            }
        }
        output
    }

    /// L1 propagation
    fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE] {
        let weights = &NETWORK.l1_weights[output_bucket];
        let biases = &NETWORK.l1_biases[output_bucket];

        // Unactivated L1 outputs in the quantized space L0_QUANT * L1_QUANT
        let mut intermediate: [i32; L2_SIZE] = [0; L2_SIZE];

        // L1 matrix multiplication
        for input_idx in 0..L1_SIZE {
            let input: i32 = input[input_idx] as i32;
            for output_idx in 0..L2_SIZE {
                let weight: i32 = weights[input_idx * output_idx] as i32;
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
            let clamped: i32 = out.clamp(0, Q);
            let activated = clamped * clamped;

            output[i] = activated;
        }


        output
    }

    /// L2 propagation
    fn propagate_l2(input: &[i32; L2_SIZE], output_bucket: usize) -> [i32; L3_SIZE] {
        let weights = &NETWORK.l2_weights[output_bucket];

        let mut out = NETWORK.l2_biases[output_bucket];
        for input_idx in 0..L2_SIZE {
            let input = input[input_idx];
            for output_idx in 0..L3_SIZE {
                let w_idx = input_idx * L3_SIZE + output_idx;
                let weight = weights[w_idx];
                out[output_idx] += input * weight;
            }
        }
        out
    }

    /// L3 propagation
    fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32 {
        let weights = &NETWORK.l3_weights[output_bucket];
        let bias = NETWORK.l3_biases[output_bucket];

        let mut output: i32 = bias;
        for (&input, &weight) in input.iter().zip(weights.iter()) {
            let clamped = input.clamp(0, Q * Q * Q);
            output += clamped * weight;
        }
        output
    }

    fn get_output_bucket(board: &Board) -> usize {
        const DIVISOR: usize = 32 / OUTPUT_BUCKET_COUNT;
        let occ_count = board.occ().count() as usize;
        (occ_count - 2) / DIVISOR
    }
}

// activte ft: q0
// l1: gets shifted by this thing: comptime Q0_BITS * 2 - 9 + Q1_BITS - Q_BITS;
