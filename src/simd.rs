#[cfg(target_feature = "avx2")]
pub(crate) mod avx2 {
    use crate::network::{HIDDEN, QA};
    use std::arch::x86_64::*;

    const CHUNK_SIZE: usize = 16;
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    pub unsafe fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        {
            let mut sum = _mm256_setzero_si256();
            for i in 0..LOOP_LENGTH {
                let features = _mm256_load_si256(features.as_ptr().add(i * CHUNK_SIZE).cast());
                let weights = _mm256_load_si256(weights.as_ptr().add(i * CHUNK_SIZE).cast());
                let clipped = clipped_relu(features);
                let v = _mm256_mullo_epi16(clipped, weights);
                let mul = _mm256_madd_epi16(v, clipped);
                sum = _mm256_add_epi32(sum, mul);
            }
            horizontal_add(sum)
        }
    }

    #[inline]
    unsafe fn horizontal_add(sum: __m256i) -> i32 {
        {
            let upper_128 = _mm256_extracti128_si256::<1>(sum);
            let lower_128 = _mm256_castsi256_si128(sum);
            let sum_128 = _mm_add_epi32(upper_128, lower_128);
            let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
            let sum_64 = _mm_add_epi32(upper_64, sum_128);
            let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
            let sum_32 = _mm_add_epi32(upper_32, sum_64);
            _mm_cvtsi128_si32(sum_32)
        }
    }

    #[inline]
    unsafe fn clipped_relu(i: __m256i) -> __m256i {
        let min = _mm256_set1_epi16(0);
        let max = _mm256_set1_epi16(QA as i16);
        _mm256_min_epi16(_mm256_max_epi16(i, min), max)
    }
}

#[cfg(target_feature = "neon")]
pub(crate) mod neon {
    use std::arch::aarch64::*;
    use crate::network::{HIDDEN, QA};

    const CHUNK_SIZE: usize = 8;
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    pub unsafe fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        // Pre-compute constants outside the loop
        let min_val = vdupq_n_s16(0);
        let max_val = vdupq_n_s16(QA as i16);

        // Use two accumulators to improve instruction-level parallelism
        let mut sum1 = vdupq_n_s32(0);
        let mut sum2 = vdupq_n_s32(0);

        // Process two iterations at once when possible
        let mut i = 0;
        while i + 1 < LOOP_LENGTH {
            // First iteration
            let features1 = vld1q_s16(features.as_ptr().add(i * CHUNK_SIZE));
            let weights1 = vld1q_s16(weights.as_ptr().add(i * CHUNK_SIZE));
            let clipped1 = vminq_s16(vmaxq_s16(features1, min_val), max_val);
            let v1 = vmulq_s16(clipped1, weights1);

            // Second iteration
            let features2 = vld1q_s16(features.as_ptr().add((i + 1) * CHUNK_SIZE));
            let weights2 = vld1q_s16(weights.as_ptr().add((i + 1) * CHUNK_SIZE));
            let clipped2 = vminq_s16(vmaxq_s16(features2, min_val), max_val);
            let v2 = vmulq_s16(clipped2, weights2);

            // Use vmlal (multiply-accumulate long) for better efficiency
            // This does: sum += v_low * clipped_low (with widening)
            sum1 = vmlal_s16(sum1, vget_low_s16(v1), vget_low_s16(clipped1));
            sum1 = vmlal_s16(sum1, vget_high_s16(v1), vget_high_s16(clipped1));

            sum2 = vmlal_s16(sum2, vget_low_s16(v2), vget_low_s16(clipped2));
            sum2 = vmlal_s16(sum2, vget_high_s16(v2), vget_high_s16(clipped2));

            i += 2;
        }

        // Handle remaining iteration if odd number of iterations
        if i < LOOP_LENGTH {
            let features = vld1q_s16(features.as_ptr().add(i * CHUNK_SIZE));
            let weights = vld1q_s16(weights.as_ptr().add(i * CHUNK_SIZE));
            let clipped = vminq_s16(vmaxq_s16(features, min_val), max_val);
            let v = vmulq_s16(clipped, weights);

            sum1 = vmlal_s16(sum1, vget_low_s16(v), vget_low_s16(clipped));
            sum1 = vmlal_s16(sum1, vget_high_s16(v), vget_high_s16(clipped));
        }

        // Combine the two accumulators
        let final_sum = vaddq_s32(sum1, sum2);
        horizontal_add(final_sum)
    }

    #[inline]
    unsafe fn horizontal_add(sum: int32x4_t) -> i32 {
        // More efficient horizontal add using vpaddq
        let sum_pair = vpaddq_s32(sum, sum);  // [a+b, c+d, a+b, c+d]
        let final_sum = vpaddq_s32(sum_pair, sum_pair);  // [a+b+c+d, *, a+b+c+d, *]
        vgetq_lane_s32(final_sum, 0)
    }

    // Alternative version for very performance-critical code
    pub unsafe fn forward_unrolled(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        let min_val = vdupq_n_s16(0);
        let max_val = vdupq_n_s16(QA as i16);

        let mut sum1 = vdupq_n_s32(0);
        let mut sum2 = vdupq_n_s32(0);
        let mut sum3 = vdupq_n_s32(0);
        let mut sum4 = vdupq_n_s32(0);

        // Unroll loop by 4 for maximum throughput
        let mut i = 0;
        while i + 3 < LOOP_LENGTH {
            // Load and process 4 chunks at once
            let f1 = vld1q_s16(features.as_ptr().add(i * CHUNK_SIZE));
            let w1 = vld1q_s16(weights.as_ptr().add(i * CHUNK_SIZE));
            let f2 = vld1q_s16(features.as_ptr().add((i + 1) * CHUNK_SIZE));
            let w2 = vld1q_s16(weights.as_ptr().add((i + 1) * CHUNK_SIZE));
            let f3 = vld1q_s16(features.as_ptr().add((i + 2) * CHUNK_SIZE));
            let w3 = vld1q_s16(weights.as_ptr().add((i + 2) * CHUNK_SIZE));
            let f4 = vld1q_s16(features.as_ptr().add((i + 3) * CHUNK_SIZE));
            let w4 = vld1q_s16(weights.as_ptr().add((i + 3) * CHUNK_SIZE));

            let c1 = vminq_s16(vmaxq_s16(f1, min_val), max_val);
            let c2 = vminq_s16(vmaxq_s16(f2, min_val), max_val);
            let c3 = vminq_s16(vmaxq_s16(f3, min_val), max_val);
            let c4 = vminq_s16(vmaxq_s16(f4, min_val), max_val);

            let v1 = vmulq_s16(c1, w1);
            let v2 = vmulq_s16(c2, w2);
            let v3 = vmulq_s16(c3, w3);
            let v4 = vmulq_s16(c4, w4);

            sum1 = vmlal_s16(sum1, vget_low_s16(v1), vget_low_s16(c1));
            sum1 = vmlal_s16(sum1, vget_high_s16(v1), vget_high_s16(c1));

            sum2 = vmlal_s16(sum2, vget_low_s16(v2), vget_low_s16(c2));
            sum2 = vmlal_s16(sum2, vget_high_s16(v2), vget_high_s16(c2));

            sum3 = vmlal_s16(sum3, vget_low_s16(v3), vget_low_s16(c3));
            sum3 = vmlal_s16(sum3, vget_high_s16(v3), vget_high_s16(c3));

            sum4 = vmlal_s16(sum4, vget_low_s16(v4), vget_low_s16(c4));
            sum4 = vmlal_s16(sum4, vget_high_s16(v4), vget_high_s16(c4));

            i += 4;
        }

        // Handle remaining iterations
        while i < LOOP_LENGTH {
            let features = vld1q_s16(features.as_ptr().add(i * CHUNK_SIZE));
            let weights = vld1q_s16(weights.as_ptr().add(i * CHUNK_SIZE));
            let clipped = vminq_s16(vmaxq_s16(features, min_val), max_val);
            let v = vmulq_s16(clipped, weights);

            sum1 = vmlal_s16(sum1, vget_low_s16(v), vget_low_s16(clipped));
            sum1 = vmlal_s16(sum1, vget_high_s16(v), vget_high_s16(clipped));

            i += 1;
        }

        // Combine all accumulators
        let combined = vaddq_s32(vaddq_s32(sum1, sum2), vaddq_s32(sum3, sum4));
        horizontal_add(combined)
    }
}

#[cfg(all(not(target_feature = "avx2"), not(target_feature = "neon")))]
pub(crate) mod scalar {
    use crate::network::{HIDDEN, QA};

    pub fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        let mut output = 0;
        for (&input, &weight) in features.iter().zip(weights.iter()) {
            let clipped = input.clamp(0, QA as i16);
            let result = clipped * weight;
            output += result as i32 * clipped as i32;
        }
        output
    }
}