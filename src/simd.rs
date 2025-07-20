#[cfg(target_feature = "avx2")]
pub(crate) mod avx2 {
    use crate::network::{HIDDEN, QA};
    use std::arch::x86_64::*;

    const CHUNK_SIZE: usize = 16;
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    pub unsafe fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        let min_val = _mm256_set1_epi16(0);
        let max_val = _mm256_set1_epi16(QA as i16);

        let mut sum1 = _mm256_setzero_si256();
        let mut sum2 = _mm256_setzero_si256();
        let mut sum3 = _mm256_setzero_si256();
        let mut sum4 = _mm256_setzero_si256();

        let mut i = 0;
        while i + 3 < LOOP_LENGTH {
            let f1 = _mm256_load_si256(features.as_ptr().add(i * CHUNK_SIZE).cast());
            let w1 = _mm256_load_si256(weights.as_ptr().add(i * CHUNK_SIZE).cast());
            let f2 = _mm256_load_si256(features.as_ptr().add((i + 1) * CHUNK_SIZE).cast());
            let w2 = _mm256_load_si256(weights.as_ptr().add((i + 1) * CHUNK_SIZE).cast());
            let f3 = _mm256_load_si256(features.as_ptr().add((i + 2) * CHUNK_SIZE).cast());
            let w3 = _mm256_load_si256(weights.as_ptr().add((i + 2) * CHUNK_SIZE).cast());
            let f4 = _mm256_load_si256(features.as_ptr().add((i + 3) * CHUNK_SIZE).cast());
            let w4 = _mm256_load_si256(weights.as_ptr().add((i + 3) * CHUNK_SIZE).cast());

            let c1 = _mm256_min_epi16(_mm256_max_epi16(f1, min_val), max_val);
            let c2 = _mm256_min_epi16(_mm256_max_epi16(f2, min_val), max_val);
            let c3 = _mm256_min_epi16(_mm256_max_epi16(f3, min_val), max_val);
            let c4 = _mm256_min_epi16(_mm256_max_epi16(f4, min_val), max_val);

            let v1 = _mm256_mullo_epi16(c1, w1);
            let v2 = _mm256_mullo_epi16(c2, w2);
            let v3 = _mm256_mullo_epi16(c3, w3);
            let v4 = _mm256_mullo_epi16(c4, w4);

            let mul1 = _mm256_madd_epi16(v1, c1);
            let mul2 = _mm256_madd_epi16(v2, c2);
            let mul3 = _mm256_madd_epi16(v3, c3);
            let mul4 = _mm256_madd_epi16(v4, c4);

            sum1 = _mm256_add_epi32(sum1, mul1);
            sum2 = _mm256_add_epi32(sum2, mul2);
            sum3 = _mm256_add_epi32(sum3, mul3);
            sum4 = _mm256_add_epi32(sum4, mul4);

            i += 4;
        }

        while i < LOOP_LENGTH {
            let features = _mm256_load_si256(features.as_ptr().add(i * CHUNK_SIZE).cast());
            let weights = _mm256_load_si256(weights.as_ptr().add(i * CHUNK_SIZE).cast());
            let clipped = _mm256_min_epi16(_mm256_max_epi16(features, min_val), max_val);
            let v = _mm256_mullo_epi16(clipped, weights);
            let mul = _mm256_madd_epi16(v, clipped);
            sum1 = _mm256_add_epi32(sum1, mul);

            i += 1;
        }

        let combined12 = _mm256_add_epi32(sum1, sum2);
        let combined34 = _mm256_add_epi32(sum3, sum4);
        let final_sum = _mm256_add_epi32(combined12, combined34);

        horizontal_add(final_sum)
    }

    #[inline]
    unsafe fn horizontal_add(sum: __m256i) -> i32 {
        let sum_128_low = _mm256_castsi256_si128(sum);
        let sum_128_high = _mm256_extracti128_si256::<1>(sum);
        let sum_128 = _mm_add_epi32(sum_128_low, sum_128_high);

        let sum_64 = _mm_hadd_epi32(sum_128, sum_128);
        let final_sum = _mm_hadd_epi32(sum_64, sum_64);

        _mm_cvtsi128_si32(final_sum)
    }

}

#[cfg(target_feature = "neon")]
pub(crate) mod neon {
    use std::arch::aarch64::*;
    use crate::network::{HIDDEN, QA};

    const CHUNK_SIZE: usize = 8;
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    pub unsafe fn forward(features: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i32 {
        let min_val = vdupq_n_s16(0);
        let max_val = vdupq_n_s16(QA as i16);

        let mut sum1 = vdupq_n_s32(0);
        let mut sum2 = vdupq_n_s32(0);
        let mut sum3 = vdupq_n_s32(0);
        let mut sum4 = vdupq_n_s32(0);

        let mut i = 0;
        while i + 3 < LOOP_LENGTH {
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

        while i < LOOP_LENGTH {
            let features = vld1q_s16(features.as_ptr().add(i * CHUNK_SIZE));
            let weights = vld1q_s16(weights.as_ptr().add(i * CHUNK_SIZE));
            let clipped = vminq_s16(vmaxq_s16(features, min_val), max_val);
            let v = vmulq_s16(clipped, weights);

            sum1 = vmlal_s16(sum1, vget_low_s16(v), vget_low_s16(clipped));
            sum1 = vmlal_s16(sum1, vget_high_s16(v), vget_high_s16(clipped));

            i += 1;
        }

        let combined = vaddq_s32(vaddq_s32(sum1, sum2), vaddq_s32(sum3, sum4));
        horizontal_add(combined)
    }

    #[inline]
    unsafe fn horizontal_add(sum: int32x4_t) -> i32 {
        let sum_pair = vpaddq_s32(sum, sum);
        let final_sum = vpaddq_s32(sum_pair, sum_pair);
        vgetq_lane_s32(final_sum, 0)
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