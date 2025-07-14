#[cfg(target_feature = "avx2")]
pub(crate) mod avx2 {
    use std::arch::x86_64::*;
    use crate::network::{HIDDEN, QA};

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
        {
            let mut sum = vdupq_n_s32(0);
            for i in 0..LOOP_LENGTH {
                let features = vld1q_s16(features.as_ptr().add(i * CHUNK_SIZE));
                let weights = vld1q_s16(weights.as_ptr().add(i * CHUNK_SIZE));
                let clipped = clipped_relu(features);
                let v = vmulq_s16(clipped, weights);

                let clipped_low = vget_low_s16(clipped);
                let clipped_high = vget_high_s16(clipped);
                let v_low = vget_low_s16(v);
                let v_high = vget_high_s16(v);

                let mul_low = vmull_s16(v_low, clipped_low);
                let mul_high = vmull_s16(v_high, clipped_high);

                sum = vaddq_s32(sum, mul_low);
                sum = vaddq_s32(sum, mul_high);
            }
            horizontal_add(sum)
        }
    }

    #[inline]
    unsafe fn horizontal_add(sum: int32x4_t) -> i32 {
        {
            let sum_pair = vpadd_s32(vget_low_s32(sum), vget_high_s32(sum));
            let final_sum = vpadd_s32(sum_pair, sum_pair);
            vget_lane_s32(final_sum, 0)
        }
    }

    #[inline]
    unsafe fn clipped_relu(i: int16x8_t) -> int16x8_t {
        let min = vdupq_n_s16(0);
        let max = vdupq_n_s16(QA as i16);
        vminq_s16(vmaxq_s16(i, min), max)
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