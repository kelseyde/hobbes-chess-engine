#[cfg(target_feature = "avx512f")]
pub(crate) mod avx512 {
    use crate::evaluation::network::{Align64, Block, HIDDEN, QA};
    use std::arch::x86_64::*;

    const CHUNK_SIZE: usize = 32; // 32 i16 elements per 512-bit vector
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    #[inline]
    pub unsafe fn forward(features: &Align64<Block>, weights: &Align64<Block>) -> i32 {
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
    use crate::evaluation::network::{Align64, Block, HIDDEN, QA};
    use std::arch::x86_64::*;

    const CHUNK_SIZE: usize = 16;
    const LOOP_LENGTH: usize = HIDDEN / CHUNK_SIZE;

    pub unsafe fn forward(features: &Align64<Block>, weights: &Align64<Block>) -> i32 {
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
    use crate::evaluation::network::{Align64, Block, QA};

    pub fn forward(features: &Align64<Block>, weights: &Align64<Block>) -> i32 {
        let mut output = 0;
        for (&input, &weight) in features.iter().zip(weights.iter()) {
            let clipped = input.clamp(0, QA as i16);
            let result = clipped * weight;
            output += result as i32 * clipped as i32;
        }
        output
    }
}
