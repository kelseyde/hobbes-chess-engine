use std::{arch::x86_64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<__m256i>() / size_of::<i16>();

pub unsafe fn splat_i16(a: i16) -> __m256i {
    _mm256_set1_epi16(a)
}

pub unsafe fn clamp_i16(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_max_epi16(_mm256_min_epi16(x, max), min)
}

pub unsafe fn shift_left_i16<const SHIFT: i32>(a: __m256i) -> __m256i {
    _mm256_slli_epi16::<SHIFT>(a)
}

pub unsafe fn mul_high_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_mulhi_epi16(a, b)
}

pub unsafe fn packus(a: __m256i, b: __m256i) -> __m256i {
    _mm256_packus_epi16(a, b)
}

pub unsafe fn permute(a: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0b11_01_10_00>(a)
}