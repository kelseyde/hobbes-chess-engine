use hobbes_nnue_arch::L0_SHIFT;
use std::{arch::x86_64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<__m256i>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<__m256i>() / size_of::<i32>();
pub const I8_LANES: usize = size_of::<__m256i>() / size_of::<i8>();

#[inline(always)]
pub unsafe fn load_u8(ptr: *const u8) -> __m256i {
    _mm256_loadu_si256(ptr as *const __m256i)
}

#[inline(always)]
pub unsafe fn store_u8(ptr: *mut u8, v: __m256i) {
    _mm256_storeu_si256(ptr as *mut __m256i, v)
}

#[inline(always)]
pub unsafe fn load_i8(ptr: *const i8) -> __m256i {
    _mm256_loadu_si256(ptr as *const __m256i)
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> __m256i {
    _mm256_set1_epi16(a)
}

#[inline(always)]
pub unsafe fn load_i16(ptr: *const i16) -> __m256i {
    _mm256_loadu_si256(ptr as *const __m256i)
}

#[inline(always)]
pub unsafe fn store_i16(ptr: *mut i16, v: __m256i) {
    _mm256_storeu_si256(ptr as *mut __m256i, v)
}

#[inline(always)]
pub unsafe fn add_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi16(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi16(a, b)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_max_epi16(_mm256_min_epi16(x, max), min)
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> __m256i {
    _mm256_set1_epi32(a)
}

#[inline(always)]
pub unsafe fn load_i32(ptr: *const i32) -> __m256i {
    _mm256_loadu_si256(ptr as *const __m256i)
}

#[inline(always)]
pub unsafe fn store_i32(ptr: *mut i32, v: __m256i) {
    _mm256_storeu_si256(ptr as *mut __m256i, v)
}

#[inline(always)]
pub unsafe fn add_i32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

#[inline(always)]
pub unsafe fn mul_i32(a: __m256i, b: __m256i) -> __m256i {
    _mm256_mullo_epi32(a, b)
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    _mm256_add_epi32(_mm256_mullo_epi32(a, b), c)
}

#[inline(always)]
pub unsafe fn clamp_i32(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_max_epi32(_mm256_min_epi32(x, max), min)
}

#[inline(always)]
pub unsafe fn shift_left_mul_high_i16(a: __m256i, b: __m256i) -> __m256i {
    const SHIFT: i32 = 16 - L0_SHIFT as i32;
    _mm256_mulhi_epi16(_mm256_slli_epi16::<SHIFT>(a), b)
}

#[inline(always)]
pub unsafe fn packus(a: __m256i, b: __m256i) -> __m256i {
    _mm256_packus_epi16(a, b)
}

#[inline(always)]
pub unsafe fn dpbusd(acc: __m256i, u8s: __m256i, i8s: __m256i) -> __m256i {
    let products = _mm256_maddubs_epi16(u8s, i8s);
    let ones = _mm256_set1_epi16(1);
    let summed = _mm256_madd_epi16(products, ones);
    _mm256_add_epi32(acc, summed)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32_single(a: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256::<1>(a);
    let lo128 = _mm256_castsi256_si128(a);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum64);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32<const N: usize>(a: [__m256i; N]) -> i32 {
    let mut acc = a[0];
    for i in 1..N {
        acc = _mm256_add_epi32(acc, a[i]);
    }
    horizontal_sum_i32_single(acc)
}
