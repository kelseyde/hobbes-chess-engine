use std::{arch::x86_64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<__m256i>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<__m256i>() / size_of::<i32>();
pub const I8_LANES: usize = size_of::<__m256i>() / size_of::<i8>();

pub unsafe fn splat_i16(a: i16) -> __m256i {
    _mm256_set1_epi16(a)
}

pub unsafe fn splat_i32(a: i32) -> __m256i {
    _mm256_set1_epi32(a)
}

pub unsafe fn clamp_i16(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_max_epi16(_mm256_min_epi16(x, max), min)
}

pub unsafe fn clamp_i32(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
    _mm256_max_epi32(_mm256_min_epi32(x, max), min)
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

/// Emulate vpdpbusd: dot-product of unsigned u8 × signed i8, accumulated into i32.
/// For each 32-bit lane: acc += u8[0]*i8[0] + u8[1]*i8[1] + u8[2]*i8[2] + u8[3]*i8[3]
#[inline(always)]
pub unsafe fn dpbusd(acc: __m256i, u8s: __m256i, i8s: __m256i) -> __m256i {
    // Treat i8s as two interleaved halves for maddubs:
    // maddubs(a, b) interprets a as u8 and b as i8, then does:
    //   for each pair of adjacent bytes: u8[2k]*i8[2k] + u8[2k+1]*i8[2k+1] → i16
    let products = _mm256_maddubs_epi16(u8s, i8s);
    // Now reduce pairs of i16 → i32 by multiplying with 1s
    let ones = _mm256_set1_epi16(1);
    let summed = _mm256_madd_epi16(products, ones);
    _mm256_add_epi32(acc, summed)
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    // No FMA for integers; just multiply and add
    _mm256_add_epi32(_mm256_mullo_epi32(a, b), c)
}

#[inline(always)]
pub unsafe fn load_i32(ptr: *const i32) -> __m256i {
    _mm256_load_si256(ptr as *const __m256i)
}

#[inline(always)]
pub unsafe fn store_i32(ptr: *mut i32, v: __m256i) {
    _mm256_store_si256(ptr as *mut __m256i, v)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32(a: [__m256i; 8]) -> i32 {
    let sum01 = _mm256_add_epi32(a[0], a[1]);
    let sum23 = _mm256_add_epi32(a[2], a[3]);
    let sum45 = _mm256_add_epi32(a[4], a[5]);
    let sum67 = _mm256_add_epi32(a[6], a[7]);
    let sum0123 = _mm256_add_epi32(sum01, sum23);
    let sum4567 = _mm256_add_epi32(sum45, sum67);
    let sum_all = _mm256_add_epi32(sum0123, sum4567);
    horizontal_sum_i32_single(sum_all)
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
