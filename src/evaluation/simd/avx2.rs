use hobbes_nnue_arch::L0_SHIFT;
use std::{arch::x86_64::*, mem::size_of};

pub const U8_LANES: usize = size_of::<__m256i>() / size_of::<u8>();
pub const I16_LANES: usize = size_of::<__m256i>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<__m256i>() / size_of::<i32>();
pub const I8_LANES: usize = size_of::<__m256i>() / size_of::<i8>();

pub type VecI32 = __m256i;
pub type VecI8 = __m256i;
pub type VecU16 = __m128i; 

#[inline(always)]
pub unsafe fn splat_u16(a: u16) -> __m128i {
    _mm_set1_epi16(a as i16)
}

#[inline(always)]
pub unsafe fn load_u16(ptr: *const u16) -> __m128i {
    _mm_loadu_si128(ptr as *const __m128i)
}

#[inline(always)]
pub unsafe fn add_u16(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi16(a, b)
}

#[inline(always)]
pub unsafe fn store_u16(ptr: *mut u16, v: __m128i) {
    _mm_storeu_si128(ptr as *mut __m128i, v)
}

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
pub unsafe fn nonzero_mask_i32(vec: __m256i) -> u16 {
    _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vec, _mm256_setzero_si256()))) as u16
}

#[inline(always)]
pub unsafe fn nonzero_mask_u8(ptr: *const u8) -> u32 {
    let chunk = _mm256_loadu_si256(ptr as *const __m256i);
    nonzero_mask_i32(chunk) as u32
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
pub unsafe fn dpbusd_x4(
    a0: __m256i,
    a1: __m256i,
    a2: __m256i,
    a3: __m256i,
    u0: __m256i,
    u1: __m256i,
    u2: __m256i,
    u3: __m256i,
    w0: __m256i,
    w1: __m256i,
    w2: __m256i,
    w3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    (
        dpbusd(a0, u0, w0),
        dpbusd(a1, u1, w1),
        dpbusd(a2, u2, w2),
        dpbusd(a3, u3, w3),
    )
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

#[inline(always)]
pub unsafe fn dpbusdx2(
    acc: __m256i,
    u1: __m256i, w1: __m256i,
    u2: __m256i, w2: __m256i,
) -> __m256i {
    let p1 = _mm256_maddubs_epi16(u1, w1);
    let p2 = _mm256_maddubs_epi16(u2, w2);
    let combined = _mm256_adds_epi16(p1, p2);
    let ones = _mm256_set1_epi16(1);
    _mm256_add_epi32(acc, _mm256_madd_epi16(combined, ones))
}

#[inline(always)]
pub unsafe fn shift_right_i32<const SHIFT: u32>(a: __m256i) -> __m256i {
    _mm256_srai_epi32::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn shift_left_i32<const SHIFT: u32>(a: __m256i) -> __m256i {
    _mm256_slli_epi32::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn load_i8x4(ptr: *const i8, stride: usize) -> (__m256i, __m256i, __m256i, __m256i) {
    (
        _mm256_loadu_si256(ptr as *const __m256i),
        _mm256_loadu_si256(ptr.add(stride) as *const __m256i),
        _mm256_loadu_si256(ptr.add(2 * stride) as *const __m256i),
        _mm256_loadu_si256(ptr.add(3 * stride) as *const __m256i),
    )
}

#[inline(always)]
pub unsafe fn splat_i32_x4(a: i32) -> (__m256i, __m256i, __m256i, __m256i) {
    let v = _mm256_set1_epi32(a);
    (v, v, v, v)
}

#[inline(always)]
pub unsafe fn splat_i32_as_u8(a: i32) -> __m256i {
    _mm256_set1_epi32(a)
}

#[inline(always)]
pub unsafe fn extract_i32(v: __m256i, lane: usize) -> i32 {
    let mut tmp = [0i32; 8];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
    tmp[lane]
}

