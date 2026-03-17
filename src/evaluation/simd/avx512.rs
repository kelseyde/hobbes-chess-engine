use std::{arch::x86_64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<__m512i>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<__m512i>() / size_of::<i32>();
pub const I8_LANES: usize = size_of::<__m512i>() / size_of::<i8>();

pub unsafe fn splat_i16(a: i16) -> __m512i {
    _mm512_set1_epi16(a)
}

pub unsafe fn splat_i32(a: i32) -> __m512i {
    _mm512_set1_epi32(a)
}

pub unsafe fn clamp_i16(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
    _mm512_max_epi16(_mm512_min_epi16(x, max), min)
}

pub unsafe fn clamp_i32(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
    _mm512_max_epi32(_mm512_min_epi32(x, max), min)
}

pub unsafe fn shift_left_i16<const SHIFT: u32>(a: __m512i) -> __m512i {
    _mm512_slli_epi16::<SHIFT>(a)
}

pub unsafe fn mul_high_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_mulhi_epi16(a, b)
}

pub unsafe fn packus(a: __m512i, b: __m512i) -> __m512i {
    _mm512_packus_epi16(a, b)
}

#[inline(always)]
pub unsafe fn dpbusd(acc: __m512i, u8s: __m512i, i8s: __m512i) -> __m512i {
    let products = _mm512_maddubs_epi16(u8s, i8s);
    let ones = _mm512_set1_epi16(1);
    let summed = _mm512_madd_epi16(products, ones);
    _mm512_add_epi32(acc, summed)
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_add_epi32(_mm512_mullo_epi32(a, b), c)
}

#[inline(always)]
pub unsafe fn load_i32(ptr: *const i32) -> __m512i {
    _mm512_load_si512(ptr as *const i32)
}

#[inline(always)]
pub unsafe fn store_i32(ptr: *mut i32, v: __m512i) {
    _mm512_store_si512(ptr as *mut i32, v)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32<const N: usize>(a: [__m512i; N]) -> i32 {
    let mut acc = a[0];
    for i in 1..N {
        acc = _mm512_add_epi32(acc, a[i]);
    }
    horizontal_sum_i32_single(acc)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32_single(a: __m512i) -> i32 {
    _mm512_reduce_add_epi32(a)
}

#[inline(always)]
pub unsafe fn add_i32(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(a, b)
}

#[inline(always)]
pub unsafe fn shr_i32<const SHIFT: i32>(a: __m512i) -> __m512i {
    _mm512_srai_epi32::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn mul_i32(a: __m512i, b: __m512i) -> __m512i {
    _mm512_mullo_epi32(a, b)
}
