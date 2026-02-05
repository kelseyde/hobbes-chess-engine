use std::{arch::x86_64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<__m512i>() / size_of::<i16>();

pub unsafe fn splat_i16(a: i16) -> __m512i {
    _mm512_set1_epi16(a)
}

pub unsafe fn clamp_i16(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
    _mm512_max_epi16(_mm512_min_epi16(x, max), min)
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

pub unsafe fn permute(a: __m512i) -> __m512i {
    _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), a)
}
