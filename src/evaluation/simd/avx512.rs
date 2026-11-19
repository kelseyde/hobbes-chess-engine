use hobbes_nnue_arch::L0_SHIFT;
use std::{arch::x86_64::*, mem::size_of};

pub const I32_LANES: usize = size_of::<__m512i>() / size_of::<i32>();

pub type VecI16 = __m512i;
pub type VecI32 = __m512i;

pub mod u8 {
    use crate::evaluation::simd;
    use std::arch::x86_64::*;

    pub const LANES: usize = size_of::<__m512i>() / size_of::<u8>();

    #[inline(always)]
    pub unsafe fn store(ptr: *mut u8, v: __m512i) {
        _mm512_storeu_si512(ptr as *mut __m512i, v)
    }

    #[inline(always)]
    pub unsafe fn nonzero_mask(ptr: *const u8) -> u32 {
        let chunk = _mm512_loadu_si512(ptr as *const __m512i);
        simd::i32::nonzero_mask(chunk) as u32
    }
}

pub mod i8 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn load(ptr: *const i8) -> __m512i {
        _mm512_loadu_si512(ptr as *const __m512i)
    }
}

pub mod u16 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn splat(a: u16) -> __m128i {
        _mm_set1_epi16(a as i16)
    }

    #[inline(always)]
    pub unsafe fn load(ptr: *const u16) -> __m128i {
        _mm_loadu_si128(ptr as *const __m128i)
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut u16, v: __m128i) {
        _mm_storeu_si128(ptr as *mut __m128i, v)
    }

    #[inline(always)]
    pub unsafe fn add(a: __m128i, b: __m128i) -> __m128i {
        _mm_add_epi16(a, b)
    }
}

pub mod i16 {
    use crate::evaluation::simd::VecI16;
    use hobbes_nnue_arch::L0_SHIFT;
    use std::{arch::x86_64::*, mem::size_of};

    pub const LANES: usize = size_of::<VecI16>() / size_of::<i16>();

    #[inline(always)]
    pub unsafe fn load(ptr: *const i16) -> __m512i {
        _mm512_loadu_si512(ptr as *const __m512i)
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut i16, v: __m512i) {
        _mm512_storeu_si512(ptr as *mut __m512i, v)
    }

    #[inline(always)]
    pub unsafe fn splat(a: i16) -> __m512i {
        _mm512_set1_epi16(a)
    }

    #[inline(always)]
    pub unsafe fn zero() -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    pub unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
        _mm512_sub_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn clamp(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
        _mm512_max_epi16(_mm512_min_epi16(x, max), min)
    }

    #[inline(always)]
    pub unsafe fn load_i8(ptr: *const i8) -> __m512i {
        _mm512_cvtepi8_epi16(_mm256_loadu_si256(ptr as *const __m256i))
    }

    #[inline(always)]
    pub unsafe fn shift_left_mul_high(a: __m512i, b: __m512i) -> __m512i {
        const SHIFT: u32 = 16 - L0_SHIFT as u32;
        _mm512_mulhi_epi16(_mm512_slli_epi16::<SHIFT>(a), b)
    }
}

pub mod i32 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn zero() -> __m512i {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    pub unsafe fn splat(a: i32) -> __m512i {
        _mm512_set1_epi32(a)
    }

    #[inline(always)]
    pub unsafe fn splat_as_u8(a: i32) -> __m512i {
        _mm512_set1_epi32(a)
    }

    #[inline(always)]
    pub unsafe fn load(ptr: *const i32) -> __m512i {
        _mm512_loadu_si512(ptr as *const __m512i)
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut i32, v: __m512i) {
        _mm512_storeu_si512(ptr as *mut __m512i, v)
    }

    #[inline(always)]
    pub unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
        _mm512_add_epi32(a, b)
    }

    #[inline(always)]
    pub unsafe fn mul(a: __m512i, b: __m512i) -> __m512i {
        _mm512_mullo_epi32(a, b)
    }

    #[inline(always)]
    pub unsafe fn mul_add(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        _mm512_add_epi32(_mm512_mullo_epi32(a, b), c)
    }

    #[inline(always)]
    pub unsafe fn clamp(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
        _mm512_max_epi32(_mm512_min_epi32(x, max), min)
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum_single(a: __m512i) -> i32 {
        _mm512_reduce_add_epi32(a)
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum<const N: usize>(a: [__m512i; N]) -> i32 {
        let mut acc = a[0];
        for i in 1..N {
            acc = _mm512_add_epi32(acc, a[i]);
        }
        horizontal_sum_single(acc)
    }

    #[inline(always)]
    pub unsafe fn shift_right<const SHIFT: u32>(a: __m512i) -> __m512i {
        _mm512_srai_epi32::<SHIFT>(a)
    }

    #[inline(always)]
    pub unsafe fn shift_left<const SHIFT: u32>(a: __m512i) -> __m512i {
        _mm512_slli_epi32::<SHIFT>(a)
    }

    #[inline(always)]
    pub unsafe fn nonzero_mask(vec: __m512i) -> u16 {
        _mm512_cmpgt_epi32_mask(vec, _mm512_setzero_si512()) as u16
    }
}

#[inline(always)]
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
pub unsafe fn dpbusdx2(
    acc: __m512i,
    u1: __m512i,
    w1: __m512i,
    u2: __m512i,
    w2: __m512i,
) -> __m512i {
    let p1 = _mm512_maddubs_epi16(u1, w1);
    let p2 = _mm512_maddubs_epi16(u2, w2);
    let combined = _mm512_adds_epi16(p1, p2);
    let ones = _mm512_set1_epi16(1);
    _mm512_add_epi32(acc, _mm512_madd_epi16(combined, ones))
}
