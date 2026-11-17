use hobbes_nnue_arch::L0_SHIFT;
use std::{arch::x86_64::*, mem::size_of};

pub const I32_LANES: usize = size_of::<__m256i>() / size_of::<i32>();

pub type VecI32 = __m256i;

pub type VecI16 = __m256i;

pub mod u8 {
    use crate::evaluation::simd;
    use std::arch::x86_64::*;

    pub const LANES: usize = size_of::<__m256i>() / size_of::<u8>();

    #[inline(always)]
    pub unsafe fn store(ptr: *mut u8, v: __m256i) {
        _mm256_storeu_si256(ptr as *mut __m256i, v)
    }

    #[inline(always)]
    pub unsafe fn nonzero_mask(ptr: *const u8) -> u32 {
        let chunk = _mm256_loadu_si256(ptr as *const __m256i);
        simd::i32::nonzero_mask(chunk) as u32
    }
}

pub mod i8 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn load(ptr: *const i8) -> __m256i {
        _mm256_loadu_si256(ptr as *const __m256i)
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
    pub unsafe fn zero() -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    pub unsafe fn splat(a: i16) -> __m256i {
        _mm256_set1_epi16(a)
    }

    #[inline(always)]
    pub unsafe fn load(ptr: *const i16) -> __m256i {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut i16, v: __m256i) {
        _mm256_storeu_si256(ptr as *mut __m256i, v)
    }

    #[inline(always)]
    pub unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn sub(a: __m256i, b: __m256i) -> __m256i {
        _mm256_sub_epi16(a, b)
    }

    #[inline(always)]
    pub unsafe fn clamp(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
        _mm256_max_epi16(_mm256_min_epi16(x, max), min)
    }

    #[inline(always)]
    pub unsafe fn load_i8(ptr: *const i8) -> __m256i {
        _mm256_cvtepi8_epi16(_mm_loadu_si128(ptr as *const __m128i))
    }

    #[inline(always)]
    pub unsafe fn shift_left_mul_high(a: __m256i, b: __m256i) -> __m256i {
        const SHIFT: i32 = 16 - L0_SHIFT as i32;
        _mm256_mulhi_epi16(_mm256_slli_epi16::<SHIFT>(a), b)
    }
}

pub mod i32 {
    use std::arch::x86_64::*;

    #[inline(always)]
    pub unsafe fn zero() -> __m256i {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    pub unsafe fn splat(a: i32) -> __m256i {
        _mm256_set1_epi32(a)
    }

    #[inline(always)]
    pub unsafe fn splat_as_u8(a: i32) -> __m256i {
        _mm256_set1_epi32(a)
    }

    #[inline(always)]
    pub unsafe fn load(ptr: *const i32) -> __m256i {
        _mm256_loadu_si256(ptr as *const __m256i)
    }

    #[inline(always)]
    pub unsafe fn store(ptr: *mut i32, v: __m256i) {
        _mm256_storeu_si256(ptr as *mut __m256i, v)
    }

    #[inline(always)]
    pub unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
        _mm256_add_epi32(a, b)
    }

    #[inline(always)]
    pub unsafe fn mul(a: __m256i, b: __m256i) -> __m256i {
        _mm256_mullo_epi32(a, b)
    }

    #[inline(always)]
    pub unsafe fn mul_add(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
        _mm256_add_epi32(_mm256_mullo_epi32(a, b), c)
    }

    #[inline(always)]
    pub unsafe fn clamp(x: __m256i, min: __m256i, max: __m256i) -> __m256i {
        _mm256_max_epi32(_mm256_min_epi32(x, max), min)
    }

    #[inline(always)]
    pub unsafe fn nonzero_mask(vec: __m256i) -> u16 {
        _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(
            vec,
            _mm256_setzero_si256(),
        ))) as u16
    }

    #[inline(always)]
    pub unsafe fn shift_right<const SHIFT: i32>(a: __m256i) -> __m256i {
        _mm256_srai_epi32::<SHIFT>(a)
    }

    #[inline(always)]
    pub unsafe fn shift_left<const SHIFT: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi32::<SHIFT>(a)
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum_single(a: __m256i) -> i32 {
        let hi_128 = _mm256_extracti128_si256::<1>(a);
        let lo_128 = _mm256_castsi256_si128(a);
        let sum_128 = _mm_add_epi32(lo_128, hi_128);
        let hi_64 = _mm_unpackhi_epi64(sum_128, sum_128);
        let sum_64 = _mm_add_epi32(sum_128, hi_64);
        let hi_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
        let sum_32 = _mm_add_epi32(sum_64, hi_32);
        _mm_cvtsi128_si32(sum_32)
    }

    #[inline(always)]
    pub unsafe fn horizontal_sum<const N: usize>(a: [__m256i; N]) -> i32 {
        let mut acc = a[0];
        for i in 1..N {
            acc = _mm256_add_epi32(acc, a[i]);
        }
        horizontal_sum_single(acc)
    }
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
pub unsafe fn dpbusdx2(
    acc: __m256i,
    u1: __m256i,
    w1: __m256i,
    u2: __m256i,
    w2: __m256i,
) -> __m256i {
    let p1 = _mm256_maddubs_epi16(u1, w1);
    let p2 = _mm256_maddubs_epi16(u2, w2);
    let combined = _mm256_adds_epi16(p1, p2);
    let ones = _mm256_set1_epi16(1);
    _mm256_add_epi32(acc, _mm256_madd_epi16(combined, ones))
}
