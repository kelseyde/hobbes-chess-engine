use hobbes_nnue_arch::{L0_SHIFT, L1_SIZE};
use std::{arch::x86_64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<__m512i>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<__m512i>() / size_of::<i32>();
pub const I8_LANES: usize = size_of::<__m512i>() / size_of::<i8>();

pub type I32Vec = __m512i;

#[inline(always)]
pub unsafe fn load_u8(ptr: *const u8) -> __m512i {
    _mm512_loadu_si512(ptr as *const __m512i)
}

#[inline(always)]
pub unsafe fn store_u8(ptr: *mut u8, v: __m512i) {
    _mm512_storeu_si512(ptr as *mut __m512i, v)
}

#[inline(always)]
pub unsafe fn load_i8(ptr: *const i8) -> __m512i {
    _mm512_loadu_si512(ptr as *const __m512i)
}

/// Reinterpret an i32 vector as an i8 vector (no-op on x86, same __m512i type).
#[inline(always)]
pub unsafe fn reinterpret_i32_as_i8(v: __m512i) -> __m512i {
    v
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> __m512i {
    _mm512_set1_epi16(a)
}

#[inline(always)]
pub unsafe fn load_i16(ptr: *const i16) -> __m512i {
    _mm512_loadu_si512(ptr as *const __m512i)
}

#[inline(always)]
pub unsafe fn store_i16(ptr: *mut i16, v: __m512i) {
    _mm512_storeu_si512(ptr as *mut __m512i, v)
}

#[inline(always)]
pub unsafe fn add_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi16(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: __m512i, b: __m512i) -> __m512i {
    _mm512_sub_epi16(a, b)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
    _mm512_max_epi16(_mm512_min_epi16(x, max), min)
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> __m512i {
    _mm512_set1_epi32(a)
}

#[inline(always)]
pub unsafe fn splat_i32_x4(a: i32) -> (__m512i, __m512i, __m512i, __m512i) {
    let v = _mm512_set1_epi32(a);
    (v, v, v, v)
}

#[inline(always)]
pub unsafe fn load_i32(ptr: *const i32) -> __m512i {
    _mm512_loadu_si512(ptr as *const __m512i)
}

#[inline(always)]
pub unsafe fn store_i32(ptr: *mut i32, v: __m512i) {
    _mm512_storeu_si512(ptr as *mut __m512i, v)
}

#[inline(always)]
pub unsafe fn add_i32(a: __m512i, b: __m512i) -> __m512i {
    _mm512_add_epi32(a, b)
}

#[inline(always)]
pub unsafe fn mul_i32(a: __m512i, b: __m512i) -> __m512i {
    _mm512_mullo_epi32(a, b)
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_add_epi32(_mm512_mullo_epi32(a, b), c)
}

#[inline(always)]
pub unsafe fn clamp_i32(x: __m512i, min: __m512i, max: __m512i) -> __m512i {
    _mm512_max_epi32(_mm512_min_epi32(x, max), min)
}

/// Fused shift-left + multiply-high.
#[inline(always)]
pub unsafe fn shift_left_mul_high_i16(a: __m512i, b: __m512i) -> __m512i {
    const SHIFT: u32 = 16 - L0_SHIFT as u32;
    _mm512_mulhi_epi16(_mm512_slli_epi16::<SHIFT>(a), b)
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
pub unsafe fn dpbusd_x4(
    a0: __m512i,
    a1: __m512i,
    a2: __m512i,
    a3: __m512i,
    u0: __m512i,
    u1: __m512i,
    u2: __m512i,
    u3: __m512i,
    w0: __m512i,
    w1: __m512i,
    w2: __m512i,
    w3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    (
        dpbusd(a0, u0, w0),
        dpbusd(a1, u1, w1),
        dpbusd(a2, u2, w2),
        dpbusd(a3, u3, w3),
    )
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32_single(a: __m512i) -> i32 {
    _mm512_reduce_add_epi32(a)
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
pub unsafe fn load_i8x4(ptr: *const i8, stride: usize) -> (__m512i, __m512i, __m512i, __m512i) {
    (
        _mm512_loadu_si512(ptr as *const __m512i),
        _mm512_loadu_si512(ptr.add(stride) as *const __m512i),
        _mm512_loadu_si512(ptr.add(2 * stride) as *const __m512i),
        _mm512_loadu_si512(ptr.add(3 * stride) as *const __m512i),
    )
}

/// Find non-zero i32 blocks in the u8 activation buffer.
/// AVX512 has 16 i32 lanes per vector. We use _mm512_test_epi32_mask to get a 16-bit mask,
/// then split it into two 8-bit halves to use the same 256-entry lookup table.
#[inline(always)]
pub unsafe fn find_nnz(
    input: &[u8; L1_SIZE],
    nnz: &mut [u16; L1_SIZE / 4],
) -> usize {
    let input_ptr = input.as_ptr() as *const __m512i;
    let num_blocks = L1_SIZE / 4;
    let num_vecs = num_blocks / I32_LANES; // 320 / 16 = 20

    let mut count: usize = 0;

    for i in 0..num_vecs {
        let vec = _mm512_loadu_si512(input_ptr.add(i));
        let mask = _mm512_test_epi32_mask(vec, vec); // 16-bit mask

        // Process low 8 bits
        let lo_mask = (mask & 0xFF) as usize;
        let lo_base = (i * I32_LANES) as u16;
        let lo_entry = &NNZ_TABLE[lo_mask];
        let lo_count = NNZ_COUNT[lo_mask] as usize;
        for j in 0..lo_count {
            *nnz.get_unchecked_mut(count + j) = lo_base + lo_entry[j];
        }
        count += lo_count;

        // Process high 8 bits
        let hi_mask = ((mask >> 8) & 0xFF) as usize;
        let hi_base = (i * I32_LANES + 8) as u16;
        let hi_entry = &NNZ_TABLE[hi_mask];
        let hi_count = NNZ_COUNT[hi_mask] as usize;
        for j in 0..hi_count {
            *nnz.get_unchecked_mut(count + j) = hi_base + hi_entry[j];
        }
        count += hi_count;
    }

    count
}

/// For an 8-bit mask (0..256), maps each mask to the indices of set bits.
const NNZ_TABLE: [[u16; 8]; 256] = {
    let mut table = [[0u16; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut idx = 0;
        let mut bit = 0;
        while bit < 8 {
            if mask & (1 << bit) != 0 {
                table[mask][idx] = bit as u16;
                idx += 1;
            }
            bit += 1;
        }
        mask += 1;
    }
    table
};

const NNZ_COUNT: [u8; 256] = {
    let mut counts = [0u8; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut c = 0u8;
        let mut bit = 0;
        while bit < 8 {
            if mask & (1 << bit) != 0 {
                c += 1;
            }
            bit += 1;
        }
        counts[mask] = c;
        mask += 1;
    }
    counts
};
