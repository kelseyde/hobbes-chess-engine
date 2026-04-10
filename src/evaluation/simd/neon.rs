use hobbes_nnue_arch::{L0_SHIFT, L1_SIZE};
use std::{arch::aarch64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<int32x4_t>() / size_of::<i32>();

#[allow(dead_code)]
pub type I32Vec = int32x4_t;

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn load_u8(ptr: *const u8) -> int8x16_t {
    vreinterpretq_s8_u8(vld1q_u8(ptr))
}

#[inline(always)]
pub unsafe fn load_i8(ptr: *const i8) -> int8x16_t {
    vld1q_s8(ptr)
}

/// Reinterpret an i32 vector as an i8 vector (for dpbusd with splatted activations).
#[inline(always)]
pub unsafe fn reinterpret_i32_as_i8(v: int32x4_t) -> int8x16_t {
    vreinterpretq_s8_s32(v)
}

#[inline(always)]
pub unsafe fn store_u8(ptr: *mut u8, v: uint8x16_t) {
    vst1q_u8(ptr, v)
}

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> int16x8_t {
    vdupq_n_s16(a)
}

#[inline(always)]
pub unsafe fn load_i16(ptr: *const i16) -> int16x8_t {
    vld1q_s16(ptr)
}

#[inline(always)]
pub unsafe fn store_i16(ptr: *mut i16, v: int16x8_t) {
    vst1q_s16(ptr, v)
}

#[inline(always)]
pub unsafe fn add_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vaddq_s16(a, b)
}

#[inline(always)]
pub unsafe fn sub_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vsubq_s16(a, b)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: int16x8_t, min: int16x8_t, max: int16x8_t) -> int16x8_t {
    vmaxq_s16(vminq_s16(x, max), min)
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> int32x4_t {
    vdupq_n_s32(a)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn splat_i32_x4(a: i32) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let v = vdupq_n_s32(a);
    (v, v, v, v)
}

#[inline(always)]
pub unsafe fn load_i32(ptr: *const i32) -> int32x4_t {
    vld1q_s32(ptr)
}

#[inline(always)]
pub unsafe fn store_i32(ptr: *mut i32, v: int32x4_t) {
    vst1q_s32(ptr, v)
}

#[inline(always)]
pub unsafe fn add_i32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vaddq_s32(a, b)
}

#[inline(always)]
pub unsafe fn mul_i32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vmulq_s32(a, b)
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vmlaq_s32(c, a, b)
}

#[inline(always)]
pub unsafe fn clamp_i32(x: int32x4_t, min: int32x4_t, max: int32x4_t) -> int32x4_t {
    vmaxq_s32(vminq_s32(x, max), min)
}

/// Fused shift-left + multiply-high using NEON's vqdmulhq_s16 (doubling multiply high).
/// vqdmulhq computes (a * b * 2) >> 16, so we shift by one less than the intended shift.
#[inline(always)]
pub unsafe fn shift_left_mul_high_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    const SHIFT: i32 = 16 - L0_SHIFT as i32 - 1;
    vqdmulhq_s16(vshlq_n_s16::<SHIFT>(a), b)
}

#[inline(always)]
pub unsafe fn packus(a: int16x8_t, b: int16x8_t) -> uint8x16_t {
    vcombine_u8(vqmovun_s16(a), vqmovun_s16(b))
}

#[inline(always)]
pub unsafe fn dpbusd(acc: int32x4_t, u8s: int8x16_t, i8s: int8x16_t) -> int32x4_t {
    #[cfg(target_feature = "dotprod")]
    {
        // NEON dotprod is unstable, so for now we use inline ASM.
        let mut result = acc;
        std::arch::asm!(
            "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inlateout(vreg) result,
            in(vreg) u8s,
            in(vreg) i8s,
            options(pure, nomem, nostack),
        );
        result
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        let lo = vmull_s8(vget_low_s8(u8s), vget_low_s8(i8s));
        let hi = vmull_high_s8(u8s, i8s);
        let pairwise = vpaddq_s16(lo, hi);
        vpadalq_s16(acc, pairwise)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub unsafe fn dpbusd_x4(
    a0: int32x4_t,
    a1: int32x4_t,
    a2: int32x4_t,
    a3: int32x4_t,
    u0: int8x16_t,
    u1: int8x16_t,
    u2: int8x16_t,
    u3: int8x16_t,
    w0: int8x16_t,
    w1: int8x16_t,
    w2: int8x16_t,
    w3: int8x16_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    (
        dpbusd(a0, u0, w0),
        dpbusd(a1, u1, w1),
        dpbusd(a2, u2, w2),
        dpbusd(a3, u3, w3),
    )
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32_single(a: int32x4_t) -> i32 {
    vaddvq_s32(a)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32<const N: usize>(a: [int32x4_t; N]) -> i32 {
    let mut acc = a[0];
    for &lane in a.iter().skip(1) {
        acc = vaddq_s32(acc, lane);
    }
    horizontal_sum_i32_single(acc)
}

#[inline(always)]
#[allow(dead_code)]
pub unsafe fn load_i8x4(
    ptr: *const i8,
    stride: usize,
) -> (int8x16_t, int8x16_t, int8x16_t, int8x16_t) {
    (
        vld1q_s8(ptr),
        vld1q_s8(ptr.add(stride)),
        vld1q_s8(ptr.add(2 * stride)),
        vld1q_s8(ptr.add(3 * stride)),
    )
}

/// Returns a bitmask of which i32 lanes in a 128-bit vector are non-zero.
/// Each bit in the result corresponds to one i32 lane (bit 0 = lane 0, etc).
#[inline(always)]
pub unsafe fn nonzero_mask_i32(v: int32x4_t) -> u32 {
    let nonzero = vtstq_u32(vreinterpretq_u32_s32(v), vreinterpretq_u32_s32(v));
    let bits: [u32; 4] = [1, 2, 4, 8];
    let mask_bits = vld1q_u32(bits.as_ptr());
    let masked = vandq_u32(nonzero, mask_bits);
    vaddvq_u32(masked)
}

/// Find non-zero i32 blocks in the u8 activation buffer.
/// Returns (nnz_indices, nnz_count) where nnz_indices[i] is the index of a
/// non-zero i32 group (4 consecutive u8 values).
///
/// Uses a lookup table to expand each nibble mask to packed indices.
#[inline(always)]
pub unsafe fn find_nnz(
    input: &[u8; L1_SIZE],
    nnz: &mut [u16; L1_SIZE / 4],
) -> usize {
    let input32 = input.as_ptr() as *const int32x4_t;
    let num_blocks = L1_SIZE / 4;      // e.g. 320
    let num_vecs = num_blocks / I32_LANES; // 320 / 4 = 80

    let mut count: usize = 0;

    for i in 0..num_vecs {
        let vec = vld1q_s32((input32 as *const i32).add(i * I32_LANES));
        let mask = nonzero_mask_i32(vec);
        let base = (i * I32_LANES) as u16;

        // Expand the mask using the NNZ lookup table
        let entry = &NNZ_TABLE[mask as usize];
        let num_set = NNZ_COUNT[mask as usize] as usize;
        for j in 0..num_set {
            *nnz.get_unchecked_mut(count + j) = base + entry[j];
        }
        count += num_set;
    }

    count
}

/// For a 4-lane i32 mask (0..16), maps each mask to the indices of set bits.
/// E.g. mask 0b1010 = 10 -> NNZ_TABLE[10] = [1, 3, 0, 0], NNZ_COUNT[10] = 2
const NNZ_TABLE: [[u16; 4]; 16] = {
    let mut table = [[0u16; 4]; 16];
    let mut mask = 0usize;
    while mask < 16 {
        let mut idx = 0;
        let mut bit = 0;
        while bit < 4 {
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

const NNZ_COUNT: [u8; 16] = {
    let mut counts = [0u8; 16];
    let mut mask = 0usize;
    while mask < 16 {
        let mut c = 0u8;
        let mut bit = 0;
        while bit < 4 {
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

