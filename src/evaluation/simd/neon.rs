use hobbes_nnue_arch::L0_SHIFT;
use std::{arch::aarch64::*, mem::size_of};

pub const U8_LANES: usize = size_of::<int8x16_t>() / size_of::<u8>();
pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<int32x4_t>() / size_of::<i32>();

pub type VecI32 = int32x4_t;

#[inline(always)]
pub unsafe fn splat_u16(a: u16) -> uint16x8_t {
    vdupq_n_u16(a)
}

#[inline(always)]
pub unsafe fn load_u16(ptr: *const u16) -> uint16x8_t {
    vld1q_u16(ptr)
}

#[inline(always)]
pub unsafe fn add_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    vaddq_u16(a, b)
}

#[inline(always)]
pub unsafe fn store_u16(ptr: *mut u16, v: uint16x8_t) {
    vst1q_u16(ptr, v)
}

#[inline(always)]
pub unsafe fn load_i8(ptr: *const i8) -> int8x16_t {
    vld1q_s8(ptr)
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
pub unsafe fn splat_i32_as_u8(a: i32) -> int8x16_t {
    vreinterpretq_s8_s32(vdupq_n_s32(a))
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
pub unsafe fn nonzero_mask_i32(vec: int32x4_t) -> u16 {
    const MASK: [u32; 4] = [1, 2, 4, 8];
    let a = std::mem::transmute(vec);
    vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32(MASK.as_ptr()))) as u16
}

#[inline(always)]
pub unsafe fn nonzero_mask_u8(ptr: *const u8) -> u32 {
    let chunk = vreinterpretq_s32_u8(vld1q_u8(ptr));
    nonzero_mask_i32(chunk) as u32
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
pub unsafe fn horizontal_sum_i32_single(a: int32x4_t) -> i32 {
    vaddvq_s32(a)
}

#[inline(always)]
pub unsafe fn dpbusdx2(
    acc: int32x4_t,
    u1: int8x16_t, w1: int8x16_t,
    u2: int8x16_t, w2: int8x16_t,
) -> int32x4_t {
    #[cfg(target_feature = "dotprod")]
    {
        dpbusd(dpbusd(acc, u1, w1), u2, w2)
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        let lo1 = vmull_s8(vget_low_s8(u1), vget_low_s8(w1));
        let hi1 = vmull_high_s8(u1, w1);
        let p1  = vpaddq_s16(lo1, hi1);

        let lo2 = vmull_s8(vget_low_s8(u2), vget_low_s8(w2));
        let hi2 = vmull_high_s8(u2, w2);
        let p2  = vpaddq_s16(lo2, hi2);

        vpadalq_s16(acc, vaddq_s16(p1, p2))
    }
}

#[inline(always)]
pub unsafe fn shift_right_i32<const SHIFT: i32>(a: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn shift_left_i32<const SHIFT: i32>(a: int32x4_t) -> int32x4_t {
    vshlq_n_s32::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32<const N: usize>(a: [int32x4_t; N]) -> i32 {
    let mut acc = a[0];
    for &lane in a.iter().skip(1) {
        acc = vaddq_s32(acc, lane);
    }
    horizontal_sum_i32_single(acc)
}