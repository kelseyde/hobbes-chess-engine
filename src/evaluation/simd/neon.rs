use std::{arch::aarch64::*, mem::size_of};
use hobbes_nnue_arch::L0_SHIFT;

pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<int32x4_t>() / size_of::<i32>();

#[inline(always)]
pub unsafe fn splat_i16(a: i16) -> int16x8_t {
    vdupq_n_s16(a)
}

#[inline(always)]
pub unsafe fn splat_i32(a: i32) -> int32x4_t {
    vdupq_n_s32(a)
}

#[inline(always)]
pub unsafe fn clamp_i16(x: int16x8_t, min: int16x8_t, max: int16x8_t) -> int16x8_t {
    vmaxq_s16(vminq_s16(x, max), min)
}

#[inline(always)]
pub unsafe fn clamp_i32(x: int32x4_t, min: int32x4_t, max: int32x4_t) -> int32x4_t {
    vmaxq_s32(vminq_s32(x, max), min)
}

#[inline(always)]
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: int16x8_t) -> int16x8_t {
    vshlq_n_s16::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn mul_high_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    // smull/smull2: widen low/high 4 i16 pairs to i32
    let low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let high = vmull_high_s16(a, b);

    // Reinterpret as i16x8 so uzp2 can extract the high 16 bits of each i32 lane
    let low_i16 = vreinterpretq_s16_s32(low);
    let high_i16 = vreinterpretq_s16_s32(high);

    // uzp2: extract odd (high) 16-bit elements -> upper 16 bits of each product
    vuzp2q_s16(low_i16, high_i16)
}

/// Fused shift-left + multiply-high using NEON's vqdmulhq_s16 (doubling multiply high).
#[inline(always)]
pub unsafe fn shift_left_mul_high_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    // vqdmulhq doubles the product, so we shift by one less than intended
    const SHIFT: i32 = 16 - L0_SHIFT as i32 - 1;
    vqdmulhq_s16(vshlq_n_s16::<SHIFT>(a), b)
}

#[inline(always)]
pub unsafe fn packus(a: int16x8_t, b: int16x8_t) -> uint8x16_t {
    let a_u8 = vqmovun_s16(a);
    let b_u8 = vqmovun_s16(b);
    vcombine_u8(a_u8, b_u8)
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
pub unsafe fn load_u8(ptr: *const u8) -> int8x16_t {
    vreinterpretq_s8_u8(vld1q_u8(ptr))
}

#[inline(always)]
pub unsafe fn load_i8(ptr: *const i8) -> int8x16_t {
    vld1q_s8(ptr)
}

#[inline(always)]
pub unsafe fn store_i8(ptr: *mut u8, v: int8x16_t) {
    vst1q_u8(ptr, vreinterpretq_u8_s8(v))
}

pub unsafe fn store_u8(ptr: *mut u8, v: uint8x16_t) {
    vst1q_u8(ptr, v)
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vmlaq_s32(c, a, b)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32<const N: usize>(a: [int32x4_t; N]) -> i32 {
    let mut acc = a[0];
    for i in 1..N {
        acc = vaddq_s32(acc, a[i]);
    }
    horizontal_sum_i32_single(acc)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32_single(a: int32x4_t) -> i32 {
    vaddvq_s32(a)
}

#[inline(always)]
pub unsafe fn dpbusd(acc: int32x4_t, u8s: int8x16_t, i8s: int8x16_t) -> int32x4_t {
    // u8s values are in [0, 127] after SCReLU, so signed multiply is safe
    let lo = vmull_s8(vget_low_s8(u8s), vget_low_s8(i8s));
    let hi = vmull_high_s8(u8s, i8s);
    let pairwise = vpaddq_s16(lo, hi);
    vpadalq_s16(acc, pairwise)
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
pub unsafe fn shr_i32<const SHIFT: i32>(a: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<SHIFT>(a)
}

#[inline(always)]
pub unsafe fn mul_i32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vmulq_s32(a, b)
}
