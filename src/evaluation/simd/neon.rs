use std::{arch::aarch64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<int32x4_t>() / size_of::<i32>();

pub unsafe fn splat_i16(a: i16) -> int16x8_t {
    vdupq_n_s16(a)
}

pub unsafe fn splat_i32(a: i32) -> int32x4_t {
    vdupq_n_s32(a)
}

pub fn add_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    unsafe { vaddq_s16(a, b) }
}

pub unsafe fn clamp_i16(x: int16x8_t, min: int16x8_t, max: int16x8_t) -> int16x8_t {
    vmaxq_s16(vminq_s16(x, max), min)
}

pub unsafe fn clamp_i32(x: int32x4_t, min: int32x4_t, max: int32x4_t) -> int32x4_t {
    vmaxq_s32(vminq_s32(x, max), min)
}

pub unsafe fn min_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    vminq_s16(a, b)
}

pub unsafe fn shift_left_i16<const SHIFT: i32>(a: int16x8_t) -> int16x8_t {
    vshlq_n_s16::<SHIFT>(a)
}

pub unsafe fn mul_high_i16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    let low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let high = vmull_s16(vget_high_s16(a), vget_high_s16(b));

    let low_hi = vshrn_n_s32::<16>(low);
    let high_hi = vshrn_n_s32::<16>(high);

    vcombine_s16(low_hi, high_hi)
}

pub unsafe fn mul_add_i32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vmlaq_s32(a, b, c)
}
pub unsafe fn packus(a: int16x8_t, b: int16x8_t) -> int8x16_t {
    let a_u8 = vqmovun_s16(a);
    let b_u8 = vqmovun_s16(b);
    vreinterpretq_s8_u8(vcombine_u8(a_u8, b_u8))
}

/// No permute needed for NEON as the pack operation already arranges bytes correctly.
pub unsafe fn permute(a: int8x16_t) -> int8x16_t {
    a
}

unsafe fn dot_bytes(u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    let u8s = vreinterpretq_u8_s32(u8s);

    let products_low = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u8s))), vmovl_s8(vget_low_s8(i8s)));
    let products_high = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u8s))), vmovl_s8(vget_high_s8(i8s)));

    let sums_low = vpaddlq_s16(products_low);
    let sums_high = vpaddlq_s16(products_high);

    vpaddq_s32(sums_low, sums_high)
}

pub unsafe fn dpbusd(i32s: int32x4_t, u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    vaddq_s32(i32s, dot_bytes(u8s, i8s))
}

pub unsafe fn horizontal_sum_i32(a: [int32x4_t; 8]) -> i32 {
    let mut total = vdupq_n_s32(0);
    for vec in &a {
        total = vaddq_s32(total, *vec);
    }
    let pair_sum = vpaddq_s32(total, total);
    let quad_sum = vpaddq_s32(pair_sum, pair_sum);
    vgetq_lane_s32(quad_sum, 0)
}