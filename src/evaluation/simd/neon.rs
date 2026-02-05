use std::{arch::aarch64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();

pub unsafe fn splat_i16(a: i16) -> int16x8_t {
    vdupq_n_s16(a)
}

pub unsafe fn clamp_i16(x: int16x8_t, min: int16x8_t, max: int16x8_t) -> int16x8_t {
    vmaxq_s16(vminq_s16(x, max), min)
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

pub unsafe fn packus(a: int16x8_t, b: int16x8_t) -> int8x16_t {
    let a_u8 = vqmovun_s16(a);
    let b_u8 = vqmovun_s16(b);
    vreinterpretq_s8_u8(vcombine_u8(a_u8, b_u8))
}

/// No permute needed for NEON as the pack operation already arranges bytes correctly.
pub unsafe fn permute(a: int8x16_t) -> int8x16_t {
    a
}