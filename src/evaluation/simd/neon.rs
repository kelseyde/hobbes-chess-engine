use std::{arch::aarch64::*, mem::size_of};

pub const I16_LANES: usize = size_of::<int16x8_t>() / size_of::<i16>();
pub const I32_LANES: usize = size_of::<int32x4_t>() / size_of::<i32>();

// For dpbusd-style L1 matmul
pub const I8_LANES: usize = size_of::<int8x16_t>() / size_of::<i8>();

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
    let low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
    let high = vmull_s16(vget_high_s16(a), vget_high_s16(b));

    let low_hi = vshrn_n_s32::<16>(low);
    let high_hi = vshrn_n_s32::<16>(high);

    vcombine_s16(low_hi, high_hi)
}

#[inline(always)]
pub unsafe fn packus(a: int16x8_t, b: int16x8_t) -> int8x16_t {
    let a_u8 = vqmovun_s16(a);
    let b_u8 = vqmovun_s16(b);
    vreinterpretq_s8_u8(vcombine_u8(a_u8, b_u8))
}

/// No permute needed for NEON as the pack operation already arranges bytes correctly.
#[inline(always)]
pub unsafe fn permute(a: int8x16_t) -> int8x16_t {
    a
}

#[inline(always)]
pub unsafe fn mul_add_i32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vmlaq_s32(c, a, b)
}

#[inline(always)]
pub unsafe fn horizontal_sum_i32(a: [int32x4_t; 8]) -> i32 {
    let sum01 = vaddq_s32(a[0], a[1]);
    let sum23 = vaddq_s32(a[2], a[3]);
    let sum45 = vaddq_s32(a[4], a[5]);
    let sum67 = vaddq_s32(a[6], a[7]);

    let sum0123 = vaddq_s32(sum01, sum23);
    let sum4567 = vaddq_s32(sum45, sum67);
    let sum = vaddq_s32(sum0123, sum4567);

    let pair = vpadd_s32(vget_low_s32(sum), vget_high_s32(sum));
    let final_sum = vpadd_s32(pair, pair);

    vget_lane_s32::<0>(final_sum)
}


#[inline(always)]
pub unsafe fn load_i8x16(ptr: *const i8) -> int8x16_t {
    vld1q_s8(ptr)
}

#[inline(always)]
pub unsafe fn load_u8x16_as_i32x4(ptr: *const u8) -> int32x4_t {
    vld1q_s32(ptr.cast())
}

#[inline(always)]
pub unsafe fn extract_lane_i32(v: int32x4_t, lane: i32) -> i32 {
    match lane {
        0 => vgetq_lane_s32::<0>(v),
        1 => vgetq_lane_s32::<1>(v),
        2 => vgetq_lane_s32::<2>(v),
        3 => vgetq_lane_s32::<3>(v),
        _ => core::hint::unreachable_unchecked(),
    }
}

#[inline(always)]
unsafe fn dot_bytes(u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    let u8s = vreinterpretq_u8_s32(u8s);

    let products_low = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u8s))),
        vmovl_s8(vget_low_s8(i8s)),
    );
    let products_high = vmulq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(u8s))),
        vmovl_s8(vget_high_s8(i8s)),
    );

    let sums_low = vpaddlq_s16(products_low);
    let sums_high = vpaddlq_s16(products_high);

    vpaddq_s32(sums_low, sums_high)
}

#[inline(always)]
pub unsafe fn dpbusd(i32s: int32x4_t, u8s: int32x4_t, i8s: int8x16_t) -> int32x4_t {
    vaddq_s32(i32s, dot_bytes(u8s, i8s))
}
