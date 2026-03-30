use crate::board::bitboard::Bitboard;
use std::arch::aarch64::*;

type Vec256 = (uint64x2_t, uint64x2_t);

pub fn knights_and_sliders_setwise(
    knights: Bitboard,
    orthos: Bitboard,
    diags: Bitboard,
    blockers: Bitboard,
) -> Bitboard {
    unsafe {
        let [a, b] = knights_setwise(knights);
        let [c, d] = sliders_setwise(orthos, diags, blockers);
        Bitboard(reduce_or(or(a, c), or(b, d)))
    }
}

fn knights_setwise(knights: Bitboard) -> [Vec256; 2] {
    let a = 0x0101010101010101i64;
    let b = a << 1;
    let g = a << 6;
    let h = a << 7;

    unsafe {
        let filemask1 = lanes(a | b, a, h, g | h);
        let filemask2 = lanes(g | h, h, a, a | b);

        let sq = broadcast(knights.0 as i64);
        let offsets = lanes(6, 15, 17, 10);
        let upper = shift_left(and_not(filemask1, sq), offsets);
        let lower = shift_right(and_not(filemask2, sq), offsets);

        [upper, lower]
    }
}

fn sliders_setwise(orth: Bitboard, diag: Bitboard, blockers: Bitboard) -> [Vec256; 2] {
    let a = 0x0101010101010101i64;
    let h = a << 7;

    unsafe {
        let shift = |n: i64| lanes(7 * n, 9 * n, 8 * n, n);

        let filemask1 = lanes(a, h, 0, h);
        let filemask2 = lanes(h, a, 0, a);

        let mut gen1 = lanes(diag.0 as i64, diag.0 as i64, orth.0 as i64, orth.0 as i64);
        let mut block1 = or(broadcast(blockers.0 as i64), filemask1);

        let mut gen2 = lanes(diag.0 as i64, diag.0 as i64, orth.0 as i64, orth.0 as i64);
        let mut block2 = or(broadcast(blockers.0 as i64), filemask2);

        gen1 = or(gen1, and_not(block1, shift_right(gen1, shift(1))));
        gen2 = or(gen2, and_not(block2, shift_left(gen2, shift(1))));

        block1 = or(block1, shift_right(block1, shift(1)));
        block2 = or(block2, shift_left(block2, shift(1)));

        gen1 = or(gen1, and_not(block1, shift_right(gen1, shift(2))));
        gen2 = or(gen2, and_not(block2, shift_left(gen2, shift(2))));

        block1 = or(block1, shift_right(block1, shift(2)));
        block2 = or(block2, shift_left(block2, shift(2)));

        gen1 = or(gen1, and_not(block1, shift_right(gen1, shift(4))));
        gen2 = or(gen2, and_not(block2, shift_left(gen2, shift(4))));

        gen1 = and_not(filemask1, shift_right(gen1, shift(1)));
        gen2 = and_not(filemask2, shift_left(gen2, shift(1)));

        [gen1, gen2]
    }
}

#[inline(always)]
unsafe fn broadcast(x: i64) -> Vec256 {
    let v = vdupq_n_u64(x as u64);
    (v, v)
}

#[inline(always)]
unsafe fn lanes(a: i64, b: i64, c: i64, d: i64) -> Vec256 {
    (
        vcombine_u64(vcreate_u64(a as u64), vcreate_u64(b as u64)),
        vcombine_u64(vcreate_u64(c as u64), vcreate_u64(d as u64)),
    )
}

#[inline(always)]
unsafe fn or(a: Vec256, b: Vec256) -> Vec256 {
    (vorrq_u64(a.0, b.0), vorrq_u64(a.1, b.1))
}

#[inline(always)]
unsafe fn and_not(a: Vec256, b: Vec256) -> Vec256 {
    (vbicq_u64(b.0, a.0), vbicq_u64(b.1, a.1))
}

#[inline(always)]
unsafe fn shift_left(a: Vec256, shift: Vec256) -> Vec256 {
    (
        vshlq_u64(a.0, vreinterpretq_s64_u64(shift.0)),
        vshlq_u64(a.1, vreinterpretq_s64_u64(shift.1)),
    )
}

#[inline(always)]
unsafe fn shift_right(a: Vec256, shift: Vec256) -> Vec256 {
    (
        vshlq_u64(a.0, vnegq_s64(vreinterpretq_s64_u64(shift.0))),
        vshlq_u64(a.1, vnegq_s64(vreinterpretq_s64_u64(shift.1))),
    )
}

#[inline(always)]
fn reduce_or(a: Vec256, b: Vec256) -> u64 {
    unsafe {
        let or = or(a, b);
        let combined = vorrq_u64(or.0, or.1);
        vgetq_lane_u64(combined, 0) | vgetq_lane_u64(combined, 1)
    }
}