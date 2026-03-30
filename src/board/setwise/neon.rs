use std::arch::aarch64::*;
use crate::board::bitboard::Bitboard;

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
        Bitboard(reduce_or2(or256(a, c), or256(b, d)))
    }
}

fn knights_setwise(knights: Bitboard) -> [Vec256; 2] {
    let a = 0x0101010101010101i64;
    let b = a << 1;
    let g = a << 6;
    let h = a << 7;

    unsafe {
        let filemask1 = setr(a | b, a, h, g | h);
        let filemask2 = setr(g | h, h, a, a | b);

        let sq = set1(knights.0 as i64);
        let offsets = setr(6, 15, 17, 10);
        let upper = sllv256(and_not256(filemask1, sq), offsets);
        let lower = srlv256(and_not256(filemask2, sq), offsets);

        [upper, lower]
    }
}

fn sliders_setwise(orth: Bitboard, diag: Bitboard, blockers: Bitboard) -> [Vec256; 2] {
    let a = 0x0101010101010101i64;
    let h = a << 7;

    unsafe {
        let shift = |n: i64| setr(7 * n, 9 * n, 8 * n, n);

        let filemask1 = setr(a, h, 0, h);
        let filemask2 = setr(h, a, 0, a);

        let mut gen1 = setr(diag.0 as i64, diag.0 as i64, orth.0 as i64, orth.0 as i64);
        let mut block1 = or256(set1(blockers.0 as i64), filemask1);

        let mut gen2 = setr(diag.0 as i64, diag.0 as i64, orth.0 as i64, orth.0 as i64);
        let mut block2 = or256(set1(blockers.0 as i64), filemask2);

        gen1 = or256(gen1, and_not256(block1, srlv256(gen1, shift(1))));
        gen2 = or256(gen2, and_not256(block2, sllv256(gen2, shift(1))));

        block1 = or256(block1, srlv256(block1, shift(1)));
        block2 = or256(block2, sllv256(block2, shift(1)));

        gen1 = or256(gen1, and_not256(block1, srlv256(gen1, shift(2))));
        gen2 = or256(gen2, and_not256(block2, sllv256(gen2, shift(2))));

        block1 = or256(block1, srlv256(block1, shift(2)));
        block2 = or256(block2, sllv256(block2, shift(2)));

        gen1 = or256(gen1, and_not256(block1, srlv256(gen1, shift(4))));
        gen2 = or256(gen2, and_not256(block2, sllv256(gen2, shift(4))));

        gen1 = and_not256(filemask1, srlv256(gen1, shift(1)));
        gen2 = and_not256(filemask2, sllv256(gen2, shift(1)));

        [gen1, gen2]
    }
}

#[inline(always)]
unsafe fn set1(x: i64) -> Vec256 {
    let v = vdupq_n_u64(x as u64);
    (v, v)
}

#[inline(always)]
unsafe fn setr(a: i64, b: i64, c: i64, d: i64) -> Vec256 {
    (
        vcombine_u64(vcreate_u64(a as u64), vcreate_u64(b as u64)),
        vcombine_u64(vcreate_u64(c as u64), vcreate_u64(d as u64)),
    )
}

#[inline(always)]
unsafe fn or256(a: Vec256, b: Vec256) -> Vec256 {
    (vorrq_u64(a.0, b.0), vorrq_u64(a.1, b.1))
}

#[inline(always)]
unsafe fn and_not256(a: Vec256, b: Vec256) -> Vec256 {
    (vbicq_u64(b.0, a.0), vbicq_u64(b.1, a.1))
}

#[inline(always)]
unsafe fn sllv256(a: Vec256, shift: Vec256) -> Vec256 {
    (
        vshlq_u64(a.0, vreinterpretq_s64_u64(shift.0)),
        vshlq_u64(a.1, vreinterpretq_s64_u64(shift.1)),
    )
}

#[inline(always)]
unsafe fn srlv256(a: Vec256, shift: Vec256) -> Vec256 {
    (
        vshlq_u64(a.0, vnegq_s64(vreinterpretq_s64_u64(shift.0))),
        vshlq_u64(a.1, vnegq_s64(vreinterpretq_s64_u64(shift.1))),
    )
}

fn reduce_or2(a: Vec256, b: Vec256) -> u64 {
    unsafe {
        let or = or256(a, b);
        let combined = vorrq_u64(or.0, or.1);
        vgetq_lane_u64(combined, 0) | vgetq_lane_u64(combined, 1)
    }
}