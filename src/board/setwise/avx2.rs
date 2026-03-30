use std::arch::x86_64::*;
use crate::board::attacks;
use crate::board::bitboard::Bitboard;

fn reduce_or2(a: __m256i, b: __m256i) -> u64 {
    unsafe {
        let or = _mm256_or_si256(a, b);
        let or = _mm_or_si128(_mm256_castsi256_si128(or), _mm256_extracti128_si256(or, 1));
        let or = _mm_or_si128(or, _mm_shuffle_epi32(or, 0xee));
        _mm_cvtsi128_si64(or) as u64
    }
}

fn knight_attacks_setwise(knights: Bitboard) -> [__m256i; 2] {
    let a = 0x0101010101010101i64;
    let b = a << 1;
    let g = a << 6;
    let h = a << 7;

    unsafe {
        let filemask1 = _mm256_setr_epi64x(a | b, a, h, g | h);
        let filemask2 = _mm256_setr_epi64x(g | h, h, a, a | b);

        let sq = _mm256_set1_epi64x(knights.0 as i64);
        let offsets = _mm256_setr_epi64x(6, 15, 17, 10);
        let upper = _mm256_sllv_epi64(_mm256_andnot_si256(filemask1, sq), offsets);
        let lower = _mm256_srlv_epi64(_mm256_andnot_si256(filemask2, sq), offsets);

        [upper, lower]
    }
}

fn slider_attacks_setwise(orth: Bitboard, diag: Bitboard, blockers: Bitboard) -> [__m256i; 2] {
    let a = 0x0101010101010101i64;
    let h = a << 7;
    unsafe {
        let shift = |n: i64| _mm256_setr_epi64x(7 * n, 9 * n, 8 * n, n);

        let filemask1 = _mm256_setr_epi64x(a, h, 0, h);
        let filemask2 = _mm256_setr_epi64x(h, a, 0, a);

        // se, sw, s, w
        let mut gen1 =
            _mm256_setr_epi64x(diag.0 as i64, diag.0 as i64, orth.0 as i64, orth.0 as i64);
        let mut block1 = _mm256_or_si256(_mm256_set1_epi64x(blockers.0 as i64), filemask1);

        // nw, ne, n, e
        let mut gen2 =
            _mm256_setr_epi64x(diag.0 as i64, diag.0 as i64, orth.0 as i64, orth.0 as i64);
        let mut block2 = _mm256_or_si256(_mm256_set1_epi64x(blockers.0 as i64), filemask2);

        gen1 = _mm256_or_si256(
            gen1,
            _mm256_andnot_si256(block1, _mm256_srlv_epi64(gen1, shift(1))),
        );
        gen2 = _mm256_or_si256(
            gen2,
            _mm256_andnot_si256(block2, _mm256_sllv_epi64(gen2, shift(1))),
        );

        block1 = _mm256_or_si256(block1, _mm256_srlv_epi64(block1, shift(1)));
        block2 = _mm256_or_si256(block2, _mm256_sllv_epi64(block2, shift(1)));

        gen1 = _mm256_or_si256(
            gen1,
            _mm256_andnot_si256(block1, _mm256_srlv_epi64(gen1, shift(2))),
        );
        gen2 = _mm256_or_si256(
            gen2,
            _mm256_andnot_si256(block2, _mm256_sllv_epi64(gen2, shift(2))),
        );

        block1 = _mm256_or_si256(block1, _mm256_srlv_epi64(block1, shift(2)));
        block2 = _mm256_or_si256(block2, _mm256_sllv_epi64(block2, shift(2)));

        gen1 = _mm256_or_si256(
            gen1,
            _mm256_andnot_si256(block1, _mm256_srlv_epi64(gen1, shift(4))),
        );
        gen2 = _mm256_or_si256(
            gen2,
            _mm256_andnot_si256(block2, _mm256_sllv_epi64(gen2, shift(4))),
        );

        gen1 = _mm256_andnot_si256(filemask1, _mm256_srlv_epi64(gen1, shift(1)));
        gen2 = _mm256_andnot_si256(filemask2, _mm256_sllv_epi64(gen2, shift(1)));

        [gen1, gen2]
    }
}

pub fn knight_and_slider_attacks_setwise(
    knights: Bitboard,
    orth: Bitboard,
    diag: Bitboard,
    blockers: Bitboard,
) -> Bitboard {
    unsafe {
        let [a, b] = knight_attacks_setwise(knights);
        let [c, d] = slider_attacks_setwise(orth, diag, blockers);
        Bitboard(reduce_or2(_mm256_or_si256(a, c), _mm256_or_si256(b, d)))
    }
}