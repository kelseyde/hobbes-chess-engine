use std::arch::x86_64::*;

use crate::board::bitboard::Bitboard;
use crate::board::file::File;
use crate::board::rank::Rank;

const A: i64 = File::A.to_bb().0 as i64;
const B: i64 = File::B.to_bb().0 as i64;
const G: i64 = File::G.to_bb().0 as i64;
const H: i64 = File::H.to_bb().0 as i64;
const R1: i64 = Rank::One.to_bb().0 as i64;
const R2: i64 = Rank::Two.to_bb().0 as i64;
const R7: i64 = Rank::Seven.to_bb().0 as i64;
const R8: i64 = Rank::Eight.to_bb().0 as i64;

fn knights_setwise(knights: Bitboard) -> __m512i {
    unsafe {
        // knight moves are done clockwise, starting at wnw.
        let rotates = _mm512_setr_epi64(6, 15, 17, 10, -6, -15, -17, -10);
        // mask containing the files+ranks that need to be removed for each shift
        // (e.g. a knight that is on files a or b or on rank 8 cannot move wnw).
        let mask = _mm512_setr_epi64(
            A | B | R8,
            A | R7 | R8,
            H | R7 | R8,
            G | H | R8,
            G | H | R1,
            H | R1 | R2,
            A | R1 | R2,
            A | B | R1,
        );

        _mm512_rolv_epi64(
            _mm512_andnot_si512(mask, _mm512_set1_epi64(knights.0 as i64)),
            rotates,
        )
    }
}

fn sliders_setwise(orth: Bitboard, diag: Bitboard, blockers: Bitboard) -> __m512i {
    unsafe {
        let (orth, diag) = (orth.0 as i64, diag.0 as i64);
        let rotate = |n: i64| _mm512_setr_epi64(-7 * n, -9 * n, 7 * n, 9 * n, n, -8 * n, -n, 8 * n);
        // se, sw, nw, ne, e, s, w, n
        let mut generate = _mm512_setr_epi64(diag, diag, diag, diag, orth, orth, orth, orth);
        let mask = _mm512_setr_epi64(A | R8, H | R8, H | R1, A | R1, A, R8, H, R1);
        let mut blockers = _mm512_or_si512(mask, _mm512_set1_epi64(blockers.0 as i64));

        // 242 <=> a | (!b & c)
        generate = _mm512_ternarylogic_epi64(
            generate,
            blockers,
            _mm512_rolv_epi64(generate, rotate(1)),
            242,
        );
        blockers = _mm512_or_si512(blockers, _mm512_rolv_epi64(blockers, rotate(1)));

        generate = _mm512_ternarylogic_epi64(
            generate,
            blockers,
            _mm512_rolv_epi64(generate, rotate(2)),
            242,
        );
        blockers = _mm512_or_si512(blockers, _mm512_rolv_epi64(blockers, rotate(2)));

        generate = _mm512_ternarylogic_epi64(
            generate,
            blockers,
            _mm512_rolv_epi64(generate, rotate(4)),
            242,
        );

        _mm512_andnot_si512(mask, _mm512_rolv_epi64(generate, rotate(1)))
    }
}

pub fn knights_and_sliders_setwise(
    knights: Bitboard,
    orthos: Bitboard,
    diags: Bitboard,
    blockers: Bitboard,
) -> Bitboard {
    unsafe {
        Bitboard(_mm512_reduce_or_epi64(_mm512_or_si512(
            knights_setwise(knights),
            sliders_setwise(orthos, diags, blockers),
        )) as u64)
    }
}