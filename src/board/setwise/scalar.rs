use crate::board::attacks;
use crate::board::bitboard::Bitboard;

fn knights_setwise(knights: Bitboard) -> Bitboard {
    knights
        .into_iter()
        .fold(Bitboard::empty(), |bb, n| bb | attacks::knight(n))
}

fn sliders_setwise(ortho: Bitboard, diag: Bitboard, blockers: Bitboard) -> Bitboard {
    let ortho_attacks = ortho
        .into_iter()
        .fold(Bitboard::empty(), |bb, o| bb | attacks::rook(o, blockers));
    let diag_attacks = diag
        .into_iter()
        .fold(Bitboard::empty(), |bb, d| bb | attacks::bishop(d, blockers));

    ortho_attacks | diag_attacks
}

pub fn knights_and_sliders_setwise(
    knights: Bitboard,
    orthos: Bitboard,
    diags: Bitboard,
    blockers: Bitboard,
) -> Bitboard {
    knights_setwise(knights) | sliders_setwise(orthos, diags, blockers)
}