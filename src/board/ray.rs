use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::file::File;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::tools::utils;
use utils::slide;

/// Precomputed table of the squares between two squares on the same ray, excluding those two squares.
static mut BETWEEN: [[Bitboard; 64]; 64] = [[Bitboard::NONE; 64]; 64];

// Pre-computed table of the squares on the ray belonging to two squares, excluding any blockers.
static mut EXTENDING: [[Bitboard; 64]; 64] = [[Bitboard::NONE; 64]; 64];

// Pre-computed table of diagonals on the board (both NE/SW diagonals and NW/SE anti-diagonals)
static mut DIAGONALS: [[Bitboard; 64]; 2] = [[Bitboard::NONE; 64]; 2];

// Pre-computed table of the squares beyond a blocker on the same ray, excluding the blocker.
static mut BEYOND: [[Bitboard; 64]; 64] = [[Bitboard::NONE; 64]; 64];

pub fn init() {
    unsafe {
        init_between();
        init_extending();
        init_diagonals();
        init_beyond();
    }
}

pub fn between(a: Square, b: Square) -> Bitboard {
    unsafe { BETWEEN[a][b] }
}

pub fn relative_diagonal(side: Side, sq: Square) -> Bitboard {
    unsafe { DIAGONALS[side as usize][sq.0 as usize] }
}

pub fn extending(a: Square, b: Square) -> Bitboard {
    unsafe { EXTENDING[a][b] }
}

pub fn beyond(a: Square, b: Square) -> Bitboard {
    unsafe { BEYOND[a][b] }
}

unsafe fn init_between() {
    for a in Square::iter() {
        for b in Square::iter() {
            if attacks::rook(a, Bitboard::NONE).contains(b) {
                let a_attacks = attacks::rook(a, Bitboard::of_sq(b));
                let b_attacks = attacks::rook(b, Bitboard::of_sq(a));
                BETWEEN[a][b] = a_attacks & b_attacks;
            }

            if attacks::bishop(a, Bitboard::NONE).contains(b) {
                let a_attacks = attacks::bishop(a, Bitboard::of_sq(b));
                let b_attacks = attacks::bishop(b, Bitboard::of_sq(a));
                BETWEEN[a][b] = a_attacks & b_attacks;
            }
        }
    }
}

unsafe fn init_extending() {
    for a in Square::iter() {
        for b in Square::iter() {
            if attacks::rook(a, Bitboard::NONE).contains(b) {
                let a_attacks = attacks::rook(a, Bitboard::NONE);
                let b_attacks = attacks::rook(b, Bitboard::NONE);
                EXTENDING[a][b] = a_attacks & b_attacks;
            }

            if attacks::bishop(a, Bitboard::NONE).contains(b) {
                let a_attacks = attacks::bishop(a, Bitboard::NONE);
                let b_attacks = attacks::bishop(b, Bitboard::NONE);
                EXTENDING[a][b] = a_attacks & b_attacks;
            }
        }
    }
}

unsafe fn init_diagonals() {
    for sq in Square::iter() {
        // NE/SW diagonal
        let ne = slide(sq, Bitboard::NONE, 9, |s| s.0 % 8 == 7 || s.0 / 8 == 7);
        let sw = slide(sq, Bitboard::NONE, -9, |s| s.0 % 8 == 0 || s.0 / 8 == 0);
        DIAGONALS[0][sq] = ne | sw;

        // NW/SE anti-diagonal
        let nw = slide(sq, Bitboard::NONE, 7, |s| s.0 % 8 == 0 || s.0 / 8 == 7);
        let se = slide(sq, Bitboard::NONE, -7, |s| s.0 % 8 == 7 || s.0 / 8 == 0);
        DIAGONALS[1][sq] = nw | se;
    }
}

unsafe fn init_beyond() {
    for sq in Square::iter() {
        let origin_file = File::of(sq) as i8;
        let origin_rank = Rank::of(sq) as i8;

        for blocker in Square::iter() {
            if sq == blocker {
                continue;
            }

            let blocker_file = File::of(blocker) as i8;
            let blocker_rank = Rank::of(blocker) as i8;

            let file_delta = blocker_file - origin_file;
            let rank_delta = blocker_rank - origin_rank;

            // Beyond ray only possible for squares on the same rank/file/diagonal
            let same_file = file_delta == 0;
            let same_rank = rank_delta == 0;
            let same_diagonal = file_delta.abs() == rank_delta.abs();
            if !(same_file || same_rank || same_diagonal) {
                continue;
            }

            let file_step = file_delta.signum();
            let rank_step = rank_delta.signum();

            // Start one square past the blocker and walk to the edge of the board
            let mut beyond = Bitboard::NONE;
            let mut file = blocker_file + file_step;
            let mut rank = blocker_rank + rank_step;
            while (0..8).contains(&file) && (0..8).contains(&rank) {
                let sq = Square((rank * 8 + file) as u8);
                beyond |= Bitboard::of_sq(sq);
                file += file_step;
                rank += rank_step;
            }

            BEYOND[sq][blocker] = beyond;
        }
    }
}
