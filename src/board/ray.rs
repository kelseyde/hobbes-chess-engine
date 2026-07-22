use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::file::File;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::tools::utils;
use utils::slide;

static mut BETWEEN: [[Bitboard; 64]; 64] = [[Bitboard(0); 64]; 64];
static mut EXTENDING: [[Bitboard; 64]; 64] = [[Bitboard(0); 64]; 64];
static mut DIAGONALS: [[Bitboard; 64]; 2] = [[Bitboard(0); 64]; 2];
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
            if attacks::rook(a, Bitboard::empty()).contains(b) {
                BETWEEN[a][b] =
                    attacks::rook(a, Bitboard::of_sq(b)) & attacks::rook(b, Bitboard::of_sq(a));
            }

            if attacks::bishop(a, Bitboard::empty()).contains(b) {
                BETWEEN[a][b] =
                    attacks::bishop(a, Bitboard::of_sq(b)) & attacks::bishop(b, Bitboard::of_sq(a));
            }
        }
    }
}

unsafe fn init_extending() {
    for a in Square::iter() {
        for b in Square::iter() {
            if attacks::rook(a, Bitboard::empty()).contains(b) {
                EXTENDING[a][b] =
                    attacks::rook(a, Bitboard::empty()) & attacks::rook(b, Bitboard::empty());
            }

            if attacks::bishop(a, Bitboard::empty()).contains(b) {
                EXTENDING[a][b] =
                    attacks::bishop(a, Bitboard::empty()) & attacks::bishop(b, Bitboard::empty());
            }
        }
    }
}

unsafe fn init_diagonals() {
    for sq in Square::iter() {
        // Diagonal (NE/SW): +9 stops at rank 8 or file H; -9 stops at rank 1 or file A
        DIAGONALS[0][sq] = Bitboard(
            slide(sq.0 as usize, 0, 9, |s| s % 8 == 7 || s / 8 == 7)
                | slide(sq.0 as usize, 0, -9, |s| s % 8 == 0 || s / 8 == 0),
        );
        // Anti-diagonal (NW/SE): +7 stops at rank 8 or file A; -7 stops at rank 1 or file H
        DIAGONALS[1][sq] = Bitboard(
            slide(sq.0 as usize, 0, 7, |s| s % 8 == 0 || s / 8 == 7)
                | slide(sq.0 as usize, 0, -7, |s| s % 8 == 7 || s / 8 == 0),
        );
    }
}

unsafe fn init_beyond() {
    for origin in Square::iter() {
        let origin_file = File::of(origin) as i8;
        let origin_rank = Rank::of(origin) as i8;

        for blocker in Square::iter() {
            if origin == blocker {
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
                beyond |= Bitboard::of_sq(Square((rank * 8 + file) as u8));
                file += file_step;
                rank += rank_step;
            }

            BEYOND[origin][blocker] = beyond;
        }
    }
}
