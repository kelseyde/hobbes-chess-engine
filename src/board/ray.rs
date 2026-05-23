use utils::slide;
use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::tools::utils;

static mut BETWEEN: [[Bitboard; 64]; 64] = [[Bitboard(0); 64]; 64];
static mut EXTENDING: [[Bitboard; 64]; 64] = [[Bitboard(0); 64]; 64];
static mut DIAGONALS: [[Bitboard; 64]; 2] = [[Bitboard(0); 64]; 2];

pub fn init() {
    unsafe {
        init_between();
        init_extending();
        init_diagonals();
    }
}

pub fn between(a: Square, b: Square) -> Bitboard {
    unsafe { BETWEEN[a][b] }
}

unsafe fn init_between() {
    for a in 0..64 {
        for b in 0..64 {
            let a = Square(a);
            let b = Square(b);

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

pub fn extending(a: Square, b: Square) -> Bitboard {
    unsafe { EXTENDING[a][b] }
}

unsafe fn init_extending() {
    for a in 0..64 {
        for b in 0..64 {
            let a = Square(a);
            let b = Square(b);

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
    for sq in 0..64usize {
        // Diagonal (NE/SW): +9 stops at rank 8 or file H; -9 stops at rank 1 or file A
        DIAGONALS[0][sq] = Bitboard(
            slide(sq, 0,  9, |s| s % 8 == 7 || s / 8 == 7) |
            slide(sq, 0, -9, |s| s % 8 == 0 || s / 8 == 0),
        );
        // Anti-diagonal (NW/SE): +7 stops at rank 8 or file A; -7 stops at rank 1 or file H
        DIAGONALS[1][sq] = Bitboard(
            slide(sq, 0,  7, |s| s % 8 == 0 || s / 8 == 7) |
            slide(sq, 0, -7, |s| s % 8 == 7 || s / 8 == 0),
        );
    }
}

pub fn relative_diagonal(side: Side, sq: Square) -> Bitboard {
    unsafe { DIAGONALS[side as usize][sq.0 as usize] }
}
