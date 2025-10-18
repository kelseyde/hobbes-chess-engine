use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::square::Square;

static mut BETWEEN: [[Bitboard; 64]; 64] = [[Bitboard(0); 64]; 64];

pub fn init() {
    unsafe {
        init_between();
        init_extending();
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
    unsafe { BETWEEN[a][b] }
}

unsafe fn init_extending() {
    for a in 0..64 {
        for b in 0..64 {
            let a = Square(a);
            let b = Square(b);

            if attacks::rook(a, Bitboard::empty()).contains(b) {
                BETWEEN[a][b] =
                    attacks::rook(a, Bitboard::empty()) & attacks::rook(b, Bitboard::empty());
            }

            if attacks::bishop(a, Bitboard::empty()).contains(b) {
                BETWEEN[a][b] =
                    attacks::bishop(a, Bitboard::empty()) & attacks::bishop(b, Bitboard::empty());
            }
        }
    }
}
