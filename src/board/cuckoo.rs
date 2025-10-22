use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::moves::{Move, MoveFlag};
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::zobrist::Zobrist;

pub struct Cuckoo;

const SIZE: usize = 8192;

// Safety: These tables are written exactly once during startup, and only read thereafter.
static mut CUCKOO_KEYS: [u64; SIZE] = [0u64; SIZE];
static mut CUCKOO_MOVES: [Move; SIZE] = [Move::NONE; SIZE];

impl Cuckoo {
    #[inline(always)]
    pub fn h1(k: u64) -> usize {
        ((k >> 32) & 0x1FFF) as usize
    }

    #[inline(always)]
    pub fn h2(k: u64) -> usize {
        ((k >> 48) & 0x1FFF) as usize
    }

    pub fn keys(idx: usize) -> u64 {
        unsafe { CUCKOO_KEYS[idx] }
    }

    pub fn moves(idx: usize) -> Move {
        unsafe { CUCKOO_MOVES[idx] }
    }

}

pub fn init() {
    unsafe {
        let mut count: usize = 0;
        let pieces = [
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ];
        let sides = [Side::White, Side::Black];

        for &piece in &pieces {
            for &side in &sides {
                for from in 0..64 {
                    for to in (from + 1)..64 {
                        let from_sq = Square(from as u8);
                        let to_sq = Square(to as u8);
                        let attacks = attacks::attacks(from_sq, piece, side, Bitboard::empty());
                        if !attacks.contains(to_sq) {
                            continue;
                        }

                        let mut mv = Move::new(from_sq, to_sq, MoveFlag::Standard);
                        let mut key_diff =
                            Zobrist::sq(piece, side, from_sq) ^ Zobrist::sq(piece, side, to_sq);
                        let mut slot = Cuckoo::h1(key_diff);

                        loop {
                            let key_temp = CUCKOO_KEYS[slot];
                            CUCKOO_KEYS[slot] = key_diff;
                            key_diff = key_temp;

                            let move_temp = CUCKOO_MOVES[slot];
                            CUCKOO_MOVES[slot] = mv;
                            mv = move_temp;

                            if mv.is_null() {
                                break;
                            }

                            let h1 = Cuckoo::h1(key_diff);
                            let h2 = Cuckoo::h2(key_diff);
                            slot = if slot == h1 { h2 } else { h1 };
                        }
                        count += 1;
                    }
                }
            }
        }

        assert_eq!(
            count, 3668,
            "Failed to initialise cuckoo tables: expected 3668, got {}",
            count
        );
    }
}
