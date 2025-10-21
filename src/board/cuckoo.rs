use std::sync::OnceLock;

use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::moves::{Move, MoveFlag};
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::zobrist::Zobrist;

// Cuckoo hashing table for quickly mapping a Zobrist move key difference (from/to for a piece)
// back to the corresponding Move. Translated from the provided Java implementation.
// It only stores standard (non-pawn, non-special) piece moves generated on an empty board.
//
// Move key diff definition: XOR of Zobrist piece-square keys for the moving piece & side
// at the from and to squares.
//
// Table size: 8192 (2^13). Two hash functions:
// h1(k) = bits 32..44 of k (13 bits)
// h2(k) = bits 48..60 of k (13 bits)
// (Mask 0x1FFF)
//
// Insertion uses classic cuckoo displacement until an empty slot (Move::NONE) is found.
// The expected total number of stored moves is asserted to be 3668 (as in Java version).
pub struct Cuckoo;

const SIZE: usize = 8192;

static TABLE: OnceLock<([u64; SIZE], [Move; SIZE])> = OnceLock::new();

impl Cuckoo {

    #[inline(always)]
    pub fn h1(k: u64) -> usize { ((k >> 32) & 0x1FFF) as usize }

    #[inline(always)]
    pub fn h2(k: u64) -> usize { ((k >> 48) & 0x1FFF) as usize }

    // Initialize the table lazily. Returns references to the keys and moves arrays.
    fn init() -> &'static ([u64; SIZE], [Move; SIZE]) {
        TABLE.get_or_init(|| {
            let mut keys = [0u64; SIZE];
            let mut moves = [Move::NONE; SIZE];
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

                            // Zobrist key difference for moving piece (remove from, add to)
                            let mut key_diff = Zobrist::sq(piece, side, from_sq) ^ Zobrist::sq(piece, side, to_sq);

                            // First candidate slot
                            let mut slot = Self::h1(key_diff);

                            loop {
                                let key_temp = keys[slot];
                                keys[slot] = key_diff;
                                key_diff = key_temp;

                                let move_temp = moves[slot];
                                moves[slot] = mv;
                                mv = move_temp;

                                if mv.is_null() {
                                    // found empty spot
                                    break;
                                }

                                // Alternate between the two hash locations for the displaced key
                                let h1 = Self::h1(key_diff);
                                let h2 = Self::h2(key_diff);
                                slot = if slot == h1 { h2 } else { h1 };
                            }
                            count += 1;
                        }
                    }
                }
            }

            assert!(count == 3668, "Failed to initialise cuckoo tables: expected 3668, got {}", count);
            (keys, moves)
        })
    }

    pub fn keys() -> &'static [u64; SIZE] { &Self::init().0 }
    pub fn moves() -> &'static [Move; SIZE] { &Self::init().1 }

    pub fn lookup(key_diff: u64) -> Option<Move> {
        let keys = Self::keys();
        let moves = Self::moves();
        let slot1 = Self::h1(key_diff);
        if keys[slot1] == key_diff { return Some(moves[slot1]); }
        let slot2 = Self::h2(key_diff);
        if keys[slot2] == key_diff { return Some(moves[slot2]); }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::Cuckoo;
    use crate::board::{attacks, bitboard::Bitboard, moves::{MoveFlag, Move}, piece::Piece, side::Side, square::Square, zobrist::Zobrist};

    #[test]
    fn test_expected_count() {
        let moves = Cuckoo::moves();
        let populated = moves.iter().filter(|m| !m.is_null()).count();
        assert_eq!(populated, 3668);
    }

    #[test]
    fn test_lookup_roundtrip() {
        let pieces = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];
        let sides = [Side::White, Side::Black];
        for &piece in &pieces {
            for &side in &sides {
                for from in 0..64 {
                    for to in (from + 1)..64 {
                        let from_sq = Square(from as u8);
                        let to_sq = Square(to as u8);
                        let attacks = attacks::attacks(from_sq, piece, side, Bitboard::empty());
                        if !attacks.contains(to_sq) { continue; }
                        let key_diff = Zobrist::sq(piece, side, from_sq) ^ Zobrist::sq(piece, side, to_sq);
                        let looked = Cuckoo::lookup(key_diff).expect("Move should exist");
                        assert_eq!(looked.from().0, from_sq.0);
                        assert_eq!(looked.to().0, to_sq.0);
                        assert!(looked.flag() == MoveFlag::Standard);
                    }
                }
            }
        }
    }

    #[test]
    fn test_missing_key_returns_none() {
        let keys = Cuckoo::keys();
        let sample = keys.iter().find(|k| **k != 0).cloned().unwrap();
        let altered = sample ^ 1;
        if altered != sample {
            if !keys.iter().any(|k| *k == altered) {
                assert!(Cuckoo::lookup(altered).is_none());
            }
        }
    }
}

