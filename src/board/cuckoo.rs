use crate::board::attacks;
use crate::board::bitboard::Bitboard;
use crate::board::moves::{Move, MoveFlag};
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::zobrist::Keys;

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
                            Keys::sq(piece, side, from_sq) ^ Keys::sq(piece, side, to_sq) ^ Keys::stm();
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

#[cfg(test)]
mod tests {
    use crate::board::cuckoo;
    use crate::board::moves::Move;
    use crate::board::ray;
    use crate::board::Board;
    use crate::search::has_upcoming_repetition;
    use crate::search::thread::ThreadData;
    use std::sync::Once;

    fn init() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            ray::init();
            cuckoo::init();
        });
    }

    fn setup(fen: &str, moves: &[&str]) -> (Board, ThreadData) {
        init();
        let mut board = Board::from_fen(fen).unwrap();
        let mut td = ThreadData::default();
        td.keys.push(board.hash());
        td.root_ply = 0;

        for mv_str in moves {
            let mv = Move::parse_uci(mv_str);
            board.make(&mv);
            td.keys.push(board.hash());
        }

        (board, td)
    }

    #[test]
    fn upcoming_rep_king_shuffle() {
        let (board, td) = setup(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            &["e1e2", "e8d8", "e2e1", "d8e8"],
        );
        assert_eq!(board.hm, 4);
        assert!(has_upcoming_repetition(&board, &td, 0));
    }

    #[test]
    fn no_upcoming_rep_at_start() {
        let (board, td) = setup(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[],
        );
        assert!(!has_upcoming_repetition(&board, &td, 0));
    }

    #[test]
    fn no_upcoming_rep_after_one_move() {
        let (board, td) = setup(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["g1f3"],
        );
        assert!(!has_upcoming_repetition(&board, &td, 0));
    }

    #[test]
    fn upcoming_rep_rook_shuffle() {
        let (board, td) = setup(
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
            &["a1a3", "e8d8", "a3a1", "d8e8"],
        );
        assert_eq!(board.hm, 4);
        assert!(has_upcoming_repetition(&board, &td, 0));
    }

    #[test]
    fn no_upcoming_rep_when_path_blocked() {
        let (board, td) = setup(
            "4k3/8/8/8/8/R7/P7/4K3 w - - 0 1",
            &["a3a4", "e8d8", "a4a3", "d8e8"],
        );
        assert!(has_upcoming_repetition(&board, &td, 0));
    }

    #[test]
    fn no_upcoming_rep_rook_blocked() {
        let (board, td) = setup(
            "4k3/8/8/8/8/8/P7/R3K3 w - - 0 1",
            &["a1a3", "e8d8", "a3a1", "d8e8", "a2a4"],
        );
        assert_eq!(board.hm, 0);
        assert!(!has_upcoming_repetition(&board, &td, 0));
    }

    #[test]
    fn upcoming_rep_after_pawn_move_resets() {
        let (board, td) = setup(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["g1f3", "g8f6", "f3g1", "f6g8", "e2e4"],
        );
        assert_eq!(board.hm, 0);
        assert!(!has_upcoming_repetition(&board, &td, 0));
    }
}
