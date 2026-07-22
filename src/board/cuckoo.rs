use crate::board::bitboard::Bitboard;
use crate::board::moves::{Move, MoveFlag};
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::zobrist::Keys;
use crate::board::{attacks, ray, Board};
use crate::search::thread::ThreadData;

/// A mechanism used to determine if a move will lead to a repetition on the next ply. This is used
/// to detect repetitions one ply earlier during search. Implementation based on this paper:
/// <http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf>
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
    const PIECES: [Piece; 5] = [
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];
    const SIDES: [Side; 2] = [Side::White, Side::Black];

    let mut count = 0usize;

    for &piece in &PIECES {
        for &side in &SIDES {
            for from in 0..64u8 {
                let from_sq = Square(from);
                let attacks = attacks::attacks(from_sq, piece, side, Bitboard::empty());

                for to in (from + 1)..64 {
                    let to_sq = Square(to);
                    if !attacks.contains(to_sq) {
                        continue;
                    }

                    let key =
                        Keys::sq(piece, side, from_sq) ^ Keys::sq(piece, side, to_sq) ^ Keys::stm();
                    let mv = Move::new(from_sq, to_sq, MoveFlag::Standard);
                    insert(key, mv);
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

fn insert(mut key: u64, mut mv: Move) {
    let mut slot = Cuckoo::h1(key);

    loop {
        unsafe {
            std::mem::swap(&mut CUCKOO_KEYS[slot], &mut key);
            std::mem::swap(&mut CUCKOO_MOVES[slot], &mut mv);
        }

        if mv.is_null() {
            break;
        }

        let h1 = Cuckoo::h1(key);
        let h2 = Cuckoo::h2(key);
        slot = if slot == h1 { h2 } else { h1 };
    }
}

impl Board {
    /// <http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf>
    pub fn has_upcoming_repetition(&self, td: &ThreadData, ply: usize) -> bool {
        // TODO plies from null?
        let hm = (self.hm as usize).min(td.keys.len().saturating_sub(1));
        if hm < 3 {
            return false;
        }

        let occ = self.occ();
        let get_prev_key = |plies: usize| td.keys[td.keys.len() - 1 - plies];
        let curr_key = self.hashes.hash();

        let mut other = curr_key ^ get_prev_key(1) ^ Keys::stm();

        for i in (3..=hm).step_by(2) {
            other ^= get_prev_key(i - 1) ^ get_prev_key(i) ^ Keys::stm();

            if other != 0 {
                continue;
            }

            let diff = curr_key ^ get_prev_key(i);
            let mut slot = Cuckoo::h1(diff);

            if Cuckoo::keys(slot) != diff {
                slot = Cuckoo::h2(diff);

                if Cuckoo::keys(slot) != diff {
                    continue;
                }
            }

            let mv = Cuckoo::moves(slot);
            let ray = ray::between(mv.from(), mv.to());

            if !(ray & occ).is_empty() {
                continue;
            }

            if ply > i {
                return true;
            }

            let from = mv.from();
            let to = mv.to();
            let mut target_sq = from;
            if self.piece_at(from).is_none() {
                target_sq = to;
            }
            let us = self.us();
            return us.contains(target_sq);
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use crate::board::cuckoo;
    use crate::board::moves::Move;
    use crate::board::observer::NullBoardObserver;
    use crate::board::ray;
    use crate::board::Board;
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
            board.make(&mv, &mut NullBoardObserver);
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
        assert!(board.has_upcoming_repetition(&td, 0));
    }

    #[test]
    fn no_upcoming_rep_at_start() {
        let (board, td) = setup(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &[],
        );
        assert!(!board.has_upcoming_repetition(&td, 0));
    }

    #[test]
    fn no_upcoming_rep_after_one_move() {
        let (board, td) = setup(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["g1f3"],
        );
        assert!(!board.has_upcoming_repetition(&td, 0));
    }

    #[test]
    fn upcoming_rep_rook_shuffle() {
        let (board, td) = setup(
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
            &["a1a3", "e8d8", "a3a1", "d8e8"],
        );
        assert_eq!(board.hm, 4);
        assert!(board.has_upcoming_repetition(&td, 0));
    }

    #[test]
    fn no_upcoming_rep_when_path_blocked() {
        let (board, td) = setup(
            "4k3/8/8/8/8/R7/P7/4K3 w - - 0 1",
            &["a3a4", "e8d8", "a4a3", "d8e8"],
        );
        assert!(board.has_upcoming_repetition(&td, 0));
    }

    #[test]
    fn no_upcoming_rep_rook_blocked() {
        let (board, td) = setup(
            "4k3/8/8/8/8/8/P7/R3K3 w - - 0 1",
            &["a1a3", "e8d8", "a3a1", "d8e8", "a2a4"],
        );
        assert_eq!(board.hm, 0);
        assert!(!board.has_upcoming_repetition(&td, 0));
    }

    #[test]
    fn upcoming_rep_after_pawn_move_resets() {
        let (board, td) = setup(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            &["g1f3", "g8f6", "f3g1", "f6g8", "e2e4"],
        );
        assert_eq!(board.hm, 0);
        assert!(!board.has_upcoming_repetition(&td, 0));
    }
}
