use crate::board::piece::Piece;
use crate::board::piece::Piece::Pawn;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;

/// Represents the zobrist hashes for various aspects of the current position. The main hash covers
/// the entire board state, with separate hashes for pawns, non-pawns for both sides, major and
/// minor pieces.
#[rustfmt::skip]
#[derive(Debug, Clone, Copy, Default)]
pub struct Hashes {
    pub hash: u64,                   // Zobrist hash for the board
    pub pawn_hash: u64,              // Zobrist hash for pawns
    pub non_pawn_hashes: [u64; 2],   // Zobrist hashes for non-pawns
    pub major_hash: u64,             // Zobrist hash for major pieces
    pub minor_hash: u64,             // Zobrist hash for minor pieces
}

/// Represents the set of random numbers used to generate the zobrist hashes.
#[rustfmt::skip]
pub struct Keys {
    pub pieces: [[u64; 64]; 12], // Zobrist keys for pieces on squares
    pub ep: [u64; 64],           // Zobrist keys for en passant squares
    pub castle: [u64; 16],       // Zobrist keys for castling rights
    pub side: u64,               // Zobrist key for side to move
    pub hm: [u64; 16]            // Zobrist keys for the half-move clock buckets
}

pub const KEYS: Keys = {
    const SEED: u64 = 0xFFAA_B58C_5833_FE89u64;
    const INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    let mut zobrist = [0; 865];
    let mut state = SEED;

    let mut i = 0;
    while i < zobrist.len() {
        state = state.wrapping_add(INCREMENT);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        zobrist[i] = z ^ (z >> 31);

        i += 1;
    }
    unsafe { std::mem::transmute(zobrist) }
};

impl Hashes {
    pub fn new(board: &Board) -> Self {
        Self {
            hash: Keys::get_hash(board),
            pawn_hash: Keys::get_pawn_hash(board),
            non_pawn_hashes: Keys::get_non_pawn_hashes(board),
            major_hash: Keys::get_major_hash(board),
            minor_hash: Keys::get_minor_hash(board),
        }
    }
}

impl Keys {

    pub fn get_hash(board: &Board) -> u64 {
        let mut hash: u64 = 0;
        // Iterate over all squares and pieces
        for sq in Square::iter() {
            if let Some(pc) = board.piece_at(sq) {
                let pc_index = Keys::piece_index(pc, board.side_at(sq).unwrap());
                hash ^= KEYS.pieces[pc_index][sq];
            }
        }

        // Add en passant square if it exists
        if let Some(ep_sq) = board.ep_sq {
            hash ^= KEYS.ep[ep_sq];
        }

        // Add castling rights
        hash ^= KEYS.castle[board.rights.hash() as usize];

        // Add side to move
        if board.stm == Side::Black {
            hash ^= KEYS.side;
        }

        hash
    }

    pub fn get_pawn_hash(board: &Board) -> u64 {
        let mut hash: u64 = 0;
        for sq in board.bb[Pawn] {
            if let Some(side) = board.side_at(sq) {
                hash ^= Self::sq(Pawn, side, sq);
            }
        }
        hash
    }

    pub fn get_non_pawn_hashes(board: &Board) -> [u64; 2] {
        let mut hashes: [u64; 2] = [0, 0];
        for sq in Square::iter() {
            if let Some(pc) = board.piece_at(sq) {
                if pc != Pawn {
                    let side = board.side_at(sq).unwrap();
                    hashes[side] ^= Self::sq(pc, side, sq);
                }
            }
        }
        hashes
    }

    pub fn get_major_hash(board: &Board) -> u64 {
        Self::get_filtered_hash(board, Piece::is_major)
    }

    pub fn get_minor_hash(board: &Board) -> u64 {
        Self::get_filtered_hash(board, Piece::is_minor)
    }

    fn get_filtered_hash(board: &Board, filter: impl Fn(Piece) -> bool) -> u64 {
        let mut hash = 0u64;
        for sq in Square::iter() {
            if let Some(pc) = board.piece_at(sq) {
                if filter(pc) {
                    let side = board.side_at(sq).unwrap();
                    hash ^= Self::sq(pc, side, sq);
                }
            }
        }
        hash
    }

    pub fn sq(pc: Piece, side: Side, sq: Square) -> u64 {
        let piece_index = Keys::piece_index(pc, side);
        KEYS.pieces[piece_index][sq]
    }

    pub fn ep(ep_sq: Square) -> u64 {
        KEYS.ep[ep_sq]
    }

    pub fn castle(castle: u8) -> u64 {
        KEYS.castle[castle as usize]
    }

    pub fn stm() -> u64 {
        KEYS.side
    }

    pub fn null_move() -> u64 {
        KEYS.side
    }

    fn piece_index(piece: Piece, side: Side) -> usize {
        piece as usize + if side == Side::White { 0 } else { 6 }
    }
}

#[cfg(test)]
mod test {
    use crate::board::moves::{Move, MoveFlag};
    use crate::board::Board;

    #[test]
    fn test_move_piece() {
        assert_hash(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            &Move::parse_uci("e2e4"),
        );
    }

    #[test]
    fn test_capture() {
        assert_hash(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
            &Move::parse_uci("e4d5"),
        );
    }

    #[test]
    fn test_castle() {
        assert_hash(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
            &Move::parse_uci_with_flag("e1g1", MoveFlag::CastleK),
        );
    }

    #[test]
    fn test_ep() {
        assert_hash(
            "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "rnbqkb1r/ppp1pppp/3P1n2/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
            &Move::parse_uci_with_flag("e5d6", MoveFlag::EnPassant),
        );
    }

    #[test]
    fn test_promotion() {
        assert_hash(
            "rnbqkb1r/pP3ppp/5n2/4p3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 5",
            "rnRqkb1r/p4ppp/5n2/4p3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 5",
            &Move::parse_uci_with_flag("b7c8r", MoveFlag::PromoR),
        );
    }

    fn assert_hash(fen1: &str, fen2: &str, m: &Move) {
        let mut board1 = Board::from_fen(fen1).unwrap();
        board1.make(m);
        let board2 = Board::from_fen(fen2).unwrap();
        assert_eq!(board1.hash(), board2.hash());
    }
}
