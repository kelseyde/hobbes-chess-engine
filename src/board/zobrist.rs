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
    board: u64,           // Zobrist hash for the entire board
    pawn: u64,            // Zobrist hash for pawns
    non_pawn: [u64; 2],   // Zobrist hashes for non-pawns
    major: u64,           // Zobrist hash for major pieces
    minor: u64,           // Zobrist hash for minor pieces
}

/// Represents the set of random numbers used to generate the zobrist hashes.
#[rustfmt::skip]
pub struct Keys {
    pieces: [[u64; 64]; 12], // Zobrist keys for pieces on squares
    ep: [u64; 64],           // Zobrist keys for en passant squares
    castle: [u64; 16],       // Zobrist keys for castling rights
    side: u64,               // Zobrist key for side to move
}

pub const KEYS: Keys = {
    const SEED: u64 = 0xFFAA_B58C_5833_FE89u64;
    const INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

    let mut zobrist = [0; 849];
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
            board: Keys::get_hash(board),
            pawn: Keys::get_pawn_hash(board),
            non_pawn: Keys::get_non_pawn_hashes(board),
            major: Keys::get_major_hash(board),
            minor: Keys::get_minor_hash(board),
        }
    }

    pub fn flip_stm(&mut self) {
        self.board ^= KEYS.side;
    }

    pub fn update_hash(&mut self, hash: u64) {
        self.board ^= hash;
    }

    pub fn update_pawn_hash(&mut self, hash: u64) {
        self.pawn ^= hash;
    }

    pub fn update_non_pawn_hash(&mut self, side: Side, hash: u64) {
        self.non_pawn[side] ^= hash;
    }

    pub fn update_major_hash(&mut self, hash: u64) {
        self.major ^= hash;
    }

    pub fn update_minor_hash(&mut self, hash: u64) {
        self.minor ^= hash;
    }

    pub const fn hash(&self) -> u64 {
        self.board
    }

    pub const fn pawn_hash(&self) -> u64 {
        self.pawn
    }

    pub fn non_pawn_hash(&self, side: Side) -> u64 {
        self.non_pawn[side]
    }

    pub const fn major_hash(&self) -> u64 {
        self.major
    }

    pub const fn minor_hash(&self) -> u64 {
        self.minor
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
        // SAFETY: piece_index is 0-11, sq.0 is 0-63, both within bounds.
        unsafe {
            *KEYS
                .pieces
                .get_unchecked(piece_index)
                .get_unchecked(sq.0 as usize)
        }
    }

    pub fn ep(ep_sq: Square) -> u64 {
        // SAFETY: ep_sq.0 is always 0-63.
        unsafe { *KEYS.ep.get_unchecked(ep_sq.0 as usize) }
    }

    pub fn castle(castle: u8) -> u64 {
        // SAFETY: castle is masked to 4 bits (0-15).
        unsafe { *KEYS.castle.get_unchecked(castle as usize) }
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
