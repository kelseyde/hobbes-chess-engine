use crate::types::side::Side::{Black, White};

use crate::board::Board;
use crate::moves::Move;
use crate::types::File;
use crate::types::piece::{Piece, PIECES};
use crate::types::piece::Piece::King;
use crate::types::side::Side;
use crate::types::square::Square;

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 1024;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub const PIECE_OFFSET: usize = 64;
pub const SIDE_OFFSET: usize = 64 * 6;
pub const MAX_ACCUMULATORS: usize = 255;

pub const BUCKETS: [usize; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 4, 5, 5, 5, 5, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7
];

static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../resources/woodpusher.nnue")) };

#[repr(C, align(64))]
pub struct Network {
    feature_weights: [i16; FEATURES * HIDDEN],
    feature_bias: [i16; HIDDEN],
    output_weights: [i16; 2 * HIDDEN],
    output_bias: i16,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: [i16; HIDDEN],
    pub black_features: [i16; HIDDEN],
}

pub struct NNUE {
    stack: [Accumulator; MAX_ACCUMULATORS],
    current: usize,
}

impl Default for NNUE {
    fn default() -> Self {
        NNUE {
            current: 0,
            stack: [Accumulator::default(); MAX_ACCUMULATORS],
        }
    }
}

impl NNUE {

    /// Evaluates the current position. Gets the 'us-perspective' and 'them-perspective' feature
    /// sets, based on the side to move. Then, passes the features through the network to get the
    /// static evaluation.
    pub fn evaluate(&mut self, board: &Board) -> i32 {

        let acc = &self.stack[self.current];

        let us = match board.stm { White => acc.white_features, Black => acc.black_features };
        let them = match board.stm { White => acc.black_features, Black => acc.white_features };

        let mut output = i32::from(NETWORK.output_bias);

        for (&input, &weight) in us.iter().zip(NETWORK.output_weights[..HIDDEN].iter()) {
            output += crelu(input) * i32::from(weight);
        }

        for (&input, &weight) in them.iter().zip(NETWORK.output_weights[HIDDEN..].iter()) {
            output += crelu(input) * i32::from(weight);
        }

        output *= SCALE;
        output /= QAB;
        output
    }

    /// Activate the entire board from scratch. This initializes the accumulators based on the
    /// current board state, iterating over all pieces and their squares. Should be called only
    /// at the top of search, and then efficiently updated with each move.
    pub fn activate(&mut self, board: &Board) {
        self.current = 0;
        self.stack[self.current] = Accumulator::default();
        for &pc in PIECES.iter() {
            for sq in board.pcs(pc) {
                let side = board.side_at(sq).unwrap();
                let w_idx = ft_idx(sq, pc, side, White);
                let b_idx = ft_idx(sq, pc, side, Black);
                self.stack[self.current].add(w_idx, b_idx);
            }
        }
    }

    /// Efficiently update the accumulators for the current move. Depending on the nature of
    /// the move (standard, capture, castle), only the relevant parts of the accumulator are
    /// updated.
    pub fn update(&mut self, mv: &Move, pc: Piece, captured: Option<Piece>, board: &Board) {

        self.current += 1;
        // TODO: This can be optimized. No need to have an extra copy
        self.stack[self.current] = self.stack[self.current - 1];
        let us = board.stm;

        let new_pc = if let Some(promo_pc) = mv.promo_piece() { promo_pc } else { pc };

        if mv.is_castle() {
            self.handle_castle(mv, us);
        } else if let Some(captured) = captured {
            self.handle_capture(mv, pc, new_pc, captured, us);
        } else {
            self.handle_standard(mv, pc, new_pc, us);
        };

    }

    /// Update the accumulator for a standard move (no castle or capture). The old piece is removed
    /// from the starting square and the new piece (potentially a promo piece) is added to the
    /// destination square.
    fn handle_standard(&mut self, mv: &Move, pc: Piece, new_pc: Piece, side: Side) {

        let pc_index_w = ft_idx(mv.from(), pc, side, White);
        let pc_index_b = ft_idx(mv.from(), pc, side, Black);

        let new_pc_idx_w = ft_idx(mv.to(), new_pc, side, White);
        let new_pc_idx_b = ft_idx(mv.to(), new_pc, side, Black);

        let white_fts = [new_pc_idx_w, pc_index_w];
        let black_fts = [new_pc_idx_b, pc_index_b];

        self.stack[self.current].add_sub(white_fts, black_fts);

    }

    /// Update the accumulator for a capture move. The old piece is removed from the starting
    /// square, the new piece (potentially a promo piece) is added to the destination square, and
    /// the captured piece (potentially an en-passant pawn) is removed from the destination square.
    fn handle_capture(&mut self, mv: &Move, pc: Piece, new_pc: Piece, captured: Piece, side: Side) {

        let capture_sq = if mv.is_ep() { Square(mv.to().0 ^ 8) } else { mv.to() };

        let pc_index_w = ft_idx(mv.from(), pc, side, White);
        let pc_index_b = ft_idx(mv.from(), pc, side, Black);

        let new_pc_idx_w = ft_idx(mv.to(), new_pc, side, White);
        let new_pc_idx_b = ft_idx(mv.to(), new_pc, side, Black);

        let capture_idx_w = ft_idx(capture_sq, captured, side.flip(), White);
        let capture_idx_b = ft_idx(capture_sq, captured, side.flip(), Black);

        let white_fts = [new_pc_idx_w, pc_index_w, capture_idx_w];
        let black_fts = [new_pc_idx_b, pc_index_b, capture_idx_b];

        self.stack[self.current].add_sub_sub(white_fts, black_fts);

    }

    /// Update the accumulator for a castling move. The king and rook are moved to their new
    /// positions, and the old positions are cleared.
    fn handle_castle(&mut self, mv: &Move, us: Side) {

        let kingside = mv.to().0 > mv.from().0;
        let is_white = us == White;
        let king_from = mv.from();
        let king_to = mv.to();
        let rook_to = Move::rook_to(kingside, is_white);
        let rook_from = Move::rook_from(kingside, is_white);

        let king_from_w = ft_idx(king_from, Piece::King, us, White);
        let king_from_b = ft_idx(king_from, Piece::King, us, Black);
        let rook_from_w = ft_idx(rook_from, Piece::Rook, us, White);
        let rook_from_b = ft_idx(rook_from, Piece::Rook, us, Black);
        let king_to_w = ft_idx(king_to, Piece::King, us, White);
        let king_to_b = ft_idx(king_to, Piece::King, us, Black);
        let rook_to_w = ft_idx(rook_to, Piece::Rook, us, White);
        let rook_to_b = ft_idx(rook_to, Piece::Rook, us, Black);

        let white_fts = [king_to_w, rook_to_w, king_from_w, rook_from_w];
        let black_fts = [king_to_b, rook_to_b, king_from_b, rook_from_b];

        self.stack[self.current].add_add_sub_sub(white_fts, black_fts);

    }

    /// Undo the last move by decrementing the current accumulator index.
    pub fn undo(&mut self) {
        self.current = self.current.saturating_sub(1);
    }

}

fn ft_idx(sq: Square, pc: Piece, side: Side, perspective: Side) -> usize {
    let sq = if perspective == White {
        sq
    } else {
        sq.flip_rank()
    };
    let pc_offset = pc as usize * PIECE_OFFSET;
    let side_offset = if side == perspective { 0 } else { SIDE_OFFSET };
    side_offset + pc_offset + sq.0 as usize
}

fn king_square(board: &Board, mv: Move, pc: Piece, side: Side) {
    if side != board.stm || pc != Piece::King {
        board.king_square(side)
    } else {
        mv.to()
    }
}

fn bucket_changed(mv: Move, pc: Piece, side: Side) -> bool {
    if pc != King {
        return false;
    }
    let prev_king_sq = mv.from();
    let new_king_sq = mv.to();
    king_bucket(prev_king_sq , side) != king_bucket(new_king_sq, side)
}

fn mirror_changed(mv: Move, pc: Piece, side: Side) -> bool {
    if pc != King {
        return false;
    }
    let prev_king_sq = mv.from();
    let new_king_sq = mv.to();
    should_mirror(prev_king_sq) != should_mirror(new_king_sq)
}

fn king_bucket(sq: Square, side: Side) -> usize {
    let sq = if side == White { sq } else { sq.flip_rank() };
    BUCKETS[sq]
}

fn should_mirror(king_sq: Square) -> bool {
    File::of(king_sq) > File::D
}

fn crelu(x: i16) -> i32 {
    0.max(x).min(QA as i16) as i32
}

fn screlu(x: i16) -> i32 {
    0.max(x).min(QA as i16).pow(2) as i32
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            white_features: NETWORK.feature_bias,
            black_features: NETWORK.feature_bias,
        }
    }
}

impl Accumulator {

    pub fn add(&mut self, w: usize, b: usize) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w * HIDDEN];
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b * HIDDEN];
        }
    }

    pub fn sub(&mut self, w: usize, b: usize) {
        for i in 0..self.white_features.len() {
            self.white_features[i] -= NETWORK.feature_weights[i + w * HIDDEN];
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] -= NETWORK.feature_weights[i + b * HIDDEN];
        }
    }

    pub fn add_sub(&mut self, w: [usize; 2], b: [usize; 2]) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + w[1] * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + b[1] * HIDDEN]);
        }
    }

    pub fn add_sub_sub(&mut self, w: [usize; 3], b: [usize; 3]) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + w[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + w[2] * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + b[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + b[2] * HIDDEN]);
        }
    }

    pub fn add_add_sub_sub(&mut self, w: [usize; 4], b: [usize; 4]) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w[0] * HIDDEN]
                .wrapping_add(NETWORK.feature_weights[i + w[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + w[2] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + w[3] * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b[0] * HIDDEN]
                .wrapping_add(NETWORK.feature_weights[i + b[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + b[2] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + b[3] * HIDDEN]);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::fen;

    use super::NNUE;

    #[test]
    fn test_startpos() {
        let board = Board::from_fen(fen::STARTPOS);
        let mut eval = NNUE::default();
        let score = eval.evaluate(&board);
        assert_eq!(score, 26);
    }
}
