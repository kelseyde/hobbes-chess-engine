use crate::types::side::Side::{Black, White};

use crate::board::Board;
use crate::moves::Move;
use crate::types::File;
use crate::types::piece::{Piece, PIECES};
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
pub const NUM_BUCKETS: usize = 8;

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
    unsafe { std::mem::transmute(*include_bytes!("../resources/calvin1024_8b.nnue")) };

#[repr(C, align(64))]
pub struct Network {
    feature_weights: [FeatureWeights; NUM_BUCKETS],
    feature_bias: [i16; HIDDEN],
    output_weights: [i16; 2 * HIDDEN],
    output_bias: i16,
}

pub type FeatureWeights = [i16; FEATURES * HIDDEN];

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: [i16; HIDDEN],
    pub black_features: [i16; HIDDEN],
    pub mirrored: [bool; 2]
}

pub struct Feature {
    pc: Piece,
    sq: Square,
    side: Side
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

        let mut output = 0;

        for (&input, &weight) in us.iter().zip(NETWORK.output_weights[..HIDDEN].iter()) {
            let clipped = input.clamp(0, QA as i16);
            let result = clipped * weight;
            output += result as i32 * clipped as i32;
        }

        for (&input, &weight) in them.iter().zip(NETWORK.output_weights[HIDDEN..].iter()) {
            let clipped = input.clamp(0, QA as i16);
            let result = clipped * weight;
            output += result as i32 * clipped as i32;
        }

        output /= QA;
        output += NETWORK.output_bias as i32;
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

        let w_mirror = should_mirror(board.king_sq(White));
        let b_mirror = should_mirror(board.king_sq(Black));

        let w_bucket = king_bucket(board.king_sq(White), White);
        let b_bucket = king_bucket(board.king_sq(Black), Black);

        self.full_refresh(board, self.current, White, w_mirror, w_bucket);
        self.full_refresh(board, self.current, Black, b_mirror, b_bucket);
    }

    pub fn full_refresh(&mut self,
                        board: &Board,
                        idx: usize,
                        perspective: Side,
                        mirror: bool,
                        bucket: usize) {

        let acc = &mut self.stack[idx];
        acc.reset(perspective);
        acc.mirrored[perspective] = mirror;
        let weights = &NETWORK.feature_weights[bucket];
        for &pc in PIECES.iter() {
            for sq in board.pcs(pc) {
                let side = board.side_at(sq).unwrap();
                let ft = Feature::new(pc, sq, side);
                acc.add(ft, weights, perspective);
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

        let w_king_sq = king_square(board, *mv, new_pc, White);
        let b_king_sq = king_square(board, *mv, new_pc, Black);

        let w_bucket = king_bucket(w_king_sq, White);
        let b_bucket = king_bucket(b_king_sq, Black);

        let w_weights = &NETWORK.feature_weights[w_bucket];
        let b_weights = &NETWORK.feature_weights[b_bucket];

        let mirror_changed = mirror_changed(*mv, new_pc);
        let bucket_changed = bucket_changed(*mv, new_pc, us);
        let refresh_required = mirror_changed || bucket_changed;

        if refresh_required {
            let bucket = if us == White { w_bucket } else { b_bucket };
            let mut mirror = should_mirror(board.king_sq(us));
            if mirror_changed {
                mirror = !mirror
            }
            self.full_refresh(board, self.current, us, mirror, bucket);
        }

        if mv.is_castle() {
            self.handle_castle(mv, us, w_weights, b_weights);
        } else if let Some(captured) = captured {
            self.handle_capture(mv, pc, new_pc, captured, us, w_weights, b_weights);
        } else {
            self.handle_standard(mv, pc, new_pc, us, w_weights, b_weights);
        };

    }

    /// Update the accumulator for a standard move (no castle or capture). The old piece is removed
    /// from the starting square and the new piece (potentially a promo piece) is added to the
    /// destination square.
    fn handle_standard(&mut self,
                       mv: &Move,
                       pc: Piece,
                       new_pc: Piece,
                       side: Side,
                       w_weights: &FeatureWeights,
                       b_weights: &FeatureWeights) {

        let pc_ft = Feature::new(pc, mv.from(), side);
        let new_pc_ft = Feature::new(new_pc, mv.to(), side);

        self.stack[self.current].add_sub(new_pc_ft, pc_ft, w_weights, b_weights);

    }

    /// Update the accumulator for a capture move. The old piece is removed from the starting
    /// square, the new piece (potentially a promo piece) is added to the destination square, and
    /// the captured piece (potentially an en-passant pawn) is removed from the destination square.
    fn handle_capture(&mut self,
                      mv: &Move,
                      pc: Piece,
                      new_pc: Piece,
                      captured: Piece,
                      side: Side,
                      w_weights: &FeatureWeights,
                      b_weights: &FeatureWeights) {

        let capture_sq = if mv.is_ep() { Square(mv.to().0 ^ 8) } else { mv.to() };

        let pc_ft = Feature::new(pc, mv.from(), side);
        let new_pc_ft = Feature::new(new_pc, mv.to(), side);
        let capture_ft = Feature::new(captured, capture_sq, side.flip());

        self.stack[self.current].add_sub_sub(new_pc_ft, pc_ft, capture_ft, w_weights, b_weights);

    }

    /// Update the accumulator for a castling move. The king and rook are moved to their new
    /// positions, and the old positions are cleared.
    fn handle_castle(&mut self,
                     mv: &Move,
                     us: Side,
                     w_weights: &FeatureWeights,
                     b_weights: &FeatureWeights) {

        let kingside = mv.to().0 > mv.from().0;
        let is_white = us == White;
        let king_from = mv.from();
        let king_to = mv.to();
        let rook_to = Move::rook_to(kingside, is_white);
        let rook_from = Move::rook_from(kingside, is_white);

        let king_from_ft = Feature::new(Piece::King, king_from, us);
        let king_to_ft = Feature::new(Piece::King, king_to, us);
        let rook_from_ft = Feature::new(Piece::Rook, rook_from, us);
        let rook_to_ft = Feature::new(Piece::Rook, rook_to, us);

        self.stack[self.current]
            .add_add_sub_sub(king_to_ft, rook_to_ft, king_from_ft, rook_from_ft, w_weights, b_weights);

    }

    /// Undo the last move by decrementing the current accumulator index.
    pub fn undo(&mut self) {
        self.current = self.current.saturating_sub(1);
    }

}

impl Feature {

    pub fn new(pc: Piece, sq: Square, side: Side) -> Self {
        Feature { pc, sq, side }
    }

    pub fn index(&self, perspective: Side, mirror: bool) -> usize {
        let sq_index = self.square_index(perspective, mirror);
        let pc_offset = self.pc as usize * PIECE_OFFSET;
        let side_offset = if self.side == perspective { 0 } else { SIDE_OFFSET };
        side_offset + pc_offset + sq_index
    }

    fn square_index(&self, perspective: Side, mirror: bool) -> usize {
        let mut sq_index = self.sq;
        if perspective != White {
            sq_index = sq_index.flip_rank();
        }
        if mirror {
            sq_index = sq_index.flip_file();
        }
        sq_index.0 as usize
    }

}

fn king_square(board: &Board, mv: Move, pc: Piece, side: Side) -> Square {
    if side != board.stm || pc != Piece::King {
        board.king_sq(side)
    } else {
        mv.to()
    }
}

fn bucket_changed(mv: Move, pc: Piece, side: Side) -> bool {
    if pc != Piece::King {
        return false;
    }
    let prev_king_sq = mv.from();
    let new_king_sq = mv.to();
    king_bucket(prev_king_sq , side) != king_bucket(new_king_sq, side)
}

fn mirror_changed(mv: Move, pc: Piece) -> bool {
    if pc != Piece::King {
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

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            white_features: NETWORK.feature_bias,
            black_features: NETWORK.feature_bias,
            mirrored: [false, false],
        }
    }
}

impl Accumulator {

    pub fn reset(&mut self, perspective: Side) {
        let feats = if perspective == White { &mut self.white_features } else { &mut self.black_features };
        *feats = NETWORK.feature_bias;
    }

    pub fn add(&mut self,
               add: Feature,
               weights: &FeatureWeights,
               perspective: Side) {

        let mirror = self.mirrored[perspective as usize];
        let idx = add.index(perspective, mirror);
        let feats = if perspective == White { &mut self.white_features } else { &mut self.black_features };

        for i in 0..feats.len() {
            feats[i] += weights[i + idx * HIDDEN];
        }
    }

    pub fn sub(&mut self,
               add: Feature,
               weights: &FeatureWeights,
               perspective: Side) {

        let mirror = self.mirrored[perspective as usize];
        let idx = add.index(White, mirror);
        let feats = if perspective == White { &mut self.white_features } else { &mut self.black_features };

        for i in 0..feats.len() {
            feats[i] -= weights[i + idx * HIDDEN];
        }
    }

    pub fn add_sub(&mut self,
                   add: Feature,
                   sub: Feature,
                   w_weights: &FeatureWeights,
                   b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[White];
        let b_mirror = self.mirrored[Black];

        let w_idx_1 = add.index(White, w_mirror);
        let b_idx_1 = add.index(Black, b_mirror);

        let w_idx_2 = sub.index(White, w_mirror);
        let b_idx_2 = sub.index(Black, b_mirror);

        for i in 0..self.white_features.len() {
            self.white_features[i] += w_weights[i + w_idx_1 * HIDDEN]
                .wrapping_sub(w_weights[i + w_idx_2 * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += b_weights[i + b_idx_1 * HIDDEN]
                .wrapping_sub(b_weights[i + b_idx_2 * HIDDEN]);
        }
    }

    pub fn add_sub_sub(&mut self,
                       add: Feature,
                       sub1: Feature,
                       sub2: Feature,
                       w_weights: &FeatureWeights,
                       b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[White];
        let b_mirror = self.mirrored[Black];

        let w_idx_1 = add.index(White, w_mirror);
        let b_idx_1 = add.index(Black, b_mirror);

        let w_idx_2 = sub1.index(White, w_mirror);
        let b_idx_2 = sub1.index(Black, b_mirror);

        let w_idx_3 = sub2.index(White, w_mirror);
        let b_idx_3 = sub2.index(Black, b_mirror);

        for i in 0..self.white_features.len() {
            self.white_features[i] += w_weights[i + w_idx_1 * HIDDEN]
                .wrapping_sub(w_weights[i + w_idx_2 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_3 * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += b_weights[i + b_idx_1 * HIDDEN]
                .wrapping_sub(b_weights[i + b_idx_2 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_3 * HIDDEN]);
        }
    }

    pub fn add_add_sub_sub(&mut self,
                           add1: Feature,
                           add2: Feature,
                           sub1: Feature,
                           sub2: Feature,
                           w_weights: &FeatureWeights,
                           b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[White];
        let b_mirror = self.mirrored[Black];

        let w_idx_1 = add1.index(White, w_mirror);
        let b_idx_1 = add1.index(Black, b_mirror);

        let w_idx_2 = add2.index(White, w_mirror);
        let b_idx_2 = add2.index(Black, b_mirror);

        let w_idx_3 = sub1.index(White, w_mirror);
        let b_idx_3 = sub1.index(Black, b_mirror);

        let w_idx_4 = sub2.index(White, w_mirror);
        let b_idx_4 = sub2.index(Black, b_mirror);

        for i in 0..self.white_features.len() {
            self.white_features[i] += w_weights[i + w_idx_1 * HIDDEN]
                .wrapping_add(w_weights[i + w_idx_2 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_3 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_4 * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += b_weights[i + b_idx_1 * HIDDEN]
                .wrapping_add(b_weights[i + b_idx_2 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_3 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_4 * HIDDEN]);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::fen;
    use crate::moves::Move;
    use crate::types::piece::Piece::Pawn;
    use crate::types::side::Side;
    use crate::types::square::Square;
    use super::{Feature, NNUE};

    #[test]
    fn test_startpos() {
        let board = Board::from_fen(fen::STARTPOS);
        let mut eval = NNUE::default();
        let score = eval.evaluate(&board);
        assert_eq!(score, 26);
    }

    #[test]
    fn make_move_standard() {

        let feat = Feature::new(Pawn, Square(8), Side::White);
        println!("index: {}", feat.index(Side::Black, false));

    }

}
