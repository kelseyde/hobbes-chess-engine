use crate::board::Board;
use crate::evaluation::accumulator::Accumulator;
use crate::evaluation::cache::InputBucketCache;
use crate::evaluation::feature::Feature;
use crate::moves::Move;
use crate::search::MAX_PLY;
use crate::types::piece::Piece;
use crate::types::piece::Piece::{Bishop, King, Knight, Pawn, Queen, Rook};
use crate::types::side::Side;
use crate::types::side::Side::{Black, White};
use crate::types::square::Square;
use crate::types::{castling, File};
use crate::utils::boxed_and_zeroed;
use arrayvec::ArrayVec;
use crate::parameters::{material_scaling_base, scale_value_bishop, scale_value_knight, scale_value_queen, scale_value_rook};

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 32;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub const PIECE_OFFSET: usize = 64;
pub const SIDE_OFFSET: usize = 64 * 6;
pub const MAX_ACCUMULATORS: usize = MAX_PLY + 8;
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

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../../hobbes.nnue")) };

#[repr(C, align(64))]
pub struct Network {
    pub feature_weights: FeatureWeights,
    pub feature_bias: [i16; HIDDEN],
    pub output_weights: [[i16; HIDDEN]; 2],
    pub output_bias: i16,
}

pub type FeatureWeights = [i16; FEATURES * HIDDEN];

pub struct NNUE {
    pub stack: Box<[Accumulator; MAX_ACCUMULATORS]>,
    pub cache: InputBucketCache,
    pub current: usize,
}

impl Default for NNUE {
    fn default() -> Self {
        let mut stack: Box<[Accumulator; MAX_ACCUMULATORS]> = unsafe { boxed_and_zeroed() };
        for i in 0..MAX_ACCUMULATORS {
            stack[i] = Accumulator::default();
        }
        NNUE {
            current: 0,
            cache: InputBucketCache::default(),
            stack,
        }
    }
}

impl NNUE {

    /// Evaluates the current position. Gets the 'us-perspective' and 'them-perspective' feature
    /// sets, based on the side to move. Then, passes the features through the network to get the
    /// static evaluation.
    pub fn evaluate(&mut self, board: &Board) -> i32 {

        let acc = &self.stack[self.current];
        let us = match board.stm { White => &acc.white_features, Black => &acc.black_features };
        let them = match board.stm { White => &acc.black_features, Black => &acc.white_features };

        let mut output = Self::forward(us, them);

        output /= QA;
        output += NETWORK.output_bias as i32;
        output *= SCALE;
        output /= QAB;
        output = scale_evaluation(board, output);
        output

    }

    /// Forward pass through the neural network. SIMD instructions are used if available to
    /// accelerate inference. Otherwise, a fall-back scalar implementation is used.
    pub(crate) fn forward(us: &[i16; HIDDEN], them: &[i16; HIDDEN]) -> i32 {
        #[cfg(target_feature = "avx2")]
        {
            use crate::evaluation::simd::avx2;
            let weights = &crate::evaluation::network::NETWORK.output_weights;
            unsafe { avx2::forward(us, &weights[0]) + avx2::forward(them, &weights[1]) }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            use crate::evaluation::simd::scalar;
            let weights = &NETWORK.output_weights;
            scalar::forward(us, &weights[0]) + scalar::forward(them, &weights[1])
        }
    }

    /// Activate the entire board from scratch. This initializes the accumulators based on the
    /// current board state, iterating over all pieces and their squares. Should be called only
    /// at the top of search, and then efficiently updated with each move.
    pub fn activate(&mut self, board: &Board) {
        self.current = 0;
        self.stack[self.current] = Accumulator::default();
        self.cache = InputBucketCache::default();

        // let w_mirror = should_mirror(board.king_sq(White));
        let w_mirror = false;
        // let b_mirror = should_mirror(board.king_sq(Black));
        let b_mirror = false;

        // let w_bucket = king_bucket(board.king_sq(White), White);
        let w_bucket = 0;
        // let b_bucket = king_bucket(board.king_sq(Black), Black);
        let b_bucket = 0;

        self.full_refresh(board, self.current, White, w_mirror, w_bucket);
        self.full_refresh(board, self.current, Black, b_mirror, b_bucket);
    }

    /// Refresh the accumulator for the given perspective, mirror state, and bucket. Retrieves
    /// the cached state for this accumulator, bucket, and perspective, and refreshes only the
    /// features of the board that have changed since the last refresh.
    pub fn full_refresh(&mut self,
                        board: &Board,
                        idx: usize,
                        perspective: Side,
                        mirror: bool,
                        bucket: usize) {

        let acc = &mut self.stack[idx];
        acc.mirrored[perspective] = mirror;
        let cache_entry = self.cache.get(perspective, mirror, bucket);
        acc.copy_from(perspective, &cache_entry.features);

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();

        for side in [White, Black] {
            for pc in [Pawn, Knight, Bishop, Rook, Queen, King] {

                let pieces = board.pieces(pc) & board.side(side);
                let cached_pieces = cache_entry.bitboards[pc] & cache_entry.bitboards[side.idx()];

                let added = pieces & !cached_pieces;
                for add in added {
                    adds.push(Feature::new(pc, add, side));
                }

                let removed = cached_pieces & !pieces;
                for sub in removed {
                    subs.push(Feature::new(pc, sub, side))
                }
            }
        }

        let weights = &NETWORK.feature_weights;

        for add in adds {
            acc.add(add, weights, perspective);
        }
        for sub in subs {
            acc.sub(sub, weights, perspective);
        }

        cache_entry.bitboards = board.bb;
        cache_entry.features = *acc.features(perspective);

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

        // let w_bucket = king_bucket(w_king_sq, White);
        let w_bucket = 0;
        // let b_bucket = king_bucket(b_king_sq, Black);
        let b_bucket = 0;

        let w_weights = &NETWORK.feature_weights;
        let b_weights = &NETWORK.feature_weights;

        // let mirror_changed = mirror_changed(board, *mv, new_pc);
        // let bucket_changed = bucket_changed(board, *mv, new_pc, us);
        // let refresh_required = mirror_changed || bucket_changed;
        //
        // if refresh_required {
        //     let bucket = if us == White { w_bucket } else { b_bucket };
        //     let mut mirror = should_mirror(board.king_sq(us));
        //     if mirror_changed {
        //         mirror = !mirror
        //     }
        //     self.full_refresh(board, self.current, us, mirror, bucket);
        // }

        if mv.is_castle() {
            self.handle_castle(board, mv, us, w_weights, b_weights);
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
        let capture_ft = Feature::new(captured, capture_sq, !side);

        self.stack[self.current].add_sub_sub(new_pc_ft, pc_ft, capture_ft, w_weights, b_weights);

    }

    /// Update the accumulator for a castling move. The king and rook are moved to their new
    /// positions, and the old positions are cleared.
    fn handle_castle(&mut self,
                     board: &Board,
                     mv: &Move,
                     us: Side,
                     w_weights: &FeatureWeights,
                     b_weights: &FeatureWeights) {

        let kingside = mv.to().0 > mv.from().0;
        let king_from = mv.from();
        let king_to = if board.is_frc() { castling::king_to(us, kingside) } else { mv.to() };
        let rook_from = if board.is_frc() { mv.to() } else { castling::rook_from(us, kingside) };
        let rook_to = castling::rook_to(us, kingside);

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

#[inline]
fn king_square(board: &Board, mv: Move, pc: Piece, side: Side) -> Square {

    if side != board.stm || pc != Piece::King {
        board.king_sq(side)
    } else {
        if mv.is_castle() && board.is_frc() {
            let kingside = castling::is_kingside(mv.from(), mv.to());
            castling::king_to(board.stm, kingside)
        } else {
            mv.to()
        }
    }
}

#[inline]
fn bucket_changed(board: &Board, mv: Move, pc: Piece, side: Side) -> bool {
    if pc != Piece::King {
        return false;
    }
    let prev_king_sq = mv.from();
    let mut new_king_sq = mv.to();
    if mv.is_castle() && board.is_frc() {
        let kingside = castling::is_kingside(mv.from(), mv.to());
        new_king_sq = castling::king_to(board.stm, kingside);
    }
    king_bucket(prev_king_sq , side) != king_bucket(new_king_sq, side)
}

#[inline]
fn mirror_changed(board: &Board, mv: Move, pc: Piece) -> bool {
    if pc != King {
        return false;
    }
    let prev_king_sq = mv.from();
    let mut new_king_sq = mv.to();
    if mv.is_castle() && board.is_frc() {
        let kingside = castling::is_kingside(mv.from(), mv.to());
        new_king_sq = castling::king_to(board.stm, kingside);
    }
    should_mirror(prev_king_sq) != should_mirror(new_king_sq)
}

#[inline(always)]
fn king_bucket(sq: Square, side: Side) -> usize {
    let sq = if side == White { sq } else { sq.flip_rank() };
    BUCKETS[sq]
}

#[inline(always)]
fn should_mirror(king_sq: Square) -> bool {
    File::of(king_sq) > File::D
}

fn scale_evaluation(board: &Board, eval: i32) -> i32 {
    let phase = material_phase(board);
    eval * (material_scaling_base() + phase) / 32768
}

fn material_phase(board: &Board) -> i32 {
    let knights = board.pieces(Knight).count();
    let bishops = board.pieces(Bishop).count();
    let rooks = board.pieces(Rook).count();
    let queens = board.pieces(Queen).count();

    scale_value_knight() * knights as i32
    + scale_value_bishop() * bishops as i32
    + scale_value_rook() * rooks as i32
    + scale_value_queen() * queens as i32
}

#[cfg(test)]
mod tests {
    use super::{Feature, NNUE};
    use crate::board::Board;
    use crate::fen;
    use crate::types::piece::Piece::Pawn;
    use crate::types::side::Side;
    use crate::types::square::Square;

    #[test]
    fn test_startpos() {
        let board = Board::from_fen(fen::STARTPOS).unwrap();
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
