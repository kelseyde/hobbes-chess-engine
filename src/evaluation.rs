pub mod accumulator;
pub mod cache;
pub mod feature;
pub mod network;
pub mod simd;

use crate::board::file::File;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::piece::Piece::{Bishop, King, Knight, Pawn, Queen, Rook};
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::board::{castling, Board};
use crate::evaluation::accumulator::{Accumulator, AccumulatorUpdate};
use crate::evaluation::cache::InputBucketCache;
use crate::evaluation::feature::Feature;
use crate::evaluation::network::{HIDDEN, NETWORK, QA, QAB, SCALE};
use crate::search::parameters::{
    material_scaling_base, scale_value_bishop, scale_value_knight, scale_value_queen,
    scale_value_rook,
};
use crate::search::MAX_PLY;
use crate::tools::utils::boxed_and_zeroed;
use arrayvec::ArrayVec;

#[rustfmt::skip]
pub const BUCKETS: [usize; 64] = [
         0,  1,  2,  3, 3, 2,  1,  0,
         4,  5,  6,  7, 7, 6,  5,  4,
         8,  9, 10, 11, 11, 10, 9,  8,
         8,  9, 10, 11, 11, 10, 9,  8,
        12, 12, 13, 13, 13, 13, 12, 12,
        12, 12, 13, 13, 13, 13, 12, 12,
        14, 14, 15, 15, 15, 15, 14, 14,
        14, 14, 15, 15, 15, 15, 14, 14,
];
pub const NUM_BUCKETS: usize = get_num_buckets(&BUCKETS);
pub const MAX_ACCUMULATORS: usize = MAX_PLY + 8;

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
        self.apply_lazy_updates(board);
        let acc = &self.stack[self.current];
        let us = match board.stm {
            White => &acc.white_features,
            Black => &acc.black_features,
        };
        let them = match board.stm {
            White => &acc.black_features,
            Black => &acc.white_features,
        };

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
        #[cfg(target_feature = "avx512f")]
        {
            use crate::evaluation::network::NETWORK;
            use crate::evaluation::simd::avx512;
            let weights = &NETWORK.output_weights;
            unsafe { avx512::forward(us, &weights[0]) + avx512::forward(them, &weights[1]) }
        }
        #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
        {
            use crate::evaluation::network::NETWORK;
            use crate::evaluation::simd::avx2;
            let weights = &NETWORK.output_weights;
            unsafe { avx2::forward(us, &weights[0]) + avx2::forward(them, &weights[1]) }
        }
        #[cfg(all(not(target_feature = "avx2"), not(target_feature = "avx512f")))]
        {
            use crate::evaluation::network::NETWORK;
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

        let w_mirror = should_mirror(board.king_sq(White));
        let b_mirror = should_mirror(board.king_sq(Black));

        let w_bucket = king_bucket(board.king_sq(White), White);
        let b_bucket = king_bucket(board.king_sq(Black), Black);

        self.full_refresh(board, self.current, White, w_mirror, w_bucket);
        self.full_refresh(board, self.current, Black, b_mirror, b_bucket);
    }

    /// Refresh the accumulator for the given perspective, mirror state, and bucket. Retrieves
    /// the cached state for this accumulator, bucket, and perspective, and refreshes only the
    /// features of the board that have changed since the last refresh.
    fn full_refresh(
        &mut self,
        board: &Board,
        idx: usize,
        perspective: Side,
        mirror: bool,
        bucket: usize,
    ) {
        let acc = &mut self.stack[idx];
        acc.mirrored[perspective] = mirror;
        let acc_features = acc.features_mut(perspective);

        let cache_entry = self.cache.get(perspective, mirror, bucket);
        let cached_features = &cache_entry.features;

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
                    subs.push(Feature::new(pc, sub, side));
                }
            }
        }

        let weights = &NETWORK.feature_weights[bucket];

        // Fuse together updates to the accumulator for efficiency using iterators.
        for chunk in adds.as_slice().chunks_exact(4) {
            accumulator::add_add_add_add(cached_features, acc_features, chunk[0], chunk[1], chunk[2], chunk[3], weights, perspective, mirror);
        }
        for &add in adds.as_slice().chunks_exact(4).remainder() {
            accumulator::add(cached_features, acc_features, add, weights, perspective, mirror);
        }

        for chunk in subs.as_slice().chunks_exact(4) {
            accumulator::sub_sub_sub_sub(cached_features, acc_features, chunk[0], chunk[1], chunk[2], chunk[3], weights, perspective, mirror);
        }
        for &sub in subs.as_slice().chunks_exact(4).remainder() {
            accumulator::sub(cached_features, acc_features, sub, weights, perspective, mirror);
        }

        acc.computed[perspective] = true;
        acc.needs_refresh[perspective] = false;

        cache_entry.bitboards = board.bb;
        cache_entry.features = *acc.features(perspective);
    }

    /// Efficiently update the accumulators for the current move. Depending on the nature of
    /// the move (standard, capture, castle), only the relevant parts of the accumulator are
    /// updated.
    pub fn update(&mut self, mv: &Move, pc: Piece, captured: Option<Piece>, board: &Board) {
        self.current += 1;
        self.stack[self.current].needs_refresh = self.stack[self.current - 1].needs_refresh;
        self.stack[self.current].mirrored = self.stack[self.current - 1].mirrored;
        self.stack[self.current].computed[White] = false;
        self.stack[self.current].computed[Black] = false;
        let us = board.stm;

        let new_pc = mv.promo_piece().unwrap_or(pc);
        let mirror_changed = mirror_changed(board, *mv, new_pc);
        let bucket_changed = bucket_changed(board, *mv, new_pc, us);
        let refresh_required = mirror_changed || bucket_changed;

        if refresh_required {
            if mirror_changed {
                self.stack[self.current].mirrored[us] = !self.stack[self.current - 1].mirrored[us];
            }
            self.stack[self.current].needs_refresh[us] = true;
        }

        self.stack[self.current].update = if mv.is_castle() {
            self.handle_castle(board, mv, us)
        } else if let Some(captured) = captured {
            self.handle_capture(mv, pc, new_pc, captured, us)
        } else {
            self.handle_standard(mv, pc, new_pc, us)
        };
    }

    fn apply_lazy_updates(&mut self, board: &Board) {
        for perspective in [White, Black] {
            if self.stack[self.current].computed[perspective] {
                continue; // already up-to-date for this perspective
            }

            let king_sq = board.king_sq(perspective);
            let mirror = should_mirror(king_sq);
            let bucket = king_bucket(king_sq, perspective);

            if self.stack[self.current].needs_refresh[perspective] {
                self.full_refresh(board, self.current, perspective, mirror, bucket);
                continue;
            }

            // Scan backwards to find the nearest parent accumulator that is computed
            // for this perspective.
            let mut curr = self.current - 1;
            while !self.stack[curr].computed[perspective]
                && !self.stack[curr].needs_refresh[perspective]
            {
                if curr == 0 {
                    break;
                }
                curr -= 1;
            }

            if self.stack[curr].needs_refresh[perspective] {
                self.full_refresh(board, self.current, perspective, mirror, bucket);
            } else {
                let weights = &NETWORK.feature_weights[bucket];
                // Apply all updates from that accumulator up to the current one
                while curr < self.current {
                    let (front, back) = self.stack.split_at_mut(curr + 1);
                    let prev_acc = front.last().unwrap();
                    let next_acc = back.first_mut().unwrap();
                    let update = next_acc.update;
                    let prev_features = prev_acc.features(perspective);
                    let next_features = next_acc.features_mut(perspective);
                    accumulator::apply_update(prev_features, next_features, weights, &update, perspective, mirror);
                    next_acc.computed[perspective] = true;
                    curr += 1;
                }
            }
        }
    }

    /// Update the accumulator for a standard move (no castle or capture). The old piece is removed
    /// from the starting square and the new piece (potentially a promo piece) is added to the
    /// destination square.
    fn handle_standard(
        &mut self,
        mv: &Move,
        pc: Piece,
        new_pc: Piece,
        side: Side,
    ) -> AccumulatorUpdate {
        let pc_ft = Feature::new(pc, mv.from(), side);
        let new_pc_ft = Feature::new(new_pc, mv.to(), side);

        let mut update = AccumulatorUpdate::default();
        update.push_sub(pc_ft);
        update.push_add(new_pc_ft);
        update
    }

    /// Update the accumulator for a capture move. The old piece is removed from the starting
    /// square, the new piece (potentially a promo piece) is added to the destination square, and
    /// the captured piece (potentially an en-passant pawn) is removed from the destination square.
    #[allow(clippy::too_many_arguments)]
    fn handle_capture(
        &mut self,
        mv: &Move,
        pc: Piece,
        new_pc: Piece,
        captured: Piece,
        side: Side,
    ) -> AccumulatorUpdate {
        let capture_sq = if mv.is_ep() {
            Square(mv.to().0 ^ 8)
        } else {
            mv.to()
        };

        let pc_ft = Feature::new(pc, mv.from(), side);
        let new_pc_ft = Feature::new(new_pc, mv.to(), side);
        let capture_ft = Feature::new(captured, capture_sq, !side);

        let mut update = AccumulatorUpdate::default();
        update.push_sub(pc_ft);
        update.push_add(new_pc_ft);
        update.push_sub(capture_ft);
        update
    }

    /// Update the accumulator for a castling move. The king and rook are moved to their new
    /// positions, and the old positions are cleared.
    fn handle_castle(&mut self, board: &Board, mv: &Move, us: Side) -> AccumulatorUpdate {
        let kingside = mv.to().0 > mv.from().0;
        let king_from = mv.from();
        let king_to = if board.is_frc() {
            castling::king_to(us, kingside)
        } else {
            mv.to()
        };
        let rook_from = if board.is_frc() {
            mv.to()
        } else {
            castling::rook_from(us, kingside)
        };
        let rook_to = castling::rook_to(us, kingside);

        let king_from_ft = Feature::new(King, king_from, us);
        let king_to_ft = Feature::new(King, king_to, us);
        let rook_from_ft = Feature::new(Rook, rook_from, us);
        let rook_to_ft = Feature::new(Rook, rook_to, us);

        let mut update = AccumulatorUpdate::default();
        update.push_sub(king_from_ft);
        update.push_add(king_to_ft);
        update.push_sub(rook_from_ft);
        update.push_add(rook_to_ft);
        update
    }

    /// Undo the last move by decrementing the current accumulator index.
    pub fn undo(&mut self) {
        self.current = self.current.saturating_sub(1);
    }
}

#[inline]
fn king_square(board: &Board, mv: Move, pc: Piece, side: Side) -> Square {
    if side != board.stm || pc != King {
        board.king_sq(side)
    } else if mv.is_castle() && board.is_frc() {
        let kingside = castling::is_kingside(mv.from(), mv.to());
        castling::king_to(board.stm, kingside)
    } else {
        mv.to()
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
    king_bucket(prev_king_sq, side) != king_bucket(new_king_sq, side)
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
    eval * (material_scaling_base() + phase) / 32768 * (200 - board.hm as i32) / 200
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

pub const fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;

    while i < N {
        if arr[i] > max {
            max = arr[i];
        }
        i += 1;
    }
    max + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fen;

    #[test]
    fn test_lazy_updates() {
        let mut nnue1 = NNUE::default();
        let mut nnue2 = NNUE::default();

        let mut board = Board::from_fen(fen::STARTPOS).unwrap();
        nnue1.activate(&board);
        nnue2.activate(&board);

        let eval1 = nnue1.evaluate(&board);
        let eval2 = nnue2.evaluate(&board);
        assert_eq!(eval1, eval2);

        let mv = Move::parse_uci("e2e4");
        let pc = Pawn;
        nnue1.update(&mv, pc, None, &board);
        board.make(&mv);

        nnue2.activate(&board);

        let eval1 = nnue1.evaluate(&board);
        let eval2 = nnue2.evaluate(&board);
        assert_eq!(eval1, eval2);
    }
}
