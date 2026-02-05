pub mod accumulator;
pub mod cache;
pub mod feature;
pub mod arch;
pub mod stats;

mod forward {
    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
    mod vectorised;
    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
    pub use vectorised::*;

    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
    mod scalar;
    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
    pub use scalar::*;
}

mod simd {
    #[cfg(target_feature = "avx512f")]
    mod avx512;
    #[cfg(target_feature = "avx512f")]
    pub use avx512::*;

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    mod avx2;
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    pub use avx2::*;

    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]
    mod neon;
    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]
    pub use neon::*;

}

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
use crate::evaluation::arch::{NETWORK, OUTPUT_BUCKET_COUNT, Q, SCALE};
use crate::search::parameters::{
    material_scaling_base, scale_value_bishop, scale_value_knight, scale_value_queen,
    scale_value_rook,
};
use crate::search::MAX_PLY;
use crate::tools::utils::boxed_and_zeroed;
use arrayvec::ArrayVec;

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

    /// Forward pass through the neural network. We apply any pending accumulator updates and end up
    /// with the pre-activations of L0 stored in the current accumulator. We activate L0 and propagate
    /// through L1, L2, and L3 to get the final output.
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

        let output_bucket = get_output_bucket(board);
        let mut output: i64;
        unsafe  {
            let l0_outputs = forward::activate_l0(us, them);
            let l1_outputs = forward::propagate_l1(&l0_outputs, output_bucket);
            let l2_outputs = forward::propagate_l2(&l1_outputs, output_bucket);
            let l3_output = forward::propagate_l3(&l2_outputs, output_bucket);
            output = l3_output as i64;
        }
        output *= SCALE;
        output /= Q * Q * Q * Q;
        output = scale_evaluation(board, output as i32) as i64;
        output as i32

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
                    subs.push(Feature::new(pc, sub, side));
                }
            }
        }

        let weights = &NETWORK.l0_weights[bucket];

        // Fuse together updates to the accumulator for efficiency using iterators.
        for chunk in adds.as_slice().chunks_exact(4) {
            acc.add_add_add_add(chunk[0], chunk[1], chunk[2], chunk[3], weights, perspective);
        }
        for &add in adds.as_slice().chunks_exact(4).remainder() {
            acc.add(add, weights, perspective);
        }

        for chunk in subs.as_slice().chunks_exact(4) {
            acc.sub_sub_sub_sub(chunk[0], chunk[1], chunk[2], chunk[3], weights, perspective);
        }
        for &sub in subs.as_slice().chunks_exact(4).remainder() {
            acc.sub(sub, weights, perspective);
        }

        acc.computed[perspective] = true;
        acc.needs_refresh[perspective] = false;

        cache_entry.bitboards = board.bb;
        cache_entry.features = *acc.features(perspective);
    }

    /// Efficiently update the accumulators for the current move. Depending on the nature of
    /// the move (standard, capture, castle), only the relevant parts of the accumulator are
    /// updated. The update is then stored on the accumulator to later be applied lazily.
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

    /// Apply any pending lazy updates to the current accumulator. For each perspective, scan
    /// backwards to find the nearest computed accumulator, and move forward applying all updates
    /// one by one. If at any point we encounter an accumulator that requires a refresh - due to
    /// bucket or mirror change - we bail out and perform a full refresh instead.
    fn apply_lazy_updates(&mut self, board: &Board) {
        for side in [White, Black] {
            // If already up-to-date for this perspective, then there is nothing to do.
            if self.stack[self.current].computed[side] {
                continue;
            }

            let king_sq = board.king_sq(side);
            let mirror = should_mirror(king_sq);
            let bucket = king_bucket(king_sq, side);

            // If the current accumulator requires a full refresh, skip lazy updates and do a refresh.
            if self.stack[self.current].needs_refresh[side] {
                self.full_refresh(board, self.current, side, mirror, bucket);
                continue;
            }

            // Scan backwards to find the nearest parent accumulator that is computed for this
            // perspective, or requires a refresh.
            let mut curr = self.current - 1;
            while !self.stack[curr].computed[side]
                && !self.stack[curr].needs_refresh[side]
            {
                if curr == 0 {
                    break;
                }
                curr -= 1;
            }

            if self.stack[curr].needs_refresh[side] {
                // If we found an accumulator that requires a full refresh, do that instead.
                self.full_refresh(board, self.current, side, mirror, bucket);
            } else {
                // Otherwise, move forward through the stack applying all updates one by one.
                let weights = &NETWORK.l0_weights[bucket];
                while curr < self.current {
                    let (front, back) = self.stack.split_at_mut(curr + 1);
                    let prev_acc = front.last().unwrap();
                    let next_acc = back.first_mut().unwrap();
                    let update = next_acc.update;
                    let prev_fts = prev_acc.features(side);
                    let next_fts = next_acc.features_mut(side);
                    accumulator::apply_update(prev_fts, next_fts, weights, &update, side, mirror);
                    next_acc.computed[side] = true;
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
    arch::BUCKETS[sq]
}

fn get_output_bucket(board: &Board) -> usize {
    const DIVISOR: usize = usize::div_ceil(32, OUTPUT_BUCKET_COUNT);
    (board.occ().count() as usize - 2) / DIVISOR
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
