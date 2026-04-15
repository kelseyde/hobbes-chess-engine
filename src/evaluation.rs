pub mod stats;
mod accumulator;
mod cache;
mod feature;
mod sparse;

mod forward {
    use hobbes_nnue_arch::{L1_SIZE, L2_SIZE, L3_SIZE};

    /// Trait grouping for four forward pass functions in Hobbes' multilayer NNUE inference. There
    /// is both a `Vectorised` and `Scalar` implementation of this trait; the appropriate one is
    /// selected at compile time.
    pub trait Forward {
        unsafe fn activate_l0(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE]) -> [u8; L1_SIZE];

        unsafe fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE * 2];

        unsafe fn propagate_l2(input: &[i32; L2_SIZE * 2], output_bucket: usize) -> [i32; L3_SIZE];

        unsafe fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32;
    }

    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
    mod vectorised;
    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]
    pub use vectorised::Vectorised as inference;

    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
    mod scalar;
    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
    pub use scalar::Scalar as inference;
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

    #[cfg(all(
        target_feature = "neon",
        not(any(target_feature = "avx2", target_feature = "avx512f"))
    ))]
    mod neon;
    #[cfg(all(
        target_feature = "neon",
        not(any(target_feature = "avx2", target_feature = "avx512f"))
    ))]
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
use crate::search::parameters::{
    material_scaling_base, scale_value_bishop, scale_value_knight, scale_value_pawn,
    scale_value_queen, scale_value_rook,
};
use crate::search::MAX_PLY;
use crate::tools::utils::boxed_and_zeroed;
use arrayvec::ArrayVec;
use hobbes_nnue_arch::{Network, BUCKETS, OUTPUT_BUCKET_COUNT, Q, SCALE};
use crate::evaluation::forward::{inference, Forward};

pub const MAX_ACCUMULATORS: usize = MAX_PLY + 8;

pub(crate) static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!(env!("NETWORK_PATH"))) };

pub struct NNUE {
    pub stack: Box<[Accumulator; MAX_ACCUMULATORS]>,
    pub cache: InputBucketCache,
    pub current: usize,
}

impl Default for NNUE {
    fn default() -> Self {
        NNUE {
            current: 0,
            cache: InputBucketCache::default(),
            stack: unsafe { boxed_and_zeroed() },
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
        let (us, them) = match board.stm {
            White => (&acc.white_features, &acc.black_features),
            Black => (&acc.black_features, &acc.white_features),
        };

        let output_bucket = get_output_bucket(board);
        let raw = unsafe {
            let l0_outputs = inference::activate_l0(us, them);
            let l1_outputs = inference::propagate_l1(&l0_outputs, output_bucket);
            let l2_outputs = inference::propagate_l2(&l1_outputs, output_bucket);
            inference::propagate_l3(&l2_outputs, output_bucket)
        };

        let output = raw as i64 * SCALE / (Q * Q * Q * Q);
        scale_evaluation(board, output as i32)
    }

    /// Activate the entire board from scratch. This initializes the accumulators based on the
    /// current board state, iterating over all pieces and their squares. Should be called only
    /// at the top of search, and then efficiently updated with each move.
    pub fn activate(&mut self, board: &Board) {
        self.current = 0;
        self.stack[self.current] = Accumulator::default();
        self.cache = InputBucketCache::default();

        for side in [White, Black] {
            let king_sq = board.king_sq(side);
            let mirror = should_mirror(king_sq);
            let bucket = king_bucket(king_sq, side);
            self.full_refresh(board, self.current, side, mirror, bucket);
        }
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
            Self::handle_castle(board, mv, us)
        } else if let Some(captured) = captured {
            Self::handle_capture(mv, pc, new_pc, captured, us)
        } else {
            Self::handle_standard(mv, pc, new_pc, us)
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
            while !self.stack[curr].computed[side] && !self.stack[curr].needs_refresh[side] {
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
    fn handle_standard(mv: &Move, pc: Piece, new_pc: Piece, side: Side) -> AccumulatorUpdate {
        let mut update = AccumulatorUpdate::default();
        update.push_sub(Feature::new(pc, mv.from(), side));
        update.push_add(Feature::new(new_pc, mv.to(), side));
        update
    }

    /// Update the accumulator for a capture move. The old piece is removed from the starting
    /// square, the new piece (potentially a promo piece) is added to the destination square, and
    /// the captured piece (potentially an en-passant pawn) is removed from the destination square.
    fn handle_capture(
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

        let mut update = AccumulatorUpdate::default();
        update.push_sub(Feature::new(pc, mv.from(), side));
        update.push_add(Feature::new(new_pc, mv.to(), side));
        update.push_sub(Feature::new(captured, capture_sq, !side));
        update
    }

    /// Update the accumulator for a castling move. The king and rook are moved to their new
    /// positions, and the old positions are cleared.
    fn handle_castle(board: &Board, mv: &Move, us: Side) -> AccumulatorUpdate {
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

        let mut update = AccumulatorUpdate::default();
        update.push_sub(Feature::new(King, king_from, us));
        update.push_add(Feature::new(King, king_to, us));
        update.push_sub(Feature::new(Rook, rook_from, us));
        update.push_add(Feature::new(Rook, rook_to, us));
        update
    }

    /// Undo the last move by decrementing the current accumulator index.
    pub fn undo(&mut self) {
        self.current = self.current.saturating_sub(1);
    }
}

/// Resolve the king's destination square, accounting for FRC castling where the move target
/// is the rook square rather than the king's final square.
#[inline]
fn king_dest_sq(board: &Board, mv: Move) -> Square {
    if mv.is_castle() && board.is_frc() {
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
    king_bucket(mv.from(), side) != king_bucket(king_dest_sq(board, mv), side)
}

#[inline]
fn mirror_changed(board: &Board, mv: Move, pc: Piece) -> bool {
    if pc != King {
        return false;
    }
    should_mirror(mv.from()) != should_mirror(king_dest_sq(board, mv))
}

#[inline(always)]
fn king_bucket(sq: Square, side: Side) -> usize {
    let sq = if side == White { sq } else { sq.flip_rank() };
    BUCKETS[sq]
}

fn get_output_bucket(board: &Board) -> usize {
    const DIVISOR: usize = usize::div_ceil(32, OUTPUT_BUCKET_COUNT);
    (board.occ().count() as usize - 2) / DIVISOR
}

#[inline(always)]
fn should_mirror(king_sq: Square) -> bool {
    File::of(king_sq) > File::D
}

#[inline]
fn scale_evaluation(board: &Board, eval: i32) -> i32 {
    let phase = material_phase(board);
    eval * (material_scaling_base() + phase) / 32768 * (200 - board.hm as i32) / 200
}

#[inline]
fn material_phase(board: &Board) -> i32 {
    let pawns = board.pieces(Pawn).count();
    let knights = board.pieces(Knight).count();
    let bishops = board.pieces(Bishop).count();
    let rooks = board.pieces(Rook).count();
    let queens = board.pieces(Queen).count();

    scale_value_pawn() * pawns as i32
        + scale_value_knight() * knights as i32
        + scale_value_bishop() * bishops as i32
        + scale_value_rook() * rooks as i32
        + scale_value_queen() * queens as i32
}
