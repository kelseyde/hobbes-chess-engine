mod accumulator;
mod cache;
pub mod feature;
mod forward;
pub mod sparse;
pub mod stats;

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
use crate::evaluation::accumulator::Accumulator;
use crate::evaluation::cache::InputBucketCache;
use crate::evaluation::forward::{inference, Forward};
use crate::search::parameters::{
    material_scaling_base, scale_value_bishop, scale_value_knight, scale_value_pawn,
    scale_value_queen, scale_value_rook,
};
use crate::search::MAX_PLY;
use crate::tools::utils::boxed_and_zeroed;
use accumulator::{psq, threat};
use hobbes_nnue_arch::{
    Network, BUCKETS, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT, Q, SCALE,
};

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
        // Apply any pending updates to the PSQ and threat accumulators.
        psq::apply_lazy_updates(self, board);
        threat::apply_lazy_updates(self, board);

        // Arrange the features of both accumulators in (stm, nstm) order.
        let acc = &self.stack[self.current];
        let (psq_us, psq_them) = acc.psq.features_relative(board.stm);
        let (threat_us, threat_them) = acc.threat.features_relative(board.stm);

        let mut l0_outputs = [0u8; L1_SIZE];
        let mut l1_outputs = [0i32; L2_SIZE * 2];
        let mut l2_outputs = [0i32; L3_SIZE];
        let output_bucket = get_output_bucket(board);

        // Pass the features through the network to retrieve the raw eval.
        let raw = unsafe {
            inference::activate_l0(psq_us, threat_us, psq_them, threat_them, &mut l0_outputs);
            #[cfg(feature = "track_l0_activations")]
            sparse::track_activations(&l0_outputs);
            inference::propagate_l1(&l0_outputs, output_bucket, &mut l1_outputs);
            inference::propagate_l2(&l1_outputs, output_bucket, &mut l2_outputs);
            inference::propagate_l3(&l2_outputs, output_bucket)
        };

        // Scale the eval and return
        let output = raw as i64 * SCALE / (Q * Q * Q * Q);
        scale_evaluation(board, output as i32)
    }

    /// Activate the entire board from scratch. This initializes the accumulators based on the
    /// current board state, iterating over all pieces and their squares. Should be called only
    /// at the top of search, and then efficiently updated with each move.
    pub fn activate(&mut self, board: &Board) {
        self.current = 0;
        self.cache = InputBucketCache::default();

        let mut acc = Accumulator::default();
        for side in [White, Black] {
            acc.psq.refresh(board, side, &mut self.cache);
            acc.threat.refresh(board, side);
        }
        self.stack[self.current] = acc;
    }

    /// Efficiently update the accumulators for the current move. Depending on the nature of
    /// the move (standard, capture, castle), only the relevant parts of the accumulator are
    /// updated. The update is then stored on the accumulator to later be applied lazily.
    pub fn update(&mut self, mv: &Move, pc: Piece, board: &Board) {
        let us = board.stm;
        self.current += 1;

        let (prev_psq_refresh, prev_threat_refresh) = (
            self.stack[self.current - 1].psq.needs_refresh,
            self.stack[self.current - 1].threat.needs_refresh,
        );
        let acc = &mut self.stack[self.current];
        acc.psq.adds.clear();
        acc.psq.subs.clear();
        acc.psq.computed = [false; 2];
        acc.threat.deltas.clear();
        acc.threat.computed = [false; 2];

        acc.psq.needs_refresh = prev_psq_refresh;
        acc.psq.needs_refresh[us] |=
            mirror_changed(board, *mv, pc) || bucket_changed(board, *mv, pc, us);
        acc.threat.needs_refresh = prev_threat_refresh;
        acc.threat.needs_refresh[us] |= mirror_changed(board, *mv, pc);
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
