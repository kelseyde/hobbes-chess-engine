use crate::board::side::Side;
use crate::board::Board;
use crate::search::parameters::*;
use crate::search::stack::SearchStack;
use crate::tools::utils::boxed_and_zeroed;
use std::marker::PhantomData;

const CORRECTION_SCALE: i32 = 256;

/// Correction history tracks how much the static evaluation of a position matched the actual search
/// score. We can use this information to 'correct' the current static eval based on the diff between
/// the static eval and the search score of previously searched positions.
pub struct CorrectionHistory<const N: usize> {
    entries: Box<[[i32; N]; 2]>,
    _marker: PhantomData<[i32; N]>,
}

pub type HashCorrectionHistory = CorrectionHistory<16384>;
pub type FromToCorrectionHistory = CorrectionHistory<4096>;

#[derive(Default)]
pub struct CorrectionHistories {
    pawn_corrhist: HashCorrectionHistory,
    nonpawn_corrhist: [HashCorrectionHistory; 2],
    countermove_corrhist: FromToCorrectionHistory,
    follow_up_move_corrhist: FromToCorrectionHistory,
    major_corrhist: HashCorrectionHistory,
    minor_corrhist: HashCorrectionHistory,
}

impl CorrectionHistories {
    pub fn update_correction_history(
        &mut self,
        board: &Board,
        ss: &SearchStack,
        depth: i32,
        ply: usize,
        static_eval: i32,
        best_score: i32,
    ) {
        let us = board.stm;
        let pawn_hash = board.keys.pawn_hash;
        let w_nonpawn_hash = board.keys.non_pawn_hashes[Side::White];
        let b_nonpawn_hash = board.keys.non_pawn_hashes[Side::Black];
        let major_hash = board.keys.major_hash;
        let minor_hash = board.keys.minor_hash;

        self.pawn_corrhist
            .update(us, pawn_hash, depth, static_eval, best_score);
        self.nonpawn_corrhist[Side::White].update(
            us,
            w_nonpawn_hash,
            depth,
            static_eval,
            best_score,
        );
        self.nonpawn_corrhist[Side::Black].update(
            us,
            b_nonpawn_hash,
            depth,
            static_eval,
            best_score,
        );
        self.major_corrhist
            .update(us, major_hash, depth, static_eval, best_score);
        self.minor_corrhist
            .update(us, minor_hash, depth, static_eval, best_score);
        self.update_countermove_correction(board, ss, ply, depth, static_eval, best_score);
        self.update_follow_up_move_correction(board, ss, ply, depth, static_eval, best_score);
    }

    #[rustfmt::skip]
    pub fn correction(&self, board: &Board, ss: &SearchStack, ply: usize) -> i32 {

        let us = board.stm;
        let phase = board.phase;
        let pawn_hash = board.keys.pawn_hash;
        let w_nonpawn_hash = board.keys.non_pawn_hashes[Side::White];
        let b_nonpawn_hash = board.keys.non_pawn_hashes[Side::Black];
        let major_hash = board.keys.major_hash;
        let minor_hash = board.keys.minor_hash;

        let pawn       = self.pawn_corrhist.get(us, pawn_hash);
        let white      = self.nonpawn_corrhist[Side::White].get(us, w_nonpawn_hash);
        let black      = self.nonpawn_corrhist[Side::Black].get(us, b_nonpawn_hash);
        let major      = self.major_corrhist.get(us, major_hash);
        let minor      = self.minor_corrhist.get(us, minor_hash);
        let counter    = self.countermove_correction(board, ss, ply);
        let follow_up  = self.follow_up_move_correction(board, ss, ply);

        ((pawn * 100 / corr_pawn_weight(phase))
            + (white * 100 / corr_non_pawn_weight(phase))
            + (black * 100 / corr_non_pawn_weight(phase))
            + (major * 100 / corr_major_weight(phase))
            + (minor * 100 / corr_minor_weight(phase))
            + (counter * 100 / corr_counter_weight(phase))
            + (follow_up * 100 / corr_follow_up_weight(phase)))
            / CORRECTION_SCALE

    }

    fn update_countermove_correction(
        &mut self,
        board: &Board,
        ss: &SearchStack,
        ply: usize,
        depth: i32,
        static_eval: i32,
        best_score: i32,
    ) {
        if ply >= 1 {
            if let Some(prev_mv) = ss[ply - 1].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                self.countermove_corrhist.update(
                    board.stm,
                    encoded_mv,
                    depth,
                    static_eval,
                    best_score,
                );
            }
        }
    }

    fn update_follow_up_move_correction(
        &mut self,
        board: &Board,
        ss: &SearchStack,
        ply: usize,
        depth: i32,
        static_eval: i32,
        best_score: i32,
    ) {
        if ply >= 2 {
            if let Some(prev_mv) = ss[ply - 2].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                self.follow_up_move_corrhist.update(
                    board.stm,
                    encoded_mv,
                    depth,
                    static_eval,
                    best_score,
                );
            }
        }
    }

    fn countermove_correction(&self, board: &Board, ss: &SearchStack, ply: usize) -> i32 {
        if ply >= 1 {
            if let Some(prev_mv) = ss[ply - 1].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                return self.countermove_corrhist.get(board.stm, encoded_mv);
            }
        }
        0
    }

    fn follow_up_move_correction(&self, board: &Board, ss: &SearchStack, ply: usize) -> i32 {
        if ply >= 2 {
            if let Some(prev_mv) = ss[ply - 2].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                return self.follow_up_move_corrhist.get(board.stm, encoded_mv);
            }
        }
        0
    }

    pub fn clear(&mut self) {
        self.pawn_corrhist.clear();
        self.nonpawn_corrhist
            .iter_mut()
            .for_each(|hist| hist.clear());
        self.countermove_corrhist.clear();
        self.follow_up_move_corrhist.clear();
        self.major_corrhist.clear();
        self.minor_corrhist.clear();
    }
}

impl<const N: usize> Default for CorrectionHistory<N> {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() },
            _marker: PhantomData,
        }
    }
}

impl<const N: usize> CorrectionHistory<N> {
    const MASK: usize = N - 1;
    const GRAIN: i32 = 256;
    const MAX: i32 = Self::GRAIN * 32;

    pub fn get(&self, stm: Side, key: u64) -> i32 {
        let idx = self.index(key);
        self.entries[stm][idx]
    }

    pub fn update(&mut self, stm: Side, key: u64, depth: i32, static_eval: i32, score: i32) {
        let idx = self.index(key);
        let entry = &mut self.entries[stm][idx];
        let new_value = (score - static_eval) * CORRECTION_SCALE;

        let new_weight = (depth + 1).min(16);
        let old_weight = CORRECTION_SCALE - new_weight;

        let update = *entry * old_weight + new_value * new_weight;
        *entry = i32::clamp(update / CORRECTION_SCALE, -Self::MAX, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[0; N]; 2]);
    }

    fn index(&self, key: u64) -> usize {
        key as usize & Self::MASK
    }
}
