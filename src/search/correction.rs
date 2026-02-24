use crate::board::side::Side;
use crate::board::Board;
use crate::search::node::NodeStack;
use crate::search::parameters::*;
use crate::tools::utils::boxed_and_zeroed;
use std::marker::PhantomData;
use Side::{Black, White};

const CORRECTION_SCALE: i32 = 280;

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
    #[rustfmt::skip]
    pub fn update_correction_history(
        &mut self,
        board: &Board,
        ss: &NodeStack,
        depth: i32,
        ply: usize,
        static_eval: i32,
        best_score: i32,
    ) {
        let us = board.stm;
        let pawn_hash = board.keys.pawn_hash;
        let w_nonpawn_hash = board.keys.non_pawn_hashes[White];
        let b_nonpawn_hash = board.keys.non_pawn_hashes[Black];
        let major_hash = board.keys.major_hash;
        let minor_hash = board.keys.minor_hash;

        let diff = best_score - static_eval;
        let pawn_bonus = pawn_corr_bonus(diff, depth);
        let nonpawn_bonus = nonpawn_corr_bonus(diff, depth);
        let major_bonus = major_corr_bonus(diff, depth);
        let minor_bonus = minor_corr_bonus(diff, depth);
        let counter_bonus = counter_corr_bonus(diff, depth);
        let follow_up_bonus = follow_up_corr_bonus(diff, depth);

        self.pawn_corrhist.update(us, pawn_hash, pawn_bonus);
        self.nonpawn_corrhist[White].update(us, w_nonpawn_hash, nonpawn_bonus);
        self.nonpawn_corrhist[Black].update( us, b_nonpawn_hash, nonpawn_bonus);
        self.major_corrhist.update(us, major_hash, major_bonus);
        self.minor_corrhist.update(us, minor_hash, minor_bonus);
        self.update_countermove_correction(board, ss, ply, counter_bonus);
        self.update_follow_up_move_correction(board, ss, ply, follow_up_bonus);
    }

    #[rustfmt::skip]
    pub fn correction(&self, board: &Board, ss: &NodeStack, ply: usize) -> i32 {

        let us = board.stm;
        let pawn_hash = board.keys.pawn_hash;
        let w_nonpawn_hash = board.keys.non_pawn_hashes[White];
        let b_nonpawn_hash = board.keys.non_pawn_hashes[Black];
        let major_hash = board.keys.major_hash;
        let minor_hash = board.keys.minor_hash;

        let pawn       = self.pawn_corrhist.get(us, pawn_hash);
        let white      = self.nonpawn_corrhist[White].get(us, w_nonpawn_hash);
        let black      = self.nonpawn_corrhist[Black].get(us, b_nonpawn_hash);
        let major      = self.major_corrhist.get(us, major_hash);
        let minor      = self.minor_corrhist.get(us, minor_hash);
        let counter    = self.countermove_correction(board, ss, ply);
        let follow_up  = self.follow_up_move_correction(board, ss, ply);

        ((pawn * 100 / corr_pawn_weight())
            + (white * 100 / corr_non_pawn_weight())
            + (black * 100 / corr_non_pawn_weight())
            + (major * 100 / corr_major_weight())
            + (minor * 100 / corr_minor_weight())
            + (counter * 100 / corr_counter_weight())
            + (follow_up * 100 / corr_follow_up_weight()))
            / CORRECTION_SCALE

    }

    fn update_countermove_correction(
        &mut self,
        board: &Board,
        ss: &NodeStack,
        ply: usize,
        bonus: i32,
    ) {
        if ply >= 1 {
            if let Some(prev_mv) = ss[ply - 1].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                self.countermove_corrhist
                    .update(board.stm, encoded_mv, bonus);
            }
        }
    }

    fn update_follow_up_move_correction(
        &mut self,
        board: &Board,
        ss: &NodeStack,
        ply: usize,
        bonus: i32,
    ) {
        if ply >= 2 {
            if let Some(prev_mv) = ss[ply - 2].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                self.follow_up_move_corrhist
                    .update(board.stm, encoded_mv, bonus);
            }
        }
    }

    fn countermove_correction(&self, board: &Board, ss: &NodeStack, ply: usize) -> i32 {
        if ply >= 1 {
            if let Some(prev_mv) = ss[ply - 1].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                return self.countermove_corrhist.get(board.stm, encoded_mv);
            }
        }
        0
    }

    fn follow_up_move_correction(&self, board: &Board, ss: &NodeStack, ply: usize) -> i32 {
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
    const MAX_HISTORY: i32 = 14734;
    const SIZE: usize = 16384;
    const MASK: usize = Self::SIZE - 1;

    pub fn get(&self, stm: Side, key: u64) -> i32 {
        let idx = self.index(key);
        self.entries[stm][idx]
    }

    pub fn update(&mut self, stm: Side, key: u64, bonus: i32) {
        let idx = self.index(key);
        let entry = &mut self.entries[stm][idx];
        *entry += bonus - bonus.abs() * (*entry) / Self::MAX_HISTORY;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[0; N]; 2]);
    }

    fn index(&self, key: u64) -> usize {
        key as usize & Self::MASK
    }
}

fn pawn_corr_bonus(diff: i32, depth: i32) -> i32 {
    corr_bonus(
        diff,
        depth,
        corr_pawn_bonus_mult(),
        corr_pawn_bonus_div(),
        corr_pawn_bonus_min(),
        corr_pawn_bonus_max(),
    )
}

fn nonpawn_corr_bonus(diff: i32, depth: i32) -> i32 {
    corr_bonus(
        diff,
        depth,
        corr_nonpawn_bonus_mult(),
        corr_nonpawn_bonus_div(),
        corr_nonpawn_bonus_min(),
        corr_nonpawn_bonus_max(),
    )
}

fn major_corr_bonus(diff: i32, depth: i32) -> i32 {
    corr_bonus(
        diff,
        depth,
        corr_major_bonus_mult(),
        corr_major_bonus_div(),
        corr_major_bonus_min(),
        corr_major_bonus_max(),
    )
}

fn minor_corr_bonus(diff: i32, depth: i32) -> i32 {
    corr_bonus(
        diff,
        depth,
        corr_minor_bonus_mult(),
        corr_minor_bonus_div(),
        corr_minor_bonus_min(),
        corr_minor_bonus_max(),
    )
}

fn counter_corr_bonus(diff: i32, depth: i32) -> i32 {
    corr_bonus(
        diff,
        depth,
        corr_counter_bonus_mult(),
        corr_counter_bonus_div(),
        corr_counter_bonus_min(),
        corr_counter_bonus_max(),
    )
}

fn follow_up_corr_bonus(diff: i32, depth: i32) -> i32 {
    corr_bonus(
        diff,
        depth,
        corr_follow_up_bonus_mult(),
        corr_follow_up_bonus_div(),
        corr_follow_up_bonus_min(),
        corr_follow_up_bonus_max(),
    )
}

fn corr_bonus(diff: i32, depth: i32, mult: i32, div: i32, min: i32, max: i32) -> i32 {
    (mult * depth * diff / div).clamp(min, max)
}
