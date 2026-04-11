use crate::board::side::Side;
use crate::board::Board;
use crate::search::node::NodeStack;
use crate::search::parameters::*;
use crate::tools::utils::boxed_and_zeroed;
use Side::{Black, White};

const CORRECTION_SCALE: i32 = 280;

/// Correction history tracks how much the static evaluation of a position matched the actual search
/// score. We can use this information to 'correct' the current static eval based on the diff between
/// the static eval and the search score of previously searched positions.
pub struct CorrectionHistory<const N: usize> {
    entries: Box<[[i32; N]; 2]>,
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
        let diff = best_score - static_eval;

        let pawn_key = board.keys.pawn_hash;
        let white_key = board.keys.non_pawn_hashes[White];
        let black_key = board.keys.non_pawn_hashes[Black];
        let major_key = board.keys.major_hash;
        let minor_key = board.keys.minor_hash;

        self.pawn_corrhist.update(us, pawn_key, pawn_corr_bonus(diff, depth));
        self.nonpawn_corrhist[White].update(us, white_key, nonpawn_corr_bonus(diff, depth));
        self.nonpawn_corrhist[Black].update(us, black_key, nonpawn_corr_bonus(diff, depth));
        self.major_corrhist.update(us, major_key, major_corr_bonus(diff, depth));
        self.minor_corrhist.update(us, minor_key, minor_corr_bonus(diff, depth));

        if let Some(key) = prev_move_key(ss, ply, 1) {
            self.countermove_corrhist.update(us, key, counter_corr_bonus(diff, depth));
        }
        if let Some(key) = prev_move_key(ss, ply, 2) {
            self.follow_up_move_corrhist.update(us, key, follow_up_corr_bonus(diff, depth));
        }
    }

    #[rustfmt::skip]
    pub fn correction(&self, board: &Board, ss: &NodeStack, ply: usize) -> i32 {
        let us = board.stm;

        let pawn      = self.pawn_corrhist.get(us, board.keys.pawn_hash);
        let white     = self.nonpawn_corrhist[White].get(us, board.keys.non_pawn_hashes[White]);
        let black     = self.nonpawn_corrhist[Black].get(us, board.keys.non_pawn_hashes[Black]);
        let major     = self.major_corrhist.get(us, board.keys.major_hash);
        let minor     = self.minor_corrhist.get(us, board.keys.minor_hash);
        let counter   = prev_move_key(ss, ply, 1).map_or(0, |k| self.countermove_corrhist.get(us, k));
        let follow_up = prev_move_key(ss, ply, 2).map_or(0, |k| self.follow_up_move_corrhist.get(us, k));

        ((pawn * 100 / corr_pawn_weight())
            + (white * 100 / corr_non_pawn_weight())
            + (black * 100 / corr_non_pawn_weight())
            + (major * 100 / corr_major_weight())
            + (minor * 100 / corr_minor_weight())
            + (counter * 100 / corr_counter_weight())
            + (follow_up * 100 / corr_follow_up_weight()))
            / CORRECTION_SCALE
    }

    pub fn clear(&mut self) {
        self.pawn_corrhist.clear();
        self.nonpawn_corrhist.iter_mut().for_each(|h| h.clear());
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
        }
    }
}

impl<const N: usize> CorrectionHistory<N> {
    const MAX_HISTORY: i32 = 14734;
    const MASK: usize = N - 1;

    /// Returns the correction value for the given key and side.
    pub fn get(&self, stm: Side, key: u64) -> i32 {
        let index = Self::index(key);
        self.entries[stm][index]
    }

    /// Updates the correction value for the given key and side.
    pub fn update(&mut self, stm: Side, key: u64, bonus: i32) {
        let index = Self::index(key);
        let entry = &mut self.entries[stm][index];
        *entry += bonus - bonus.abs() * (*entry) / Self::MAX_HISTORY;
    }

    /// Clears the correction history tables.
    pub fn clear(&mut self) {
        self.entries[0].fill(0);
        self.entries[1].fill(0);
    }

    #[inline(always)]
    const fn index(key: u64) -> usize {
        key as usize & Self::MASK
    }
}

/// Returns the encoded key of the move played `offset` plies ago, if any.
fn prev_move_key(ss: &NodeStack, ply: usize, offset: usize) -> Option<u64> {
    if ply >= offset {
        ss[ply - offset].mv.map(|mv| mv.encoded() as u64)
    } else {
        None
    }
}

#[rustfmt::skip]
mod bonuses {
    use super::*;

    macro_rules! corr_bonus {
        ($name:ident, $mult:ident, $div:ident, $min:ident, $max:ident) => {
            pub fn $name(diff: i32, depth: i32) -> i32 {
                ($mult() * depth * diff / $div()).clamp($min(), $max())
            }
        };
    }

    corr_bonus!(pawn_corr_bonus,      corr_pawn_bonus_mult,      corr_pawn_bonus_div,      corr_pawn_bonus_min,      corr_pawn_bonus_max);
    corr_bonus!(nonpawn_corr_bonus,   corr_nonpawn_bonus_mult,   corr_nonpawn_bonus_div,   corr_nonpawn_bonus_min,   corr_nonpawn_bonus_max);
    corr_bonus!(major_corr_bonus,     corr_major_bonus_mult,     corr_major_bonus_div,     corr_major_bonus_min,     corr_major_bonus_max);
    corr_bonus!(minor_corr_bonus,     corr_minor_bonus_mult,     corr_minor_bonus_div,     corr_minor_bonus_min,     corr_minor_bonus_max);
    corr_bonus!(counter_corr_bonus,   corr_counter_bonus_mult,   corr_counter_bonus_div,   corr_counter_bonus_min,   corr_counter_bonus_max);
    corr_bonus!(follow_up_corr_bonus, corr_follow_up_bonus_mult, corr_follow_up_bonus_div, corr_follow_up_bonus_min, corr_follow_up_bonus_max);
}
use bonuses::*;
