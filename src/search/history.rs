use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;
use crate::search::parameters::{
    capt_hist_bonus_max, capt_hist_bonus_offset, capt_hist_bonus_scale, capt_hist_malus_max,
    capt_hist_malus_offset, capt_hist_malus_scale, cont_hist_bonus_max, cont_hist_bonus_offset,
    cont_hist_bonus_scale, cont_hist_malus_max, cont_hist_malus_offset, cont_hist_malus_scale,
    lmr_cont_hist_bonus_max, lmr_cont_hist_bonus_offset, lmr_cont_hist_bonus_scale,
    lmr_cont_hist_malus_max, lmr_cont_hist_malus_offset, lmr_cont_hist_malus_scale, pcm_bonus_max,
    pcm_bonus_offset, pcm_bonus_scale, quiet_hist_bonus_max, quiet_hist_bonus_offset,
    quiet_hist_bonus_scale, quiet_hist_malus_max, quiet_hist_malus_offset, quiet_hist_malus_scale,
};
use crate::search::stack::SearchStack;
use crate::tools::utils::boxed_and_zeroed;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];

pub struct QuietHistory {
    entries: Box<[[[FromToHistory<i16>; 2]; 2]; 2]>,
}

pub struct CaptureHistory {
    entries: Box<[PieceToHistory<[i16; 6]>; 2]>,
}

pub struct ContinuationHistory {
    entries: Box<PieceToHistory<PieceToHistory<i16>>>,
}

#[derive(Default)]
pub struct Histories {
    pub quiet_history: QuietHistory,
    pub capture_history: CaptureHistory,
    pub cont_history: ContinuationHistory,
}

impl Histories {
    #[allow(clippy::too_many_arguments)]
    pub fn history_score(
        &self,
        board: &Board,
        ss: &SearchStack,
        mv: &Move,
        ply: usize,
        threats: Bitboard,
        pc: Piece,
        captured: Option<Piece>,
    ) -> i32 {
        if let Some(captured) = captured {
            self.capture_history_score(board, ss, mv, pc, captured, ply)
        } else {
            self.quiet_history_score(board, ss, mv, ply, threats)
        }
    }

    pub fn quiet_history_score(
        &self,
        board: &Board,
        ss: &SearchStack,
        mv: &Move,
        ply: usize,
        threats: Bitboard,
    ) -> i32 {
        let pc = board.piece_at(mv.from()).unwrap();
        let quiet_score = self.quiet_history.get(board.stm, *mv, threats) as i32;
        let mut cont_score = 0;
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply {
                if let (Some(prev_mv), Some(prev_pc)) =
                    (ss[ply - prev_ply].mv, ss[ply - prev_ply].pc)
                {
                    cont_score += self.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                }
            }
        }
        quiet_score + cont_score
    }

    pub fn capture_history_score(
        &self,
        board: &Board,
        ss: &SearchStack,
        mv: &Move,
        pc: Piece,
        captured: Piece,
        ply: usize,
    ) -> i32 {
        let capture_score = self.capture_history.get(board.stm, pc, mv.to(), captured) as i32;
        let mut cont_score = 0;
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply {
                if let (Some(prev_mv), Some(prev_pc)) =
                    (ss[ply - prev_ply].mv, ss[ply - prev_ply].pc)
                {
                    cont_score += self.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                }
            }
        }
        capture_score + cont_score
    }

    pub fn update_continuation_history(
        &mut self,
        ss: &SearchStack,
        ply: usize,
        mv: &Move,
        pc: Piece,
        bonus: i16,
    ) {
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply {
                if let (Some(prev_mv), Some(prev_pc)) =
                    (ss[ply - prev_ply].mv, ss[ply - prev_ply].pc)
                {
                    self.cont_history.update(&prev_mv, prev_pc, mv, pc, bonus);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.quiet_history.clear();
        self.capture_history.clear();
        self.cont_history.clear();
    }
}

impl Default for QuietHistory {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() },
        }
    }
}

impl Default for CaptureHistory {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() },
        }
    }
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() },
        }
    }
}

impl QuietHistory {
    const MAX: i16 = 16384;

    pub fn get(&self, stm: Side, mv: Move, threats: Bitboard) -> i16 {
        let threat_index = ThreatIndex::new(mv, threats);
        self.entries[stm][threat_index.from()][threat_index.to()][mv.from()][mv.to()]
    }

    pub fn update(&mut self, stm: Side, mv: &Move, threats: Bitboard, bonus: i16) {
        let threat_index = ThreatIndex::new(*mv, threats);
        let entry =
            &mut self.entries[stm][threat_index.from()][threat_index.to()][mv.from()][mv.to()];
        *entry = gravity(*entry as i32, bonus as i32, Self::MAX as i32) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[[0; 64]; 64]; 2]; 2]; 2]);
    }
}

impl CaptureHistory {
    const MAX: i16 = 16384;

    pub fn get(&self, stm: Side, pc: Piece, sq: Square, captured: Piece) -> i16 {
        self.entries[stm][pc][sq][captured]
    }

    pub fn update(&mut self, stm: Side, pc: Piece, sq: Square, captured: Piece, bonus: i16) {
        let entry = &mut self.entries[stm][pc][sq][captured];
        *entry = gravity(*entry as i32, bonus as i32, Self::MAX as i32) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[0; 6]; 64]; 6], [[[0; 6]; 64]; 6]]);
    }
}

impl ContinuationHistory {
    const MAX: i16 = 16384;

    pub fn get(&self, prev_mv: Move, prev_pc: Piece, mv: &Move, pc: Piece) -> i16 {
        self.entries[prev_pc][prev_mv.to()][pc][mv.to()]
    }

    pub fn update(&mut self, prev_mv: &Move, prev_pc: Piece, mv: &Move, pc: Piece, bonus: i16) {
        let entry = &mut self.entries[prev_pc][prev_mv.to()][pc][mv.to()];
        *entry = gravity(*entry as i32, bonus as i32, Self::MAX as i32) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[0; 64]; 6]; 64]; 6])
    }
}

pub struct ThreatIndex {
    pub from_attacked: bool,
    pub to_attacked: bool,
}

impl ThreatIndex {
    pub fn new(mv: Move, threats: Bitboard) -> Self {
        let from_attacked = threats.contains(mv.from());
        let to_attacked = threats.contains(mv.to());
        ThreatIndex {
            from_attacked,
            to_attacked,
        }
    }

    pub fn from(&self) -> usize {
        self.from_attacked as usize
    }

    pub fn to(&self) -> usize {
        self.to_attacked as usize
    }
}

pub fn quiet_history_bonus(depth: i32) -> i16 {
    let scale = quiet_hist_bonus_scale() as i16;
    let offset = quiet_hist_bonus_offset() as i16;
    let max = quiet_hist_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn quiet_history_malus(depth: i32) -> i16 {
    let scale = quiet_hist_malus_scale() as i16;
    let offset = quiet_hist_malus_offset() as i16;
    let max = quiet_hist_malus_max() as i16;
    history_malus(depth, scale, offset, max)
}

pub fn capture_history_bonus(depth: i32) -> i16 {
    let scale = capt_hist_bonus_scale() as i16;
    let offset = capt_hist_bonus_offset() as i16;
    let max = capt_hist_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn capture_history_malus(depth: i32) -> i16 {
    let scale = capt_hist_malus_scale() as i16;
    let offset = capt_hist_malus_offset() as i16;
    let max = capt_hist_malus_max() as i16;
    history_malus(depth, scale, offset, max)
}

pub fn cont_history_bonus(depth: i32) -> i16 {
    let scale = cont_hist_bonus_scale() as i16;
    let offset = cont_hist_bonus_offset() as i16;
    let max = cont_hist_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn cont_history_malus(depth: i32) -> i16 {
    let scale = cont_hist_malus_scale() as i16;
    let offset = cont_hist_malus_offset() as i16;
    let max = cont_hist_malus_max() as i16;
    history_malus(depth, scale, offset, max)
}

pub fn prior_countermove_bonus(depth: i32) -> i16 {
    let scale = pcm_bonus_scale() as i16;
    let offset = pcm_bonus_offset() as i16;
    let max = pcm_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn lmr_conthist_bonus(depth: i32, good: bool) -> i16 {
    if good {
        let scale = lmr_cont_hist_bonus_scale() as i16;
        let offset = lmr_cont_hist_bonus_offset() as i16;
        let max = lmr_cont_hist_bonus_max() as i16;
        history_bonus(depth, scale, offset, max)
    } else {
        let scale = lmr_cont_hist_malus_scale() as i16;
        let offset = lmr_cont_hist_malus_offset() as i16;
        let max = lmr_cont_hist_malus_max() as i16;
        history_malus(depth, scale, offset, max)
    }
}

fn history_bonus(depth: i32, scale: i16, offset: i16, max: i16) -> i16 {
    (scale * depth as i16 - offset).min(max)
}

fn history_malus(depth: i32, scale: i16, offset: i16, max: i16) -> i16 {
    -(scale * depth as i16 - offset).min(max)
}

fn gravity(current: i32, update: i32, max: i32) -> i32 {
    current + update - current * update.abs() / max
}
