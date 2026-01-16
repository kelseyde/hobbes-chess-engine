use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;
use crate::search::parameters::*;
use crate::search::node::NodeStack;
use crate::tools::utils::boxed_and_zeroed;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];
type ThreatBucket<T> = [[T; 2]; 2];

pub struct QuietHistory {
    from_to_entries: Box<[FromToHistory<QuietHistoryEntry>; 2]>,
    piece_to_entries: Box<[PieceToHistory<QuietHistoryEntry>; 2]>,
}

pub struct CaptureHistory {
    entries: Box<[PieceToHistory<[i16; 6]>; 2]>,
}

pub struct ContinuationHistory {
    entries: Box<PieceToHistory<PieceToHistory<i16>>>,
}

pub struct SquareHistory {
    pub entries: Box<[[i16; 64]; 2]>,
}

#[derive(Default, Copy, Clone)]
struct QuietHistoryEntry {
    factoriser: i16,
    bucket: ThreatBucket<i16>,
}

#[derive(Default)]
pub struct Histories {
    pub quiet_history: QuietHistory,
    pub capture_history: CaptureHistory,
    pub cont_history: ContinuationHistory,
    pub from_history: SquareHistory,
    pub to_history: SquareHistory,
}

impl Histories {
    #[allow(clippy::too_many_arguments)]
    pub fn history_score(
        &self,
        board: &Board,
        ss: &NodeStack,
        mv: &Move,
        ply: usize,
        threats: Bitboard,
        pc: Piece,
        captured: Option<Piece>,
    ) -> i32 {
        if let Some(captured) = captured {
            self.capture_history_score(board, mv, pc, captured)
        } else {
            let quiet_score = self.quiet_history_score(board, mv, pc, threats);
            let cont_score = self.cont_history_score(board, ss, mv, ply);
            let from_score = self.from_history.get(board.stm, mv.from()) as i32;
            let to_score = self.to_history.get(board.stm, mv.to()) as i32;
            quiet_score + cont_score + from_score + to_score
        }
    }

    pub fn quiet_history_score(
        &self,
        board: &Board,
        mv: &Move,
        pc: Piece,
        threats: Bitboard,
    ) -> i32 {
        self.quiet_history.get(board.stm, *mv, pc, threats) as i32
    }

    pub fn cont_history_score(&self, board: &Board, ss: &NodeStack, mv: &Move, ply: usize) -> i32 {
        let pc = board.piece_at(mv.from()).unwrap();
        let mut cont_score = 0;
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply {
                let prev_mv = ss[ply - prev_ply].mv;
                let prev_pc = ss[ply - prev_ply].pc;
                if let (Some(prev_mv), Some(prev_pc)) = (prev_mv, prev_pc) {
                    cont_score += self.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                }
            }
        }
        cont_score
    }

    pub fn capture_history_score(
        &self,
        board: &Board,
        mv: &Move,
        pc: Piece,
        captured: Piece,
    ) -> i32 {
        self.capture_history.get(board.stm, pc, mv.to(), captured) as i32
    }

    pub fn update_continuation_history(
        &mut self,
        ss: &NodeStack,
        ply: usize,
        mv: &Move,
        pc: Piece,
        bonus: i16,
    ) {
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply {
                let prev_mv = ss[ply - prev_ply].mv;
                let prev_pc = ss[ply - prev_ply].pc;
                if let (Some(prev_mv), Some(prev_pc)) = (prev_mv, prev_pc) {
                    self.cont_history.update(&prev_mv, prev_pc, mv, pc, bonus);
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.quiet_history.clear();
        self.capture_history.clear();
        self.cont_history.clear();
        self.from_history.clear();
        self.to_history.clear();
    }
}

impl Default for QuietHistory {
    fn default() -> Self {
        Self {
            from_to_entries: unsafe { boxed_and_zeroed() },
            piece_to_entries: unsafe { boxed_and_zeroed() },
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

impl Default for SquareHistory {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() },
        }
    }
}

impl QuietHistoryEntry {
    #[inline]
    fn score(&self, threat_index: &ThreatIndex) -> i16 {
        self.factoriser + self.bucket[threat_index.from()][threat_index.to()]
    }

    #[inline]
    fn update(&mut self, threat_index: &ThreatIndex, bonus: i16) {
        self.factoriser = gravity(
            self.factoriser as i32,
            bonus as i32,
            QuietHistory::FACTORISER_MAX,
        ) as i16;

        let bucket_entry = &mut self.bucket[threat_index.from()][threat_index.to()];
        *bucket_entry = gravity(*bucket_entry as i32, bonus as i32, QuietHistory::BUCKET_MAX) as i16;
    }
}

impl QuietHistory {
    const FACTORISER_MAX: i32 = 8192;
    const BUCKET_MAX: i32 = 16384;
    const BONUS_MAX: i16 = Self::BUCKET_MAX as i16 / 4;

    pub fn get(&self, stm: Side, mv: Move, pc: Piece, threats: Bitboard) -> i16 {
        let threat_index = ThreatIndex::new(mv, threats);

        let from_to_score =
            self.from_to_entries[stm][mv.from()][mv.to()].score(&threat_index) as i32;
        let piece_to_score = self.piece_to_entries[stm][pc][mv.to()].score(&threat_index) as i32;

        let lerp_factor = quiet_hist_lerp_factor();
        ((from_to_score * (100 - lerp_factor) + piece_to_score * lerp_factor) / 100) as i16
    }

    pub fn update(
        &mut self,
        stm: Side,
        mv: &Move,
        pc: Piece,
        threats: Bitboard,
        ft_bonus: i16,
        pt_bonus: i16) {
        let ft_bonus = ft_bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        let pt_bonus = pt_bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        let threat_index = ThreatIndex::new(*mv, threats);

        self.from_to_entries[stm][mv.from()][mv.to()].update(&threat_index, ft_bonus);
        self.piece_to_entries[stm][pc][mv.to()].update(&threat_index, pt_bonus);
    }

    pub fn clear(&mut self) {
        self.from_to_entries = Box::new([[[QuietHistoryEntry::default(); 64]; 64]; 2]);
        self.piece_to_entries = Box::new([[[QuietHistoryEntry::default(); 64]; 6]; 2]);
    }
}

impl CaptureHistory {
    const MAX: i32 = 16384;
    const BONUS_MAX: i16 = Self::MAX as i16 / 4;

    pub fn get(&self, stm: Side, pc: Piece, sq: Square, captured: Piece) -> i16 {
        self.entries[stm][pc][sq][captured]
    }

    pub fn update(&mut self, stm: Side, pc: Piece, sq: Square, captured: Piece, bonus: i16) {
        let entry = &mut self.entries[stm][pc][sq][captured];
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        *entry = gravity(*entry as i32, bonus as i32, Self::MAX) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[0; 6]; 64]; 6], [[[0; 6]; 64]; 6]]);
    }
}

impl ContinuationHistory {
    const MAX: i32 = 16384;
    const BONUS_MAX: i16 = Self::MAX as i16 / 4;

    pub fn get(&self, prev_mv: Move, prev_pc: Piece, mv: &Move, pc: Piece) -> i16 {
        self.entries[prev_pc][prev_mv.to()][pc][mv.to()]
    }

    pub fn update(&mut self, prev_mv: &Move, prev_pc: Piece, mv: &Move, pc: Piece, bonus: i16) {
        let entry = &mut self.entries[prev_pc][prev_mv.to()][pc][mv.to()];
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        *entry = gravity(*entry as i32, bonus as i32, Self::MAX) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[0; 64]; 6]; 64]; 6])
    }
}

impl SquareHistory {
    const MAX: i32 = 4096;
    const BONUS_MAX: i16 = Self::MAX as i16 / 4;

    pub fn get(&self, stm: Side, sq: Square) -> i16 {
        self.entries[stm][sq]
    }

    pub fn update(&mut self, stm: Side, sq: Square, bonus: i16) {
        let entry = &mut self.entries[stm][sq];
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        *entry = gravity(*entry as i32, bonus as i32, Self::MAX) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[0; 64]; 2]);
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

/// Bonus for the from/to-indexed quiet history table.
pub fn quiet_history_ft_bonus(depth: i32) -> i16 {
    let scale = quiet_hist_ft_bonus_scale() as i16;
    let offset = quiet_hist_ft_bonus_offset() as i16;
    let max = quiet_hist_ft_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

/// Malus for the from/to-indexed quiet history table.
pub fn quiet_history_ft_malus(depth: i32) -> i16 {
    let scale = quiet_hist_ft_malus_scale() as i16;
    let offset = quiet_hist_ft_malus_offset() as i16;
    let max = quiet_hist_ft_malus_max() as i16;
    history_malus(depth, scale, offset, max)
}

/// Bonus for the piece/to-indexed quiet history table.
pub fn quiet_history_pt_bonus(depth: i32) -> i16 {
    let scale = quiet_hist_pt_bonus_scale() as i16;
    let offset = quiet_hist_pt_bonus_offset() as i16;
    let max = quiet_hist_pt_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

/// Malus for the piece/to-indexed quiet history table.
pub fn quiet_history_pt_malus(depth: i32) -> i16 {
    let scale = quiet_hist_pt_malus_scale() as i16;
    let offset = quiet_hist_pt_malus_offset() as i16;
    let max = quiet_hist_pt_malus_max() as i16;
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

pub fn prior_countermove_ft_bonus(depth: i32) -> i16 {
    let scale = pcm_ft_bonus_scale() as i16;
    let offset = pcm_ft_bonus_offset() as i16;
    let max = pcm_ft_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn prior_countermove_pt_bonus(depth: i32) -> i16 {
    let scale = pcm_pt_bonus_scale() as i16;
    let offset = pcm_pt_bonus_offset() as i16;
    let max = pcm_pt_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn from_history_bonus(depth: i32) -> i16 {
    let scale = from_hist_bonus_scale() as i16;
    let offset = from_hist_bonus_offset() as i16;
    let max = from_hist_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn from_history_malus(depth: i32) -> i16 {
    let scale = from_hist_malus_scale() as i16;
    let offset = from_hist_malus_offset() as i16;
    let max = from_hist_malus_max() as i16;
    history_malus(depth, scale, offset, max)
}

pub fn to_history_bonus(depth: i32) -> i16 {
    let scale = to_hist_bonus_scale() as i16;
    let offset = to_hist_bonus_offset() as i16;
    let max = to_hist_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn to_history_malus(depth: i32) -> i16 {
    let scale = to_hist_malus_scale() as i16;
    let offset = to_hist_malus_offset() as i16;
    let max = to_hist_malus_max() as i16;
    history_malus(depth, scale, offset, max)
}

pub fn qs_capthist_bonus(depth: i32) -> i16 {
    let scale = qs_capt_hist_bonus_scale() as i16;
    let offset = qs_capt_hist_bonus_offset() as i16;
    let max = qs_capt_hist_bonus_max() as i16;
    history_bonus(depth, scale, offset, max)
}

pub fn qs_capthist_malus(depth: i32) -> i16 {
    let scale = qs_capt_hist_malus_scale() as i16;
    let offset = qs_capt_hist_malus_offset() as i16;
    let max = qs_capt_hist_malus_max() as i16;
    history_malus(depth, scale, offset, max)
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

pub fn dynamic_policy_ft_bonus(static_eval: i32, prev_eval: i32) -> i16 {
    let mult = dynamic_policy_ft_mult();
    let min = dynamic_policy_ft_min();
    let max = dynamic_policy_ft_max();
    dynamic_policy_bonus(static_eval, prev_eval, mult, min, max)
}

pub fn dynamic_policy_pt_bonus(static_eval: i32, prev_eval: i32) -> i16 {
    let mult = dynamic_policy_pt_mult();
    let min = dynamic_policy_pt_min();
    let max = dynamic_policy_pt_max();
    dynamic_policy_bonus(static_eval, prev_eval, mult, min, max)
}

fn dynamic_policy_bonus(static_eval: i32, prev_eval: i32, mult: i32, min: i32, max: i32) -> i16 {
    let value = mult * -(static_eval + prev_eval);
    value.clamp(min, max) as i16
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
