use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;
use crate::search::node::NodeStack;
use crate::search::parameters::{capt_hist_bonus_max, capt_hist_bonus_offset, capt_hist_bonus_scale, capt_hist_lerp_factor, capt_hist_malus_max, capt_hist_malus_offset, capt_hist_malus_scale, cont_hist_1_bonus_max, cont_hist_1_bonus_offset, cont_hist_1_bonus_scale, cont_hist_1_malus_max, cont_hist_1_malus_offset, cont_hist_1_malus_scale, cont_hist_2_bonus_max, cont_hist_2_bonus_offset, cont_hist_2_bonus_scale, cont_hist_2_malus_max, cont_hist_2_malus_offset, cont_hist_2_malus_scale, from_hist_bonus_max, from_hist_bonus_offset, from_hist_bonus_scale, from_hist_malus_max, from_hist_malus_offset, from_hist_malus_scale, lmr_cont_1_bonus_max, lmr_cont_1_bonus_offset, lmr_cont_1_bonus_scale, lmr_cont_1_malus_max, lmr_cont_1_malus_offset, lmr_cont_1_malus_scale, lmr_cont_2_bonus_max, lmr_cont_2_bonus_offset, lmr_cont_2_bonus_scale, lmr_cont_2_malus_max, lmr_cont_2_malus_offset, lmr_cont_2_malus_scale, pcm_bonus_max, pcm_bonus_offset, pcm_bonus_scale, qs_capt_hist_bonus_max, qs_capt_hist_bonus_offset, qs_capt_hist_bonus_scale, qs_capt_hist_malus_max, qs_capt_hist_malus_offset, qs_capt_hist_malus_scale, quiet_fact_bonus_max, quiet_fact_bonus_offset, quiet_fact_bonus_scale, quiet_fact_malus_max, quiet_fact_malus_offset, quiet_fact_malus_scale, quiet_hist_bonus_max, quiet_hist_bonus_offset, quiet_hist_bonus_scale, quiet_hist_lerp_factor, quiet_hist_malus_max, quiet_hist_malus_offset, quiet_hist_malus_scale, to_hist_bonus_max, to_hist_bonus_offset, to_hist_bonus_scale, to_hist_malus_max, to_hist_malus_offset, to_hist_malus_scale};
use crate::tools::utils::boxed_and_zeroed;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];
type ThreatBucket<T> = [[T; 2]; 2];

pub struct QuietHistory {
    from_to_entries: Box<[FromToHistory<QuietHistoryEntry>; 2]>,
    piece_to_entries: Box<[PieceToHistory<QuietHistoryEntry>; 2]>,
}

pub struct CaptureHistory {
    piece_to_entries: Box<[PieceToHistory<[i16; 6]>; 2]>,
    from_to_entries: Box<[FromToHistory<i16>; 2]>,
}

pub struct ContinuationHistory {
    entries: Box<[PieceToHistory<PieceToHistory<i16>>; 2]>,
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
        ContinuationHistory::PLIES
            .iter()
            .filter(|&&prev_ply| ply >= prev_ply)
            .filter_map(|&prev_ply| {
                let prev = ss[ply - prev_ply];
                let (prev_mv, prev_pc) = (prev.mv?, prev.pc?);
                Some(self.cont_history.get(prev_mv, prev_pc, mv, pc, prev_ply) as i32)
            })
            .sum()
    }

    pub fn capture_history_score(
        &self,
        board: &Board,
        mv: &Move,
        pc: Piece,
        captured: Piece,
    ) -> i32 {
        self.capture_history.get(board.stm, pc, *mv, captured) as i32
    }

    pub fn update_continuation_history(
        &mut self,
        board: &Board,
        ss: &NodeStack,
        ply: usize,
        mv: &Move,
        pc: Piece,
        bonuses: &[i16; ContinuationHistory::PLY_COUNT],
    ) {
        let total_score = self.cont_history_score(board, ss, mv, ply);
        let updates = ContinuationHistory::PLIES
            .iter()
            .zip(bonuses)
            .filter(|(&prev_ply, _)| ply >= prev_ply)
            .filter_map(|(&prev_ply, &bonus)| {
                let prev = ss[ply - prev_ply];
                let (prev_mv, prev_pc) = (prev.mv?, prev.pc?);
                Some((prev_mv, prev_pc, bonus, prev_ply))
            });

        for (prev_mv, prev_pc, bonus, prev_ply) in updates {
            self.cont_history
                .update(&prev_mv, prev_pc, mv, pc, total_score, bonus, prev_ply);
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
            piece_to_entries: unsafe { boxed_and_zeroed() },
            from_to_entries: unsafe { boxed_and_zeroed() },
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
    fn update(&mut self, threat_index: &ThreatIndex, bonus: i16, factoriser_bonus: i16) {
        self.factoriser = gravity(
            self.factoriser as i32,
            factoriser_bonus as i32,
            QuietHistory::FACTORISER_MAX,
        ) as i16;

        let bucket_entry = &mut self.bucket[threat_index.from()][threat_index.to()];
        *bucket_entry =
            gravity(*bucket_entry as i32, bonus as i32, QuietHistory::BUCKET_MAX) as i16;
    }
}

impl QuietHistory {
    const FACTORISER_MAX: i32 = 8192;
    const BUCKET_MAX: i32 = 16384;
    const BONUS_MAX: i16 = Self::BUCKET_MAX as i16 / 4;

    pub fn get(&self, stm: Side, mv: Move, pc: Piece, threats: Bitboard) -> i16 {
        let threat_idx = ThreatIndex::new(mv, threats);
        let from_to_score = self.from_to_entries[stm][mv.from()][mv.to()].score(&threat_idx) as i32;
        let piece_to_score = self.piece_to_entries[stm][pc][mv.to()].score(&threat_idx) as i32;
        lerp(from_to_score, piece_to_score, quiet_hist_lerp_factor()) as i16
    }

    pub fn update(
        &mut self,
        stm: Side,
        mv: &Move,
        pc: Piece,
        threats: Bitboard,
        bonus: i16,
        factoriser_bonus: i16,
    ) {
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        let threat_index = ThreatIndex::new(*mv, threats);

        self.from_to_entries[stm][mv.from()][mv.to()].update(
            &threat_index,
            bonus,
            factoriser_bonus,
        );

        self.piece_to_entries[stm][pc][mv.to()].update(&threat_index, bonus, factoriser_bonus);
    }

    pub fn clear(&mut self) {
        self.from_to_entries = Box::new([[[QuietHistoryEntry::default(); 64]; 64]; 2]);
        self.piece_to_entries = Box::new([[[QuietHistoryEntry::default(); 64]; 6]; 2]);
    }
}

impl CaptureHistory {
    const MAX: i32 = 16384;
    const BONUS_MAX: i16 = Self::MAX as i16 / 4;

    pub fn get(&self, stm: Side, pc: Piece, mv: Move, captured: Piece) -> i16 {
        let piece_to_score = self.piece_to_entries[stm][pc][mv.to()][captured] as i32;
        let from_to_score = self.from_to_entries[stm][mv.from()][mv.to()] as i32;
        lerp(from_to_score, piece_to_score, capt_hist_lerp_factor()) as i16
    }

    pub fn update(&mut self, stm: Side, pc: Piece, mv: &Move, captured: Piece, bonus: i16) {
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        update_entry(&mut self.piece_to_entries[stm][pc][mv.to()][captured], bonus, Self::MAX);
        update_entry(&mut self.from_to_entries[stm][mv.from()][mv.to()], bonus, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.piece_to_entries = Box::new([[[[0; 6]; 64]; 6], [[[0; 6]; 64]; 6]]);
        self.from_to_entries = Box::new([[[0; 64]; 64]; 2]);
    }
}

impl ContinuationHistory {
    const PLIES: [usize; 2] = [1, 2];
    const PLY_COUNT: usize = Self::PLIES.len();
    const MAX: i32 = 16384;
    const BONUS_MAX: i16 = Self::MAX as i16 / 4;

    pub fn get(&self, prev_mv: Move, prev_pc: Piece, mv: &Move, pc: Piece, prev_ply: usize) -> i16 {
        let index = (prev_ply & 1 == 0) as usize;
        self.entries[index][prev_pc][prev_mv.to()][pc][mv.to()]
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        prev_mv: &Move,
        prev_pc: Piece,
        mv: &Move,
        pc: Piece,
        total_score: i32,
        bonus: i16,
        prev_ply: usize,
    ) {
        let index = (prev_ply & 1 == 0) as usize;
        let entry = &mut self.entries[index][prev_pc][prev_mv.to()][pc][mv.to()];
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        *entry = gravity_with_base(*entry as i32, bonus as i32, total_score, Self::MAX) as i16;
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[[0; 64]; 6]; 64]; 6]; 2])
    }
}

impl SquareHistory {
    const MAX: i32 = 4096;
    const BONUS_MAX: i16 = Self::MAX as i16 / 4;

    pub fn get(&self, stm: Side, sq: Square) -> i16 {
        self.entries[stm][sq]
    }

    pub fn update(&mut self, stm: Side, sq: Square, bonus: i16) {
        let bonus = bonus.clamp(-Self::BONUS_MAX, Self::BONUS_MAX);
        update_entry(&mut self.entries[stm][sq], bonus, Self::MAX);
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
        ThreatIndex {
            from_attacked: threats.contains(mv.from()),
            to_attacked:   threats.contains(mv.to()),
        }
    }

    pub fn from(&self) -> usize {
        self.from_attacked as usize
    }

    pub fn to(&self) -> usize {
        self.to_attacked as usize
    }
}

/// Compute a history bonus, using the formula `scale * depth - offset`, clamped to the maximum bonus.
fn history_bonus(depth: i32, scale: i16, offset: i16, max: i16) -> i16 {
    (scale * depth as i16 - offset).min(max)
}

/// Compute a history malus, using the formula `-(scale * depth - offset)`, clamped to the maximum malus.
fn history_malus(depth: i32, scale: i16, offset: i16, max: i16) -> i16 {
    -(scale * depth as i16 - offset).min(max)
}

/// Apply the gravity formula to a single `i16` table entry in-place.
#[inline]
fn update_entry(entry: &mut i16, bonus: i16, max: i32) {
    *entry = gravity(*entry as i32, bonus as i32, max) as i16;
}

/// Gravity formula for history updates, using the current value of the entry as the base for the update.
fn gravity(current: i32, update: i32, max: i32) -> i32 {
    gravity_with_base(current, update, current, max)
}

/// Gravity formula for history updates, weighting the update by the current value of the entry.
fn gravity_with_base(current: i32, update: i32, base: i32, max: i32) -> i32 {
    current + update - base * update.abs() / max
}

/// Linearly interpolate between two history scores, using the provided factor (0-100).
fn lerp(a: i32, b: i32, factor: i32) -> i32 {
    (a * (100 - factor) + b * factor) / 100
}


// Defines a history bonus and malus function pair, using the provided tunable parameters.
macro_rules! history_bonuses {
    (
        $bonus_fn:ident ($bs:ident, $bo:ident, $bm:ident),
        $malus_fn:ident ($ms:ident, $mo:ident, $mm:ident)
    ) => {
        pub fn $bonus_fn(depth: i32) -> i16 {
            history_bonus(depth, $bs() as i16, $bo() as i16, $bm() as i16)
        }
        pub fn $malus_fn(depth: i32) -> i16 {
            history_malus(depth, $ms() as i16, $mo() as i16, $mm() as i16)
        }
    };
}

history_bonuses!(
    quiet_history_bonus (quiet_hist_bonus_scale, quiet_hist_bonus_offset, quiet_hist_bonus_max),
    quiet_history_malus (quiet_hist_malus_scale, quiet_hist_malus_offset, quiet_hist_malus_max)
);
history_bonuses!(
    quiet_factoriser_bonus (quiet_fact_bonus_scale, quiet_fact_bonus_offset, quiet_fact_bonus_max),
    quiet_factoriser_malus (quiet_fact_malus_scale, quiet_fact_malus_offset, quiet_fact_malus_max)
);
history_bonuses!(
    capture_history_bonus (capt_hist_bonus_scale, capt_hist_bonus_offset, capt_hist_bonus_max),
    capture_history_malus (capt_hist_malus_scale, capt_hist_malus_offset, capt_hist_malus_max)
);
history_bonuses!(
    cont_history_1_bonus (cont_hist_1_bonus_scale, cont_hist_1_bonus_offset, cont_hist_1_bonus_max),
    cont_history_1_malus (cont_hist_1_malus_scale, cont_hist_1_malus_offset, cont_hist_1_malus_max)
);
history_bonuses!(
    cont_history_2_bonus (cont_hist_2_bonus_scale, cont_hist_2_bonus_offset, cont_hist_2_bonus_max),
    cont_history_2_malus (cont_hist_2_malus_scale, cont_hist_2_malus_offset, cont_hist_2_malus_max)
);
history_bonuses!(
    from_history_bonus (from_hist_bonus_scale, from_hist_bonus_offset, from_hist_bonus_max),
    from_history_malus (from_hist_malus_scale, from_hist_malus_offset, from_hist_malus_max)
);
history_bonuses!(
    to_history_bonus (to_hist_bonus_scale, to_hist_bonus_offset, to_hist_bonus_max),
    to_history_malus (to_hist_malus_scale, to_hist_malus_offset, to_hist_malus_max)
);
history_bonuses!(
    qs_capthist_bonus (qs_capt_hist_bonus_scale, qs_capt_hist_bonus_offset, qs_capt_hist_bonus_max),
    qs_capthist_malus (qs_capt_hist_malus_scale, qs_capt_hist_malus_offset, qs_capt_hist_malus_max)
);
history_bonuses!(
    lmr_conthist_1_bonus (lmr_cont_1_bonus_scale, lmr_cont_1_bonus_offset, lmr_cont_1_bonus_max),
    lmr_conthist_1_malus (lmr_cont_1_malus_scale, lmr_cont_1_malus_offset, lmr_cont_1_malus_max)
);
history_bonuses!(
    lmr_conthist_2_bonus (lmr_cont_2_bonus_scale, lmr_cont_2_bonus_offset, lmr_cont_2_bonus_max),
    lmr_conthist_2_malus (lmr_cont_2_malus_scale, lmr_cont_2_malus_offset, lmr_cont_2_malus_max)
);
history_bonuses!(
    prior_countermove_bonus (pcm_bonus_scale, pcm_bonus_offset, pcm_bonus_max),
    prior_countermove_malus (pcm_bonus_scale, pcm_bonus_offset, pcm_bonus_max)
);

