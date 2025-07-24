use crate::board::Board;
use crate::moves::Move;
use crate::search::SearchStack;
use crate::types::bitboard::Bitboard;
use crate::types::piece::Piece;
use crate::types::side::Side;
use crate::types::square::Square;
use crate::utils::boxed_and_zeroed;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];

pub struct QuietHistory {
    entries: Box<[[[FromToHistory<i16>; 2]; 2]; 2]>,
}

pub struct ContinuationHistory {
    entries: Box<PieceToHistory<PieceToHistory<i16>>>,
}

pub struct CaptureHistory {
    entries: Box<[PieceToHistory<[i16; 6]>; 2]>,
}

#[derive(Default)]
pub struct Histories {
    pub quiet_history: QuietHistory,
    pub capture_history: CaptureHistory,
    pub cont_history: ContinuationHistory,
}

impl Histories {

    pub fn history_score(&self,
                         board: &Board,
                         ss: &SearchStack,
                         mv: &Move, ply: usize,
                         threats: Bitboard,
                         pc: Piece,
                         captured: Option<Piece>) -> i32 {
        if let Some(captured) = captured {
            self.capture_history_score(board, mv, pc, captured)
        } else {
            self.quiet_history_score(board, ss, mv, ply, threats)
        }
    }

    pub fn quiet_history_score(&self,
                               board: &Board,
                               ss: &SearchStack,
                               mv: &Move,
                               ply: usize,
                               threats: Bitboard) -> i32 {
        let pc = board.piece_at(mv.from()).unwrap();
        let quiet_score = self.quiet_history.get(board.stm, *mv, threats) as i32;
        let mut cont_score = 0;
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply  {
                if let (Some(prev_mv), Some(prev_pc)) = (ss[ply - prev_ply].mv, ss[ply - prev_ply].pc) {
                    cont_score += self.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                }
            }
        }
        quiet_score + cont_score
    }

    pub fn capture_history_score(&self, board: &Board, mv: &Move, pc: Piece, captured: Piece) -> i32 {
        self.capture_history.get(board.stm, pc, mv.to(), captured) as i32
    }

    pub fn update_continuation_history(&mut self, ss: &SearchStack, ply: usize, mv: &Move, pc: Piece, bonus: i16) {
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply {
                if let (Some(prev_mv), Some(prev_pc)) = (ss[ply - prev_ply].mv, ss[ply - prev_ply].pc) {
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
            entries: unsafe { boxed_and_zeroed() }
        }
    }
}

impl Default for CaptureHistory {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() }
        }
    }
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() }
        }
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

impl QuietHistory {
    const MAX: i16 = 16384;

    pub fn get(&self, stm: Side, mv: Move, threats: Bitboard) -> i16 {
        let threat_index = ThreatIndex::new(mv, threats);
        self.entries[stm][threat_index.from()][threat_index.to()][mv.from()][mv.to()]
    }

    pub fn update(&mut self, stm: Side, mv: &Move, threats: Bitboard, bonus: i16) {
        let threat_index = ThreatIndex::new(*mv, threats);
        let entry = &mut self.entries[stm][threat_index.from()][threat_index.to()][mv.from()][mv.to()];
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

pub struct ThreatIndex {
    pub from_attacked: bool,
    pub to_attacked: bool
}

impl ThreatIndex {

    pub fn new(mv: Move, threats: Bitboard) -> Self {
        let from_attacked = threats.contains(mv.from());
        let to_attacked = threats.contains(mv.to());
        ThreatIndex { from_attacked, to_attacked }
    }

    pub fn from(&self) -> usize {
        self.from_attacked as usize
    }

    pub fn to(&self) -> usize {
        self.to_attacked as usize
    }

}

fn gravity(current: i32, update: i32, max: i32) -> i32 {
    current + update - current * update.abs() / max
}