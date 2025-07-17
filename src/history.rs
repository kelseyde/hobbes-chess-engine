use crate::moves::Move;
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

pub struct CorrectionHistory {
    entries: Box<[[i32; CorrectionHistory::SIZE]; 2]>,
}

pub struct CaptureHistory {
    entries: Box<[PieceToHistory<[i16; 6]>; 2]>,
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

impl Default for CorrectionHistory {
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
        *entry = gravity(*entry, bonus, Self::MAX);
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
        *entry = gravity(*entry, bonus, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[[0; 64]; 64]; 2]; 2]; 2]);
    }
}

impl CorrectionHistory {
    const SIZE: usize = 16384;
    const MASK: usize = Self::SIZE - 1;
    const SCALE: i32 = 256;
    const GRAIN: i32 = 256;
    const MAX: i32 = Self::GRAIN * 32;

    pub fn get(&self, stm: Side, key: u64) -> i32 {
        let idx = self.index(key);
        self.entries[stm][idx] / Self::SCALE
    }

    pub fn update(&mut self, stm: Side, key: u64, depth: i32, static_eval: i32, score: i32) {
        let idx = self.index(key);
        let entry = &mut self.entries[stm][idx];
        let new_value = (score - static_eval) * Self::SCALE;

        let new_weight = (depth + 1).min(16);
        let old_weight = Self::SCALE - new_weight;

        let update = *entry * old_weight + new_value * new_weight;
        *entry = i32::clamp(update / Self::SCALE, -Self::MAX, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[0; Self::SIZE]; 2]);
    }

    fn index(&self, key: u64) -> usize {
        key as usize & Self::MASK
    }
}

impl CaptureHistory {
    const MAX: i16 = 16384;

    pub fn get(&self, stm: Side, pc: Piece, sq: Square, captured: Piece) -> i16 {
        self.entries[stm][pc][sq][captured]
    }

    pub fn update(&mut self, stm: Side, pc: Piece, sq: Square, captured: Piece, bonus: i16) {
        let entry = &mut self.entries[stm][pc][sq][captured];
        *entry = gravity(*entry, bonus, Self::MAX);
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

fn gravity(current: i16, update: i16, max: i16) -> i16 {
    current + update - (current as i32 * update.abs() as i32 / max as i32) as i16
}
