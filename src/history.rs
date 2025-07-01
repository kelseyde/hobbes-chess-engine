use crate::consts::{Piece, Side};
use crate::moves::Move;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];

pub struct QuietHistory {
    entries: Box<[FromToHistory<i16>; 2]>,
}

pub struct ContinuationHistory {
    entries: Box<PieceToHistory<PieceToHistory<i16>>>,
}

pub struct CorrectionHistory {
    entries: Box<[[i32; CorrectionHistory::SIZE]; 2]>,
}

impl ContinuationHistory {
    const MAX: i16 = 16384;

    pub fn new() -> Self {
        ContinuationHistory {
            entries: Box::new([[[[0; 64]; 6]; 64]; 6])
        }
    }

    pub fn get(&self, prev_mv: Move, prev_pc: Piece, mv: Move, pc: Piece) -> i16 {
        self.entries[prev_pc][prev_mv.to()][pc][mv.to()]
    }

    pub fn update(&mut self, prev_mv: Move, prev_pc: Piece, mv: Move, pc: Piece, bonus: i16) {
        let entry = &mut self.entries[prev_pc][prev_mv.to()][pc][mv.to()];
        *entry = gravity(*entry, bonus, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[[0; 64]; 6]; 64]; 6])
    }

}

impl QuietHistory {
    const MAX: i16 = 16384;

    pub fn new() -> Self {
        QuietHistory {
            entries: Box::new([[[0; 64]; 64], [[0; 64]; 64] ]),
        }
    }

    pub fn get(&self, stm: Side, mv: Move) -> i16 {
        self.entries[stm][mv.from()][mv.to()]
    }

    pub fn update(&mut self, stm: Side, mv: &Move, bonus: i16) {
        let entry = &mut self.entries[stm][mv.from()][mv.to()];
        *entry = gravity(*entry, bonus, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[[0; 64]; 64], [[0; 64]; 64]]);
    }
}

impl CorrectionHistory {
    const SIZE: usize = 16384;
    const MASK: usize = Self::SIZE - 1;
    const SCALE: i32 = 64;
    const MAX: i32 = Self::SCALE * 32;

    pub fn new() -> Self {
        CorrectionHistory {
            entries: Box::new([[0; Self::SIZE]; 2])
        }
    }

    pub fn get(&self, stm: Side, key: u64) -> i32 {
        let index = self.index(key);
        self.entries[stm][index] / Self::SCALE
    }

    pub fn update(&mut self, stm: Side, key: u64, depth: i32, diff: i32) {
        let index = self.index(key);

        let old_value = &mut self.entries[stm][index];
        let new_value = diff * Self::SCALE;

        let new_weight = 16.min(depth + 1);
        let old_weight = Self::SCALE - new_weight;

        let update = (*old_value * old_weight + new_value * new_value) / Self::SCALE;
        *old_value = update.clamp(-Self::MAX, Self::MAX);
    }

    pub fn clear(&mut self) {
        self.entries = Box::new([[0; Self::SIZE]; 2]);
    }

    fn index(&self, key: u64) -> usize {
        key as usize & Self::MASK
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self { entries: Box::new([[0; Self::SIZE]; 2]) }
    }
}

fn gravity(current: i16, update: i16, max: i16) -> i16 {
    current + update - current * update.abs() / max
}