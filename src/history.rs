use crate::consts::{Piece, Side};
use crate::moves::Move;
use crate::types::square::Square;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];

pub struct QuietHistory {
    entries: Box<[FromToHistory<i16>; 2]>,
}

pub struct ContinuationHistory {
    entries: Box<PieceToHistory<PieceToHistory<i16>>>,
}

pub struct CaptureHistory {
    entries: Box<[PieceToHistory<[i16; 6]>; 2]>,
}

impl ContinuationHistory {
    const MAX: i16 = 16384;

    pub fn new() -> Self {
        ContinuationHistory {
            entries: Box::new([[[[0; 64]; 6]; 64]; 6])
        }
    }

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

    pub fn new() -> Self {
        QuietHistory {
            entries: Box::new([[[0; 64]; 64], [[0; 64]; 64]]),
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

impl CaptureHistory {

    const MAX: i16 = 16384;

    pub fn new() -> Self {
        CaptureHistory {
            entries: Box::new([[[[0; 6]; 64]; 6], [[[0; 6]; 64]; 6]]),
        }
    }

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

fn gravity(current: i16, update: i16, max: i16) -> i16 {
    current + update - current * update.abs() / max
}