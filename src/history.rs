use crate::consts::{Piece, Side};
use crate::moves::Move;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];

pub struct QuietHistory {
    entries: Box<[FromToHistory<i16>; 2]>,
}

pub struct ContinuationHistory {
    // [previous_piece][previous_to][current_piece][current_to]
    entries: Box<[[[[i32; 64]; 12]; 64]; 12]>,
}

impl ContinuationHistory {
    const MAX_HISTORY: i32 = 16384;

    pub fn get(&self, prev_pc: Piece, prev_mv: Move, pc: Piece, mv: Move) -> i32 {
        self.entries[prev_pc][prev_mv.to()][pc][mv.to()]
    }

    pub fn update(&mut self, prev_mv: Move, prev_pc: Piece, mv: Move, pc: Piece, bonus: i32) {
        let entry = &mut self.entries[prev_pc][prev_mv.to()][pc][mv.to()];
        *entry += bonus - bonus.abs() * (*entry) / Self::MAX_HISTORY;
    }
}

impl QuietHistory {
    const HISTORY_MAX: i16 = 16384;

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
        *entry = gravity(*entry, bonus, Self::HISTORY_MAX);
    }


    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            for row in entry.iter_mut() {
                for col in row.iter_mut() {
                    *col = 0;
                }
            }
        }
    }
}

fn gravity(current: i16, update: i16, max: i16) -> i16 {
    current + update - current * update.abs() / max
}