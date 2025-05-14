use crate::consts::Side;
use crate::moves::Move;

type FromToHistory<T> = [[T; 64]; 64];
type PieceToHistory<T> = [[T; 64]; 6];

pub struct QuietHistory {
    entries: Box<[FromToHistory<i16>; 2]>,
}

impl QuietHistory {
    const HISTORY_MAX: i16 = 16384;

    pub fn new() -> Self {
        QuietHistory {
            entries: Box::new([[[0; 64]; 64], [[0; 64]; 64] ]),
        }
    }

    pub fn get(&self, stm: Side, mv: Move) -> i16 {
        self.entries[stm as usize][mv.from() as usize][mv.to() as usize]
    }

    pub fn update(&mut self, stm: Side, mv: Move, bonus: i16) {
        let entry = &mut self.entries[stm as usize][mv.from() as usize][mv.to() as usize];
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