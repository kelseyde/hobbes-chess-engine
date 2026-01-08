use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::search::{Score, MAX_PLY};
use std::ops::{Index, IndexMut};

pub struct SearchStack {
    data: [StackEntry; MAX_PLY + 8],
}

#[derive(Copy, Clone)]
pub struct StackEntry {
    pub mv: Option<Move>,
    pub pc: Option<Piece>,
    pub captured: Option<Piece>,
    pub killers: Killers,
    pub singular: Option<Move>,
    pub threats: Bitboard,
    pub raw_eval: i32,
    pub static_eval: i32,
    pub reduction: i32,
}

impl Default for SearchStack {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchStack {
    pub fn new() -> Self {
        SearchStack {
            data: [StackEntry {
                mv: None,
                pc: None,
                captured: None,
                killers: Killers::default(),
                singular: None,
                threats: Bitboard::empty(),
                raw_eval: Score::MIN,
                static_eval: Score::MIN,
                reduction: 0,
            }; MAX_PLY + 8],
        }
    }
}

impl Index<usize> for SearchStack {
    type Output = StackEntry;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl IndexMut<usize> for SearchStack {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

#[derive(Copy, Clone)]
pub struct Killers {
    pub entries: [Move; 2],
}

impl Default for Killers {
    fn default() -> Self {
        Killers {
            entries: [Move::NONE; 2],
        }
    }
}

impl Killers {

    pub fn add(&mut self, mv: Move) {
        self.entries[1] = self.entries[0];
        self.entries[0] = mv;
    }

    pub fn is_killer(&self, mv: Move) -> bool {
        self.entries.iter().any(|&x| x == mv)
    }

    pub fn killer_index(&self, mv: Move) -> Option<usize> {
        for (i, &killer) in self.entries.iter().enumerate() {
            if killer == mv {
                return Some(i);
            }
        }
        None
    }

    pub fn clear(&mut self) {
        self.entries = [Move::NONE; 2];
    }

}
