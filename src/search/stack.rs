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
    pub killer: Option<Move>,
    pub singular: Option<Move>,
    pub threats: Bitboard,
    pub raw_eval: i32,
    pub static_eval: i32,
    pub reduction: i32,
    pub dextensions: i32,
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
                killer: None,
                singular: None,
                threats: Bitboard::empty(),
                raw_eval: Score::MIN,
                static_eval: Score::MIN,
                reduction: 0,
                dextensions: 0,
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
