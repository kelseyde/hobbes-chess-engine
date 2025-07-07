use std::ops::{Index, IndexMut};

pub const MAX_DEPTH: i32 = 255;

pub struct Score {}

impl Score {
    pub const DRAW: i32 = 0;
    pub const MAX: i32 = 32767;
    pub const MIN: i32 = -32767;
    pub const MATE: i32 = 32766;

    pub fn is_mate(score: i32) -> bool {
        score.abs() >= Score::MATE - MAX_DEPTH
    }
}


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5
}

pub const PIECES: [Piece; 6] = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];

impl Piece {
    pub fn iter() -> impl Iterator<Item = Piece> {
        PIECES.iter().copied()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Side {
    #[default]
    White,
    Black
}

impl Side {

    pub fn flip(&self) -> Side {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White
        }
    }

    pub const fn idx(&self) -> usize {
        *self as usize + 6
    }

}


impl<T, const N: usize> Index<Piece> for [T; N] {
    type Output = T;

    fn index(&self, pc: Piece) -> &Self::Output {
        &self[pc as usize]
    }
}

impl<T, const N: usize> IndexMut<Piece> for [T; N] {
    fn index_mut(&mut self, pc: Piece) -> &mut Self::Output {
        &mut self[pc as usize]
    }
}

impl<T, const N: usize> Index<Side> for [T; N] {
    type Output = T;

    fn index(&self, stm: Side) -> &Self::Output {
        &self[stm as usize]
    }
}

impl<T, const N: usize> IndexMut<Side> for [T; N] {
    fn index_mut(&mut self, stm: Side) -> &mut Self::Output {
        &mut self[stm as usize]
    }
}