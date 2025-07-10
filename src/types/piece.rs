use std::ops::{Index, IndexMut};

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