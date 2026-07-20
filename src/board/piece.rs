use crate::board::side::Side;
use std::ops::{Index, IndexMut};

/// Enum representing each chess piece type.
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

pub const PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

impl Piece {
    pub const COUNT: usize = 6;

    #[inline(always)]
    pub const fn is_major(self) -> bool {
        matches!(self, Piece::Queen | Piece::Rook | Piece::King)
    }

    #[inline(always)]
    pub const fn is_minor(self) -> bool {
        matches!(self, Piece::Bishop | Piece::Knight | Piece::King)
    }

    /// Provides the index of this piece as a coloured piece (0-5 for white, 6-11 for black).
    pub fn coloured_index(&self, side: Side) -> usize {
        *self as usize + 6 * side as usize
    }

    /// Provides an iterator of (side, piece) pairs for all pieces of both sides.
    pub fn coloured_pieces() -> impl Iterator<Item = (Side, Piece)> {
        [Side::White, Side::Black].into_iter().flat_map(|side| {
            [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ]
            .into_iter()
            .map(move |pc| (side, pc))
        })
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
