use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;

/// Represents a single feature used by the neural network. A feature is a piece on a square on the
/// board, with a colour (white or black). The feature can either be activated - meaning the piece
/// is present on that square - or not  activated - meaning the piece is not present on that square.
/// The presence or absence of a feature is represented by a 1 or 0 respectively in the input layer.
#[derive(Copy, Clone)]
pub struct Feature {
    pc: Piece,
    sq: Square,
    side: Side,
}

impl Feature {
    #[inline]
    pub fn new(pc: Piece, sq: Square, side: Side) -> Self {
        Feature { pc, sq, side }
    }

    #[inline]
    pub fn index(&self, perspective: Side, mirror: bool) -> usize {
        let (mut sq, mut color) = match perspective {
            Side::White => (self.sq, self.side),
            Side::Black => (self.sq.flip_rank(), !self.side),
        };

        // Horizontal mirroring
        if mirror {
            sq = sq.flip_file();
        }

        // Merged king planes
        if self.pc == Piece::King {
            color = Side::White;
        }

        let sq = sq.0 as usize;
        let pc = self.pc as usize;
        let color = color as usize;

        sq + Square::COUNT as usize * (pc + Piece::COUNT * color)
    }


}
