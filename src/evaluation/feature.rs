use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::side::Side::White;
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

const PIECE_OFFSET: usize = 64;
const SIDE_OFFSET: usize = 64 * 6;

impl Feature {
    pub fn new(pc: Piece, sq: Square, side: Side) -> Self {
        Feature { pc, sq, side }
    }

    pub fn index(&self, perspective: Side, mirror: bool) -> usize {
        let sq_index = self.square_index(perspective, mirror);
        let pc_offset = self.pc as usize * PIECE_OFFSET;
        let side_offset = if self.side == perspective {
            0
        } else {
            SIDE_OFFSET
        };
        side_offset + pc_offset + sq_index
    }

    fn square_index(&self, perspective: Side, mirror: bool) -> usize {
        let mut sq_index = self.sq;
        if perspective != White {
            sq_index = sq_index.flip_rank();
        }
        if mirror {
            sq_index = sq_index.flip_file();
        }
        sq_index.0 as usize
    }
}
