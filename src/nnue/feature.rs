use crate::network::{PIECE_OFFSET, SIDE_OFFSET};
use crate::types::piece::Piece;
use crate::types::side::Side;
use crate::types::side::Side::White;
use crate::types::square::Square;

pub struct Feature {
    pc: Piece,
    sq: Square,
    side: Side
}

impl Feature {

    pub fn new(pc: Piece, sq: Square, side: Side) -> Self {
        Feature { pc, sq, side }
    }

    pub fn index(&self, perspective: Side, mirror: bool) -> usize {
        let sq_index = self.square_index(perspective, mirror);
        let pc_offset = self.pc as usize * PIECE_OFFSET;
        let side_offset = if self.side == perspective { 0 } else { SIDE_OFFSET };
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