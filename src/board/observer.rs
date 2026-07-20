use crate::board::Board;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;

/// Observer trait that reacts to changes to the `Board` during makemove. Used to update the NNUE
/// threat accumulator on the fly.
pub trait BoardObserver {
    fn on_piece_create(&mut self, board: &Board, pc: Piece, side: Side, sq: Square);
    fn on_piece_destroy(&mut self, board: &Board, pc: Piece, side: Side, sq: Square);
    fn on_piece_teleport(&mut self, board: &Board, pc: Piece, side: Side, from: Square, to: Square);
    fn on_piece_transform(&mut self, board: &Board, old_pc: Piece, old_side: Side, new_pc: Piece, side: Side, sq: Square);
}

pub struct NullBoardObserver;

impl BoardObserver for NullBoardObserver {
    fn on_piece_create(&mut self, _: &Board, _: Piece, _: Side, _: Square) {}
    fn on_piece_destroy(&mut self, _: &Board, _: Piece, _: Side, _: Square) {}
    fn on_piece_teleport(&mut self, _: &Board, _: Piece, _: Side, _: Square, _: Square) {}
    fn on_piece_transform(&mut self, _: &Board, _: Piece, _: Side, _: Piece, _: Side, _: Square) {}
}