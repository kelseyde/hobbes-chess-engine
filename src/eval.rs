use crate::board::Board;
use crate::consts::Piece;

pub const PAWN_SCORE: i32 = 100;
pub const KNIGHT_SCORE: i32 = 320;
pub const BISHOP_SCORE: i32 = 330;
pub const ROOK_SCORE: i32 = 500;
pub const QUEEN_SCORE: i32 = 900;

pub fn eval(board: &Board) -> i32 {
    let us = board.side(board.stm);
    let them = board.side(board.stm.flip());
    count_material(board, us) - count_material(board, them)
}

fn count_material(board: &Board, us: u64) -> i32 {
    let mut score = 0;
    score += PAWN_SCORE   * (board.pcs(Piece::Pawn) & us).count_ones() as i32;
    score += KNIGHT_SCORE * (board.pcs(Piece::Knight) & us).count_ones() as i32;
    score += BISHOP_SCORE * (board.pcs(Piece::Bishop) & us).count_ones() as i32;
    score += ROOK_SCORE   * (board.pcs(Piece::Rook) & us).count_ones() as i32;
    score += QUEEN_SCORE  * (board.pcs(Piece::Queen) &us).count_ones() as i32;
    score
}