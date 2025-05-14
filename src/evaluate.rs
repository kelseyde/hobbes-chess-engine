use crate::bits;
use crate::board::Board;
use crate::consts::Side;

pub const PIECE_VALUES: [u32; 6] = [100, 300, 300, 500, 900, 0];

pub fn evaluate(board: &Board) -> i32 {
    let white_eval = side_eval(board, Side::White) as i32;
    let black_eval = side_eval(board, Side::Black) as i32;
    if board.stm == Side::White {
        white_eval - black_eval
    } else {
        black_eval - white_eval
    }
}

fn side_eval(board: &Board, side: Side) -> u32 {
    bits::count(board.pawns(side)) * PIECE_VALUES[0] +
    bits::count(board.knights(side)) * PIECE_VALUES[1] +
    bits::count(board.bishops(side)) * PIECE_VALUES[2] +
    bits::count(board.rooks(side)) * PIECE_VALUES[3] +
    bits::count(board.queens(side)) * PIECE_VALUES[4]
}