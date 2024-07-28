use std::time::Instant;

use crate::board::Board;
use crate::consts::MAX_DEPTH;
use crate::movegen::{gen_moves, is_check, MoveFilter};
use crate::moves::Move;
use crate::thread::ThreadData;
use crate::tt::TTFlag;

pub fn search(board: &Board, td: &mut ThreadData) -> (Move, i32) {

    td.time = Instant::now();
    td.best_move = Move::NONE;

    let mut depth = 1;
    let mut alpha = i16::MIN;
    let mut beta = i16::MAX;
    let mut best_move = Move::NONE;
    let mut score = 0;


    while depth < MAX_DEPTH && !td.cancelled() {
        let eval = alpha_beta(board, td, depth, 0, alpha, beta);
        best_move = td.best_move;
        score = eval;

        depth += 1;
    }

    (td.best_move, score)

}

fn alpha_beta(board: &Board, td: &mut ThreadData, depth: u8, ply: u8, mut alpha: i32, mut beta: i32) -> i32 {

    // If timeout is reached, exit immediately
    if td.cancelled() { return alpha }

    // If depth is reached, drop into quiescence search
    if depth == 0 { return qs(board, td, alpha, beta) }

    let root = ply == 0;
    // let pv_node = alpha != beta - 1;

    if !root {
        let tt_entry = td.tt.probe(board.hash);
        match tt_entry {
            Some(entry) => {
                if entry.depth() >= depth {
                    if entry.flag() == TTFlag::Exact {
                        return entry.score()
                    } else if entry.flag() == TTFlag::Lower {
                        alpha = alpha.max(entry.score())
                    } else if entry.flag() == TTFlag::Upper {
                        beta = beta.min(entry.score())
                    }
                    if alpha >= beta {
                        return entry.score()
                    }
                }
            }
            None => {}
        }
    }

    let mut best_move: &Move = &Move::NONE;
    let mut moves = gen_moves(board, MoveFilter::All);
    let mut flag = TTFlag::Upper;

    for mv in moves.iter() {
        let mut board = *board;
        board.make(mv);
        if is_check(&board, board.stm) {
            continue
        }
        let score = -alpha_beta(&board, td, depth - 1, ply + 1, -beta, -alpha);

        if score >= beta {
            td.tt.insert(board.hash, *mv, score, depth, TTFlag::Lower);
            return score;
        }

        if score > alpha {
            alpha = score;
            best_move = mv;
            td.best_move = *mv;
            flag = TTFlag::Exact;
        }
    }

    td.tt.insert(board.hash, *best_move, alpha, depth, flag);
    alpha
}

fn qs(board: &Board, td: &mut ThreadData, mut alpha: i32, beta: i32) -> i32 {

    let stand_pat = board.eval();
    if stand_pat >= beta {
        return beta
    }
    if stand_pat > alpha {
        alpha = stand_pat
    }

    let mut moves = gen_moves(board, MoveFilter::Captures);
    for mv in moves.iter() {
        let mut board = *board;
        board.make(mv);
        if is_check(&board, board.stm) {
            continue
        }
        let score = -qs(&board, td, -beta, -alpha);
        if score >= beta {
            return beta
        }
        if score > alpha {
            alpha = score
        }
    }

    alpha
}

fn is_cancelled(time: Instant) -> bool {
    Instant::now() >= time
}