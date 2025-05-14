use std::time::Instant;

use crate::board::Board;
use crate::consts::{Score, MAX_DEPTH};
use crate::movegen::{gen_moves, is_check, MoveFilter};
use crate::moves::Move;
use crate::thread::ThreadData;

pub fn search(board: &Board, td: &mut ThreadData) -> (Move, i32) {

    td.time = Instant::now();
    td.best_move = Move::NONE;

    let alpha = Score::Min as i32;
    let beta = Score::Max as i32;
    let mut score = 0;

    while td.depth < MAX_DEPTH && !td.abort() {
        let eval = alpha_beta(board, td, td.depth, 0, alpha, beta);
        score = eval;

        if td.main {
            if td.best_move.exists() {
                println!("info depth {} score cp {} pv {}", td.depth, score, td.best_move.to_uci());
            } else {
                println!("info depth {} score cp {}", td.depth, score);
            }
        }

        td.depth += 1;
    }

    (td.best_move, score)

}

fn alpha_beta(board: &Board, td: &mut ThreadData, depth: u8, ply: u8, mut alpha: i32, beta: i32) -> i32 {

    // If search is aborted, exit immediately
    if td.abort() { return alpha }

    // If depth is reached, drop into quiescence search
    if depth == 0 { return td.evaluator.evaluate(&board) }

    if depth == MAX_DEPTH { return td.evaluator.evaluate(&board) }

    let root = ply == 0;

    let moves = gen_moves(board, MoveFilter::All);
    let mut legals = 0;

    for mv in moves.iter() {
        let mut board = *board;
        board.make(&mv);
        if is_check(&board, board.stm.flip()) {
            continue
        }
        legals += 1;
        td.nodes += 1;
        let score = -alpha_beta(&board, td, depth - 1, ply + 1, -beta, -alpha);

        if td.abort() { break; }

        if score >= beta {
            return score;
        }

        if score > alpha {
            alpha = score;
            if root {
                td.best_move = mv.clone();
            }
        }
    }

    // handle checkmate / stalemate
    if legals == 0 {
        return if is_check(board, board.stm) { ply as i32 - Score::Max as i32 } else { Score::Draw as i32 }
    }

    alpha
}

fn is_cancelled(time: Instant) -> bool {
    Instant::now() >= time
}