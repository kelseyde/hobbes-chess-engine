use std::time::Instant;

use crate::board::Board;
use crate::consts::{Score, MAX_DEPTH};
use crate::movegen::{gen_moves, is_check, MoveFilter};
use crate::moves::Move;
use crate::ordering::score;
use crate::thread::ThreadData;
use crate::tt::TTFlag;

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

fn alpha_beta(board: &Board, td: &mut ThreadData, mut depth: u8, ply: u8, mut alpha: i32, mut beta: i32) -> i32 {

    // If search is aborted, exit immediately
    if td.abort() { return alpha }

    let in_check = is_check(board, board.stm);

    // If depth is reached, drop into quiescence search
    if depth <= 0 && !in_check { return qs(&board, td, alpha, beta) }

    if depth < 0 { depth = 0; }

    if depth == MAX_DEPTH { return td.evaluator.evaluate(&board) }

    let root = ply == 0;
    let mut tt_move = Move::NONE;

    if !root {
        let tt_entry = td.tt.probe(board.hash);
        match tt_entry {
            Some(entry) => {
                tt_move = entry.best_move();
                if entry.depth() >= depth {
                    if entry.flag() == TTFlag::Exact {
                        return entry.score() as i32
                    } else if entry.flag() == TTFlag::Lower {
                        alpha = alpha.max(entry.score() as i32)
                    } else if entry.flag() == TTFlag::Upper {
                        beta = beta.min(entry.score() as i32)
                    }
                    if alpha >= beta {
                        return entry.score() as i32
                    }
                }
            }
            None => {}
        }
    }

    let mut moves = gen_moves(board, MoveFilter::All);
    let scores = score(&board, &moves, &tt_move);
    moves.sort(&scores);

    let mut legals = 0;
    let mut best_score = Score::Min as i32;
    let mut best_move = Move::NONE;
    let mut flag = TTFlag::Lower;

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

        if score > best_score {
            best_score = score;
        }

        if score > alpha {
            alpha = score;
            best_move = *mv;
            flag = TTFlag::Exact;
            if root {
                td.best_move = mv.clone();
            }

            if score >= beta {
                flag = TTFlag::Upper;
                break;
            }
        }
    }

    // handle checkmate / stalemate
    if legals == 0 {
        return if in_check { ply as i32 - Score::Max as i32 } else { Score::Draw as i32 }
    }

    if !root {
        td.tt.insert(board.hash, &best_move, best_score, depth, flag);
    }

    best_score
}

fn qs(board: &Board, td: &mut ThreadData, mut alpha: i32, mut beta: i32) -> i32 {

    // If search is aborted, exit immediately
    if td.abort() { return alpha }

    let in_check = is_check(board, board.stm);

    let tt_entry = td.tt.probe(board.hash);
    let mut tt_move = Move::NONE;
    match tt_entry {
        Some(entry) => {
            tt_move = entry.best_move();
            if entry.flag() == TTFlag::Exact {
                return entry.score() as i32
            } else if entry.flag() == TTFlag::Lower {
                alpha = alpha.max(entry.score() as i32)
            } else if entry.flag() == TTFlag::Upper {
                beta = beta.min(entry.score() as i32)
            }
            if alpha >= beta {
                return entry.score() as i32
            }
        }
        None => {}
    }

    if !in_check {
        let eval = td.evaluator.evaluate(&board);
        if eval > alpha {
            alpha = eval
        }
        if alpha >= beta {
            return alpha;
        }
    }

    let filter = if in_check { MoveFilter::All } else { MoveFilter::Captures };
    let mut moves = gen_moves(board, filter);
    let scores = score(&board, &moves, &tt_move);
    moves.sort(&scores);
    let mut legals = 0;

    let mut best_score = alpha;

    for mv in moves.iter() {
        let mut board = *board;
        board.make(&mv);
        if is_check(&board, board.stm.flip()) {
            continue
        }
        legals += 1;
        td.nodes += 1;
        let score = -qs(&board, td, -beta, -alpha);

        if td.abort() { break; }

        if score > best_score {
            best_score = score;
        }

        if score > alpha {
            alpha = score;

            if score >= beta {
                return score;
            }
        }
    }

    best_score
}

fn is_cancelled(time: Instant) -> bool {
    Instant::now() >= time
}