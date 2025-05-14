use std::time::Instant;
use arrayvec::ArrayVec;
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

    let mut alpha = Score::Min as i32;
    let mut beta = Score::Max as i32;
    let mut score = 0;
    let mut delta = 24;

    while td.depth < MAX_DEPTH && !td.abort() {

        if td.depth >= 4 {
            alpha = (score - delta).max(Score::Min as i32);
            beta = (score + delta).min(Score::Max as i32);
        }

        loop {
            score = alpha_beta(board, td, td.depth, 0, alpha, beta);

            if td.main {
                if td.best_move.exists() {
                    println!("info depth {} score cp {} pv {}", td.depth, score, td.best_move.to_uci());
                } else {
                    println!("info depth {} score cp {}", td.depth, score);
                }
            }

            if td.abort() || Score::is_mate(score) {
                break;
            }

            match score {
                s if s <= alpha => {
                    beta = (alpha + beta) / 2;
                    alpha = (score - delta).max(Score::Min as i32);
                }
                s if s >= beta => {
                    beta = (score + delta).min(Score::Max as i32);
                }
                _ => break,
            }

            delta += delta / 2;
        }

        td.depth += 1;
    }

    (td.best_move, score)

}

fn alpha_beta(board: &Board, td: &mut ThreadData, mut depth: i32, ply: i32, mut alpha: i32, mut beta: i32) -> i32 {

    // If search is aborted, exit immediately
    if td.abort() { return alpha }

    let in_check = is_check(board, board.stm);

    // If depth is reached, drop into quiescence search
    if depth <= 0 && !in_check { return qs(&board, td, alpha, beta, ply) }

    if depth < 0 { depth = 0; }

    if depth == MAX_DEPTH { return td.evaluator.evaluate(&board) }

    let root = ply == 0;
    let mut tt_move = Move::NONE;

    if !root {
        if let Some(entry) = td.tt.probe(board.hash) {
            tt_move = entry.best_move();

            if entry.depth() >= depth as u8 {
                let score = entry.score(ply) as i32;
                match entry.flag() {
                    TTFlag::Exact => return score,
                    TTFlag::Lower => alpha = alpha.max(score),
                    TTFlag::Upper => beta = beta.min(score),
                }

                if alpha >= beta {
                    return score;
                }
            }
        }
    }

    let static_eval = if in_check {Score::Min as i32} else { td.evaluator.evaluate(&board) };

    if !root
        && !in_check
        && depth <= 8
        && static_eval - 80 * depth >= beta {
        return static_eval;
    }

    if !root
        && !in_check
        && depth >= 3
        && static_eval >= beta
        && board.has_non_pawns() {

        let mut board = *board;
        board.make_null_move();
        td.nodes += 1;

        let score = -alpha_beta(&board, td, depth - 3, ply + 1, -beta, -beta + 1);

        if score >= beta {
            return score;
        }

    }

    let mut moves = gen_moves(board, MoveFilter::All);
    let scores = score(&td, &board, &moves, &tt_move);
    moves.sort(&scores);

    let mut move_count = 0;
    let mut best_score = Score::Min as i32;
    let mut best_move = Move::NONE;
    let mut flag = TTFlag::Upper;

    let mut quiet_moves = ArrayVec::<Move, 32>::new();

    for mv in moves.iter() {
        let mut board = *board;
        board.make(&mv);
        if is_check(&board, board.stm.flip()) {
            continue
        }
        let captured = board.captured(&mv);
        let is_quiet = captured.is_none();
        move_count += 1;
        td.nodes += 1;

        let score = -alpha_beta(&board, td, depth - 1, ply + 1, -beta, -alpha);

        if is_quiet {
            quiet_moves.push(*mv);
        }

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
                flag = TTFlag::Lower;
                td.quiet_history.update(board.stm, &mv, (120 * depth as i16 - 75).min(1200));
                break;
            }
        }
    }

    // handle checkmate / stalemate
    if move_count == 0 {
        return if in_check { -(Score::Mate as i32) + ply } else { Score::Draw as i32 }
    }

    if !root {
        td.tt.insert(board.hash, &best_move, best_score, depth as u8, ply, flag);
    }

    best_score
}

fn qs(board: &Board, td: &mut ThreadData, mut alpha: i32, mut beta: i32, ply: i32) -> i32 {

    // If search is aborted, exit immediately
    if td.abort() { return alpha }

    let in_check = is_check(board, board.stm);

    let tt_entry = td.tt.probe(board.hash);
    let mut tt_move = Move::NONE;
    if let Some(entry) = tt_entry {
        tt_move = entry.best_move();
        let score = entry.score(ply) as i32;
        match entry.flag() {
            TTFlag::Exact => return score,
            TTFlag::Lower => alpha = alpha.max(score),
            TTFlag::Upper => beta = beta.min(score),
        }

        if alpha >= beta {
            return score;
        }
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
    let scores = score(&td, &board, &moves, &tt_move);
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
        let score = -qs(&board, td, -beta, -alpha, ply + 1);

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