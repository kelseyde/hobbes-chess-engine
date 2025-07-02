use crate::board::Board;
use crate::consts::{Piece, Score, MAX_DEPTH};
use crate::movegen::{gen_moves, is_check, is_legal, MoveFilter};
use crate::moves::Move;
use crate::ordering::score;
use crate::see;
use crate::see::see;
use crate::thread::ThreadData;
use crate::tt::TTFlag;
use arrayvec::ArrayVec;
use std::ops::{Index, IndexMut};
use std::time::Instant;
use crate::time::LimitType::{Hard, Soft};

pub const MAX_PLY: usize = 256;

pub fn search(board: &Board, td: &mut ThreadData) -> (Move, i32) {

    td.start_time = Instant::now();
    td.best_move = Move::NONE;

    let mut alpha = Score::MIN;
    let mut beta = Score::MAX;
    let mut score = 0;
    let mut delta = 24;

    while td.depth < MAX_DEPTH && !td.should_stop(Soft) {

        if td.depth >= 4 {
            alpha = (score - delta).max(Score::MIN);
            beta = (score + delta).min(Score::MAX);
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

            if td.should_stop(Hard) || Score::is_mate(score) {
                break;
            }

            match score {
                s if s <= alpha => {
                    beta = (alpha + beta) / 2;
                    alpha = (score - delta).max(Score::MIN);
                }
                s if s >= beta => {
                    beta = (score + delta).min(Score::MAX);
                }
                _ => break,
            }

            delta += delta / 2;
        }

        td.depth += 1;
    }

    (td.best_move, score)

}

fn alpha_beta(board: &Board, td: &mut ThreadData, mut depth: i32, ply: usize, mut alpha: i32, mut beta: i32) -> i32 {

    // If search is aborted, exit immediately
    if td.should_stop(Hard) { return alpha }

    let in_check = is_check(board, board.stm);

    // If depth is reached, drop into quiescence search
    if depth <= 0 && !in_check { return qs(&board, td, alpha, beta, ply) }

    if depth < 0 { depth = 0; }

    if ply > 0 && is_draw(&td, &board) {
        return Score::DRAW;
    }

    if depth == MAX_DEPTH { return td.nnue.evaluate(&board) }

    let root_node = ply == 0;
    let pv_node = beta - alpha > 1;

    let mut tt_move = Move::NONE;

    if !root_node {
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

    let mut raw_eval = Score::MIN;
    let mut static_eval = Score::MIN;

    if !in_check {
        raw_eval = td.nnue.evaluate(&board);
        static_eval = raw_eval + td.correction(board);
    };

    if !root_node && !pv_node && !in_check {

        if depth <= 8
            && static_eval - 80 * depth >= beta {
            return static_eval;
        }

        if depth >= 3
            && static_eval >= beta
            && board.has_non_pawns() {

            let mut board = *board;
            board.make_null_move();
            td.nodes += 1;
            td.keys.push(board.hash);
            let score = -alpha_beta(&board, td, depth - 3, ply + 1, -beta, -beta + 1);
            td.keys.pop();

            if score >= beta {
                return score;
            }
        }

    }

    let mut moves = gen_moves(board, MoveFilter::All);
    let scores = score(&td, &board, &moves, &tt_move, ply);
    moves.sort(&scores);

    let mut move_count = 0;
    let mut quiet_count = 0;
    let mut best_score = Score::MIN;
    let mut best_move = Move::NONE;
    let mut flag = TTFlag::Upper;

    let mut quiet_moves = ArrayVec::<Move, 32>::new();

    for mv in moves.iter() {

        if !is_legal(&board, &mv) {
            continue;
        }

        let pc = board.piece_at(mv.from());
        let captured = board.captured(&mv);
        let is_quiet = captured.is_none();
        let is_mate_score = Score::is_mate(best_score);

        if !pv_node
            && !root_node
            && !in_check
            && is_quiet
            && depth < 6
            && !is_mate_score
            && static_eval + 100 * depth.max(1) + 150 <= alpha {
            continue;
        }

        let see_threshold = if is_quiet { -56 * depth } else { -36 * depth * depth };
        if !pv_node
            && depth <= 8
            && move_count >= 1
            && !Score::is_mate(best_score)
            && !see(&board, &mv, see_threshold) {
            continue;
        }

        let mut board = *board;
        board.make(&mv);

        td.ss[ply].mv = Some(*mv);
        td.ss[ply].pc = pc;
        td.keys.push(board.hash);

        move_count += 1;
        td.nodes += 1;

        let mut extension = 0;
        if in_check {
            extension = 1;
        }

        let new_depth = depth - 1 + extension;

        let mut score = Score::MIN;
        if depth >= 3 && move_count > 3 + root_node as i32 + pv_node as i32 && is_quiet {
            let reduction = 1;

            let reduced_depth = (new_depth - reduction).max(1).min(new_depth);

            score = -alpha_beta(&board, td, reduced_depth, ply + 1, -alpha - 1, -alpha);

            if score > alpha && new_depth > reduced_depth {
                score = -alpha_beta(&board, td, new_depth, ply + 1, -alpha - 1, -alpha);
            }
        } else if !pv_node || move_count > 1 {
            score = -alpha_beta(&board, td, new_depth, ply + 1, -alpha - 1, -alpha);
        }

        if pv_node && (move_count == 1 || score > alpha) {
            score = -alpha_beta(&board, td, new_depth, ply + 1, -beta, -alpha);
        }

        if is_quiet && quiet_count < 32 {
            quiet_moves.push(*mv);
            quiet_count += 1;
        }

        td.ss[ply].mv = None;
        td.ss[ply].pc = None;
        td.keys.pop();

        if td.should_stop(Hard) { break; }

        if score > best_score {
            best_score = score;
        }

        if score > alpha {
            alpha = score;
            best_move = *mv;
            flag = TTFlag::Exact;
            if root_node {
                td.best_move = mv.clone();
            }

            if score >= beta {
                flag = TTFlag::Lower;
                break;
            }
        }
    }

    if best_move.exists() && !board.is_noisy(&best_move) {
        td.ss[ply].killer = Some(best_move);
        td.quiet_history.update(board.stm, &best_move, (120 * depth as i16 - 75).min(1200));
        if ply > 0 {
            if let Some(prev_mv) = td.ss[ply - 1].mv {
                let prev_pc = td.ss[ply - 1].pc.unwrap();
                let pc = board.piece_at(best_move.from()).unwrap();
                td.cont_history.update(prev_mv, prev_pc, best_move, pc, (120 * depth as i16 - 75).min(1200));
            }
        }
    }

    // handle checkmate / stalemate
    if move_count == 0 {
        return if in_check { -Score::MATE + ply as i32} else { Score::DRAW }
    }

    if !in_check
        && !Score::is_mate(best_score)
        && !(flag == TTFlag::Upper && best_score >= static_eval)
        && !(flag == TTFlag::Lower && best_score <= static_eval)
        && (!best_move.exists() || !board.is_noisy(&best_move)) {

        let bonus = (depth * (best_score - static_eval)).clamp(-3900, 3300);
        td.pawn_corrhist.update(board.stm, board.pawn_hash, bonus);
    }

    if !root_node {
        td.tt.insert(board.hash, &best_move, best_score, depth as u8, ply, flag);
    }

    best_score
}

fn qs(board: &Board, td: &mut ThreadData, mut alpha: i32, mut beta: i32, ply: usize) -> i32 {

    // If search is aborted, exit immediately
    if td.should_stop(Hard) { return alpha }

    if ply > 0 && is_draw(&td, &board) {
        return Score::DRAW;
    }

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
        let eval = td.nnue.evaluate(&board) + td.correction(board);
        if eval > alpha {
            alpha = eval
        }
        if alpha >= beta {
            return alpha;
        }
    }

    let filter = if in_check { MoveFilter::All } else { MoveFilter::Captures };
    let mut moves = gen_moves(board, filter);
    let scores = score(&td, &board, &moves, &tt_move, ply);
    moves.sort(&scores);
    let mut move_count = 0;

    let mut best_score = alpha;

    for mv in moves.iter() {

        if !is_legal(&board, &mv) {
            continue;
        }

        let pc = board.piece_at(mv.from());

        // SEE Pruning
        if !in_check && !see::see(&board, &mv, 0) {
            continue;
        }

        let mut board = *board;
        board.make(&mv);
        td.ss[ply].mv = Some(*mv);
        td.ss[ply].pc = pc;
        td.keys.push(board.hash);

        move_count += 1;
        td.nodes += 1;

        let score = -qs(&board, td, -beta, -alpha, ply + 1);

        td.ss[ply].mv = None;
        td.ss[ply].pc = None;
        td.keys.pop();

        if td.should_stop(Hard) { break; }

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

    if move_count == 0 && in_check {
        return -Score::MATE + ply as i32;
    }

    best_score
}

fn is_draw(td: &ThreadData, board: &Board) -> bool {
    board.is_fifty_move_rule() || board.is_insufficient_material() || td.is_repetition(&board)
}

pub struct LmrTable {
    table: [[i32; 64]; 64],
}

impl LmrTable {
    pub fn reduction(&self, depth: i32, move_count: i32) -> i32 {
        self.table[depth.min(63) as usize][move_count.min(63) as usize]
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        let mut table = [[0; 64]; 64];

        for depth in 1..64 {
            for move_count in 1..64 {
                let reduction = 820.0 + 455.0 * (depth as f32).ln() * (move_count as f32).ln();
                table[depth as usize][move_count as usize] = reduction as i32;
            }
        }

        Self { table }
    }
}

pub struct SearchStack {
    data: [StackEntry; MAX_PLY + 8],
}

#[derive(Copy, Clone)]
pub struct StackEntry {
    pub mv: Option<Move>,
    pub pc: Option<Piece>,
    pub killer: Option<Move>,
}

impl SearchStack {

    pub fn new() -> Self {
        SearchStack { data: [StackEntry { mv: None, pc: None, killer: None }; MAX_PLY + 8] }
    }

}

impl Index<usize> for SearchStack {
    type Output = StackEntry;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl IndexMut<usize> for SearchStack {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}