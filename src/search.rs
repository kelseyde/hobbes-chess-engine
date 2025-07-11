use crate::board::Board;
use crate::movegen::MoveFilter;
use crate::movepicker::{MovePicker, Stage};
use crate::moves::Move;
use crate::see::see;
use crate::thread::ThreadData;
use crate::time::LimitType::{Hard, Soft};
use crate::tt::TTFlag;
use crate::tt::TTFlag::{Lower, Upper};
use crate::types::piece::Piece;
use crate::{movegen, see};
use arrayvec::ArrayVec;
use std::ops::{Index, IndexMut};
use std::time::Instant;
use TTFlag::Exact;

pub const MAX_PLY: usize = 256;

pub fn search(board: &Board, td: &mut ThreadData) -> (Move, i32) {
    td.start_time = Instant::now();
    td.best_move = Move::NONE;
    td.nnue.activate(board);

    let mut alpha = Score::MIN;
    let mut beta = Score::MAX;
    let mut score = 0;
    let mut delta = 24;

    // Iterative Deepening
    while td.depth < MAX_DEPTH && !td.should_stop(Soft) {

        // Aspiration Windows
        if td.depth >= 4 {
            alpha = (score - delta).max(Score::MIN);
            beta = (score + delta).min(Score::MAX);
        }

        loop {
            score = alpha_beta(board, td, td.depth, 0, alpha, beta, false);

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

#[rustfmt::skip]
fn alpha_beta(board: &Board, td: &mut ThreadData, mut depth: i32, ply: usize, mut alpha: i32, beta: i32, cut_node: bool) -> i32 {

    // If search is aborted, exit immediately
    if td.should_stop(Hard) {
        return alpha;
    }

    let threats = movegen::calc_threats(board, board.stm);
    let in_check = threats.contains(board.king_sq(board.stm));

    // If depth is reached, drop into quiescence search
    if depth <= 0 && !in_check {
        return qs(board, td, alpha, beta, ply);
    }

    if depth < 0 {
        depth = 0;
    }

    if ply > 0 && is_draw(td, board) {
        return Score::DRAW;
    }

    if ply >= MAX_PLY {
        return td.nnue.evaluate(board);
    }

    let root_node = ply == 0;
    let pv_node = beta - alpha > 1;

    let singular = td.ss[ply].singular;
    let singular_search = singular.is_some();

    let mut tt_hit = false;
    let mut tt_move = Move::NONE;
    let mut tt_score = Score::MIN;
    let mut tt_flag = Lower;
    let mut tt_depth = 0;

    // Transposition Table probe
    if !singular_search {
        if let Some(entry) = td.tt.probe(board.hash) {
            tt_hit = true;
            tt_score = entry.score(ply) as i32;
            tt_depth = entry.depth() as i32;
            tt_flag = entry.flag();
            if can_use_tt_move(board, &entry.best_move()) {
                tt_move = entry.best_move();
            }

            if !root_node
                && tt_depth >= depth
                && bounds_match(entry.flag(), tt_score, alpha, beta) {
                return tt_score;
            }

        }
    }

    let mut static_eval = Score::MIN;

    // Static Evaluation
    if !in_check {
        static_eval = td.nnue.evaluate(board) + td.correction(board, ply);
    };

    td.ss[ply].static_eval = Some(static_eval);

    let improving = is_improving(td, ply, static_eval);

    if !root_node && !pv_node && !in_check && !singular_search{

        // Reverse Futility Pruning
        if depth <= 8 && static_eval - 80 * (depth - improving as i32) >= beta {
            return beta + (static_eval - beta) / 3;
        }

        // Null Move Pruning
        if depth >= 3 && static_eval >= beta && board.has_non_pawns() {
            let r = 3 + depth / 3 + ((static_eval - beta) / 210).min(4);
            let mut board = *board;
            board.make_null_move();
            td.nodes += 1;
            td.keys.push(board.hash);
            let score = -alpha_beta(&board, td, depth - r, ply + 1, -beta, -beta + 1, !cut_node);
            td.keys.pop();

            if score >= beta {
                return score;
            }
        }

    }

    // Internal Iterative Reductions
    if !root_node
        && depth >= 5
        && (pv_node || cut_node)
        && (!tt_hit || tt_move.is_null() || tt_depth < depth - 4) {
        depth -= 1;
    }

    let mut move_picker = MovePicker::new(tt_move, ply, threats);

    let mut legal_moves = 0;
    let mut searched_moves = 0;
    let mut quiet_count = 0;
    let mut capture_count = 0;
    let mut best_score = Score::MIN;
    let mut best_move = Move::NONE;
    let mut flag = TTFlag::Upper;

    let mut quiets = ArrayVec::<Move, 32>::new();
    let mut captures = ArrayVec::<Move, 32>::new();

    while let Some(mv) = move_picker.next(board, td) {

        if !board.is_legal(&mv) {
            continue;
        }

        legal_moves += 1;

        if singular.is_some_and(|s| s == mv) {
            continue;
        }

        let pc = board.piece_at(mv.from()).unwrap();
        let captured = board.captured(&mv);
        let is_quiet = captured.is_none();
        let is_mate_score = Score::is_mate(best_score);
        let history_score = td.history_score(board, &mv, ply, threats, pc, captured);
        let base_reduction = td.lmr.reduction(depth, legal_moves);
        let lmr_depth = depth.saturating_sub(base_reduction);

        let mut extension = 0;
        if in_check {
            extension = 1;
        }

        // Futility Pruning
        if !pv_node
            && !root_node
            && !in_check
            && is_quiet
            && lmr_depth < 6
            && !is_mate_score
            && static_eval + 100 * lmr_depth + 150 <= alpha {
            move_picker.skip_quiets = true;
            continue;
        }

        // Late Move Pruning
        if !pv_node
            && !root_node
            && !is_mate_score
            && is_quiet
            && depth <= 8
            && searched_moves > late_move_threshold(depth, improving) {
            move_picker.skip_quiets = true;
            continue;
        }

        // History Pruning
        if !pv_node
            && !root_node
            && !in_check
            && !is_mate_score
            && is_quiet
            && depth <= 4
            && history_score < -2048 * depth * depth {
            move_picker.skip_quiets = true;
            continue
        }

        // Bad Noisy Pruning
        let futility_margin = static_eval + 128 * lmr_depth;
        if !pv_node
            && !in_check
            && lmr_depth < 6
            && move_picker.stage == Stage::BadNoisies
            && futility_margin <= alpha {
            break;
        }

        // SEE Pruning
        let see_threshold = if is_quiet { -56 * depth } else { -36 * depth * depth };
        if !pv_node
            && depth <= 8
            && searched_moves >= 1
            && !Score::is_mate(best_score)
            && !see(board, &mv, see_threshold) {
            continue;
        }

        // Singular Extensions
        if !root_node
            && !singular_search
            && tt_hit
            && mv == tt_move
            && depth >= 8
            && tt_flag != Upper
            && tt_depth >= depth - 3 {

            let s_beta = (tt_score - depth * 32 / 16).max(-Score::MATE + 1);
            let s_depth = (depth - 1) / 2;

            td.ss[ply].singular = Some(mv);
            let score = alpha_beta(&board, td, s_depth, ply, s_beta - 1, s_beta, cut_node);
            td.ss[ply].singular = None;

            if score < s_beta {
                extension = 1;
                extension += (!pv_node && score < s_beta - 20) as i32;
            } else if tt_score >= beta {
                extension = -1;
            }

        }

        let mut board = *board;
        td.nnue.update(&mv, pc, captured, &board);
        board.make(&mv);

        td.ss[ply].mv = Some(mv);
        td.ss[ply].pc = Some(pc);
        td.keys.push(board.hash);

        searched_moves += 1;
        td.nodes += 1;

        let initial_nodes = td.nodes;
        let new_depth = depth - 1 + extension;

        let mut score = Score::MIN;

        // Principal Variation Search
        if depth >= 3 && searched_moves > 3 + root_node as i32 + pv_node as i32 && is_quiet {
            // Late Move Reductions
            let mut reduction = base_reduction;
            reduction += cut_node as i32;
            reduction += !improving as i32;
            if is_quiet {
                reduction -= (history_score - 512) / 16384;
            }

            let reduced_depth = (new_depth - reduction).clamp(1, new_depth);

            // Reduced-depth search
            score = -alpha_beta(&board, td, reduced_depth, ply + 1, -alpha - 1, -alpha, true);

            // Re-search if we reduced depth and score beat alpha
            if score > alpha && new_depth > reduced_depth {
                score = -alpha_beta(&board, td, new_depth, ply + 1, -alpha - 1, -alpha, !cut_node);
            }
        } else if !pv_node || searched_moves > 1 {
            score = -alpha_beta(&board, td, new_depth, ply + 1, -alpha - 1, -alpha, !cut_node);
        }

        if pv_node && (searched_moves == 1 || score > alpha) {
            score = -alpha_beta(&board, td, new_depth, ply + 1, -beta, -alpha, false);
        }

        if is_quiet && quiet_count < 32 {
            quiets.push(mv);
            quiet_count += 1;
        } else if captured.is_some() && capture_count < 32 {
            captures.push(mv);
            capture_count += 1;
        }

        td.ss[ply].mv = None;
        td.ss[ply].pc = None;
        td.keys.pop();
        td.nnue.undo();

        if root_node {
            td.node_table.add(&mv, td.nodes - initial_nodes);
        }

        if td.should_stop(Hard) {
            break;
        }

        if score > best_score {
            best_score = score;
        }

        if score > alpha {
            alpha = score;
            best_move = mv;
            flag = Exact;
            if root_node {
                td.best_move = mv;
            }

            if score >= beta {
                flag = TTFlag::Lower;
                break;
            }
        }
    }

    // Update history tables
    if best_move.exists() {
        let pc = board.piece_at(best_move.from()).unwrap();

        let quiet_bonus = (120 * depth as i16 - 75).min(1200);
        let quiet_malus = (120 * depth as i16 - 75).min(1200);

        let capt_bonus = (120 * depth as i16 - 75).min(1200);
        let capt_malus = (120 * depth as i16 - 75).min(1200);

        let cont_bonus = (120 * depth as i16 - 75).min(1200);
        let cont_malus = (120 * depth as i16 - 75).min(1200);

        if let Some(captured) = board.captured(&best_move) {
            td.capture_history.update(board.stm, pc, best_move.to(), captured, capt_bonus);
        } else {
            td.ss[ply].killer = Some(best_move);

            td.quiet_history.update(board.stm, &best_move, threats, quiet_bonus);
            update_continuation_history(td, ply, &best_move, pc, cont_bonus);

            for mv in quiets.iter() {
                if mv != &best_move {
                    td.quiet_history.update(board.stm, mv, threats, -quiet_malus);
                    update_continuation_history(td, ply, mv, pc, -cont_malus);
                }
            }
        }

        for mv in captures.iter() {
            if mv != &best_move {
                if let Some(captured) = board.captured(mv) {
                    td.capture_history.update(board.stm, pc, mv.to(), captured, -capt_malus);
                }
            }
        }
    }

    // Handle checkmate / stalemate
    if legal_moves == 0 {
        return if singular_search {
            alpha
        } else if in_check {
            -Score::MATE + ply as i32
        } else {
            Score::DRAW
        };
    }

    // Update static eval correction history
    if !in_check
        && !singular_search
        && !Score::is_mate(best_score)
        && bounds_match(flag, best_score, static_eval, static_eval)
        && (!best_move.exists() || !board.is_noisy(&best_move)) {
        td.update_correction_history(board, depth, ply, static_eval, best_score);
    }

    // Write to transposition table
    if !singular_search && !td.hard_limit_reached(){
        td.tt.insert(board.hash, best_move, best_score, depth as u8, ply, flag);
    }

    best_score
}

fn qs(board: &Board, td: &mut ThreadData, mut alpha: i32, beta: i32, ply: usize) -> i32 {

    // If search is aborted, exit immediately
    if td.should_stop(Hard) {
        return alpha;
    }

    if ply > 0 && is_draw(td, board) {
        return Score::DRAW;
    }

    if ply >= MAX_PLY {
        return td.nnue.evaluate(board);
    }

    let threats = movegen::calc_threats(board, board.stm);
    let in_check = threats.contains(board.king_sq(board.stm));

    let tt_entry = td.tt.probe(board.hash);
    let mut tt_move = Move::NONE;
    if let Some(entry) = tt_entry {
        if can_use_tt_move(board, &entry.best_move()) {
            tt_move = entry.best_move();
        }
        let score = entry.score(ply) as i32;

        if bounds_match(entry.flag(), score, alpha, beta) {
            return score;
        }
    }

    if !in_check {
        let static_eval = td.nnue.evaluate(board) + td.correction(board, ply);

        if static_eval > alpha {
            alpha = static_eval
        }
        if alpha >= beta {
            return alpha;
        }
    }

    let filter = if in_check {
        MoveFilter::All
    } else {
        MoveFilter::Captures
    };
    let mut move_picker = MovePicker::new_qsearch(tt_move, filter, ply, threats);

    let mut move_count = 0;

    let mut best_score = alpha;

    while let Some(mv) = move_picker.next(board, td) {

        if !board.is_legal(&mv) {
            continue;
        }

        let pc = board.piece_at(mv.from()).unwrap();
        let captured = board.captured(&mv);

        // SEE Pruning
        if !in_check && !see::see(&board, &mv, 0) {
            continue;
        }

        let mut board = *board;
        td.nnue.update(&mv, pc, captured, &board);

        board.make(&mv);
        td.ss[ply].mv = Some(mv);
        td.ss[ply].pc = Some(pc);
        td.keys.push(board.hash);

        move_count += 1;
        td.nodes += 1;

        let score = -qs(&board, td, -beta, -alpha, ply + 1);

        td.ss[ply].mv = None;
        td.ss[ply].pc = None;
        td.keys.pop();
        td.nnue.undo();

        if td.should_stop(Hard) {
            break;
        }

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
    board.is_fifty_move_rule() || board.is_insufficient_material() || td.is_repetition(board)
}

fn is_improving(td: &ThreadData, ply: usize, static_eval: i32) -> bool {
    if static_eval == Score::MIN {
        return false;
    }
    if ply > 1 {
        if let Some(prev_eval) = td.ss[ply - 2]
            .static_eval
            .filter(|eval| *eval != Score::MIN) {
            return static_eval > prev_eval;
        }
    }
    if ply > 3 {
        if let Some(prev_eval) = td.ss[ply - 4]
            .static_eval
            .filter(|eval| *eval != Score::MIN) {
            return static_eval > prev_eval;
        }
    }
    true
}

fn late_move_threshold(depth: i32, improving: bool) -> i32 {
    let base = if improving { 3 } else { 1 };
    let scale = if improving { 87 } else { 39 };
    (base + depth * scale) / 10
}

fn update_continuation_history(td: &mut ThreadData, ply: usize, mv: &Move, pc: Piece, bonus: i16) {
    for &prev_ply in &[1, 2] {
        if ply >= prev_ply {
            if let (Some(prev_mv), Some(prev_pc)) = (td.ss[ply - prev_ply].mv, td.ss[ply - prev_ply].pc) {
                td.cont_history.update(&prev_mv, prev_pc, mv, pc, bonus);
            }
        }
    }
}

pub struct LmrTable {
    table: [[i32; 64]; 256],
}

impl LmrTable {
    pub fn reduction(&self, depth: i32, move_count: i32) -> i32 {
        self.table[depth.min(255) as usize][move_count.min(63) as usize]
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        let base = 0.92;
        let divisor = 3.11;

        let mut table = [[0; 64]; 256];

        for depth in 1..256 {
            for move_count in 1..64 {
                let ln_depth = (depth as f32).ln();
                let ln_move_count = (move_count as f32).ln();
                let reduction = (base + (ln_depth * ln_move_count / divisor)) as i32;
                table[depth as usize][move_count as usize] = reduction;
            }
        }

        Self { table }
    }
}

fn bounds_match(flag: TTFlag, score: i32, lower: i32, upper: i32) -> bool {
    match flag {
        Exact => true,
        Lower => score >= upper,
        Upper => score <= lower,
    }
}

fn can_use_tt_move(board: &Board, tt_move: &Move) -> bool {
    tt_move.exists() && board.is_pseudo_legal(tt_move) && board.is_legal(tt_move)
}

pub struct SearchStack {
    data: [StackEntry; MAX_PLY + 8],
}

#[derive(Copy, Clone)]
pub struct StackEntry {
    pub mv: Option<Move>,
    pub pc: Option<Piece>,
    pub killer: Option<Move>,
    pub singular: Option<Move>,
    pub static_eval: Option<i32>,
}

impl Default for SearchStack {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchStack {
    pub fn new() -> Self {
        SearchStack {
            data: [StackEntry {
                mv: None,
                pc: None,
                killer: None,
                static_eval: None,
                singular: None,
            }; MAX_PLY + 8],
        }
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

pub const MAX_DEPTH: i32 = 255;

pub struct Score {}

impl Score {
    pub const DRAW: i32 = 0;
    pub const MAX: i32 = 32767;
    pub const MIN: i32 = -32767;
    pub const MATE: i32 = 32766;

    pub fn is_mate(score: i32) -> bool {
        score.abs() >= Score::MATE - MAX_DEPTH
    }
}
