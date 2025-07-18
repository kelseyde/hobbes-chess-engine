use std::time::Instant;

use crate::board::Board;
use crate::history::{CaptureHistory, ContinuationHistory, CorrectionHistory, QuietHistory};
use crate::moves::Move;
use crate::network::NNUE;
use crate::parameters::{corr_counter_weight, corr_follow_up_weight, corr_major_weight, corr_minor_weight, corr_non_pawn_weight, corr_pawn_weight};
use crate::search::{LmrTable, Score, SearchStack};
use crate::time::{LimitType, SearchLimits};
use crate::tt::TranspositionTable;
use crate::types::bitboard::Bitboard;
use crate::types::piece::Piece;
use crate::types::side::Side;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub tt: TranspositionTable,
    pub ss: SearchStack,
    pub nnue: NNUE,
    pub keys: Vec<u64>,
    pub root_ply: usize,
    pub quiet_history: QuietHistory,
    pub capture_history: CaptureHistory,
    pub cont_history: ContinuationHistory,
    pub pawn_corrhist: CorrectionHistory,
    pub nonpawn_corrhist: [CorrectionHistory; 2],
    pub countermove_corrhist: CorrectionHistory,
    pub follow_up_move_corrhist: CorrectionHistory,
    pub major_corrhist: CorrectionHistory,
    pub minor_corrhist: CorrectionHistory,
    pub lmr: LmrTable,
    pub node_table: NodeTable,
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub nodes: u64,
    pub depth: i32,
    pub best_move: Move,
    pub best_score: i32,
}

impl Default for ThreadData {
    fn default() -> Self {
        ThreadData {
            id: 0,
            main: true,
            tt: TranspositionTable::new(64),
            ss: SearchStack::new(),
            nnue: NNUE::default(),
            keys: Vec::new(),
            root_ply: 0,
            quiet_history: QuietHistory::default(),
            capture_history: CaptureHistory::default(),
            cont_history: ContinuationHistory::default(),
            pawn_corrhist: CorrectionHistory::default(),
            nonpawn_corrhist: [CorrectionHistory::default(), CorrectionHistory::default()],
            countermove_corrhist: CorrectionHistory::default(),
            follow_up_move_corrhist: CorrectionHistory::default(),
            major_corrhist: CorrectionHistory::default(),
            minor_corrhist: CorrectionHistory::default(),
            lmr: LmrTable::default(),
            node_table: NodeTable::new(),
            limits: SearchLimits::new(None, None, None, None, None),
            start_time: Instant::now(),
            nodes: 0,
            depth: 0,
            best_move: Move::NONE,
            best_score: Score::MIN,
        }
    }
}

impl ThreadData {

    pub fn reset(&mut self) {
        self.ss = SearchStack::new();
        self.node_table.clear();
        self.nodes = 0;
        self.depth = 1;
        self.best_move = Move::NONE;
        self.best_score = 0;
    }

    pub fn clear(&mut self) {
        self.tt.clear();
        self.keys.clear();
        self.root_ply = 0;
        self.quiet_history.clear();
        self.capture_history.clear();
        self.cont_history.clear();
        self.pawn_corrhist.clear();
        self.nonpawn_corrhist[Side::White].clear();
        self.nonpawn_corrhist[Side::Black].clear();
        self.countermove_corrhist.clear();
        self.follow_up_move_corrhist.clear();
        self.major_corrhist.clear();
        self.minor_corrhist.clear();
    }

    pub fn is_repetition(&self, board: &Board) -> bool {
        let curr_hash = board.hash;
        let mut repetitions = 0;
        let end = self.keys.len().saturating_sub(board.hm as usize + 1);
        for ply in (end..self.keys.len().saturating_sub(2)).rev() {
            let hash = self.keys[ply];
            repetitions += u8::from(curr_hash == hash);

            // Two-fold repetition of positions within the search tree
            if repetitions == 1 && ply >= self.root_ply {
                return true;
            }

            // Three-fold repetition including positions before search root
            if repetitions == 2 {
                return true;
            }
        }
        false
    }

    pub fn update_correction_history(&mut self, board: &Board, depth: i32, ply: usize, static_eval: i32, best_score: i32) {
        let us = board.stm;
        let pawn_hash = board.pawn_hash;
        let w_nonpawn_hash = board.non_pawn_hashes[Side::White];
        let b_nonpawn_hash = board.non_pawn_hashes[Side::Black];

        self.pawn_corrhist.update(us, pawn_hash, depth, static_eval, best_score);
        self.nonpawn_corrhist[Side::White].update(us, w_nonpawn_hash, depth, static_eval, best_score);
        self.nonpawn_corrhist[Side::Black].update(us, b_nonpawn_hash, depth, static_eval, best_score);
        self.major_corrhist.update(us, board.major_hash, depth, static_eval, best_score);
        self.minor_corrhist.update(us, board.minor_hash, depth, static_eval, best_score);

        if ply >= 1 {
            if let Some(prev_mv) = self.ss[ply - 1].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                self.countermove_corrhist.update(board.stm, encoded_mv, depth, static_eval, best_score);
            }
        }
        if ply >= 2 {
            if let Some(prev_mv) = self.ss[ply - 2].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                self.follow_up_move_corrhist.update(board.stm, encoded_mv, depth, static_eval, best_score);
            }
        }
    }

    #[rustfmt::skip]
    pub fn correction(&self, board: &Board, ply: usize) -> i32 {

        let pawn       = self.pawn_corrhist.get(board.stm, board.pawn_hash);
        let white      = self.nonpawn_corrhist[Side::White].get(board.stm, board.non_pawn_hashes[Side::White]);
        let black      = self.nonpawn_corrhist[Side::Black].get(board.stm, board.non_pawn_hashes[Side::Black]);
        let major      = self.major_corrhist.get(board.stm, board.major_hash);
        let minor      = self.minor_corrhist.get(board.stm, board.minor_hash);
        let counter    = self.countermove_correction(board, ply);
        let follow_up  = self.follow_up_move_correction(board, ply);

        (pawn * 100 / corr_pawn_weight())
            + (white * 100 / corr_non_pawn_weight())
            + (black * 100 / corr_non_pawn_weight())
            + (major * 100 / corr_major_weight())
            + (minor * 100 / corr_minor_weight())
            + (counter * 100 / corr_counter_weight())
            + (follow_up * 100 / corr_follow_up_weight())

    }

    fn countermove_correction(&self, board: &Board, ply: usize) -> i32 {
        if ply >= 1 {
            if let Some(prev_mv) = self.ss[ply - 1].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                return self.countermove_corrhist.get(board.stm, encoded_mv);
            }
        }
        0
    }

    fn follow_up_move_correction(&self, board: &Board, ply: usize) -> i32 {
        if ply >= 2 {
            if let Some(prev_mv) = self.ss[ply - 2].mv {
                let encoded_mv = prev_mv.encoded() as u64;
                return self.follow_up_move_corrhist.get(board.stm, encoded_mv);
            }
        }
        0
    }

    pub fn history_score(&self, board: &Board, mv: &Move, ply: usize, threats: Bitboard, pc: Piece, captured: Option<Piece>) -> i32 {
        if let Some(captured) = captured {
            self.capture_history_score(board, mv, pc, captured)
        } else {
            self.quiet_history_score(board, mv, ply, threats)
        }
    }

    pub fn quiet_history_score(&self, board: &Board, mv: &Move, ply: usize, threats: Bitboard) -> i32 {
        let pc = board.piece_at(mv.from()).unwrap();
        let quiet_score = self.quiet_history.get(board.stm, *mv, threats) as i32;
        let mut cont_score = 0;
        for &prev_ply in &[1, 2] {
            if ply >= prev_ply  {
                if let (Some(prev_mv), Some(prev_pc)) = (self.ss[ply - prev_ply].mv, self.ss[ply - prev_ply].pc) {
                    cont_score += self.cont_history.get(prev_mv, prev_pc, mv, pc) as i32;
                }
            }
        }
        quiet_score + cont_score
    }

    pub fn capture_history_score(&self, board: &Board, mv: &Move, pc: Piece, captured: Piece) -> i32 {
        self.capture_history.get(board.stm, pc, mv.to(), captured) as i32
    }

    pub fn time(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn should_stop(&self, limit_type: LimitType) -> bool {
        if self.depth <= 1 {
            // Always clear the first depth, to ensure at least one legal move
            return false;
        }
        match limit_type {
            LimitType::Soft => self.soft_limit_reached(),
            LimitType::Hard => self.hard_limit_reached(),
        }
    }

    pub fn soft_limit_reached(&self) -> bool {
        let best_move_nodes = self.node_table.get(&self.best_move);

        if let Some(soft_time) = self.limits.scaled_soft_limit(self.depth, self.nodes, best_move_nodes) {
            if self.start_time.elapsed() >= soft_time {
                return true;
            }
        }

        if let Some(soft_nodes) = self.limits.soft_nodes {
            if self.nodes >= soft_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = self.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }

    pub fn hard_limit_reached(&self) -> bool {
        if let Some(hard_time) = self.limits.hard_time {
            if self.start_time.elapsed() >= hard_time {
                return true;
            }
        }

        if let Some(hard_nodes) = self.limits.hard_nodes {
            if self.nodes >= hard_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = self.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }
}

pub struct NodeTable {
    table: [[u64; 64]; 64],
}

impl NodeTable {

    pub fn new() -> Self {
        NodeTable { table: [[0; 64]; 64] }
    }

    pub fn add(&mut self, mv: &Move, nodes: u64) {
        self.table[mv.from()][mv.to()] += nodes;
    }

    pub fn get(&self, mv: &Move) -> u64 {
        self.table[mv.from()][mv.to()]
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }
}


#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::moves::Move;
    use crate::thread::ThreadData;

    #[test]
    fn test_twofold_rep_after_root() {
        let mut td = ThreadData::default();
        let mut board = Board::new();
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g1f3");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g8f6");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f3g1");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f6g8");
        assert!(td.is_repetition(&board));
    }

    #[test]
    fn test_twofold_rep_before_root() {
        let mut td = ThreadData::default();
        let mut board = Board::new();
        td.root_ply = 3;
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g1f3");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g8f6");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f3g1");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f6g8");
        assert!(!td.is_repetition(&board));
    }

    #[test]
    fn test_threefold_rep_before_root() {
        let mut td = ThreadData::default();
        let mut board = Board::new();
        td.root_ply = 7;
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g1f3");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g8f6");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f3g1");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f6g8");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g1f3");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "g8f6");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f3g1");
        assert!(!td.is_repetition(&board));

        make_move(&mut td, &mut board, "f6g8");
        assert!(td.is_repetition(&board));
    }

    fn make_move(td: &mut ThreadData, board: &mut Board, mv: &str) {
        let mv = Move::parse_uci(mv);
        board.make(&mv);
        td.keys.push(board.hash);
    }
}
