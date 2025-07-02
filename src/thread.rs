use std::time::Instant;

use crate::board::Board;
use crate::history::{ContinuationHistory, QuietHistory};
use crate::moves::Move;
use crate::network::NNUE;
use crate::search::{LmrTable, SearchStack};
use crate::time::{LimitType, SearchLimits};
use crate::tt::TranspositionTable;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub tt: TranspositionTable,
    pub ss: SearchStack,
    pub nnue: NNUE,
    pub keys: Vec<u64>,
    pub root_ply: usize,
    pub quiet_history: QuietHistory,
    pub cont_history: ContinuationHistory,
    pub lmr: LmrTable,
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub nodes: u64,
    pub depth: i32,
    pub best_move: Move,
    pub eval: i32,
}

impl ThreadData {

    pub fn new() -> Self {
        ThreadData {
            id: 0,
            main: true,
            tt: TranspositionTable::new(64),
            ss: SearchStack::new(),
            nnue: NNUE::new(),
            keys: Vec::new(),
            root_ply: 0,
            quiet_history: QuietHistory::new(),
            cont_history: ContinuationHistory::new(),
            lmr: LmrTable::default(),
            limits: SearchLimits::new(None, None, None, None, None),
            start_time: Instant::now(),
            nodes: 0,
            depth: 0,
            best_move: Move::NONE,
            eval: 0,
        }
    }

    pub fn with_depth_limit(depth: i32) -> Self {
        ThreadData {
            id: 0,
            main: true,
            tt: TranspositionTable::new(64),
            ss: SearchStack::new(),
            nnue: NNUE::new(),
            keys: Vec::new(),
            root_ply: 0,
            quiet_history: QuietHistory::new(),
            cont_history: ContinuationHistory::new(),
            lmr: LmrTable::default(),
            limits: SearchLimits::new(None, None, None, None, Some(depth as u64)),
            start_time: Instant::now(),
            nodes: 0,
            depth: 1,
            best_move: Move::NONE,
            eval: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ss = SearchStack::new();
        self.start_time = Instant::now();
        self.nodes = 0;
        self.depth = 1;
        self.best_move = Move::NONE;
        self.eval = 0;
    }

    pub fn is_repetition(&self, board: &Board) -> bool {

        let curr_hash = board.hash;
        let mut repetitions = 0;
        let end = self.keys.len() - board.hm as usize - 1;
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

    pub fn time(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn should_stop(&self, limit_type: LimitType) -> bool {
        match limit_type {
            LimitType::Soft => { self.soft_limit_reached() },
            LimitType::Hard => { self.hard_limit_reached() },
        }
    }

    pub fn soft_limit_reached(&self) -> bool {
        if let Some(soft_time) = self.limits.soft_time {
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


#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::moves::Move;
    use crate::thread::ThreadData;

    #[test]
    fn test_twofold_rep_after_root() {

        let mut td = ThreadData::new();
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

        let mut td = ThreadData::new();
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

        let mut td = ThreadData::new();
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