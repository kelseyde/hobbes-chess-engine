use std::time::{Duration, Instant};

use crate::board::Board;
use crate::history::{ContinuationHistory, QuietHistory};
use crate::moves::Move;
use crate::network::NNUE;
use crate::search::{LmrTable, SearchStack};
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
    pub time: Instant,
    pub time_limit: Duration,
    pub nodes: u64,
    pub node_limit: u64,
    pub depth: i32,
    pub depth_limit: i32,
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
            time: Instant::now(),
            time_limit: Duration::MAX,
            nodes: 0,
            node_limit: 0,
            depth: 1,
            depth_limit: 0,
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
            time: Instant::now(),
            time_limit: Duration::MAX,
            nodes: 0,
            node_limit: 0,
            depth: 1,
            depth_limit: depth,
            best_move: Move::NONE,
            eval: 0,
        }
    }

    pub fn reset(&mut self) {
        self.ss = SearchStack::new();
        self.time = Instant::now();
        self.time_limit = Duration::MAX;
        self.nodes = 0;
        self.node_limit = 0;
        self.depth = 1;
        self.depth_limit = 0;
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
        self.time.elapsed().as_millis()
    }

    pub fn abort(&self) -> bool {
        self.time.elapsed() >= self.time_limit
            || self.node_limit > 0 && self.nodes >= self.node_limit
            || self.depth_limit > 0 && self.depth >= self.depth_limit
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