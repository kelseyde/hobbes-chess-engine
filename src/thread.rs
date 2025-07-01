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
        self.keys.clear();
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
        for &hash in self.keys
            .iter()
            .rev()
            .skip(3)
            .step_by(2) {
            repetitions += u8::from(hash == curr_hash);
            if repetitions == 1 {
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
    fn test_repetition() {

        let mut td = ThreadData::new();
        let mut board = Board::new();
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("g1f3"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("g8f6"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("f3g1"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("f6g8"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("g1f3"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("g8f6"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("f3g1"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("f6g8"));
        td.keys.push(board.hash);
        assert!(!td.is_repetition(&board));

        board.make(&Move::parse_uci("f3g1"));
        td.keys.push(board.hash);
        assert!(td.is_repetition(&board));
    }

}