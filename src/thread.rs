use std::time::{Duration, Instant};

use crate::board::Board;
use crate::history::QuietHistory;
use crate::moves::Move;
use crate::network::NNUE;
use crate::search::LmrTable;
use crate::tt::TranspositionTable;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub tt: TranspositionTable,
    pub nnue: NNUE,
    pub board_history: Vec<Board>,
    pub quiet_history: QuietHistory,
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
            nnue: NNUE::new(),
            board_history: Vec::new(),
            quiet_history: QuietHistory::new(),
            lmr: LmrTable::default(),
            time: Instant::now(),
            time_limit: Duration::MAX,
            nodes: 0,
            node_limit: 0,
            depth: 0,
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
            nnue: NNUE::new(),
            board_history: Vec::new(),
            quiet_history: QuietHistory::new(),
            lmr: LmrTable::default(),
            time: Instant::now(),
            time_limit: Duration::MAX,
            nodes: 0,
            node_limit: 0,
            depth: 0,
            depth_limit: depth,
            best_move: Move::NONE,
            eval: 0,
        }
    }

    pub fn reset(&mut self) {
        self.time = Instant::now();
        self.time_limit = Duration::MAX;
        self.nodes = 0;
        self.node_limit = 0;
        self.depth = 0;
        self.depth_limit = 0;
        self.best_move = Move::NONE;
        self.eval = 0;
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