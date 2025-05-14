use std::time::{Duration, Instant};

use crate::board::Board;
use crate::evaluate::Evaluator;
use crate::moves::Move;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub board_history: Vec<Board>,
    pub evaluator: Evaluator,
    pub time: Instant,
    pub time_limit: Duration,
    pub nodes: u64,
    pub node_limit: u64,
    pub depth: u8,
    pub depth_limit: u8,
    pub best_move: Move,
    pub eval: i32,
}

impl ThreadData {

    pub fn new() -> Self {
        ThreadData {
            id: 0,
            main: true,
            board_history: Vec::new(),
            evaluator: Evaluator::new(),
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

    pub fn with_depth_limit(depth: u8) -> Self {
        ThreadData {
            id: 0,
            main: true,
            board_history: Vec::new(),
            evaluator: Evaluator::new(),
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