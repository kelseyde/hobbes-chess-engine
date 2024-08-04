use std::time::{Duration, Instant};
use crate::board::Board;
use crate::moves::Move;
use crate::tt::TT;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub board_history: Vec<Board>,
    pub tt: TT,
    pub time: Instant,
    pub time_limit: Duration,
    pub nodes: u64,
    pub node_limit: u64,
    pub depth: u8,
    pub depth_limit: u8,
    pub best_move: Move,
    pub eval: i32,
    pub cancelled: bool,
}

impl ThreadData {

    pub fn new(tt: TT) -> Self {
        ThreadData {
            id: 0,
            main: true,
            board_history: Vec::new(),
            tt,
            time: Instant::now(),
            time_limit: Duration::MAX,
            nodes: 0,
            node_limit: 0,
            depth: 0,
            depth_limit: 0,
            best_move: Move::NONE,
            eval: 0,
            cancelled: false,
        }
    }

    pub fn time(&self) -> u128 {
        self.time.elapsed().as_millis()
    }

    pub fn abort(&self) -> bool {
        self.cancelled || self.time.elapsed() >= self.time_limit
    }

}