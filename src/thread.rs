use std::time::{Duration, Instant};
use crate::moves::Move;
use crate::tt::TT;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub tt: TT,
    pub time: Instant,
    pub time_limit: Duration,
    pub best_move: Move,
    pub cancelled: bool,
}

impl ThreadData {

    pub fn new(tt: TT) -> Self {
        ThreadData {
            id: 0,
            main: true,
            tt,
            time: Instant::now(),
            time_limit: Duration::MAX,
            best_move: Move::NONE,
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