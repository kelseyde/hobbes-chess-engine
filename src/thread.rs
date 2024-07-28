use std::time::Instant;
use crate::moves::Move;
use crate::tt::TT;

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub tt: TT,
    pub time: Instant,
    pub time_limit: Instant,
    pub best_move: Move
}

impl ThreadData {

    pub fn time(&self) -> u128 {
        self.time.elapsed().as_millis()
    }

    pub fn cancelled(&self) -> bool {
        self.time() >= self.time_limit.elapsed().as_millis()
    }

}