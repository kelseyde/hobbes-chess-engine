use std::time::{Duration, Instant};
use crate::board::Board;
use crate::moves::Move;
use crate::thread::ThreadData;

pub fn search(board: &Board, td: &mut ThreadData, time: Instant) {

}

fn alpha_beta(board: &Board, td: &mut ThreadData, depth: u8, alpha: i16, beta: i16) -> Move {
    Move(0)
}

fn quiesce(board: &Board, td: &mut ThreadData, alpha: i16, beta: i16) -> Move {
    Move(0)
}