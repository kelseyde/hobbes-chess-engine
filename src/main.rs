use crate::uci::UCI;

pub mod attacks;
pub mod bench;
pub mod board;
pub mod fen;
pub mod magics;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod search;
pub mod thread;
pub mod uci;
pub mod zobrist;
pub mod tt;
pub mod history;
pub mod see;
pub mod types;
pub mod network;
pub mod time;
pub mod movepicker;
pub mod nnue;
pub mod simd;
pub mod parameters;
pub mod utils;
mod correction;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);
}


