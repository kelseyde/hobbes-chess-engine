use crate::uci::UCI;

mod attacks;
mod bench;
mod bits;
mod board;
mod consts;
mod fen;
mod magics;
mod movegen;
mod moves;
mod perft;
mod search;
mod thread;
mod uci;
mod zobrist;
mod evaluate;
mod ordering;
mod tt;
mod history;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);
}


