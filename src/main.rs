use crate::uci::UCI;

mod board;
mod movegen;
mod moves;
mod consts;
mod bits;
mod zobrist;
mod fen;
mod attacks;
mod magics;
mod perft;
mod search;
mod tt;
mod network;
mod thread;
mod uci;
mod eval;

fn main() {
    UCI::new().run();
}


