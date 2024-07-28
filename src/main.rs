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

fn main() {
    uci::run();
}


