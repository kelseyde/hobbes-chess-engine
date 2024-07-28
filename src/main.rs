mod board;
mod movegen;
mod moves;
mod piece;
mod bits;
mod zobrist;
mod fen;
mod attacks;
mod magics;
mod perft;
mod search;
mod tt;
mod nnue;
mod thread;
mod uci;

fn main() {
    uci::run();
}


