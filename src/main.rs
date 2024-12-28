mod attacks;
mod bench;
mod bits;
mod board;
mod consts;
mod datagen;
mod fen;
mod magics;
mod movegen;
mod moves;
mod network;
mod perft;
mod search;
mod thread;
mod time;
mod tt;
mod uci;
mod zobrist;


fn main() {
    uci::UCI::new().run();
}


