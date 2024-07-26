use crate::magics::init_magics;

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

fn main() {
    init_magics();
}


