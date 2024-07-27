use std::time::Instant;

use crate::board::Board;
use crate::perft::perft;

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

fn main() {

    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let board = Board::from_fen(fen);
    println!("Starting perft");
    let start = Instant::now();
    let nodes = perft(&board, 4, 4, false);
    println!("Nodes: {}, Time: {}", nodes, start.elapsed().as_millis());

}


