use crate::board::{cuckoo, magics, Board};
use crate::evaluation::feature::threat;
use crate::tools::uci::UCI;
use board::ray;
use crate::evaluation::NNUE;

pub const AUTHOR: &str = "Dan Kelsey";
pub const CONTRIBUTORS: &str = "Jonathan Hallström, Mattia Giambirtone";
pub const VERSION: &str = "3.0";

/// The board module contains board representation, move generation, move legality checking, and
/// everything related to the rules of chess.
pub mod board;

/// The evaluation module contains everything required to interact with the NNUE (Efficiently
/// Updatable Neural Network), including accumulators, bucket caches, and SIMD operations.
pub mod evaluation;

/// The search module contains the search algorithm, move ordering heuristics, transposition and
/// history tables, and everything required to traverse the game tree.
pub mod search;

/// The tools module contains various utilities not strictly related to the engine itself, including
/// perft, datagen, fen and scharnagl parsing, and UCI (Universal Chess Interface) support.
pub mod tools;

fn main() {
    // Initialise static data
    magics::init();
    ray::init();
    cuckoo::init();
    threat::init();

    let mut nnue = NNUE::default();
    for fen in [
        // "b6n/B1k5/8/4KN1r/1Q6/7R/6Pp/5q2 b - - 0 1",
        // "6k1/7r/8/bB3K1N/1R1q4/4Q3/2nP1p2/8 w - - 0 1",
        // "Q6R/8/2B1q3/3N1nK1/2kb4/P7/r6p/8 w - - 0 1",
        // "8/p5r1/k7/6PK/3b4/2B5/n4qQ1/3N2R1 w - - 0 1",
        // "4kb2/6r1/K7/p7/6n1/2N5/2BP1qR1/7Q w - - 0 1",
        // "6q1/1BN5/1K3P2/3br1np/3R4/Q7/8/5k2 w - - 0 1",
        // "5n2/5q2/1NK5/k1P3r1/3p4/7Q/B6b/1R6 w - - 0 1",
        // "B3r3/3p4/N2K2k1/1Q6/2R5/1bP5/1q5n/8 w - - 0 1",
        // "BR2Q3/4N3/1n2K3/k7/1p1b1q2/8/5P2/7r b - - 0 1",
        // "1k6/7R/5K1N/1pQ5/1n6/P4b2/1r6/6qB b - - 0 1",
        // "8/3k4/3NnPK1/3QR3/3r2pB/8/4b3/q7 w - - 0 1",
        // "1Q6/4q3/NB5K/1R1r4/3P4/bp1k4/6n1/8 w - - 0 1",
        // "3Br3/K7/2q1N3/7n/8/4PbRQ/1p1k4/8 w - - 0 1",
        // "R2r4/pK1b4/1n4NB/7P/8/3Q4/6k1/4q3 b - - 0 1",
        // "3N2r1/2KP4/8/1B1p4/2b5/3RQq2/2k5/7n w - - 0 1",
        "5q2/1N1KB3/5b2/p4R2/4k3/P7/Q7/4n1r1 b - - 0 1",
        "NR6/4K3/1q3r2/3Q3P/3n2k1/8/7p/B5b1 b - - 0 1",
        "q7/1N1B1K2/1Q6/5b2/5pP1/6r1/n6k/R7 w - - 0 1",
        "2R5/2n1k1K1/5r2/3P4/2Q4p/2q5/6NB/7b w - - 0 1",
        "3n1Qr1/3p3K/8/3B4/R5b1/4P3/1qN4k/8 w - - 0 1",
        "K7/3k4/3n2b1/1P2r3/8/p2Bq3/3R4/3QN3 b - - 0 1",
        "1K6/8/3rRN2/1BP3b1/3p4/8/k2n2q1/5Q2 w - - 0 1",
        "2K5/6Bn/p4r2/2P1Q3/1qb5/8/2R5/3kN3 w - - 0 1",
        "3K4/8/2bP4/1qN5/2n3B1/3R4/4Qrp1/6k1 b - - 0 1",
        "1B2K1k1/P3b3/5q2/3R4/1pQ2r1n/8/8/6N1 b - - 0 1",
        "5K2/p4P1b/5QB1/4q3/6k1/8/4r3/R1n1N3 b - - 0 1",
        "6K1/8/b6R/N2p2P1/8/q1Q5/6r1/2Bk3n b - - 0 1",
        "7K/r2R3b/1Q6/8/2q5/1nPB2k1/N3p3/8 w - - 0 1",
    ] {
        let board = Board::from_fen(fen).unwrap();
        nnue.activate(&board);
        let eval = nnue.evaluate(&board);
        println!("FEN: {}", fen);
        println!("EVAL: {}", eval)
    }

    // // Start up the UCI (Universal Chess Interface)
    // let args: Vec<String> = std::env::args().collect();
    // UCI::new().run(&args);
}
