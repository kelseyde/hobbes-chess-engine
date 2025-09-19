use crate::tools::uci::UCI;
use board::ray;

pub const AUTHOR: &str = "Dan Kelsey";
pub const CONTRIBUTORS: &str = "Jonathan Hallstrom";
pub const VERSION: f32 = 0.1;

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
    ray::init();

    // Start up the UCI (Universal Chess Interface)
    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);

}