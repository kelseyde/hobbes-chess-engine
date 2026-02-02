use crate::tools::uci::UCI;
use board::ray;
use crate::board::Board;
use crate::evaluation::NNUE;

pub const AUTHOR: &str = "Dan Kelsey";
pub const CONTRIBUTORS: &str = "Jonathan Hallström, Mattia Giambirtone";
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

    let nnue = &mut NNUE::default();
    let board = Board::new();
    nnue.activate(&board);
    let eval = nnue.evaluate(&board);
    println!("Initial position evaluation: {}", eval);
    // let mut correct = 0;
    //
    // correct += check_eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 90, nnue) as i32;
    // correct += check_eval("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", -39, nnue) as i32;
    // correct += check_eval("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 654, nnue) as i32;
    // correct += check_eval("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2", -903, nnue) as i32;
    // correct += check_eval("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 32, nnue) as i32;
    // correct += check_eval("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 288, nnue) as i32;
    // correct += check_eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1", -903, nnue) as i32;
    // correct += check_eval("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 748, nnue) as i32;
    //
    // if correct != 8 {
    //     panic!("{} / 8 evaluations correct", correct);
    // }

    // Initialise static data
    // ray::init();

    // // Start up the UCI (Universal Chess Interface)
    // let args: Vec<String> = std::env::args().collect();
    // UCI::new().run(&args);
}

fn check_eval(fen: &str, expected: i32, nnue: &mut NNUE) -> bool {
    let board = board::Board::from_fen(fen).unwrap();
    nnue.activate(&board);
    let eval = nnue.evaluate(&board);
    println!("expected: {}, actual: {}", expected, eval);
    eval == expected
}
