use std::fs;
use crate::board::Board;
use crate::perft::perft;
use crate::types::ray;
use crate::uci::UCI;

pub mod attacks;
pub mod bench;
pub mod board;
pub mod fen;
pub mod magics;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod search;
pub mod thread;
pub mod uci;
pub mod zobrist;
pub mod tt;
pub mod history;
pub mod see;
pub mod types;
pub mod time;
pub mod movepicker;
pub mod evaluation;
pub mod parameters;
pub mod utils;
pub mod correction;

fn main() {

    // Initialise static data
    ray::init();

    let board =
        Board::from_fen("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9").unwrap();

    println!("Running perft tests...");
    assert_eq!(479, perft(&board, 2, 2));
    println!("Perft 2: 479");
    assert_eq!(10471, perft(&board, 3, 3));
    println!("Perft 3: 10471");
    assert_eq!(273318, perft(&board, 4, 4));
    println!("Perft 4: 273318");
    assert_eq!(6417013, perft(&board, 5, 5));
    println!("Perft 5: 6417013");

    println!("reading file...");
    let perft_suite = fs::read_to_string("resources/pawnocchio.epd").unwrap();
    println!("parsed file!");

    for perft_test in perft_suite.lines() {
        let parts: Vec<&str> = perft_test.split(";").collect();

        println!("Parts: {:?}", parts);
        let fen = parts[0];

        // Process all depth parts (skip the FEN part at index 0)
        for depth_part in &parts[1..] {
            let mut depth_nodes_str = depth_part.split_whitespace();
            let depth_str = depth_nodes_str.next().unwrap();
            let nodes_str = depth_nodes_str.last().unwrap();
            let depth: u8 = depth_str[1..].parse().unwrap();
            let nodes: u64 = nodes_str.parse().unwrap();

            println!("Running test on fen for depth {}: {}", depth, fen);
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(perft(&board, depth, depth), nodes, "Failed test: {} at depth {}", fen, depth);
        }
    }

    // Start up the UCI (Universal Chess Interface)
    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);

}


