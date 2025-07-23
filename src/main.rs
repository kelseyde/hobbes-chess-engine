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

    // Start up the UCI (Universal Chess Interface)
    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);

}


