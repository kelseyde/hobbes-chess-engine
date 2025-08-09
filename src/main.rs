use crate::tools::uci::UCI;
use board::ray;

pub mod board;
pub mod evaluation;
pub mod search;
pub mod tools;

fn main() {

    // Initialise static data
    ray::init();

    // Start up the UCI (Universal Chess Interface)
    let args: Vec<String> = std::env::args().collect();
    UCI::new().run(&args);

}


