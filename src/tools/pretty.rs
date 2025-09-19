use crate::{AUTHOR, CONTRIBUTORS, VERSION};

const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

pub fn print_uci_info() {
    let version_formatted = format!("{CYAN}{VERSION}{RESET}");
    let author_formatted = format!("{CYAN}{}{}", AUTHOR, RESET);
    let contributors_formatted = format!("{CYAN}{}{}", CONTRIBUTORS, RESET);
    println!("┌───────────────────────────────────────────────────────────────────┐");
    println!("│ Hobbes                                                            │");
    println!("│ Version: {:<66}│", version_formatted);
    println!("│ Author: {:<67}│", author_formatted);
    println!("│ Contributors: {:<61}│", contributors_formatted);
    println!("│ OpenBench: CalvinBench https://kelseyde.pythonanywhere.com/index/ │");
    println!("│            MattBench https://chess.n9x.co/index/                  │");
    println!("└───────────────────────────────────────────────────────────────────┘");
    println!("Type 'help' for a list of commands.");
}