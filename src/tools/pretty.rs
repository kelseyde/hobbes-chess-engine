use crate::{AUTHOR, VERSION};

const RED: &str = "\x1b[31m";
const ORANGE: &str = "\x1b[38;2;255;165;0m";
const RESET: &str = "\x1b[0m";

pub fn print_uci_info() {
    let version_formatted = format!("[{ORANGE}{VERSION}{RESET}]");
    let author_formatted = format!("{ORANGE}{}{}", AUTHOR, RESET);
    println!("┌──────────────────────┐");
    println!("│ Hobbes               │");
    println!("│ Version: {:<33}│", version_formatted);
    println!("│ Author: {:<34}│", author_formatted);
    println!("└──────────────────────┘");
}