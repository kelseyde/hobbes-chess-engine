use std::io;

use crate::types::side::Side::{Black, White};

use crate::bench::bench;
use crate::board::Board;
use crate::fen;
use crate::movegen::{gen_moves, MoveFilter};
use crate::moves::Move;
use crate::network::NNUE;
use crate::perft::perft;
use crate::search::search;
use crate::thread::ThreadData;
use crate::time::SearchLimits;

pub struct UCI {
    pub board: Board,
    pub td: Box<ThreadData>,
    pub nnue: Box<NNUE>,
}

impl Default for UCI {
    fn default() -> Self {
        Self::new()
    }
}

impl UCI {
    pub fn new() -> UCI {
        UCI {
            board: Board::new(),
            td: ThreadData::default().into(),
            nnue: NNUE::default().into(),
        }
    }

    pub fn run(&mut self, args: &[String]) {
        if args.len() > 1 && args[1] == "bench" {
            println!("Running benchmark...");
            self.handle_bench();
            return;
        }

        println!("🐅🐅🐅 Hobbes by Dan Kelsey 🐅🐅🐅");
        println!("(type 'help' for a list of commands)");

        loop {
            let mut command = String::new();
            io::stdin()
                .read_line(&mut command)
                .expect("info error failed to parse command");

            let tokens: Vec<String> = command
                .split_whitespace()
                .map(|v| v.trim().to_string())
                .collect();

            match command.split_ascii_whitespace().next().unwrap() {
                "uci" => self.handle_uci(),
                "isready" => self.handle_isready(),
                "ucinewgame" => self.handle_ucinewgame(),
                "bench" => self.handle_bench(),
                "position" => self.handle_position(tokens),
                "go" => self.handle_go(tokens),
                "stop" => self.handle_stop(),
                "fen" => self.handle_fen(),
                "eval" => self.handle_eval(),
                "perft" => self.handle_perft(tokens),
                "help" => self.handle_help(),
                "quit" => self.handle_quit(),
                _ => println!("info error: unknown command"),
            }
        }
    }

    fn handle_uci(&self) {
        println!("id name Hobbes");
        println!("id author Dan Kelsey");
        println!("uciok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_ucinewgame(&mut self) {
        self.td.clear();
    }

    fn handle_bench(&self) {
        bench();
    }

    fn handle_position(&mut self, tokens: Vec<String>) {
        if tokens.len() < 2 {
            println!("info error: missing position command");
            return;
        }

        let fen = match tokens[1].as_str() {
            "startpos" => fen::STARTPOS.to_string(), // Convert to owned String
            "fen" => tokens
                .iter()
                .skip(2)
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .join(" "), // Returns owned String
            _ => {
                println!("info error: invalid position command");
                return;
            }
        };

        self.board = Board::from_fen(&fen);

        let moves: Vec<Move> = if let Some(index) = tokens.iter().position(|x| x == "moves") {
            tokens
                .iter()
                .skip(index + 1)
                .map(|m| Move::parse_uci(m))
                .collect()
        } else {
            Vec::new()
        };

        self.td.keys.clear();
        self.td.root_ply = 0;
        self.td.keys.push(self.board.hash);

        moves.iter().for_each(|m| {
            let mut legal_moves = gen_moves(&self.board, MoveFilter::All);
            let legal_move = legal_moves.iter()
                .map(|entry| entry.mv)
                .find(|lm| lm.matches(m));
            match legal_move {
                Some(m) => {
                    self.board.make(&m);
                    self.td.keys.push(self.board.hash);
                    self.td.root_ply += 1;
                }
                None => {
                    println!("info error: illegal move {}", m.to_uci());
                }
            }
        });
    }

    fn handle_go(&mut self, tokens: Vec<String>) {
        self.td.reset();

        if tokens.contains(&String::from("movetime")) {
            match self.parse_int(&tokens, "movetime") {
                Ok(movetime) => {
                    self.td.limits = SearchLimits::new(None, Some(movetime), None, None, None)
                }
                Err(_) => {
                    println!("info error: movetime is not a valid number");
                    return;
                }
            }
        } else if tokens.contains(&String::from("wtime")) {
            let wtime = match self.parse_int(&tokens, "wtime") {
                Ok(wtime) => wtime,
                Err(_) => {
                    println!("info error: wtime is not a valid number");
                    return;
                }
            };

            let btime = match self.parse_int(&tokens, "btime") {
                Ok(btime) => btime,
                Err(_) => {
                    println!("info error: btime is not a valid number");
                    return;
                }
            };

            let winc = match self.parse_int(&tokens, "winc") {
                Ok(winc) => winc,
                Err(_) => {
                    println!("info error: winc is not a valid number");
                    return;
                }
            };

            let binc = match self.parse_int(&tokens, "binc") {
                Ok(binc) => binc,
                Err(_) => {
                    println!("info error: binc is not a valid number");
                    return;
                }
            };

            let (time, inc) = match self.board.stm {
                White => (wtime, winc),
                Black => (btime, binc),
            };

            self.td.limits = SearchLimits::new(Some((time, inc)), None, None, None, None);
        }

        // Perform the search
        search(&self.board, &mut self.td);

        // Print the best move
        println!("bestmove {}", self.td.best_move.to_uci());
    }

    fn handle_eval(&mut self) {
        let eval: i32 = self.nnue.evaluate(&self.board);
        println!("{}", eval);
    }

    fn handle_fen(&self) {
        println!("{}", self.board.to_fen());
    }

    fn handle_perft(&self, tokens: Vec<String>) {
        if tokens.len() < 2 {
            println!("info error: missing depth argument");
            return;
        }

        let depth = match tokens[1].parse::<u8>() {
            Ok(d) => d,
            Err(_) => {
                println!("info error: depth argument is not a valid number");
                return;
            }
        };

        let start = std::time::Instant::now();
        let nodes = perft(&self.board, depth);
        let elapsed = start.elapsed().as_millis();
        println!("info nodes {}", nodes);
        println!("info ms {}", elapsed);
    }

    fn handle_stop(&mut self) {
        //self.td.cancelled = true;
    }

    fn handle_help(&self) {
        println!("the following commands are available:");
        println!("uci         -- print engine info");
        println!("isready     -- check if engine is ready");
        println!("setoption   -- set engine options");
        println!("ucinewgame  -- clear the board and set up a new game");
        println!("position    -- set up the board position");
        println!("go          -- start searching for the best move");
        println!("stop        -- stop searching and return the best move");
        println!("eval        -- evaluate the current position");
        println!("perft       -- run perft on the current position");
        println!("quit        -- exit the application");
    }

    fn handle_quit(&self) {
        std::process::exit(0);
    }

    fn parse_int(&self, tokens: &[String], name: &str) -> Result<u64, String> {
        match tokens.iter().position(|x| x == name) {
            Some(index) => match tokens.get(index + 1) {
                Some(value) => match value.parse::<u64>() {
                    Ok(num) => Ok(num),
                    Err(_) => Err(format!("info error: {} is not a valid number", name)),
                },
                None => Err(format!("info error: {} is missing a value", name)),
            },
            None => Err(format!("info error: {} is missing", name)),
        }
    }
}
