use crate::types::side::Side::{Black, White};
use std::io;
use std::time::Instant;

use crate::bench::bench;
use crate::board::Board;
use crate::datagen::generate_random_openings;
use crate::fen;
use crate::movegen::{gen_moves, MoveFilter};
use crate::moves::Move;
use crate::perft::perft;
use crate::search::search;
use crate::thread::ThreadData;
use crate::time::SearchLimits;

#[cfg(feature = "tuning")]
use crate::parameters::{list_params, print_params_ob, set_param};

pub struct UCI {
    pub board: Board,
    pub td: Box<ThreadData>,
}

impl UCI {
    pub fn new() -> UCI {
        UCI {
            board: Board::new(),
            td: Box::new(ThreadData::default())
        }
    }

    pub fn run(&mut self, args: &[String]) {

        if args.len() > 1 && args[1] == "bench" {
            println!("Running benchmark...");
            self.handle_bench();
            return;
        }
        if args.len() > 1 && args[1].contains("genfens") {
            let tokens = self.split_args(args[1].clone());
            self.handle_genfens(tokens);
            return;
        }

        println!("🐅🐅🐅 Hobbes by Dan Kelsey 🐅🐅🐅");
        println!("(type 'help' for a list of commands)");

        loop {
            let mut command = String::new();
            io::stdin()
                .read_line(&mut command)
                .expect("info error failed to parse command");

            let tokens = self.split_args(command.clone());

            match command.split_ascii_whitespace().next().unwrap() {
                "uci" => self.handle_uci(),
                "isready" => self.handle_isready(),
                "setoption" => self.handle_setoption(tokens),
                "ucinewgame" => self.handle_ucinewgame(),
                "bench" => self.handle_bench(),
                "position" => self.handle_position(tokens),
                "go" => self.handle_go(tokens),
                "stop" => self.handle_stop(),
                "fen" => self.handle_fen(),
                "eval" => self.handle_eval(),
                "perft" => self.handle_perft(tokens),
                "genfens" => self.handle_genfens(tokens),
                "help" => self.handle_help(),
                #[cfg(feature = "tuning")]
                "params" => print_params_ob(),
                "quit" => self.handle_quit(),
                _ => println!("info error: unknown command"),
            }
        }
    }

    fn handle_uci(&self) {
        println!("id name Hobbes");
        println!("id author Dan Kelsey");
        println!("option name Hash type spin default {} min 1 max 1024", self.td.tt.size_mb());
        println!("option name UCI_Chess960 type check default {}", self.board.is_frc());
        println!("option name Minimal type check default false");
        println!("option name UseSoftNodes type check default false");
        #[cfg(feature = "tuning")]
        list_params();
        println!("uciok");
    }

    fn handle_isready(&self) {
        println!("readyok");
    }

    fn handle_setoption(&mut self, tokens: Vec<String>) {
        let tokens: Vec<String> = tokens.iter().map(|s| s.to_lowercase()).collect();
        let tokens: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

        match tokens.as_slice() {
            ["setoption", "name", "hash", "value", size_str] => self.set_hash_size(size_str),
            ["setoption", "name", "threads", "value", _] => return, // TODO set threads
            ["setoption", "name", "uci_chess960", "value", bool_str] => self.set_chess_960(bool_str),
            ["setoption", "name", "minimal", "value", bool_str] => self.set_minimal(bool_str),
            ["setoption", "name", "usesoftnodes", "value", bool_str] => self.set_use_soft_nodes(bool_str),
            #[cfg(feature = "tuning")]
            ["setoption", "name", name, "value", value_str] => self.set_tunable(name, *value_str),
            _ => { println!("info error unknown option"); }
        }
    }

    fn set_hash_size(&mut self, value_str: &str) {
        let value: usize = match value_str.parse() {
            Ok(v) => v,
            Err(_) => {
                println!("info error: invalid value '{}'", value_str);
                return;
            }
        };
        self.td.tt.resize(value);
        println!("info string Hash {}", value);
    }

    fn set_chess_960(&mut self, bool_str: &str) {
        let value = match bool_str {
            "true" => true,
            "false" => false,
            _ => {
                println!("info error: invalid value '{}'", bool_str);
                return;
            }
        };
        self.board.set_frc(value);
        println!("info string Chess960 {}", value);
    }

    fn set_minimal(&mut self, bool_str: &str) {
        let value = match bool_str {
            "true" => true,
            "false" => false,
            _ => {
                println!("info error: invalid value '{}'", bool_str);
                return;
            }
        };
        self.td.minimal_output = value;
        println!("info string Minimal {}", value);
    }

    fn set_use_soft_nodes(&mut self, bool_str: &str) {
        let value = match bool_str {
            "true" => true,
            "false" => false,
            _ => {
                println!("info error: invalid value '{}'", bool_str);
                return;
            }
        };
        self.td.use_soft_nodes = value;
        println!("info string UseSoftNodes {}", value);
    }

    #[cfg(feature = "tuning")]
    fn set_tunable(&self, name: &str, value_str: &str) {
        let value: i32 = match value_str.parse() {
            Ok(v) => v,
            Err(_) => {
                println!("info error: invalid value '{}'", value_str);
                return;
            }
        };
        set_param(name, value);
    }

    fn handle_ucinewgame(&mut self) {
        self.td.clear();
    }

    fn handle_bench(&mut self) {
        bench(&mut self.td);
    }

    fn handle_position(&mut self, tokens: Vec<String>) {
        if tokens.len() < 2 {
            println!("info error: missing position command");
            return;
        }

        let fen = match tokens[1].as_str() {
            "startpos" => fen::STARTPOS.to_string(),
            "fen" => tokens
                .iter()
                .skip(2)
                .take_while(|&token| token != "moves")
                .map(|s| s.as_str())
                .collect::<Vec<&str>>()
                .join(" "), // Returns owned String
            _ => {
                println!("info error: invalid position command");
                return;
            }
        };

        self.board = match Board::from_fen(&fen) {
            Ok(board) => board,
            Err(e) => {
                println!("info error invalid fen: {}", e);
                return;
            }
        };

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
        self.td.keys.push(self.board.hash());

        moves.iter().for_each(|m| {
            let mut legal_moves = gen_moves(&self.board, MoveFilter::All);
            let legal_move = legal_moves.iter()
                .map(|entry| entry.mv)
                .find(|lm| lm.matches(m));
            match legal_move {
                Some(m) => {
                    self.board.make(&m);
                    self.td.keys.push(self.board.hash());
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
        self.td.start_time = Instant::now();
        self.td.tt.birthday();

        let mut nodes = if tokens.contains(&String::from("nodes")) && !self.td.use_soft_nodes {
            match self.parse_uint(&tokens, "nodes") {
                Ok(nodes) => Some(nodes),
                Err(_) => {
                    println!("info error: nodes is not a valid number");
                    return;
                }
            }
        } else {
            None
        };

        let movetime = if tokens.contains(&String::from("movetime")) {
            match self.parse_uint(&tokens, "movetime") {
                Ok(movetime) => Some(movetime),
                Err(_) => {
                    println!("info error: movetime is not a valid number");
                    Some(500)
                }
            }
        } else {
            None
        };

        let fischer = if tokens.contains(&String::from("wtime")) {
            let wtime = self.parse_uint(&tokens, "wtime").unwrap_or_else(|_| {
                println!("info error: wtime is not a valid number");
                500
            });

            let btime = self.parse_uint(&tokens, "btime").unwrap_or_else(|_| {
                println!("info error: btime is not a valid number");
                500
            });

            let winc = self.parse_uint(&tokens, "winc").unwrap_or_else(|_| {
                println!("info error: winc is not a valid number");
                0
            });

            let binc = self.parse_uint(&tokens, "binc").unwrap_or_else(|_| {
                println!("info error: binc is not a valid number");
                0
            });

            let (time, inc) = match self.board.stm {
                White => (wtime, winc),
                Black => (btime, binc),
            };

            Some((time, inc))
        } else {
            None
        };

        let softnodes = if tokens.contains(&String::from("softnodes")) {
            match self.parse_uint(&tokens, "softnodes") {
                Ok(softnodes) => Some(softnodes),
                Err(_) => {
                    println!("info error: softnodes is not a valid number");
                    return;
                }
            }
        } else if tokens.contains(&String::from("nodes")) && self.td.use_soft_nodes {
            match self.parse_uint(&tokens, "nodes") {
                Ok(nodes) => Some(nodes),
                Err(_) => {
                    println!("info error: nodes is not a valid number");
                    return;
                }
            }
        } else {
            None
        };

        let depth = if tokens.contains(&String::from("depth")) {
            match self.parse_uint(&tokens, "depth") {
                Ok(depth) => Some(depth),
                Err(_) => {
                    println!("info error: depth is not a valid number");
                    return;
                }
            }
        } else {
            None
        };

        if let Some(soft_nodes) = softnodes {
            if nodes.is_none() {
                // When doing a soft-nodes search, always ensure a hard node limit is set.
                nodes = Some(soft_nodes * 10);
            }
        }

        self.td.limits = SearchLimits::new(fischer, movetime, softnodes, nodes, depth);

        // Perform the search
        search(&self.board, &mut self.td);

        // Print the best move
        println!("bestmove {}", self.td.best_move.to_uci());
    }

    fn handle_eval(&mut self) {
        self.td.nnue.activate(&self.board);
        let eval: i32 = self.td.nnue.evaluate(&self.board);
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

        let start = Instant::now();
        let nodes = perft(&self.board, depth, depth);
        let elapsed = start.elapsed().as_millis();
        println!("info nodes {}", nodes);
        println!("info ms {}", elapsed);
    }

    /// Handle genfens command, an OpenBench utility that generates random openings from a seed to
    /// be used in an OB datagen workload.
    fn handle_genfens(&mut self, tokens: Vec<String>) {

        let count = self.parse_uint(&tokens, "genfens").unwrap_or_else(|_| {
            println!("info error: count is not a valid number");
            0
        }) as usize;

        let seed = self.parse_uint(&tokens, "seed").unwrap_or_else(|_| {
            println!("info error: seed is not a valid number");
            0
        });

        let random_moves = self.parse_uint(&tokens, "random_moves").unwrap_or_else(|_| {
            8
        }) as usize;

        for opening in generate_random_openings(&mut self.td, count, seed, random_moves) {
            println!("info string genfens {}", opening);
        }

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

    fn parse_uint(&self, tokens: &[String], name: &str) -> Result<u64, String> {
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

    fn split_args(&self, args_str: String) -> Vec<String> {
        args_str.split_whitespace()
            .map(|v| v.trim().to_string())
            .collect()
    }

}
