use crate::board::movegen::MoveFilter;
use crate::board::moves::{Move, MoveList};
use crate::board::side::Side::{Black, White};
use crate::board::Board;
use crate::evaluation::stats;
#[cfg(feature = "tuning")]
use crate::search::parameters::{list_params, print_params_ob, set_param};
use crate::search::{search, tt};
use crate::search::thread::ThreadData;
use crate::search::time::SearchLimits;
use crate::tools::bench::bench;
use crate::tools::datagen::generate_random_openings;
use crate::tools::perft::perft;
use crate::tools::{fen, pretty};
use crate::VERSION;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

pub struct UCI {
    pub board: Board,
    pub td: Option<Box<ThreadData>>,
    pub abort: Arc<AtomicBool>,
    pub search_handle: Option<JoinHandle<Box<ThreadData>>>,
    pub frc: bool,
}

impl Default for UCI {
    fn default() -> Self {
        Self::new()
    }
}

impl UCI {
    pub fn new() -> UCI {
        let td = Box::new(ThreadData::default());
        let abort = Arc::clone(&td.abort);
        UCI {
            board: Board::new(),
            td: Some(td),
            abort,
            search_handle: None,
            frc: false,
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

        pretty::print_uci_info();

        loop {
            let mut line = String::new();
            io::stdin()
                .read_line(&mut line)
                .expect("info error failed to parse command");
            let tokens = self.split_args(line.clone());

            self.try_reclaim_search();

            if let Some(command) = line.split_ascii_whitespace().next() {
                // If a search is running, only the 'stop', 'quit', and 'isready' commands are handled.
                // All other commands are ignored until the search is finished.
                match command {
                    "stop" => self.handle_stop(),
                    "quit" => self.handle_quit(),
                    "isready" => self.handle_isready(),
                    _ if self.searching() => continue,
                    _ => match command {
                        "uci" => self.handle_uci(),
                        "setoption" => self.handle_setoption(tokens),
                        "ucinewgame" => self.handle_ucinewgame(),
                        "bench" => self.handle_bench(),
                        "position" => self.handle_position(tokens),
                        "go" => self.handle_go(tokens),
                        "fen" => self.handle_fen(),
                        "eval" => self.handle_eval(),
                        "eval_stats" => self.handle_eval_stats(tokens),
                        "perft" => self.handle_perft(tokens),
                        "genfens" => self.handle_genfens(tokens),
                        "help" => self.handle_help(),
                        #[cfg(feature = "tuning")]
                        "params" => print_params_ob(),
                        _ => println!("info error: unknown command"),
                    },
                }
            } else {
                continue;
            }
        }
    }

    fn handle_uci(&self) {
        println!("id name Hobbes {}", VERSION);
        println!("id author Dan Kelsey");
        println!("option name Threads type spin default 1 min 1 max 1");
        println!(
            "option name Hash type spin default {} min 1 max 1024", 
            tt::DEFAULT_TT_SIZE
        );
        println!(
            "option name UCI_Chess960 type check default {}",
            self.board.is_frc()
        );
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
            ["setoption", "name", "threads", "value", _] => (), // TODO set threads
            ["setoption", "name", "uci_chess960", "value", bool_str] => {
                self.set_chess_960(bool_str)
            }
            ["setoption", "name", "minimal", "value", bool_str] => self.set_minimal(bool_str),
            ["setoption", "name", "usesoftnodes", "value", bool_str] => {
                self.set_use_soft_nodes(bool_str)
            }
            #[cfg(feature = "tuning")]
            ["setoption", "name", name, "value", value_str] => self.set_tunable(name, *value_str),
            _ => {
                println!("info error unknown option");
            }
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
        self.td.as_mut().unwrap().resize_tt(value);
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
        self.frc = value;
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
        self.td.as_mut().unwrap().minimal_output = value;
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
        self.td.as_mut().unwrap().use_soft_nodes = value;
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
        self.td.as_mut().unwrap().clear();
    }

    fn handle_bench(&mut self) {
        self.abort.store(false, Relaxed);
        bench(self.td.as_mut().unwrap());
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
        self.board.set_frc(self.frc);

        let moves: Vec<Move> = if let Some(index) = tokens.iter().position(|x| x == "moves") {
            tokens
                .iter()
                .skip(index + 1)
                .map(|m| Move::parse_uci(m))
                .collect()
        } else {
            Vec::new()
        };

        let start_hash = self.board.hash();
        let td = self.td.as_mut().unwrap();
        td.keys.clear();
        td.root_ply = 0;
        td.keys.push(start_hash);

        moves.iter().for_each(|m| {
            let mut legal_moves = MoveList::new();
            self.board.gen_moves(MoveFilter::All, &mut legal_moves);
            let legal_move = legal_moves
                .iter()
                .map(|entry| entry.mv)
                .find(|lm| lm.matches(m));
            match legal_move {
                Some(m) => {
                    self.board.make(&m);
                    let hash = self.board.hash();
                    let td = self.td.as_mut().unwrap();
                    td.keys.push(hash);
                    td.root_ply += 1;
                }
                None => {
                    println!("info error: illegal move {}", m.to_uci());
                }
            }
        });
    }

    fn handle_go(&mut self, tokens: Vec<String>) {
        let use_soft_nodes = self.td.as_ref().unwrap().use_soft_nodes;

        let mut nodes = if tokens.contains(&String::from("nodes")) && !use_soft_nodes {
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
        } else if tokens.contains(&String::from("nodes")) && use_soft_nodes {
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

        let limits = SearchLimits::new(
            fischer,
            movetime,
            softnodes,
            nodes,
            depth,
            self.board.fm as usize,
        );

        // Take ownership of the thread data and configure it for the new search.
        let mut td = self.td.take().unwrap();
        td.reset();
        td.start_time = Instant::now();
        td.tt().birthday();
        td.limits = limits;

        // Clear the abort flag and launch the search thread.
        self.abort.store(false, Relaxed);
        let board = self.board;
        self.search_handle = Some(std::thread::spawn(move || {
            search(&board, &mut td);
            println!("bestmove {}", td.best_move.to_uci());
            td
        }));
    }

    fn handle_eval(&mut self) {
        let board = self.board;
        let td = self.td.as_mut().unwrap();
        td.nnue.activate(&board);
        let eval: i32 = td.nnue.evaluate(&board);
        println!("{}", eval);
    }

    fn handle_eval_stats(&mut self, tokens: Vec<String>) {
        if tokens.len() < 2 {
            println!("info error: missing input file argument");
            return;
        }

        let input_path = Path::new(&tokens[1]);
        stats::eval_stats(self.td.as_mut().unwrap(), input_path);
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

        let bulk = match tokens.get(2).map(|s| s.as_str()) {
            Some("false") => false,
            Some("true") | None => true,
            Some(other) => {
                println!(
                    "info error: bulk argument '{}' is not a valid boolean",
                    other
                );
                return;
            }
        };

        let t = Instant::now();
        let n = if bulk {
            perft::<true>(&self.board, depth)
        } else {
            perft::<false>(&self.board, depth)
        };
        let d = t.elapsed();
        let mnps = (n as f64) / d.as_secs_f64() / 1e6;
        println!("info nodes: {n}");
        println!("info {d:.2?} ({mnps:.2}Mnps)\n");
    }

    /// Handle genfens command, an OpenBench utility that generates random openings from a seed to
    /// be used in an OB datagen workload.
    fn handle_genfens(&mut self, tokens: Vec<String>) {
        let count = self.parse_uint(&tokens, "genfens").unwrap_or({
            println!("info error: count is not a valid number");
            0
        }) as usize;

        let seed = self.parse_uint(&tokens, "seed").unwrap_or({
            println!("info error: seed is not a valid number");
            0
        });

        let random_moves = self.parse_uint(&tokens, "random_moves").unwrap_or(8) as usize;

        let dfrc = self.parse_bool(&tokens, "dfrc", false).unwrap_or_else(|_| {
            println!("info error: dfrc is not a valid boolean");
            false
        });
        for opening in
            generate_random_openings(self.td.as_mut().unwrap(), count, seed, random_moves, dfrc)
        {
            println!("info string genfens {}", opening);
        }
    }

    fn handle_stop(&mut self) {
        // Signal the search thread to abort. It will print its best move and finish; the next
        // command that needs the ThreadData will reclaim it via `join_search`.
        self.abort.store(true, Relaxed);
    }

    /// Wait for any in-progress search to finish, reclaiming ownership of the thread data in the process.
    fn join_search(&mut self) {
        if let Some(handle) = self.search_handle.take() {
            let td = handle.join().expect("search thread panicked");
            self.td = Some(td);
        }
    }

    /// Reclaim the thread data from the search thread, but only if it has already finished. Unlike
    /// `join_search`, this never blocks the UCI loop waiting for an in-progress search.
    fn try_reclaim_search(&mut self) {
        if self.search_handle.as_ref().is_some_and(|h| h.is_finished()) {
            self.join_search();
        }
    }

    /// Whether a search is currently running (i.e. the search thread still owns the thread data).
    fn searching(&self) -> bool {
        self.search_handle.is_some()
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

    fn handle_quit(&mut self) {
        // Abort and join any running search so it finishes cleanly before we exit.
        self.abort.store(true, Relaxed);
        self.join_search();
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

    fn parse_bool(&self, tokens: &[String], name: &str, default: bool) -> Result<bool, String> {
        match tokens.iter().position(|x| x == name) {
            Some(index) => match tokens.get(index + 1) {
                Some(value) => match value.as_str() {
                    "true" => Ok(true),
                    "false" => Ok(false),
                    _ => Err(format!("info error: {} is not a valid boolean", name)),
                },
                None => Err(format!("info error: {} is missing a value", name)),
            },
            None => Ok(default),
        }
    }

    fn split_args(&self, args_str: String) -> Vec<String> {
        args_str
            .split_whitespace()
            .map(|v| v.trim().to_string())
            .collect()
    }
}
