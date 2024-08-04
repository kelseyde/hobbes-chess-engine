use std::io;
use std::time::Duration;

use crate::board::Board;
use crate::fen;
use crate::movegen::{gen_moves, MoveFilter};
use crate::moves::Move;
use crate::perft::perft;
use crate::search::search;
use crate::thread::ThreadData;
use crate::tt::TT;

pub struct UCI {
    pub tt: TT,
    pub td: ThreadData,
    pub board: Board,
}

impl UCI {

    pub fn new() -> UCI {
        UCI {
            tt: TT::default(),
            td: ThreadData::new(TT::default()),
            board: Board::new(),
        }
    }

    pub fn run(&mut self) {

        println!("🐅🐅🐅 Hobbes by Dan Kelsey 🐅🐅🐅");
        println!("(type 'help' for a list of commands)");

        loop {
            let mut command = String::new();
            io::stdin()
                .read_line(&mut command)
                .expect("info error failed to parse command");

            let tokens: Vec<String> = command.split_whitespace().map(|v| v.trim().to_string()).collect();

            match command.split_ascii_whitespace().next().unwrap() {
                "uci" =>          self.handle_uci(),
                "isready" =>      self.handle_isready(),
                "setoption" =>    self.handle_setoption(tokens),
                "ucinewgame" =>   self.handle_ucinewgame(),
                "position" =>     self.handle_position(tokens),
                "go" =>           self.handle_go(tokens),
                "stop" =>         self.handle_stop(),
                "ponderhit" =>    self.handle_ponderhit(),
                "fen" =>          self.handle_fen(),
                "eval" =>         self.handle_eval(),
                "perft" =>        self.handle_perft(tokens),
                "datagen" =>      self.handle_datagen(tokens),
                "help" =>         self.handle_help(),
                "quit" =>         self.handle_quit(),
                _ =>              println!("info error: unknown command")
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

    fn handle_setoption(&self, tokens: Vec<String>) {
        //TODO
    }

    fn handle_ucinewgame(&self) {
        //TODO
    }

    fn handle_position(&mut self, tokens: Vec<String>) {

        if tokens.len() < 2 {
            println!("info error: missing position command");
            return;
        }

        let fen = match tokens[1].as_str() {
            "startpos" => fen::STARTPOS,
            "fen" => &*tokens.iter().skip(2).map(|s| s.as_str()).collect::<Vec<&str>>().join(" "),
            _ => {
                println!("info error: invalid position command");
                return;
            }
        };

        self.board = Board::from_fen(fen);

        let moves: Vec<Move> = if let Some(index) = tokens.iter().position(|x| x == "moves") {
            tokens.iter().skip(index + 1).map(|m| Move::parse_uci(m)).collect()
        } else {
            Vec::new()
        };

        moves.iter().for_each(|m| {
            let legal_moves = gen_moves(&self.board, MoveFilter::All);
            let legal_move = legal_moves.iter().find(|lm| lm.matches(m));
            match legal_move {
                Some(m) => {
                    self.td.board_history.push(self.board.clone());
                    self.board.make(m)
                },
                None => {
                    println!("info error: illegal move {}", m.to_uci());
                    return;
                }
            }
        });

    }

    fn handle_go(&mut self, tokens: Vec<String>) {
        // Split the command into words

        // Check if "movetime" is in the command arguments
        if tokens.contains(&String::from("movetime")) {
            // Attempt to parse the "movetime" argument
            match tokens[tokens.iter().position(|x| x == "movetime").unwrap() + 1].parse::<u64>() {
                Ok(movetime) => self.td.time_limit = Duration::from_millis(movetime),
                Err(_) => println!("info error: movetime is not a valid number")
            }
        } else {
            // Default time limit if "movetime" is not specified
            self.td.time_limit = Duration::from_millis(1000);
        }

        // Perform the search
        search(&self.board, &mut self.td);

        // Print the best move
        println!("bestmove {}", self.td.best_move.to_uci());
    }

    fn handle_ponderhit(&self) {

    }

    fn handle_eval(&self) {

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
        let nodes = perft(&self.board, depth, depth, false);
        let elapsed = start.elapsed().as_millis();
        println!("info nodes {}", nodes);
        println!("info ms {}", elapsed);
    }

    fn handle_datagen(&self, tokens: Vec<String>) {

    }

    fn handle_stop(&mut self) {
        self.td.cancelled = true;
        println!("bestmove {}", self.td.best_move.to_uci());
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
        println!("ponderhit   -- opponent played the expected move");
        println!("eval        -- evaluate the current position");
        println!("perft       -- run perft on the current position");
        println!("datagen     -- generate training data for neural network");
        println!("quit        -- exit the application");
    }

    fn handle_quit(&self) {
        std::process::exit(0);
    }

}



