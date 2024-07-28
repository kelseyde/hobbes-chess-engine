use std::io;
use std::time::Duration;

use crate::board::Board;
use crate::fen;
use crate::moves::Move;
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
        println!("(type help for a list of commands)");

        loop {
            let mut command = String::new();
            io::stdin()
                .read_line(&mut command)
                .expect("info error failed to parse command");

            match command.split_ascii_whitespace().next().unwrap() {
                "uci" =>          self.handle_uci(),
                "isready" =>      self.handle_isready(),
                "setoption" =>    self.handle_setoption(command),
                "ucinewgame" =>   self.handle_ucinewgame(),
                "position" =>     self.handle_position(command),
                "go" =>           self.handle_go(command),
                "stop" =>         self.handle_stop(),
                "ponderhit" =>    self.handle_ponderhit(),
                "eval" =>         self.handle_eval(),
                "datagen" =>      self.handle_datagen(command),
                "help" =>         self.handle_help(),
                "quit" =>         self.handle_quit(),
                _ => {}
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

    fn handle_setoption(&self, command: String) {

    }

    fn handle_ucinewgame(&self) {

    }

    fn handle_position(&mut self, command: String) {

        if command.contains("startpos") {
            let fen = fen::STARTPOS;
            self.board = Board::from_fen(fen);
        }
        else if command.contains("fen") {
            let fen = command.split_ascii_whitespace().skip(2).collect::<Vec<&str>>().join(" ");
            self.board = Board::from_fen(fen.as_str());
        }

        let moves = if let Some(index) = command.find("moves") {
            let moves_str = &command[index + "moves".len()..].trim();
            moves_str.split_ascii_whitespace()
                .collect::<Vec<_>>().iter()
                .map(|m| Move::parse_uci(m))
                .collect()
        } else {
            Vec::new()
        };

        moves.iter().for_each(|m| self.board.make(m));

    }

    fn handle_go(&mut self, command: String) {

        let time = if let Some(index) = command.find("movetime") {
            command[index + "movetime".len()..].trim().parse::<u64>().unwrap()
        } else {
            1000
        };
        self.td.time_limit = Duration::from_millis(time);
        search(&self.board, &mut self.td);
        println!("bestmove {}", self.td.best_move.to_uci());

    }

    fn handle_ponderhit(&self) {

    }

    fn handle_eval(&self) {

    }

    fn handle_datagen(&self, command: String) {

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
        println!("datagen     -- generate training data for neural network");
        println!("quit        -- exit the application");
    }

    fn handle_quit(&self) {
        std::process::exit(0);
    }

}

