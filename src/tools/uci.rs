use crate::board::movegen::MoveFilter;
use crate::board::moves::Move;
use crate::board::side::Side::{Black, White};
use crate::board::Board;
#[cfg(feature = "tuning")]
use crate::search::parameters::{list_params, print_params_ob, set_param};
use crate::search::search;
use crate::search::thread::ThreadData;
use crate::search::time::SearchLimits;
use crate::tools::bench::bench;
use crate::tools::datagen::generate_random_openings;
use crate::tools::perft::perft;
use crate::tools::{fen, pretty};
use crate::VERSION;
use std::io;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Instant;

pub struct UCI {
    pub board: Board,
    pub td: Arc<Mutex<ThreadData>>, // shared thread data
    pub frc: bool,
    pub search_handle: Option<JoinHandle<()>>, // active search thread
}

impl Default for UCI {
    fn default() -> Self { Self::new() }
}

impl UCI {
    pub fn new() -> UCI {
        // Default TT size 16 MB (matches TranspositionTable default)
        let abort = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let td = ThreadData::new(0, 16, abort.clone());
        UCI { board: Board::new(), td: Arc::new(Mutex::new(td)), frc: false, search_handle: None }
    }

    pub fn run(&mut self, args: &[String]) {
        if args.len() > 1 && args[1] == "bench" { println!("Running benchmark..."); self.handle_bench(); return; }
        if args.len() > 1 && args[1].contains("genfens") { let tokens = self.split_args(args[1].clone()); self.handle_genfens(tokens); return; }
        pretty::print_uci_info();
        loop {
            let mut command = String::new();
            let bytes = match io::stdin().read_line(&mut command) { Ok(n) => n, Err(_) => { println!("info error: failed to read line"); continue; } };
            if bytes == 0 { // EOF
                self.handle_stop();
                break;
            }
            if let Some(first) = command.split_ascii_whitespace().next() {
                let tokens = self.split_args(command.clone());
                match first {
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
                    #[cfg(feature = "tuning")] "params" => print_params_ob(),
                    "quit" => { self.handle_stop(); self.handle_quit(); },
                    _ => println!("info error: unknown command"),
                }
            }
        }
    }

    fn handle_uci(&self) {
        let td = self.td.lock().unwrap();
        println!("id name Hobbes {}", VERSION);
        println!("id author Dan Kelsey");
        println!("option name Hash type spin default {} min 1 max 1024", td.tt.size_mb());
        println!("option name UCI_Chess960 type check default {}", self.board.is_frc());
        println!("option name Minimal type check default false");
        println!("option name UseSoftNodes type check default false");
        #[cfg(feature = "tuning")] list_params();
        println!("uciok");
    }

    fn handle_isready(&self) { println!("readyok"); }

    fn handle_setoption(&mut self, tokens: Vec<String>) {
        let tokens: Vec<String> = tokens.iter().map(|s| s.to_lowercase()).collect();
        let tokens: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        match tokens.as_slice() {
            ["setoption","name","hash","value", size_str] => self.set_hash_size(size_str),
            ["setoption","name","threads","value", _] => (),
            ["setoption","name","uci_chess960","value", b] => self.set_chess_960(b),
            ["setoption","name","minimal","value", b] => self.set_minimal(b),
            ["setoption","name","usesoftnodes","value", b] => self.set_use_soft_nodes(b),
            #[cfg(feature="tuning")]
            ["setoption","name", name, "value", val] => self.set_tunable(name, *val),
            _ => println!("info error unknown option"),
        }
    }

    fn set_hash_size(&mut self, value_str: &str) {
        let value: usize = match value_str.parse() { Ok(v)=>v, Err(_)=>{ println!("info error: invalid value '{}'", value_str); return; } };
        let mut td = self.td.lock().unwrap();
        td.tt.resize(value);
        println!("info string Hash {}", value);
    }

    fn set_chess_960(&mut self, bool_str: &str) { if let Some(v)=self.parse_bool_str(bool_str){ self.frc=v; self.board.set_frc(v); println!("info string Chess960 {}", v);} }
    fn set_minimal(&mut self, bool_str: &str) { if let Some(v)=self.parse_bool_str(bool_str){ let mut td=self.td.lock().unwrap(); td.minimal_output=v; println!("info string Minimal {}", v);} }
    fn set_use_soft_nodes(&mut self, bool_str: &str) { if let Some(v)=self.parse_bool_str(bool_str){ let mut td=self.td.lock().unwrap(); td.use_soft_nodes=v; println!("info string UseSoftNodes {}", v);} }
    fn parse_bool_str(&self, s:&str)->Option<bool>{ match s {"true"=>Some(true),"false"=>Some(false), _=>{ println!("info error: invalid value '{}'", s); None}} }

    #[cfg(feature = "tuning")]
    fn set_tunable(&self, name: &str, value_str: &str) { if let Ok(value)=value_str.parse() { set_param(name, value); } else { println!("info error: invalid value '{}'", value_str); } }

    fn handle_ucinewgame(&mut self) { let mut td=self.td.lock().unwrap(); td.clear(); }
    fn handle_bench(&mut self) { let mut td=self.td.lock().unwrap(); bench(&mut td); }

    fn handle_position(&mut self, tokens: Vec<String>) {
        if tokens.len() < 2 { println!("info error: missing position command"); return; }
        let fen = match tokens[1].as_str() { "startpos" => fen::STARTPOS.to_string(), "fen" => tokens.iter().skip(2).take_while(|&t| t!="moves").map(|s| s.as_str()).collect::<Vec<&str>>().join(" "), _ => { println!("info error: invalid position command"); return; } };
        self.board = match Board::from_fen(&fen) { Ok(b)=>b, Err(e)=>{ println!("info error invalid fen: {}", e); return; } }; self.board.set_frc(self.frc);
        let moves: Vec<Move> = if let Some(idx)=tokens.iter().position(|x| x=="moves") { tokens.iter().skip(idx+1).map(|m| Move::parse_uci(m)).collect() } else { Vec::new() };
        let mut td=self.td.lock().unwrap();
        td.keys.clear(); td.root_ply=0; td.keys.push(self.board.hash());
        for m in moves.iter() { let mut legal = self.board.gen_moves(MoveFilter::All); if let Some(lm)=legal.iter().map(|e| e.mv).find(|lm| lm.matches(m)) { self.board.make(&lm); td.keys.push(self.board.hash()); td.root_ply += 1; } else { println!("info error: illegal move {}", m.to_uci()); } }
    }

    fn handle_go(&mut self, tokens: Vec<String>) {
        // If search already running, stop it first
        if self.search_handle.is_some() { self.handle_stop(); }
        {
            let mut td = self.td.lock().unwrap();
            td.abort.store(false, Ordering::Relaxed);
            td.reset();
            td.start_time = Instant::now();
            td.tt.birthday();
            // parse limits inside lock to set on td afterwards
        }
        // Parse outside lock
        let mut nodes = if tokens.contains(&"nodes".to_string()) && { let td=self.td.lock().unwrap(); !td.use_soft_nodes } { match self.parse_uint(&tokens, "nodes") { Ok(v)=>Some(v), Err(_)=>{ println!("info error: nodes is not a valid number"); return; } } } else { None };
        let movetime = if tokens.contains(&"movetime".to_string()) { match self.parse_uint(&tokens, "movetime") { Ok(v)=>Some(v), Err(_)=>{ println!("info error: movetime is not a valid number"); Some(500) } } } else { None };
        let fischer = if tokens.contains(&"wtime".to_string()) { let wtime=self.parse_uint(&tokens,"wtime").unwrap_or_else(|_|{ println!("info error: wtime is not a valid number"); 500 }); let btime=self.parse_uint(&tokens,"btime").unwrap_or_else(|_|{ println!("info error: btime is not a valid number"); 500 }); let winc=self.parse_uint(&tokens,"winc").unwrap_or(0); let binc=self.parse_uint(&tokens,"binc").unwrap_or(0); let (time,inc)=match self.board.stm { White => (wtime,winc), Black => (btime,binc) }; Some((time,inc)) } else { None };
        let softnodes = if tokens.contains(&"softnodes".to_string()) { match self.parse_uint(&tokens,"softnodes") { Ok(v)=>Some(v), Err(_)=>{ println!("info error: softnodes is not a valid number"); return; } } } else if tokens.contains(&"nodes".to_string()) && { let td=self.td.lock().unwrap(); td.use_soft_nodes } { match self.parse_uint(&tokens,"nodes") { Ok(v)=>Some(v), Err(_)=>{ println!("info error: nodes is not a valid number"); return; } } } else { None };
        let depth = if tokens.contains(&"depth".to_string()) { match self.parse_uint(&tokens,"depth") { Ok(v)=>Some(v), Err(_)=>{ println!("info error: depth is not a valid number"); return; } } } else { None };
        if let Some(sn)=softnodes { if nodes.is_none() { nodes = Some(sn * 10); } }
        {
            let mut td = self.td.lock().unwrap();
            td.limits = SearchLimits::new(fischer, movetime, softnodes, nodes, depth);
        }
        let td_arc = self.td.clone();
        let board = self.board; // Copy
        self.search_handle = Some(std::thread::spawn(move || {
            {
                let mut td = td_arc.lock().unwrap();
                search(&board, &mut td);
                println!("bestmove {}", td.best_move.to_uci());
            }
        }));
    }

    fn handle_eval(&mut self) { let mut td=self.td.lock().unwrap(); td.nnue.activate(&self.board); let eval: i32 = td.nnue.evaluate(&self.board); println!("{}", eval); }
    fn handle_fen(&self) { println!("{}", self.board.to_fen()); }

    fn handle_perft(&self, tokens: Vec<String>) {
        if tokens.len() < 2 { println!("info error: missing depth argument"); return; }
        let depth = match tokens[1].parse::<u8>() { Ok(d)=>d, Err(_)=>{ println!("info error: depth argument is not a valid number"); return; } };
        let start = Instant::now(); let nodes = perft(&self.board, depth, depth); let elapsed = start.elapsed().as_millis(); println!("info nodes {}", nodes); println!("info ms {}", elapsed);
    }

    fn handle_genfens(&mut self, tokens: Vec<String>) {
        let count = self.parse_uint(&tokens, "genfens").unwrap_or({ println!("info error: count is not a valid number"); 0 }) as usize;
        let seed = self.parse_uint(&tokens, "seed").unwrap_or({ println!("info error: seed is not a valid number"); 0 });
        let random_moves = self.parse_uint(&tokens, "random_moves").unwrap_or(8) as usize;
        let dfrc = self.parse_bool(&tokens, "dfrc", false).unwrap_or_else(|_| { println!("info error: dfrc is not a valid boolean"); false });
        let mut td=self.td.lock().unwrap();
        for opening in generate_random_openings(&mut td, count, seed, random_moves, dfrc) { println!("info string genfens {}", opening); }
    }

    fn handle_stop(&mut self) {
        if let Some(handle) = self.search_handle.take() {
            {
                let td = self.td.lock().unwrap();
                td.abort.store(true, Ordering::Relaxed);
            }
            let _ = handle.join();
            // abort flag will be reset on next go
        }
    }

    fn handle_help(&self) { println!("the following commands are available:"); println!("uci         -- print engine info"); println!("isready     -- check if engine is ready"); println!("setoption   -- set engine options"); println!("ucinewgame  -- clear the board and set up a new game"); println!("position    -- set up the board position"); println!("go          -- start searching for the best move"); println!("stop        -- stop searching and return the best move"); println!("eval        -- evaluate the current position"); println!("perft       -- run perft on the current position"); println!("quit        -- exit the application"); }
    fn handle_quit(&self) { std::process::exit(0); }

    fn parse_uint(&self, tokens: &[String], name: &str) -> Result<u64, String> {
        match tokens.iter().position(|x| x==name) { Some(index) => match tokens.get(index+1) { Some(v)=> match v.parse::<u64>() { Ok(n)=>Ok(n), Err(_)=>Err(format!("info error: {} is not a valid number", name)) }, None=>Err(format!("info error: {} is missing a value", name)) }, None=>Err(format!("info error: {} is missing", name)) }
    }

    fn parse_bool(&self, tokens: &[String], name: &str, default: bool) -> Result<bool, String> { match tokens.iter().position(|x| x==name) { Some(i)=> match tokens.get(i+1) { Some(v)=> match v.as_str() { "true"=>Ok(true), "false"=>Ok(false), _=>Err(format!("info error: {} is not a valid boolean", name)) }, None=>Err(format!("info error: {} is missing a value", name)) }, None=>Ok(default) } }

    fn split_args(&self, args_str: String) -> Vec<String> { args_str.split_whitespace().map(|v| v.trim().to_string()).collect() }
}
