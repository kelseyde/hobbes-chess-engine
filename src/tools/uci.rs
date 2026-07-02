use crate::board::movegen::MoveFilter;
use crate::board::moves::{Move, MoveList};
use crate::board::side::Side::{Black, White};
use crate::board::Board;
use crate::search::engine::{Engine, MAX_THREADS};
#[cfg(feature = "tuning")]
use crate::search::parameters::{list_params, print_params_ob, set_param};
use crate::search::time::SearchLimits;
use crate::search::tt;
use crate::tools::bench::bench;
use crate::tools::datagen::generate_random_openings;
use crate::tools::perft::perft;
use crate::tools::{fen, pretty};
use crate::VERSION;
use std::fmt;
use std::io;
use std::str::FromStr;
use std::time::Instant;

/// A parsed UCI command. Each variant carries the data required to execute it, decoded up-front by
/// [`Command::parse`]. Dispatch to the relevant handler is centralised in [`Command::execute`].
enum Command {
    Uci,
    IsReady,
    NewGame,
    SetOption(UciOption),
    Position { board: Box<Board>, keys: Vec<u64> },
    Go(GoOptions),
    Stop,
    Quit,
    Bench,
    Eval,
    Fen,
    Perft { depth: u8, bulk: bool },
    Genfens(GenfensOptions),
    Help,
    #[cfg(feature = "tuning")]
    Params,
}

/// A parsed `setoption` payload, resolved to a concrete, typed option.
enum UciOption {
    Hash(usize),
    Threads(usize),
    Chess960(bool),
    Minimal(bool),
    UseSoftNodes(bool),
    #[cfg(feature = "tuning")]
    Tunable { name: String, value: i32 },
}

/// The raw limits parsed from a `go` command. These are interpreted into a [`SearchLimits`] by
/// [`UCI::handle_go`], which needs engine state (soft-node mode) and the side to move.
#[derive(Default)]
struct GoOptions {
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movetime: Option<u64>,
    nodes: Option<u64>,
    softnodes: Option<u64>,
    depth: Option<u64>,
}

/// Options for the `genfens` datagen utility.
struct GenfensOptions {
    count: usize,
    seed: u64,
    random_moves: usize,
    dfrc: bool,
}

/// An error encountered while parsing a UCI command line.
enum ParseError {
    Empty,
    UnknownCommand(String),
    MissingPositionType,
    InvalidFen(String),
    InvalidMove(String),
    MissingOptionName,
    MissingOptionValue,
    UnknownOption(String),
    MissingValue(&'static str),
    InvalidInt(String),
    InvalidBool(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Empty => write!(f, "empty command"),
            ParseError::UnknownCommand(c) => write!(f, "unknown command '{c}'"),
            ParseError::MissingPositionType => write!(f, "expected 'startpos' or 'fen'"),
            ParseError::InvalidFen(s) => write!(f, "invalid fen '{s}'"),
            ParseError::InvalidMove(s) => write!(f, "invalid or illegal move '{s}'"),
            ParseError::MissingOptionName => write!(f, "missing option name"),
            ParseError::MissingOptionValue => write!(f, "missing option value"),
            ParseError::UnknownOption(s) => write!(f, "unknown option '{s}'"),
            ParseError::MissingValue(n) => write!(f, "missing value for '{n}'"),
            ParseError::InvalidInt(s) => write!(f, "'{s}' is not a valid number"),
            ParseError::InvalidBool(s) => write!(f, "'{s}' is not a valid boolean"),
        }
    }
}

impl Command {
    /// Parse a single line of input into a [`Command`]. `frc` is the engine's current Chess960
    /// setting, used to encode castling correctly for freshly parsed positions.
    fn parse(line: &str, frc: bool) -> Result<Command, ParseError> {
        let mut reader = line.split_ascii_whitespace();
        let cmd = reader.next().ok_or(ParseError::Empty)?;

        match cmd {
            "uci" => Ok(Command::Uci),
            "isready" => Ok(Command::IsReady),
            "ucinewgame" => Ok(Command::NewGame),
            "stop" => Ok(Command::Stop),
            "quit" => Ok(Command::Quit),
            "eval" => Ok(Command::Eval),
            "fen" => Ok(Command::Fen),
            "help" => Ok(Command::Help),
            "bench" => Ok(Command::Bench),
            #[cfg(feature = "tuning")]
            "params" => Ok(Command::Params),
            "perft" => parse_perft(reader),
            "position" => parse_position(reader, frc),
            "go" => parse_go(reader),
            "setoption" => parse_setoption(reader),
            "genfens" => parse_genfens(reader),
            other => Err(ParseError::UnknownCommand(other.to_string())),
        }
    }

    /// Whether this command should still be processed while a search is running. All other
    /// commands are ignored until the search completes.
    fn runs_during_search(&self) -> bool {
        matches!(self, Command::Stop | Command::Quit | Command::IsReady)
    }

    /// Dispatch the command to its handler. This is the single mapping from command to behaviour.
    fn execute(self, uci: &mut UCI) {
        match self {
            Command::Uci => uci.handle_uci(),
            Command::IsReady => uci.handle_isready(),
            Command::NewGame => uci.engine.new_game(),
            Command::SetOption(opt) => uci.apply_option(opt),
            Command::Position { board, keys } => uci.handle_position(board, keys),
            Command::Go(opts) => uci.handle_go(opts),
            Command::Stop => uci.engine.stop(),
            Command::Quit => uci.handle_quit(),
            Command::Bench => uci.handle_bench(),
            Command::Eval => uci.handle_eval(),
            Command::Fen => uci.handle_fen(),
            Command::Perft { depth, bulk } => uci.handle_perft(depth, bulk),
            Command::Genfens(opts) => uci.handle_genfens(opts),
            Command::Help => uci.handle_help(),
            #[cfg(feature = "tuning")]
            Command::Params => print_params_ob(),
        }
    }
}

pub struct UCI {
    pub board: Board,
    pub engine: Engine,
    pub frc: bool,
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
            engine: Engine::new(),
            frc: false,
        }
    }

    pub fn run(&mut self, args: &[String]) {
        // Non-interactive CLI invocations (e.g. OpenBench `bench` / `genfens`): treat the argument
        // as a single command line, run it, and exit.
        if let Some(arg) = args.get(1) {
            if arg == "bench" || arg.contains("genfens") {
                match Command::parse(arg, self.frc) {
                    Ok(cmd) => cmd.execute(self),
                    Err(e) => println!("info error: {e}"),
                }
                return;
            }
        }

        pretty::print_uci_info();

        loop {
            let mut line = String::new();
            match io::stdin().read_line(&mut line) {
                Ok(0) => self.handle_quit(), // EOF: shut down gracefully (never returns)
                Ok(_) => {}
                Err(_) => {
                    println!("info error: failed to read input");
                    continue;
                }
            }

            // Reclaim finished search threads before handling the next command.
            self.engine.try_reclaim();

            match Command::parse(&line, self.frc) {
                Ok(cmd) => {
                    // While a search is running, only a select few commands are handled.
                    if self.engine.searching() && !cmd.runs_during_search() {
                        continue;
                    }
                    cmd.execute(self);
                }
                Err(ParseError::Empty) => {}
                Err(e) => println!("info error: {e}"),
            }
        }
    }

    #[rustfmt::skip]
    fn handle_uci(&self) {
        println!("id name Hobbes {}", VERSION);
        println!("id author Dan Kelsey");
        println!("option name Threads type spin default 1 min 1 max {}", MAX_THREADS);
        println!("option name Hash type spin default {} min 1 max 1024", tt::DEFAULT_TT_SIZE);
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

    fn apply_option(&mut self, opt: UciOption) {
        match opt {
            UciOption::Hash(mb) => {
                self.engine.set_hash(mb);
                println!("info string Hash {mb}");
            }
            UciOption::Threads(n) => {
                self.engine.set_threads(n);
                println!("info string Threads {}", self.engine.num_threads());
            }
            UciOption::Chess960(value) => {
                self.frc = value;
                self.board.set_frc(value);
                println!("info string Chess960 {value}");
            }
            UciOption::Minimal(value) => {
                self.engine.set_minimal_output(value);
                println!("info string Minimal {value}");
            }
            UciOption::UseSoftNodes(value) => {
                self.engine.set_use_soft_nodes(value);
                println!("info string UseSoftNodes {value}");
            }
            #[cfg(feature = "tuning")]
            UciOption::Tunable { name, value } => set_param(&name, value),
        }
    }

    fn handle_bench(&mut self) {
        println!("Running benchmark...");
        bench(self.engine.td_mut());
    }

    fn handle_position(&mut self, board: Box<Board>, keys: Vec<u64>) {
        self.board = *board;
        let td = self.engine.td_mut();
        td.root_ply = keys.len().saturating_sub(1);
        td.keys = keys;
    }

    fn handle_go(&mut self, o: GoOptions) {
        let use_soft = self.engine.use_soft_nodes();

        // A hard node limit only applies in normal (non-soft) node mode.
        let mut hard_nodes = if !use_soft { o.nodes } else { None };

        // Fischer time control: pick the clock for the side to move.
        let fischer = if o.wtime.is_some() || o.btime.is_some() {
            let (time, inc) = match self.board.stm {
                White => (o.wtime.unwrap_or(500), o.winc.unwrap_or(0)),
                Black => (o.btime.unwrap_or(500), o.binc.unwrap_or(0)),
            };
            Some((time, inc))
        } else {
            None
        };

        // In soft-node mode, `nodes` acts as the soft limit.
        let soft_nodes = match o.softnodes {
            Some(sn) => Some(sn),
            None if use_soft => o.nodes,
            None => None,
        };

        // When a soft-node limit is set, always ensure a hard node limit is present too.
        if let Some(sn) = soft_nodes {
            if hard_nodes.is_none() {
                hard_nodes = Some(sn.saturating_mul(10));
            }
        }

        let limits = SearchLimits::new(
            fischer,
            o.movetime,
            soft_nodes,
            hard_nodes,
            o.depth,
            self.board.fm as usize,
        );
        self.engine.go(self.board, limits);
    }

    fn handle_eval(&mut self) {
        let eval = self.engine.eval(self.board);
        println!("{}", eval);
    }

    fn handle_fen(&self) {
        println!("{}", self.board.to_fen());
    }

    fn handle_perft(&self, depth: u8, bulk: bool) {
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
    fn handle_genfens(&mut self, o: GenfensOptions) {
        let openings =
            generate_random_openings(self.engine.td_mut(), o.count, o.seed, o.random_moves, o.dfrc);
        for opening in openings {
            println!("info string genfens {}", opening);
        }
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

    fn handle_quit(&mut self) -> ! {
        self.engine.stop();
        self.engine.join();
        std::process::exit(0);
    }
}

// ---------------------------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------------------------

fn parse_int<T: FromStr>(tok: &str) -> Result<T, ParseError> {
    tok.parse::<T>()
        .map_err(|_| ParseError::InvalidInt(tok.to_string()))
}

fn parse_bool(tok: &str) -> Result<bool, ParseError> {
    match tok {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(ParseError::InvalidBool(tok.to_string())),
    }
}

fn parse_perft<'a>(mut reader: impl Iterator<Item = &'a str>) -> Result<Command, ParseError> {
    let depth: u8 = parse_int(next_string(&mut reader, "perft depth")?)?;
    let bulk = match reader.next() {
        Some(tok) => parse_bool(tok)?,
        None => true,
    };
    Ok(Command::Perft { depth, bulk })
}

fn parse_position<'a>(
    reader: impl Iterator<Item = &'a str>,
    frc: bool,
) -> Result<Command, ParseError> {
    let tokens: Vec<&str> = reader.collect();
    let moves_idx = tokens.iter().position(|t| *t == "moves");
    let head = &tokens[..moves_idx.unwrap_or(tokens.len())];
    let move_tokens: &[&str] = moves_idx.map_or(&[], |i| &tokens[i + 1..]);

    let mut board = match head.first().copied() {
        Some("startpos") => {
            Board::from_fen(fen::STARTPOS).map_err(|e| ParseError::InvalidFen(e.to_string()))?
        }
        Some("fen") => {
            let fen_str = head[1..].join(" ");
            if fen_str.is_empty() {
                return Err(ParseError::MissingPositionType);
            }
            Board::from_fen(&fen_str).map_err(|_| ParseError::InvalidFen(fen_str))?
        }
        _ => return Err(ParseError::MissingPositionType),
    };
    board.set_frc(frc);

    // Replay the moves, validating legality and recording the hash after each ply so that search
    // has the full key history for repetition detection.
    let mut keys = Vec::with_capacity(move_tokens.len() + 1);
    keys.push(board.hash());

    for m in move_tokens {
        let mut legal = MoveList::new();
        board.gen_moves(MoveFilter::All, &mut legal);
        let parsed = Move::parse_uci(m);
        let mv = legal
            .iter()
            .map(|e| e.mv)
            .find(|lm| lm.matches(&parsed))
            .ok_or_else(|| ParseError::InvalidMove((*m).to_string()))?;
        board.make(&mv);
        keys.push(board.hash());
    }

    Ok(Command::Position {
        board: Box::new(board),
        keys,
    })
}

fn parse_go<'a>(mut reader: impl Iterator<Item = &'a str>) -> Result<Command, ParseError> {
    let mut o = GoOptions::default();
    while let Some(tok) = reader.next() {
        match tok {
            "wtime" => o.wtime = Some(next_u64(&mut reader, "wtime")?),
            "btime" => o.btime = Some(next_u64(&mut reader, "btime")?),
            "winc" => o.winc = Some(next_u64(&mut reader, "winc")?),
            "binc" => o.binc = Some(next_u64(&mut reader, "binc")?),
            "movetime" => o.movetime = Some(next_u64(&mut reader, "movetime")?),
            "nodes" => o.nodes = Some(next_u64(&mut reader, "nodes")?),
            "softnodes" => o.softnodes = Some(next_u64(&mut reader, "softnodes")?),
            "depth" => o.depth = Some(next_u64(&mut reader, "depth")?),
            "infinite" => {} // no limits
            _ => {}          // ignore unknown tokens
        }
    }
    Ok(Command::Go(o))
}

fn parse_setoption<'a>(mut reader: impl Iterator<Item = &'a str>) -> Result<Command, ParseError> {
    if reader.next() != Some("name") {
        return Err(ParseError::MissingOptionName);
    }
    let name = reader
        .next()
        .ok_or(ParseError::MissingOptionName)?
        .to_lowercase();

    // Optional "value <value>" clause.
    let value = match reader.next() {
        Some("value") => Some(
            reader
                .next()
                .ok_or(ParseError::MissingOptionValue)?
                .to_lowercase(),
        ),
        Some(_) => return Err(ParseError::MissingOptionValue),
        None => None,
    };
    let val = value.ok_or(ParseError::MissingOptionValue)?;

    let opt = match name.as_str() {
        "hash" => UciOption::Hash(parse_int(&val)?),
        "threads" => UciOption::Threads(parse_int(&val)?),
        "uci_chess960" => UciOption::Chess960(parse_bool(&val)?),
        "minimal" => UciOption::Minimal(parse_bool(&val)?),
        "usesoftnodes" => UciOption::UseSoftNodes(parse_bool(&val)?),
        _ => {
            #[cfg(feature = "tuning")]
            {
                UciOption::Tunable {
                    name: name.clone(),
                    value: parse_int(&val)?,
                }
            }
            #[cfg(not(feature = "tuning"))]
            {
                return Err(ParseError::UnknownOption(name.clone()));
            }
        }
    };
    Ok(Command::SetOption(opt))
}

fn parse_genfens<'a>(reader: impl Iterator<Item = &'a str>) -> Result<Command, ParseError> {
    let tokens: Vec<&str> = reader.collect();
    let count = tokens.first().and_then(|t| t.parse().ok()).unwrap_or(0);
    let seed = parse_after(&tokens, "seed", 0);
    let random_moves = parse_after(&tokens, "random_moves", 8);
    let dfrc = bool_after(&tokens, "dfrc");

    Ok(Command::Genfens(GenfensOptions {
        count,
        seed,
        random_moves,
        dfrc,
    }))
}


/// Consume the next token, erroring with `MissingValue(name)` if it is absent.
fn next_string<'a>(
    reader: &mut impl Iterator<Item = &'a str>,
    name: &'static str,
) -> Result<&'a str, ParseError> {
    reader.next().ok_or(ParseError::MissingValue(name))
}

/// Consume the next token and parse it as a `u64`, erroring if it is missing or malformed.
fn next_u64<'a>(
    reader: &mut impl Iterator<Item = &'a str>,
    name: &'static str,
) -> Result<u64, ParseError> {
    parse_int(next_string(reader, name)?)
}

/// Return the token immediately following `name` in `tokens`, if present.
fn value_after<'a>(tokens: &[&'a str], name: &str) -> Option<&'a str> {
    tokens.iter()
        .position(|t| *t == name)
        .and_then(|i| tokens.get(i + 1))
        .copied()
}

/// Return the token following `name` parsed as `T`, falling back to `default` if the token is
/// absent or fails to parse.
fn parse_after<T: FromStr>(tokens: &[&str], name: &str, default: T) -> T {
    value_after(tokens, name)
        .and_then(|t| t.parse().ok())
        .unwrap_or(default)
}

/// Return `true` only if the token following `name` is exactly `"true"`. An absent token or any
/// other value yields `false`.
fn bool_after(tokens: &[&str], name: &str) -> bool {
    value_after(tokens, name) == Some("true")
}

