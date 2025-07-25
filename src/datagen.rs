use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use rand::{Rng};
use rand::rngs::ThreadRng;
use crate::board::Board;
use crate::{fen, movegen, search};
use ctrlc;
use chrono::{Utc};
use viriformat::chess::board::{DrawType, GameOutcome, WinType};
use viriformat::chess::chessmove;
use viriformat::chess::chessmove::MoveFlags;
use viriformat::dataformat::Game;
use crate::moves::{Move, MoveFlag};
use crate::search::search;
use crate::thread::ThreadData;
use crate::time::SearchLimits;
use crate::types::side::Side;
use crate::types::square::Square;

const DFRC_PERCENT: usize = 10;

static ABORT: AtomicBool = AtomicBool::new(false);
static FENS_GENERATED: AtomicU64 = AtomicU64::new(0);

pub struct DataGenOptions {
    // An optional description to append to the filename
    description: Option<String>,
    // Number of games to generate
    num_games: usize,
    // Number of threads to use
    num_threads: usize,
    // The soft node limit to use during search
    soft_nodes: usize,
}

// #[cfg(feature = "dategen")]
pub fn generate(options: DataGenOptions) -> Result<String, String> {

    // Reset the datagen global data for a new datagen run
    FENS_GENERATED.store(0, Ordering::SeqCst);
    ABORT.store(false, Ordering::SeqCst);
    ctrlc::set_handler(move || {
        ABORT.store(true, Ordering::SeqCst);
        println!("aborting datagen...")
    }).expect("failed to set ctrlc handler");

    // Generate the ID for this datagen run
    let run_id = format!("hobbes{}{}",
                         options.description.map(|d| format!("-{}-", d)).unwrap_or("-".to_string()),
                         Utc::now().format("%Y-%m-%d_%H-%M-%S"));

    // Create the directory for the data
    let data_dir = PathBuf::from("data").join(run_id);
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("failed to create data directory: {}", e))?;

    let mut counters = Vec::new();
    let num_threads = options.num_threads;

    std::thread::scope(|s| {
        let handles = (0..num_threads)
            .map(|id| {
                // Using a different rng per thread guarantees
                // that each thread gets a unique sequence.
                s.spawn(move || {
                    let rng = rand::rng();
                    generate_for_thread(id, rng, &options, &data_dir);
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            if let Ok(res) = handle.join() {
                counters.push(res);
            } else {
                println!("failed to join thread")
            }
        }
    });

    Ok("datagen complete".to_string())

}

fn generate_for_thread(id: usize,
                       rng: &mut ThreadRng,
                       options: &DataGenOptions,
                       data_dir: &Path) -> usize {

    let mut td = ThreadData::new(2);

    let num_games = options.num_games / options.num_threads;
    let mut output_file = File::create(data_dir.join(format!("thread_{id}.bin")))
        .expect("failed to create output file!");
    let mut output_buffer = BufWriter::new(&mut output_file);
    td.limits = SearchLimits::node_limits(
        options.soft_nodes as u64,
        (options.soft_nodes * 8) as u64
    );


    let start = Instant::now();
    for game in 0..num_games {

        // Reset thread state for a new game
        td.clear();
        let mut board = generate_startpos(rng).expect("failed to generate startpos!");

        // Skip wildly unbalanced exits
        td.nnue.activate(&board);
        if td.nnue.evaluate(&board).abs() > 1000 {
            continue;
        }

        let mut game: Game = Game::new(board.into());

        let mut win_adj_counter;
        let mut draw_adj_counter;
        let outcome = loop {

            let outcome = game_outcome(&td, &board);
            if outcome != GameOutcome::Ongoing {
                break outcome;
            }
            td.reset();
            td.start_time = Instant::now();
            td.tt.birthday();

            let (score, best_move) = search(&board, &mut td);
        }



    }


    // TODO to_fen for dfrc


    0

}

fn generate_startpos(rng: &mut ThreadRng) -> Result<Board, String> {
    let random = rng.random_range(0..100);
    let mut board = if random < DFRC_PERCENT {
        let random = rng.random_range(0..960 * 960);
        Board::from_dfrc_idx(random)
    } else {
        Board::from_fen(fen::STARTPOS)?
    };

    let random_plies = if rng.random_bool(0.5) { 8 } else { 9 };
    for _ in 0..random_plies {
        let mut moves = movegen::gen_legal_moves(&board);
        if moves.is_empty() {
            // Recursively try again
            return generate_startpos(rng);
        }
        let mv = moves.get(rng.random_range(0..moves.len()));
        board.make(&mv.unwrap().mv);
    }

    Ok(board)
}

fn game_outcome(td: &ThreadData, board: &Board) -> GameOutcome {

    if board.is_fifty_move_rule() {
        return GameOutcome::Draw(DrawType::FiftyMoves);
    }
    if board.is_insufficient_material() {
        return GameOutcome::Draw(DrawType::InsufficientMaterial);
    }
    if search::is_repetition(board, td, false) {
        return GameOutcome::Draw(DrawType::Repetition);
    }
    let legal_moves = movegen::gen_legal_moves(board);
    if legal_moves.is_empty() {
        return if movegen::is_check(board, board.stm) {
            match board.stm {
                Side::White => GameOutcome::BlackWin(WinType::Mate),
                Side::Black => GameOutcome::WhiteWin(WinType::Mate)
            }
        } else {
            GameOutcome::Draw(DrawType::Stalemate)
        }
    }
    GameOutcome::Ongoing

}

// TODO cfg features datagen
impl Into<viriformat::chess::board::Board> for Board {

    fn into(self) -> viriformat::chess::board::Board {
        let mut board = viriformat::chess::board::Board::new();
        board.set_from_fen(self.to_fen().as_str()).expect("failed to set from fen!");
        board
    }

}

impl Into<viriformat::chess::types::Square> for Square {
    fn into(self) -> viriformat::chess::types::Square {
        viriformat::chess::types::Square::new(self.0).unwrap()
    }
}

impl Into<chessmove::MoveFlags> for MoveFlag {
    fn into(self) -> MoveFlags {
        todo!()
    }
}

impl Into<chessmove::Move> for Move {
    fn into(self) -> chessmove::Move {
        let from = self.from().into();
        let to = self.to().into();
        match self.flag() {
            MoveFlag::Standard | MoveFlag::DoublePush => {
                chessmove::Move::new(from, to)
            },
            MoveFlag::CastleK | MoveFlag::CastleQ => {
                chessmove::Move::new_with_flags(from, to, MoveFlags::Castle)
            }
            MoveFlag::EnPassant => {
                chessmove::Move::new_with_flags(from, to, MoveFlags::EnPassant)
            },

        }
    }
}