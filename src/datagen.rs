use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use rand::{Rng};
use rand::rngs::ThreadRng;
use crate::board::Board;
use crate::{fen, movegen};
use ctrlc;
use chrono::{Utc}

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

    // Generate the ID for this datagen run
    let run_id = format!("hobbes-data-{}-{}",
                         options.description.unwrap_or("default".to_string()),
                         Utc::now().format("%Y-%m-%d_%H-%M-%S"));

    // Create the directory for the data
    let data_dir = PathBuf::from("data").join(run_id);
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("failed to create data directory: {}", e))?;

    let mut counters = Vec::new();

    std::thread::scope(|s| {
        let handles = (0..options.num_threads)
            .map(|id| {
                // Using a different rng per thread guarantees
                // that each thread gets a unique sequence.
                let rng = rand::rng();
            })
    })

    Ok("datagen complete".to_string())

}

fn generate_for_thread(options: &DataGenOptions, data_dir: &Path)

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
            return Err("reached a terminal position".to_string());
        }
        let mv = moves.get(rng.random_range(0..moves.len()));
        board.make(&mv.unwrap().mv);
    }

    Ok(board)
}

/// Reset the datagen global data for a new datagen run
fn reset() {
    FENS_GENERATED.store(0, Ordering::SeqCst);
    ABORT.store(false, Ordering::SeqCst);
    ctrlc::set_handler(move || {
        ABORT.store(true, Ordering::SeqCst);
        println!("aborting datagen...")
    }).expect("failed to set ctrlc handler");
}