use crate::board::Board;
use crate::moves::{Move, MoveFlag};
use crate::search::{search, Score};
use crate::thread::ThreadData;
use crate::time::SearchLimits;
use crate::types::piece::{Piece, PIECES};
use crate::types::side::Side;
use crate::types::square::Square;
use crate::{fen, movegen, search};
use chrono::Utc;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use viriformat::chess::board::{DrawType, GameOutcome, WinType};
use viriformat::chess::chessmove::MoveFlags;
use viriformat::chess::piece::PieceType;
use viriformat::chess::chessmove;
use viriformat::chess::piecelayout::PieceLayout;
use viriformat::chess::squareset::SquareSet;
use viriformat::dataformat::Game;

/// Generates training data for Hobbes' NNUE neural network.
/// Stores data in viriformat using the viriformat crate, credit to Viridithas author.

const DFRC_PERCENT: usize = 10;

static FEN_COUNT: AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone)]
pub struct DataGenOptions {
    // Number of games to generate
    pub num_games: usize,
    // Number of threads to use
    pub num_threads: usize,
    // The soft node limit to use during search
    pub soft_nodes: usize,
}

#[derive(Copy, Clone)]
pub struct Arbiter {
    win_adj_counter: usize,
    win_adj_ply_threshold: usize,
    win_adj_score_threshold: i32,

    draw_adj_counter: usize,
    draw_adj_ply_threshold: usize,
    draw_adj_score_threshold: i32,
}

// #[cfg(feature = "dategen")]
pub fn generate(options: DataGenOptions) -> Result<String, String> {

    // Reset the datagen global data for a new datagen run
    FEN_COUNT.store(0, Ordering::SeqCst);

    // Generate the ID for this datagen run
    let timestamp = Utc::now().format("%Y-%m-%d_%H-%M-%S");
    let run_id = format!("hobbes-{}", timestamp);

    // Create the directory for the data
    let data_dir = PathBuf::from("data").join(run_id);
    let path = data_dir.as_path();
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("failed to create data directory: {}", e))?;

    println!("starting datagen with {} games, {} threads, soft nodes: {}",
             options.num_games, options.num_threads, options.soft_nodes);

    std::thread::scope(|s| {
        let handles = (0..options.num_threads)
            .map(|id| s.spawn(move || {
                let mut rng = rand::rng();
                let opt_ref = &options;
                generate_for_thread(id, &mut rng, opt_ref, path);
            }))
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    });

    Ok("datagen complete".to_string())

}

fn generate_for_thread(id: usize,
                       rng: &mut ThreadRng,
                       options: &DataGenOptions,
                       data_dir: &Path) -> u32 {

    let mut td = ThreadData::new(2);
    td.print = false;

    let num_games = options.num_games / options.num_threads;
    let mut output_file = File::create(data_dir.join(format!("thread_{id}.bin")))
        .expect("failed to create output file!");
    let mut output_buffer = BufWriter::new(&mut output_file);

    let soft_nodes = options.soft_nodes as u64;
    let hard_nodes = (options.soft_nodes * 8) as u64;
    td.limits = SearchLimits::node_limits(soft_nodes, hard_nodes);
    let start = Instant::now();

    'game_loop: for game in 0..num_games {

        if id == 0 && game > 0 {
            let percentage = game * 100_000 / num_games;
            let percentage = percentage as f64 / 1000.0;
            let time_per_game = start.elapsed().as_secs_f64() / game as f64;
            let games_to_go = num_games as f64 - game as f64;
            let time_remaining = games_to_go * time_per_game;
            let fens = FEN_COUNT.load(Ordering::Relaxed);
            let fps = FEN_COUNT.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64();
            println!("main thread: \
                games: {game}  ({percentage:.1}%), \
                gps: {time_per_game:.2}, \
                fens: {fens}, \
                fps: {fps:.2} \
                time remaining: {time_remaining:.2} seconds");
        }

        // Reset thread state for a new game
        td.clear();
        let mut board = generate_startpos(rng).expect("failed to generate startpos!");

        // Skip wildly unbalanced exits
        td.nnue.activate(&board);
        if td.nnue.evaluate(&board).abs() > 1000 {
            continue;
        }

        let mut game: Game = Game::new(&board.into());

        let mut arbiter = Arbiter::default();
        let outcome = loop {

            let outcome = arbiter.game_outcome(&td, &board);
            if outcome != GameOutcome::Ongoing {
                break outcome;
            }
            td.reset();
            td.start_time = Instant::now();
            td.tt.birthday();

            let (best_move, score) = search(&board, &mut td);
            if !best_move.exists() {
                println!("error: search returned null best move");
                continue 'game_loop;
            }

            game.add_move(best_move.into(), score as i16);

            let outcome = arbiter.adjudicate(&board, score);
            if outcome != GameOutcome::Ongoing {
                break outcome;
            }

            board.make(&best_move);

        };

        let count = game.len();
        game.set_outcome(outcome);

        game.serialise_into(&mut output_buffer)
            .expect("Failed to serialise game into output buffer.");

        FEN_COUNT.fetch_add(count as u64, Ordering::SeqCst);

    }

    output_buffer
        .flush()
        .expect("failed to flush output buffer to file");

    0

}

fn generate_startpos(rng: &mut ThreadRng) -> Result<Board, String> {
    let random = rng.random_range(0..100);
    let mut board = if random < DFRC_PERCENT {
        let scharnagl_idx = rng.random_range(0..960 * 960);
        Board::from_dfrc_idx(scharnagl_idx)
    } else {
        Board::from_fen(fen::STARTPOS)?
    };

    let random_plies = if rng.random_bool(0.5) { 8 } else { 9 };
    for _ in 0..random_plies {
        let moves = movegen::gen_legal_moves(&board);
        if moves.is_empty() {
            // If we reached a terminal position,
            // recursively generate a new startpos
            return generate_startpos(rng);
        }
        let mv = moves.get(rng.random_range(0..moves.len()));
        board.make(&mv.unwrap().mv);
    }

    println!("startpos: {}", board.to_fen());

    Ok(board)
}

impl Default for Arbiter {
    fn default() -> Self {
        Arbiter {
            win_adj_counter: 0,
            win_adj_score_threshold: 2500,
            win_adj_ply_threshold: 4,
            draw_adj_counter: 0,
            draw_adj_ply_threshold: 12,
            draw_adj_score_threshold: 4,
        }
    }
}

impl Arbiter {

    pub fn game_outcome(&self, td: &ThreadData, board: &Board) -> GameOutcome {

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

    pub fn adjudicate(&mut self, board: &Board, score: i32) -> GameOutcome {
        let abs_score = score.abs();
        if abs_score >= self.win_adj_score_threshold {
            self.win_adj_counter += 1;
            self.draw_adj_counter = 0;
        } else if abs_score <= self.draw_adj_score_threshold {
            self.draw_adj_counter += 1;
            self.win_adj_counter = 0;
        } else {
            self.win_adj_counter = 0;
            self.draw_adj_counter = 0;
        }

        if self.win_adj_counter >= self.win_adj_ply_threshold {
            return if score > 0 {
                GameOutcome::WhiteWin(WinType::Adjudication)
            } else {
                GameOutcome::BlackWin(WinType::Adjudication)
            };
        }
        if self.draw_adj_counter >= self.draw_adj_ply_threshold {
            return GameOutcome::Draw(DrawType::Adjudication);
        }

        if Score::is_mate(score) {
            return if score.is_positive() {
                if board.stm == Side::White {
                    GameOutcome::WhiteWin(WinType::Mate)
                } else {
                    GameOutcome::BlackWin(WinType::Mate)
                }
            } else {
                if board.stm == Side::White {
                    GameOutcome::BlackWin(WinType::Mate)
                } else {
                    GameOutcome::WhiteWin(WinType::Mate)
                }
            }
        };
        GameOutcome::Ongoing
    }

}

// TODO cfg features datagen
impl Into<viriformat::chess::board::Board> for Board {

    fn into(self) -> viriformat::chess::board::Board {
        let mut board = viriformat::chess::board::Board::new();

        let pieces: [SquareSet; 6] = PIECES.iter()
            .map(|&piece| SquareSet::from_inner(self.pieces(piece).0))
            .collect::<Vec<SquareSet>>()
            .try_into()
            .expect("failed to convert piece bbs to array");
        let colours: [SquareSet; 2] = [Side::White, Side::Black]
            .iter()
            .map(|&side| SquareSet::from_inner(self.side(side).0))
            .collect::<Vec<SquareSet>>()
            .try_into()
            .expect("failed to convert side bbs to array");

        board.pieces = PieceLayout { pieces, colours };
        board.piece_array = Square::iter()
            .map(|sq| {
                self.piece_at(sq).map(|pc| {
                    let colour = match self.side_at(sq).unwrap() {
                        Side::White => viriformat::chess::piece::Colour::White,
                        Side::Black => viriformat::chess::piece::Colour::Black,
                    };
                    let piece_type = match pc {
                        Piece::Pawn => PieceType::Pawn,
                        Piece::Knight => PieceType::Knight,
                        Piece::Bishop => PieceType::Bishop,
                        Piece::Rook => PieceType::Rook,
                        Piece::Queen => PieceType::Queen,
                        Piece::King => PieceType::King,
                    };
                    viriformat::chess::piece::Piece::new(colour, piece_type)
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("failed to convert piece array to array of Option<Piece>");


        // todo dfrc tofen
        // board.set_from_fen(self.to_fen().as_str()).expect("failed to set from fen!");
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
            MoveFlag::PromoQ | MoveFlag::PromoR | MoveFlag::PromoB | MoveFlag::PromoN => {
                let pc = match self.promo_piece().unwrap() {
                    Piece::Queen => PieceType::Queen,
                    Piece::Rook => PieceType::Rook,
                    Piece::Bishop => PieceType::Bishop,
                    Piece::Knight => PieceType::Knight,
                    _ => panic!("invalid promotion piece")
                };
                chessmove::Move::new_with_promo(from, to, pc)
            }
        }
    }
}