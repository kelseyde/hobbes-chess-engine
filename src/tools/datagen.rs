use crate::board::movegen::MoveFilter;
use crate::board::moves::MoveList;
use crate::board::Board;
use crate::search::thread::ThreadData;
use crate::tools::fen;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn generate_random_openings(
    td: &mut ThreadData,
    count: usize,
    seed: u64,
    random_moves: usize,
    dfrc: bool,
) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| generate_random_opening(td, &mut rng, random_moves, dfrc))
        .collect::<Vec<String>>()
}

fn generate_random_opening(
    td: &mut ThreadData,
    rng: &mut StdRng,
    random_moves: usize,
    dfrc: bool,
) -> String {
    let mut board = startpos(rng, dfrc);

    // ensure an equal distribution of white stm and black stm exits
    let random_moves = if rng.random_bool(0.5) {
        random_moves
    } else {
        random_moves + 1
    };

    for _ in 0..random_moves {
        let mut legal_moves = MoveList::new();
        board.gen_moves(MoveFilter::All, &mut legal_moves);

        // If we reached a terminal position, retry recursively
        if legal_moves.is_empty() {
            return generate_random_opening(td, rng, random_moves, dfrc);
        }

        let mv = legal_moves
            .get(rng.random_range(0..legal_moves.len()))
            .unwrap();
        board.make(&mv.mv);
    }

    // Skip wildly imbalanced exits
    td.nnue.activate(&board);
    if td.nnue.evaluate(&board, 0).abs() > 1000 {
        return generate_random_opening(td, rng, random_moves, dfrc);
    }

    let mut legal_moves = MoveList::new();
    board.gen_moves(MoveFilter::All, &mut legal_moves);
    // If we reached a terminal position, retry recursively
    if legal_moves.is_empty() {
        return generate_random_opening(td, rng, random_moves, dfrc);
    }

    board.to_fen()
}

fn startpos(rng: &mut StdRng, dfrc: bool) -> Board {
    if !dfrc {
        Board::from_fen(fen::STARTPOS).unwrap()
    } else {
        let dfrc_seed = rng.random_range(0..960 * 960);
        Board::from_dfrc_idx(dfrc_seed)
    }
}
