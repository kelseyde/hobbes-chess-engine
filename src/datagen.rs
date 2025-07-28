use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::board::Board;
use crate::{fen, movegen};
use crate::thread::ThreadData;

pub fn generate_random_openings(td: &mut ThreadData,
                                count: usize,
                                seed: u64,
                                random_moves: usize) -> Vec<String> {


    let mut rng = StdRng::seed_from_u64(seed);

    let openings = (0..count)
        .map(|_| generate_random_opening(td, &mut rng, random_moves))
        .collect::<Vec<String>>();

    openings
}

fn generate_random_opening(td: &mut ThreadData,
                           rng: &mut StdRng,
                           random_moves: usize) -> String {

    let mut board = Board::from_fen(fen::STARTPOS).unwrap();

    // ensure an equal distribution of white stm and black stm exits
    let random_moves = if rng.random_bool(0.5) { random_moves } else { random_moves + 1 };

    for _ in 0..random_moves {
        let legal_moves = movegen::gen_legal_moves(&board);

        // If we reached a terminal position, retry recursively
        if legal_moves.is_empty() {
            return generate_random_opening(td, rng, random_moves);
        }

        let mv = legal_moves.get(rng.random_range(0..legal_moves.len())).unwrap();
        board.make(&mv.mv);
    }

    // Skip wildly imbalanced exits
    td.nnue.activate(&board);
    if td.nnue.evaluate(&board).abs() > 1000 {
        return generate_random_opening(td, rng, random_moves);
    }

    board.to_fen()
}


