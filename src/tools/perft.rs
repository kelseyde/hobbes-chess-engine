use crate::board::movegen::MoveFilter;
use crate::board::moves::MoveList;
use crate::board::{Board, NullBoardObserver};

pub fn perft<const BULK: bool>(board: &Board, depth: u8) -> u64 {
    let mut moves = MoveList::new();
    board.gen_moves(MoveFilter::All, &mut moves);
    let mut total = 0;

    let mut entries: Vec<(String, u64)> = moves
        .iter()
        .map(|entry| {
            let mv = entry.mv;
            let mut child = *board;
            child.make(&mv, &mut NullBoardObserver);
            let nodes = if depth <= 1 {
                1
            } else {
                perft_inner::<BULK>(&child, depth - 1)
            };
            total += nodes;
            (mv.to_uci(), nodes)
        })
        .collect();

    entries.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (mv, count) in &entries {
        println!("{} - {}", mv, count);
    }

    total
}

fn perft_inner<const BULK: bool>(board: &Board, depth: u8) -> u64 {
    let mut moves = MoveList::new();
    board.gen_moves(MoveFilter::All, &mut moves);

    if BULK && depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for entry in moves.iter() {
        let mv = entry.mv;
        let mut child = *board;
        child.make(&mv, &mut NullBoardObserver);
        nodes += if depth == 1 {
            1
        } else {
            perft_inner::<BULK>(&child, depth - 1)
        };
    }
    nodes
}
