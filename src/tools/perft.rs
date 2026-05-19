use crate::board::Board;
use crate::board::movegen::MoveFilter;
use crate::board::moves::MoveList;

pub fn perft<const BULK: bool>(board: &Board, depth: u8) -> u64 {
    let mut moves = MoveList::new();
    board.gen_moves(MoveFilter::All, &mut moves);
    let mut total = 0;

    let mut entries: Vec<(String, u64)> = moves
        .iter()
        .filter_map(|entry| {
            let mv = entry.mv;
            if !board.is_legal(&mv) {
                return None;
            }
            let mut child = *board;
            child.make(&mv);
            let nodes = if depth <= 1 {
                1
            } else {
                perft_inner::<BULK>(&child, depth - 1)
            };
            total += nodes;
            Some((mv.to_uci(), nodes))
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
        return moves.iter().filter(|e| board.is_legal(&e.mv)).count() as u64;
    }

    let mut nodes = 0;
    for entry in moves.iter() {
        let mv = entry.mv;
        if !board.is_legal(&mv) {
            continue;
        }
        let mut child = *board;
        child.make(&mv);
        nodes += if depth == 1 {
            1
        } else {
            perft_inner::<BULK>(&child, depth - 1)
        };
    }
    nodes
}
