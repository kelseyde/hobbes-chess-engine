use crate::board::movegen::MoveFilter;
use crate::board::Board;

pub fn perft<const BULK: bool>(board: &Board, depth: u8) -> u64 {
    let moves = board.gen_moves(MoveFilter::All);
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
            let nodes = if depth <= 1 { 1 } else { perft_inner::<BULK>(&child, depth - 1) };
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

    if depth == 0 {
        return 1;
    }

    let moves = board.gen_moves(MoveFilter::All);

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
        nodes += perft_inner::<BULK>(&child, depth - 1);
    }
    nodes
}

#[cfg(test)]
mod test {
    use crate::board::Board;
    use crate::tools::perft::perft;
    use std::fs;

    // #[test]
    fn test_perft_suite() {
        println!("reading file...");
        let perft_suite = fs::read_to_string("resources/standard.epd").unwrap();
        println!("parsed file!");

        for perft_test in perft_suite.lines() {
            let parts: Vec<&str> = perft_test.split(";").collect();

            println!("Parts: {:?}", parts);
            let fen = parts[0];

            let mut depth_nodes_str = parts.last().unwrap().split_whitespace();
            let depth_str = depth_nodes_str.next().unwrap();
            let nodes_str = depth_nodes_str.last().unwrap();
            let depth: u8 = depth_str[1..].parse().unwrap();
            let nodes: u64 = nodes_str.parse().unwrap();

            println!("Running test on fen for depth {}: {}", depth, fen);
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(perft::<false>(&board, depth), nodes, "Failed test: {}", fen);
        }
    }
}
