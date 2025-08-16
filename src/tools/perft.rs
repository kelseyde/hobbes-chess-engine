use crate::board::movegen::{is_check, MoveFilter};
use crate::board::Board;
use std::collections::HashMap;

pub fn perft(board: &Board, depth: u8, original_depth: u8) -> u64 {
    let moves = board.gen_moves(MoveFilter::All);

    let mut move_counts = if depth == original_depth { Some(HashMap::new()) } else { None };

    if depth == 1 {
        let mut nodes = 0;
        for i in 0..moves.len {
            let mv = moves.list[i].mv;
            let mut new_board = *board;
            new_board.make(&mv);
            if !is_check(&new_board, board.stm) {
                nodes += 1;
                if let Some(ref mut counts) = move_counts {
                    *counts.entry(mv.to_uci()).or_insert(0) += 1;
                }
            }
        }
        if let Some(counts) = move_counts {
            let mut entries: Vec<_> = counts.into_iter().collect();
            entries.sort_by_key(|(mv, _)| mv.clone());

            for (mv, count) in entries {
                println!("{} - {}", mv, count);
            }
        }
        return nodes;
    }

    let mut nodes = 0;
    for i in 0..moves.len {
        let mv = moves.list[i].mv;
        let mut new_board = *board;
        new_board.make(&mv);
        if is_check(&new_board, board.stm) {
            continue;
        }
        let new_nodes = perft(&new_board, depth - 1, original_depth);
        if let Some(ref mut counts) = move_counts {
            *counts.entry(mv.to_uci()).or_insert(0) += new_nodes;
        }
        nodes += new_nodes;
    }

    if let Some(counts) = move_counts {
        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_by_key(|(mv, _)| mv.clone());

        for (mv, count) in entries {
            println!("{} - {}", mv, count);
        }
    }

    nodes
}

#[cfg(test)]
mod test {
    use crate::board::ray;
    use crate::board::Board;
    use crate::moves::{Move, MoveFlag};
    use crate::perft::perft;
    use std::fs;

    #[test]
    fn test_perft_suite() {

        println!("reading file...");
        let perft_suite = fs::read_to_string("resources/standard.epd.epd").unwrap();
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
            assert_eq!(perft(&board, depth, depth), nodes, "Failed test: {}", fen);
        }
    }

    #[test]
    fn test_debug() {
        let fen = "b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9";
        let mut board = Board::from_fen(fen).unwrap();

        ray::init();
        board.make(&Move::parse_uci_with_flag("f1g1", MoveFlag::CastleK));
        println!("{}", board.to_fen());

    }

}