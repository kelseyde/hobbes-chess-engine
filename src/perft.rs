use crate::board::Board;
use crate::movegen::{gen_moves, is_check, MoveFilter};

pub fn perft(board: &Board, depth: u8) -> u64 {
    let moves = gen_moves(board, MoveFilter::All);
    if depth == 1 {
        let mut nodes = 0;
        for i in 0..moves.len {
            let mv = moves.list[i];
            let mut new_board = *board;
            new_board.make(&mv);
            if !is_check(&new_board, board.stm) {
                nodes += 1;
            }
        }
        return nodes;
    }

    let mut nodes = 0;
    for i in 0..moves.len {
        let mv = moves.list[i];
        let mut new_board = *board;
        new_board.make(&mv);
        if is_check(&new_board, board.stm) {
            continue;
        }
        let new_nodes = perft(&new_board, depth - 1);
        nodes += new_nodes;
    }

    nodes
}

#[cfg(test)]
mod test {
    use crate::board::Board;
    use crate::perft::perft;
    use std::fs;

    pub const PERFT_SUITE: [(&str, &str, u8, u64); 3] = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("en_passant_funhouse", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083)
    ];

    #[test]
    fn test_perft_suite() {

        println!("reading file...");
        let perft_suite = fs::read_to_string("resources/perft_suite.epd").unwrap();
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
            let board = Board::from_fen(fen);
            assert_eq!(perft(&board, depth), nodes, "Failed test: {}", fen);
        }
    }

    #[test]
    fn test_debug() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(fen);
        assert_eq!(perft(&board, 5), 4865609);
    }

}