use std::collections::HashMap;
use crate::board::Board;
use crate::movegen::{gen_moves, is_check};

pub fn perft(board: &Board, depth: u8, start_depth: u8, debug: bool) -> u64 {
    let moves = gen_moves(board);
    if depth == 1 {
        let mut nodes = 0;
        for i in 0..moves.len {
            let mv = moves.list[i];
            let mut new_board = *board;
            new_board.make(&mv);
            if !is_check(&new_board, board.stm) {
                if debug {
                    println!("{}", &mv.to_uci());
                }
                nodes += 1;
            }
        }
        return nodes;
    }
    // if depth == 0 {
    //     return 1;
    // }

    let mut node_count: HashMap<String, u64> = HashMap::new();

    let mut nodes = 0;
    for i in 0..moves.len {
        let mv = moves.list[i];
        let mut new_board = *board;
        new_board.make(&mv);
        if is_check(&new_board, board.stm) {
            continue;
        }
        let new_nodes = perft(&new_board, depth - 1, start_depth, debug);
        node_count.insert(mv.to_uci(), new_nodes);
        nodes += new_nodes;
    }

    if debug && depth == start_depth {
        for (k, v) in node_count.iter() {
            println!("{}: {}", k, v);
        }
    }

    nodes
}

#[cfg(test)]
mod test {
    use crate::board::Board;
    use crate::perft::perft;

    #[test]
    fn test_startpos() {
        let board = Board::new();
        // assert_eq!(perft(board, 1, 1, false), 20);
        // assert_eq!(perft(board, 2, 2, false), 400);
        // assert_eq!(perft(board, 3, 3, true), 8902);
        // assert_eq!(perft(&board, 4, 4, false), 197281);
        // assert_eq!(perft(&board, 5, 5, true), 4865609);
        assert_eq!(perft(&board, 6, 6, false), 119060324);
    }

    #[test]
    fn test_kiwipete() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(fen);
        // assert_eq!(perft(&board, 1, 1, false), 48);
        // assert_eq!(perft(&board, 2, 2, false), 2039);
        // assert_eq!(perft(&board, 3, 3, true), 97862);
        assert_eq!(perft(&board, 4, 4, false), 4085603);
        // assert_eq!(perft(board, 5, 5, false), 193690690);
        // assert_eq!(perft(board, 6, 6, false), 8031647685);
    }

    #[test]
    fn test_debug() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/4P3/1pN2Q1p/PPPBBPPP/2R1K2R w Kkq - 0 2";
        let board = Board::from_fen(fen);
        assert_eq!(perft(&board, 1, 1, true), 47);
    }

}