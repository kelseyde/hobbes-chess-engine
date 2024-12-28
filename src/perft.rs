use crate::board::Board;
use crate::movegen::{gen_moves, is_check, MoveFilter};

pub fn perft(board: &Board, depth: u8, start_depth: u8, debug: bool) -> u64 {
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

    //let mut node_count: HashMap<String, u64> = HashMap::new();

    let mut nodes = 0;
    for i in 0..moves.len {
        let mv = moves.list[i];
        let mut new_board = *board;
        new_board.make(&mv);
        if is_check(&new_board, board.stm) {
            continue;
        }
        let new_nodes = perft(&new_board, depth - 1, start_depth, debug);
        //node_count.insert(mv.to_uci(), new_nodes);
        nodes += new_nodes;
    }

    // if debug && depth == start_depth {
    //     for (k, v) in node_count.iter() {
    //         println!("{}: {}", k, v);
    //     }
    // }

    nodes
}

#[cfg(test)]
mod test {
    use crate::board::Board;
    use crate::perft::perft;

    pub const PERFT_SUITE: [(&str, &str, u8, u64); 3] = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
        ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("en_passant_funhouse", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083)
    ];

    #[test]
    fn test_perft_suite() {
        for (name, fen, depth, nodes) in PERFT_SUITE.iter() {
            let board = Board::from_fen(fen);
            assert_eq!(perft(&board, *depth, *depth, false), *nodes, "Failed test: {}", name);
        }
    }

    #[test]
    fn test_debug() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(fen);
        assert_eq!(perft(&board, 5, 5, false), 4865609);
    }

}