use crate::board::Board;
use crate::movegen::gen_moves;

pub fn perft(board: Board, depth: u8) -> u64 {
    let moves = gen_moves(board);
    if depth == 1 {
        return moves.len as u64;
    }

    let mut nodes = 0;
    for mv in moves.iter() {
        let mut board = board.clone();
        board.make(mv);
        nodes += perft(board, depth - 1);
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
        assert_eq!(perft(board, 1), 20);
        assert_eq!(perft(board, 2), 400);
        assert_eq!(perft(board, 3), 8902);
        assert_eq!(perft(board, 4), 197281);
        assert_eq!(perft(board, 5), 4865609);
    }

}