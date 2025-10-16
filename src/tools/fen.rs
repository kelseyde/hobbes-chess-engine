use crate::board::castling::Rights;
use crate::board::file::File;
use crate::board::piece::Piece;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::board::zobrist::Zobrist;
use crate::board::Board;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Board {

    pub fn from_fen(fen: &str) -> Result<Board, String> {

        if fen.is_empty() {
            return Err("FEN string cannot be empty".to_string());
        }

        let mut board = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();

        if parts.len() < 4 || parts.len() > 6 {
            return Err("FEN string has an invalid number of parts".to_string());
        }

        let board_part = parts[0];
        let slash_count = board_part.matches('/').count();
        if slash_count != 7 {
            return Err("FEN string does not have exactly 8 ranks".to_string());
        }

        let white_kings = board_part.matches('K').count();
        let black_kings = board_part.matches('k').count();
        let king_count = white_kings + black_kings;
        if king_count != 2 {
            return Err("FEN string must have exactly one white and one black king".to_string());
        }
        if white_kings > 1 || black_kings > 1 {
            return Err("FEN string cannot have more than one king of the same color".to_string());
        }

        let rows: Vec<&str> = parts[0].split('/').collect();
        if rows.len() != 8 {
            panic!("Invalid FEN string");
        }

        for (rank, row) in rows.iter().enumerate() {

            let mut file = 0;
            for ch in row.chars() {
                if file >= 8 {
                    return Err("FEN string has too many squares in a rank".to_string());
                }
                match ch {
                    '1'..='8' => {
                        let squares = ch.to_digit(10).unwrap() as usize;
                        file += squares;
                    }
                    'P' | 'N' | 'B' | 'R' | 'Q' | 'K' | 'p' | 'n' | 'b' | 'r' | 'q' | 'k' => {
                        let sq = Square::from(File::parse(file), Rank::parse(7 - rank));
                        let piece = parse_piece(ch);
                        let side = if ch.is_uppercase() { White } else { Black };
                        board.toggle_sq(sq, piece, side);
                        file += 1;
                    }
                    _ => panic!("Invalid character in FEN string"),
                }
            }
            if file != 8 {
                return Err("FEN string does not add up to 8 squares in a rank".to_string());
            }
        }

        let stm_part = parts[1];
        if stm_part != "w" && stm_part != "b" {
            return Err("FEN string has an invalid side to move".to_string());
        }
        board.stm = parse_stm(stm_part);

        let rights_part = parts[2];
        (board.rights, board.frc) = parse_castle_rights(&board, rights_part);

        let ep_part = parts[3];
        board.ep_sq = parse_ep_sq(ep_part);

        let hm_part = parts.get(4).unwrap_or(&"0");
        board.hm = hm_part.parse().unwrap_or(0);

        let fm_part = parts.get(5).unwrap_or(&"0");
        board.fm = fm_part.parse().unwrap_or(0);

        board.keys.hash = Zobrist::get_hash(&board);
        board.keys.pawn_hash = Zobrist::get_pawn_hash(&board);
        board.keys.non_pawn_hashes = Zobrist::get_non_pawn_hashes(&board);
        board.keys.major_hash = Zobrist::get_major_hash(&board);
        board.keys.minor_hash = Zobrist::get_minor_hash(&board);
        Ok(board)
    }

    pub fn to_fen(self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty_squares = 0;
            for file in 0..8 {
                let sq = Square::from(File::parse(file), Rank::parse(rank));
                match self.piece_at(sq) {
                    Some(piece) => {
                        if empty_squares > 0 {
                            fen.push_str(&empty_squares.to_string());
                            empty_squares = 0;
                        }
                        fen.push(piece_to_char(
                            piece,
                            self.side_at(sq).expect("Square should be occupied"),
                        ));
                    }
                    None => {
                        empty_squares += 1;
                    }
                }
            }
            if empty_squares > 0 {
                fen.push_str(&empty_squares.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.stm == White { 'w' } else { 'b' });

        fen.push(' ');
        fen.push_str(self.rights.to_string(self.frc).as_str());

        fen.push(' ');
        if let Some(ep_sq) = self.ep_sq {
            fen.push((b'a' + (ep_sq.0 % 8)) as char);
            fen.push((b'1' + (ep_sq.0 / 8)) as char);
        } else {
            fen.push('-');
        }

        fen.push(' ');
        fen.push_str(&self.hm.to_string());
        fen.push(' ');
        fen.push_str(&self.fm.to_string());
        fen
    }
}


fn parse_castle_rights(board: &Board, castle: &str) -> (Rights, bool) {
    let white_king_sq = board.king_sq(White);
    let black_king_sq = board.king_sq(Black);
    let white_king_file = white_king_sq.file();
    let black_king_file = black_king_sq.file();

    let mut rights = Rights::default();
    let mut frc = false;

    for c in castle.chars() {
        match c {
            // Standard FEN notation
            'K' => if let Some(file) = find_rook_file(board, White, true){
                rights.set_kingside(White, file)
            },
            'Q' => if let Some(file) = find_rook_file(board, White, false) {
                rights.set_queenside(White, file)
            }
            'k' => if let Some(file) = find_rook_file(board, Black, true) {
                rights.set_kingside(Black, file)
            }
            'q' => if let Some(file) = find_rook_file(board, Black, false) {
                rights.set_queenside(Black, file)
            }
            '-' => (),

            // Shredder FEN notation
            'A'..='H' => {
                frc = true;
                let rook_file = File::from_char(c.to_ascii_lowercase()).unwrap();
                if rook_file > white_king_file {
                    // Rook is to the right of king = kingside
                    rights.set_kingside(White, rook_file);
                } else {
                    // Rook is to the left of king = queenside
                    rights.set_queenside(White, rook_file);
                }
            },

            // Shredder FEN notation
            'a'..='h' => {
                frc = true;
                let rook_file = File::from_char(c).unwrap();
                if rook_file > black_king_file {
                    // Rook is to the right of king = kingside
                    rights.set_kingside(Black, rook_file);
                } else {
                    // Rook is to the left of king = queenside
                    rights.set_queenside(Black, rook_file);
                }
            },

            _ => panic!("Invalid character in castle rights: {}", c),
        }
    }

    (rights, frc)
}

fn parse_ep_sq(ep_sq: &str) -> Option<Square> {
    if ep_sq == "-" {
        None
    } else {
        Some(parse_square(ep_sq))
    }
}

fn parse_stm(stm: &str) -> Side {
    match stm {
        "w" => White,
        "b" => Black,
        _ => panic!("Invalid side to move in FEN string"),
    }
}

fn parse_piece(c: char) -> Piece {
    match c.to_uppercase().next().unwrap() {
        'P' => Piece::Pawn,
        'N' => Piece::Knight,
        'B' => Piece::Bishop,
        'R' => Piece::Rook,
        'Q' => Piece::Queen,
        'K' => Piece::King,
        _ => panic!("Invalid piece character"),
    }
}

fn parse_square(s: &str) -> Square {
    let file = s.chars().next().unwrap() as usize - 'a' as usize;
    let rank = s.chars().nth(1).unwrap() as usize - '1' as usize;
    Square::from(File::parse(file), Rank::parse(rank))
}

fn piece_to_char(piece: Piece, side: Side) -> char {
    let ch = match piece {
        Piece::Pawn => 'p',
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::King => 'k',
    };
    if side == White {
        ch.to_ascii_uppercase()
    } else {
        ch
    }
}

fn find_rook_file(board: &Board, side: Side, kingside: bool) -> Option<File> {
    let king_sq = board.king_sq(side);
    let rooks = board.rooks(side);
    let candidate_files = if kingside {
        [File::H, File::G, File::F, File::E, File::D, File::C]
    } else {
        [File::A, File::B, File::C, File::D, File::E, File::F]
    };
    for &file in &candidate_files {
        let sq = Square::from(file, king_sq.rank());
        if rooks.contains(sq) {
            return Some(file);
        }
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null() {
        assert!(Board::from_fen("").is_err());
    }

    #[test]
    fn test_empty() {
        assert!(Board::from_fen("").is_err());
    }

    #[test]
    fn test_random_string() {
        assert!(Board::from_fen("random string").is_err());
    }

    #[test]
    fn test_too_few_parts() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w").is_err());
    }

    #[test]
    fn test_too_many_parts() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra").is_err());
    }

    #[test]
    fn test_board_has_too_few_ranks() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_too_many_ranks() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/rnbqkbnr w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_no_kings() {
        assert!(Board::from_fen("rnbqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQNBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_one_king() {
        assert!(Board::from_fen("rnbqkbkr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_too_many_kings() {
        assert!(Board::from_fen("rnbqkbnr/ppppkppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_two_white_kings() {
        assert!(Board::from_fen("rnbqKbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_two_black_kings() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQkBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_rank_does_not_add_up_to_eight() {
        assert!(Board::from_fen("rnbqkbnr/ppp2ppp/8/8/4p/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_empty_rank() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp//8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_rank_with_too_many_pieces() {
        assert!(Board::from_fen("rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_rank_with_too_few_pieces() {
        assert!(Board::from_fen("rnbqkbn/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_invalid_turn() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR white KQkq - 0 1").is_err());
    }

    #[test]
    fn test_valid() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_ok());
    }

    #[test]
    fn test_valid_shredder_fen() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1").is_ok());
    }

    #[test]
    fn test_valid_with_en_passant() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1").is_ok());
    }

    #[test]
    fn test_valid_missing_full_move() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0").is_ok());
    }

    #[test]
    fn test_valid_missing_half_move() {
        assert!(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0").is_ok());
    }

    #[test]
    fn test_dfrc_fens() {

        let board1 =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1").unwrap();
        assert!(board1.has_kingside_rights(White)
            && board1.has_queenside_rights(White)
            && board1.has_kingside_rights(Black)
            && board1.has_queenside_rights(Black));
        assert_eq!(Some(File::H), board1.rights.kingside(White));
        assert_eq!(Some(File::A), board1.rights.queenside(White));
        assert_eq!(Some(File::H), board1.rights.kingside(Black));
        assert_eq!(Some(File::A), board1.rights.queenside(Black));
        assert_eq!("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1", board1.to_fen());

        let board2 =
            Board::from_fen("bnnrkbqr/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBQR w HDhd -").unwrap();
        assert!(board2.has_kingside_rights(White)
            && board2.has_queenside_rights(White)
            && board2.has_kingside_rights(Black)
            && board2.has_queenside_rights(Black));
        assert_eq!(Some(File::H), board2.rights.kingside(White));
        assert_eq!(Some(File::D), board2.rights.queenside(White));
        assert_eq!(Some(File::H), board2.rights.kingside(Black));
        assert_eq!(Some(File::D), board2.rights.queenside(Black));
        assert_eq!("bnnrkbqr/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBQR w HDhd - 0 0", board2.to_fen());

        let board3 =
            Board::from_fen("nrkqbrnb/pppppppp/8/8/8/8/PPPPPPPP/NRKQBRNB w FBfb -").unwrap();
        assert!(board3.has_kingside_rights(White)
            && board3.has_queenside_rights(White)
            && board3.has_kingside_rights(Black)
            && board3.has_queenside_rights(Black));
        assert_eq!(Some(File::F), board3.rights.kingside(White));
        assert_eq!(Some(File::B), board3.rights.queenside(White));
        assert_eq!(Some(File::F), board3.rights.kingside(Black));
        assert_eq!(Some(File::B), board3.rights.queenside(Black));
        assert_eq!("nrkqbrnb/pppppppp/8/8/8/8/PPPPPPPP/NRKQBRNB w FBfb - 0 0", board3.to_fen());

    }
}