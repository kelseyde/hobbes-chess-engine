use crate::board::castling::Rights;
use crate::board::file::File;
use crate::board::piece::Piece;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::board::zobrist::Hashes;
use crate::board::Board;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Board {

    /// Parses a FEN into a `Board`. Accepts both standard castling notation (`KQkq`) and Shredder
    /// notation (e.g. `HAha`) for DFRC positions. The half-move and full-move counters are optional
    /// and default to 0 if omitted.
    pub fn from_fen(fen: &str) -> Result<Board, String> {
        if fen.is_empty() {
            return Err("FEN string is empty".to_string());
        }

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return Err("FEN string is empty".to_string());
        }
        if parts.len() < 4 || parts.len() > 6 {
            return Err(format!(
                "FEN string must have 4 to 6 fields, found {}",
                parts.len()
            ));
        }

        let mut board = Board::empty();
        parse_board(&mut board, parts[0])?;
        board.stm = parse_stm(parts[1])?;
        board.rights = parse_castle_rights(&board, parts[2])?;
        board.ep_sq = parse_ep_sq(&board, parts[3])?;
        board.hm = parse_counter(parts.get(4).copied().unwrap_or("0"), "half-move clock")?;
        board.fm = parse_counter(parts.get(5).copied().unwrap_or("0"), "full-move clock")?;

        board.hashes = Hashes::new(&board);
        board.threats = board.calc_threats(board.stm);
        board.checkers = board.calc_checkers(board.stm);
        board.pinned = board.calc_both_pinned();
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

fn parse_board(board: &mut Board, board_part: &str) -> Result<(), String> {
    let ranks: Vec<&str> = board_part.split('/').collect();
    if ranks.len() != 8 {
        return Err(format!(
            "FEN board must have 8 ranks separated by '/', found {}",
            ranks.len()
        ));
    }

    let (mut w_kings, mut b_kings) = (0, 0);

    for (i, rank_str) in ranks.iter().enumerate() {
        // Ranks are ordered 8-to-1.
        let rank_number = 8 - i;
        let mut file = 0usize;

        for ch in rank_str.chars() {
            match ch {
                '1'..='8' => {
                    file += ch.to_digit(10).unwrap() as usize;
                    if file > 8 {
                        return Err(format!(
                            "rank {} has too many squares, expected 8",
                            rank_number
                        ));
                    }
                }
                _ => {
                    if file >= 8 {
                        return Err(format!(
                            "rank {} has too many squares, expected 8",
                            rank_number
                        ));
                    }
                    let piece = parse_piece(ch).ok_or_else(|| {
                        format!("invalid character '{}' in rank {}", ch, rank_number)
                    })?;
                    let side = if ch.is_ascii_uppercase() {
                        White
                    } else {
                        Black
                    };

                    if piece == Piece::Pawn && (rank_number == 1 || rank_number == 8) {
                        return Err(format!("pawns cannot be placed on rank {}", rank_number));
                    }
                    if piece == Piece::King {
                        match side {
                            White => w_kings += 1,
                            Black => b_kings += 1,
                        }
                    }

                    let sq = Square::from(File::parse(file), Rank::parse(rank_number - 1));
                    board.toggle_sq(sq, piece, side);
                    file += 1;
                }
            }
        }

        if file != 8 {
            return Err(format!(
                "rank {} describes {} squares, expected 8",
                rank_number, file
            ));
        }
    }

    match (w_kings, b_kings) {
        (1, 1) => Ok(()),
        (0, _) => Err("FEN is missing a white king!".to_string()),
        (_, 0) => Err("FEN is missing a black king!".to_string()),
        (w, _) if w > 1 => Err(format!("FEN has {} white kings!", w)),
        (_, b) => Err(format!("FEN has {} black kings!", b)),
    }
}

fn parse_stm(part: &str) -> Result<Side, String> {
    match part {
        "w" => Ok(White),
        "b" => Ok(Black),
        _ => Err(format!("side to move must be 'w' or 'b', found '{}'", part)),
    }
}

fn parse_castle_rights(board: &Board, castle: &str) -> Result<Rights, String> {
    if castle == "-" {
        return Ok(Rights::default());
    }
    if castle.contains('-') {
        return Err("castling rights '-' cannot be combined with other rights".to_string());
    }

    let w_king_file = board.king_sq(White).file();
    let b_king_file = board.king_sq(Black).file();

    let mut rights = Rights::default();

    for c in castle.chars() {
        match c {
            // Standard FEN notation
            'K' => {
                let file = find_rook_file(board, White, true).ok_or_else(|| {
                    "no white rook available for kingside castling ('K')".to_string()
                })?;
                rights.set_kingside(White, file);
            }
            'Q' => {
                let file = find_rook_file(board, White, false).ok_or_else(|| {
                    "no white rook available for queenside castling ('Q')".to_string()
                })?;
                rights.set_queenside(White, file);
            }
            'k' => {
                let file = find_rook_file(board, Black, true).ok_or_else(|| {
                    "no black rook available for kingside castling ('k')".to_string()
                })?;
                rights.set_kingside(Black, file);
            }
            'q' => {
                let file = find_rook_file(board, Black, false).ok_or_else(|| {
                    "no black rook available for queenside castling ('q')".to_string()
                })?;
                rights.set_queenside(Black, file);
            }

            // Shredder FEN notation
            'A'..='H' => {
                let rook_file = File::from_char(c.to_ascii_lowercase()).unwrap();
                set_shredder_right(board, &mut rights, White, w_king_file, rook_file, c)?;
            }
            'a'..='h' => {
                let rook_file = File::from_char(c).unwrap();
                set_shredder_right(board, &mut rights, Black, b_king_file, rook_file, c)?;
            }

            _ => {
                return Err(format!(
                    "invalid character '{}' in castling rights field",
                    c
                ))
            }
        }
    }

    Ok(rights)
}

fn set_shredder_right(
    board: &Board,
    rights: &mut Rights,
    side: Side,
    king_file: File,
    rook_file: File,
    c: char,
) -> Result<(), String> {
    if rook_file == king_file {
        return Err(format!(
            "castling right '{}' refers to the king's own file",
            c
        ));
    }

    let king_rank = board.king_sq(side).rank();
    let rook_sq = Square::from(rook_file, king_rank);
    if !board.rooks(side).contains(rook_sq) {
        return Err(format!(
            "castling right '{}' has no matching rook on {}{}",
            c,
            rook_file.to_char(),
            king_rank.to_char()
        ));
    }

    if rook_file > king_file {
        rights.set_kingside(side, rook_file);
    } else {
        rights.set_queenside(side, rook_file);
    }
    Ok(())
}

/// Parses the en passant target square field.
fn parse_ep_sq(board: &Board, ep: &str) -> Result<Option<Square>, String> {
    if ep == "-" {
        return Ok(None);
    }

    let chars: Vec<char> = ep.chars().collect();
    if chars.len() != 2 {
        return Err(format!(
            "en passant square must be '-' or two characters, found '{}'",
            ep
        ));
    }

    let file = File::from_char(chars[0])
        .ok_or_else(|| format!("en passant file must be 'a'-'h', found '{}'", chars[0]))?;
    let rank = Rank::from_char(chars[1])
        .ok_or_else(|| format!("en passant rank must be '3' or '6', found '{}'", chars[1]))?;
    if rank != Rank::Three && rank != Rank::Six {
        return Err(format!(
            "en passant rank must be '3' or '6', found '{}'",
            chars[1]
        ));
    }

    let expected_rank = if board.stm == White {
        Rank::Six
    } else {
        Rank::Three
    };
    if rank != expected_rank {
        return Err(format!(
            "en passant square '{}' is inconsistent with side to move '{}'",
            ep,
            if board.stm == White { "w" } else { "b" }
        ));
    }

    let sq = Square::from(file, rank);
    // The side that just moved (and could be captured en passant) is the opposite of `stm`.
    let mover = !board.stm;
    let pawn_sq = if mover == White {
        sq.plus(8)
    } else {
        sq.minus(8)
    };
    let origin_sq = if mover == White {
        sq.minus(8)
    } else {
        sq.plus(8)
    };

    if board.piece_at(sq).is_some() {
        return Err(format!("en passant square '{}' is not empty", ep));
    }
    if board.piece_at(pawn_sq) != Some(Piece::Pawn) || !board.colours[mover].contains(pawn_sq) {
        return Err(format!(
            "en passant square '{}' has no pawn able to make that capture",
            ep
        ));
    }
    if board.piece_at(origin_sq).is_some() {
        return Err(format!(
            "en passant square '{}' is inconsistent with the board position",
            ep
        ));
    }

    Ok(Some(sq))
}

/// Parses the half-move clock / full-move clock fields.
fn parse_counter(part: &str, name: &str) -> Result<u8, String> {
    part.parse::<u8>().map_err(|_| {
        format!("{} is invalid, found '{}'", name, part)
    })
}

fn parse_piece(c: char) -> Option<Piece> {
    match c.to_ascii_uppercase() {
        'P' => Some(Piece::Pawn),
        'N' => Some(Piece::Knight),
        'B' => Some(Piece::Bishop),
        'R' => Some(Piece::Rook),
        'Q' => Some(Piece::Queen),
        'K' => Some(Piece::King),
        _ => None,
    }
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

    fn init_tables() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            crate::board::magics::init();
            crate::board::ray::init();
        });
    }

    fn from_fen(fen: &str) -> Result<Board, String> {
        init_tables();
        Board::from_fen(fen)
    }

    fn expect_err(result: Result<Board, String>) -> String {
        match result {
            Err(e) => e,
            Ok(_) => panic!("expected FEN parsing to fail"),
        }
    }

    #[test]
    fn test_null() {
        assert!(from_fen("").is_err());
    }

    #[test]
    fn test_whitespace_only() {
        assert!(from_fen("   ").is_err());
    }

    #[test]
    fn test_random_string() {
        assert!(from_fen("random string").is_err());
    }

    #[test]
    fn test_too_few_parts() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w").is_err());
    }

    #[test]
    fn test_too_many_parts() {
        assert!(
            from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra").is_err()
        );
    }

    #[test]
    fn test_extra_whitespace_between_fields_is_tolerated() {
        assert!(
            from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR   w   KQkq  -  0  1").is_ok()
        );
    }

    #[test]
    fn test_leading_and_trailing_whitespace_is_tolerated() {
        assert!(from_fen("  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1  ").is_ok());
    }

    #[test]
    fn test_board_has_too_few_ranks() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_too_many_ranks() {
        assert!(
            from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/rnbqkbnr w KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn test_board_missing_white_king() {
        let err = expect_err(from_fen(
            "rnbqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQNBNR w KQkq - 0 1",
        ));
        assert!(err.contains("white king"), "unexpected error: {}", err);
    }

    #[test]
    fn test_board_missing_black_king() {
        let err = expect_err(from_fen(
            "rnbqbbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ));
        assert!(err.contains("black king"), "unexpected error: {}", err);
    }

    #[test]
    fn test_board_has_too_many_kings() {
        assert!(from_fen("rnbqkbnr/ppppkppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_board_has_two_white_kings() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/3K4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ));
        assert!(err.contains("white king"), "unexpected error: {}", err);
    }

    #[test]
    fn test_board_has_two_black_kings() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/3k4/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ));
        assert!(err.contains("black king"), "unexpected error: {}", err);
    }

    #[test]
    fn test_rank_does_not_add_up_to_eight() {
        assert!(from_fen("rnbqkbnr/ppp2ppp/8/8/4p/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_empty_rank() {
        assert!(from_fen("rnbqkbnr/pppppppp//8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_rank_with_too_many_pieces() {
        assert!(from_fen("rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_rank_with_too_few_pieces() {
        assert!(from_fen("rnbqkbn/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_invalid_piece_character() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPx/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_pawn_on_first_rank_is_rejected() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBPR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_pawn_on_last_rank_is_rejected() {
        assert!(from_fen("rnbqkbPr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err());
    }

    #[test]
    fn test_invalid_turn() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR white KQkq - 0 1").is_err());
    }

    #[test]
    fn test_turn_is_case_sensitive() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR W KQkq - 0 1").is_err());
    }

    #[test]
    fn test_invalid_castling_character() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkx - 0 1").is_err());
    }

    #[test]
    fn test_dash_combined_with_other_castling_rights_is_rejected() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w K- - 0 1").is_err());
    }

    #[test]
    fn test_castling_right_without_matching_rook_is_rejected() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBN1 w KQkq - 0 1",
        ));
        assert!(err.contains("rook"), "unexpected error: {}", err);
    }

    #[test]
    fn test_shredder_castling_right_without_matching_rook_is_rejected() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBN1 w AHah - 0 1",
        ));
        assert!(err.contains("rook"), "unexpected error: {}", err);
    }

    #[test]
    fn test_shredder_castling_right_on_kings_own_file_is_rejected() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Eeq - 0 1").is_err());
    }

    #[test]
    fn test_valid_with_en_passant() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").is_ok());
    }

    #[test]
    fn test_ep_square_wrong_length() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e33 0 1").is_err());
    }

    #[test]
    fn test_ep_square_invalid_file() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq z3 0 1").is_err());
    }

    #[test]
    fn test_ep_square_invalid_rank() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e9 0 1").is_err());
    }

    #[test]
    fn test_ep_square_rank_not_three_or_six() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1").is_err());
    }

    #[test]
    fn test_ep_square_inconsistent_with_side_to_move() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1",
        ));
        assert!(err.contains("side to move"), "unexpected error: {}", err);
    }

    #[test]
    fn test_ep_square_with_no_pawn_present_is_rejected() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq e3 0 1",
        ));
        assert!(err.contains("pawn"), "unexpected error: {}", err);
    }

    #[test]
    fn test_ep_square_with_occupied_origin_square_is_rejected() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPNPPP/RNBQKB1R b KQkq e3 0 1",
        ));
        assert!(err.contains("board position"), "unexpected error: {}", err);
    }

    #[test]
    fn test_ep_square_not_empty_is_rejected() {
        let err = expect_err(from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/4N3/PPPP1PPP/RNBQKB1R b KQkq e3 0 1",
        ));
        assert!(err.contains("not empty"), "unexpected error: {}", err);
    }

    #[test]
    fn test_valid_missing_full_move() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0").is_ok());
    }

    #[test]
    fn test_valid_missing_half_move() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -").is_ok());
    }

    #[test]
    fn test_half_move_clock_not_numeric() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - abc 1").is_err());
    }

    #[test]
    fn test_half_move_clock_overflows_u8() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 256 1").is_err());
    }

    #[test]
    fn test_half_move_clock_negative_is_rejected() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1").is_err());
    }

    #[test]
    fn test_full_move_number_not_numeric() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 abc").is_err());
    }

    #[test]
    fn test_full_move_number_at_u8_max_is_accepted() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 255").is_ok());
    }

    #[test]
    fn test_valid() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_ok());
    }

    #[test]
    fn test_valid_shredder_fen() {
        assert!(from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1").is_ok());
    }

    #[test]
    fn test_dfrc() {
        let mut board1 =
            from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1").unwrap();
        board1.set_frc(true);
        assert!(
            board1.has_kingside_rights(White)
                && board1.has_queenside_rights(White)
                && board1.has_kingside_rights(Black)
                && board1.has_queenside_rights(Black)
        );
        assert_eq!(Some(File::H), board1.rights.kingside(White));
        assert_eq!(Some(File::A), board1.rights.queenside(White));
        assert_eq!(Some(File::H), board1.rights.kingside(Black));
        assert_eq!(Some(File::A), board1.rights.queenside(Black));
        assert_eq!(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1",
            board1.to_fen()
        );

        let mut board2 = from_fen("bnnrkbqr/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBQR w HDhd -").unwrap();
        board2.set_frc(true);
        assert!(
            board2.has_kingside_rights(White)
                && board2.has_queenside_rights(White)
                && board2.has_kingside_rights(Black)
                && board2.has_queenside_rights(Black)
        );
        assert_eq!(Some(File::H), board2.rights.kingside(White));
        assert_eq!(Some(File::D), board2.rights.queenside(White));
        assert_eq!(Some(File::H), board2.rights.kingside(Black));
        assert_eq!(Some(File::D), board2.rights.queenside(Black));
        assert_eq!(
            "bnnrkbqr/pppppppp/8/8/8/8/PPPPPPPP/BNNRKBQR w HDhd - 0 0",
            board2.to_fen()
        );

        let mut board3 = from_fen("nrkqbrnb/pppppppp/8/8/8/8/PPPPPPPP/NRKQBRNB w FBfb -").unwrap();
        board3.set_frc(true);
        assert!(
            board3.has_kingside_rights(White)
                && board3.has_queenside_rights(White)
                && board3.has_kingside_rights(Black)
                && board3.has_queenside_rights(Black)
        );
        assert_eq!(Some(File::F), board3.rights.kingside(White));
        assert_eq!(Some(File::B), board3.rights.queenside(White));
        assert_eq!(Some(File::F), board3.rights.kingside(Black));
        assert_eq!(Some(File::B), board3.rights.queenside(Black));
        assert_eq!(
            "nrkqbrnb/pppppppp/8/8/8/8/PPPPPPPP/NRKQBRNB w FBfb - 0 0",
            board3.to_fen()
        );
    }
}
