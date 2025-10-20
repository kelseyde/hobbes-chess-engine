use crate::board::castling::Rights;
use crate::board::file::File;
use crate::board::piece::Piece;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::zobrist::Zobrist;
use crate::board::Board;

/// Derive the starting positions of the backrank pieces from a scharnagl index. Used for creating
/// a starting position for FRC and DFRC chess. Implementation based on Viridithas. Original source:
/// https://en.wikipedia.org/wiki/Fischer_random_chess_numbering_scheme#Direct_derivation
fn get_scharnagl_backrank(scharnagl: usize) -> [Piece; 8] {
    let mut backrank = [None; 8];
    let n = scharnagl;
    let (n2, b1) = (n / 4, n % 4);
    match b1 {
        0 => backrank[File::B] = Some(Piece::Bishop),
        1 => backrank[File::D] = Some(Piece::Bishop),
        2 => backrank[File::F] = Some(Piece::Bishop),
        3 => backrank[File::H] = Some(Piece::Bishop),
        _ => unreachable!(),
    }
    let (n3, b2) = (n2 / 4, n2 % 4);
    match b2 {
        0 => backrank[File::A] = Some(Piece::Bishop),
        1 => backrank[File::C] = Some(Piece::Bishop),
        2 => backrank[File::E] = Some(Piece::Bishop),
        3 => backrank[File::G] = Some(Piece::Bishop),
        _ => unreachable!(),
    }
    let (n4, mut q) = (n3 / 6, n3 % 6);
    for (idx, &piece) in backrank.iter().enumerate() {
        if piece.is_none() {
            if q == 0 {
                backrank[idx] = Some(Piece::Queen);
                break;
            }
            q -= 1;
        }
    }
    let remaining = backrank.iter_mut().filter(|pc| pc.is_none());
    let selection = match n4 {
        0 => [0, 1],
        1 => [0, 2],
        2 => [0, 3],
        3 => [0, 4],
        4 => [1, 2],
        5 => [1, 3],
        6 => [1, 4],
        7 => [2, 3],
        8 => [2, 4],
        9 => [3, 4],
        _ => unreachable!(),
    };
    for (i, slot) in remaining.enumerate() {
        if i == selection[0] || i == selection[1] {
            *slot = Some(Piece::Knight);
        }
    }

    backrank
        .iter_mut()
        .filter(|piece| piece.is_none())
        .zip([Piece::Rook, Piece::King, Piece::Rook])
        .for_each(|(slot, piece)| *slot = Some(piece));

    backrank.map(Option::unwrap)
}

impl Board {
    pub fn from_frc_idx(scharnagl: usize) -> Board {
        assert!(
            scharnagl < 960,
            "Scharnagl index must be in the range [0, 959)"
        );

        let mut board = Board::empty();
        board.frc = true;

        let backrank = get_scharnagl_backrank(scharnagl);

        // Set white pieces
        for (pc, file) in backrank.iter().zip(File::iter()) {
            let sq = Square::from(file, Rank::One);
            board.toggle_sq(sq, *pc, Side::White)
        }

        // Set white pawns
        for file in File::iter() {
            let sq = Square::from(file, Rank::Two);
            board.toggle_sq(sq, Piece::Pawn, Side::White);
        }

        // Set black pieces
        for (pc, file) in backrank.iter().zip(File::iter()) {
            let sq = Square::from(file, Rank::Eight);
            board.toggle_sq(sq, *pc, Side::Black)
        }

        // Set black pawns
        for file in File::iter() {
            let sq = Square::from(file, Rank::Seven);
            board.toggle_sq(sq, Piece::Pawn, Side::Black);
        }

        let mut rook_indices = backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == Piece::Rook {
                Some(i)
            } else {
                None
            }
        });
        let queenside_file = rook_indices.next().unwrap();
        let kingside_file = rook_indices.next().unwrap();
        board.rights = Rights::new(
            Some(File::parse(kingside_file)),
            Some(File::parse(queenside_file)),
            Some(File::parse(kingside_file)),
            Some(File::parse(queenside_file)),
        );

        board.keys.hash = Zobrist::get_hash(&board);
        board.keys.pawn_hash = Zobrist::get_pawn_hash(&board);
        board.keys.non_pawn_hashes = Zobrist::get_non_pawn_hashes(&board);
        board.keys.major_hash = Zobrist::get_major_hash(&board);
        board.keys.minor_hash = Zobrist::get_minor_hash(&board);
        board.threats = board.calc_threats(board.stm);
        board.checkers = board.calc_checkers(board.stm);
        board.pinned = board.calc_both_pinned();

        board
    }

    pub fn from_dfrc_idx(scharnagl: usize) -> Board {
        assert!(
            scharnagl < 960 * 960,
            "Double scharnagl index must be in the range [0, 921599)"
        );

        let mut board = Board::empty();
        board.frc = true;

        let white_backrank = get_scharnagl_backrank(scharnagl % 960);
        let black_backrank = get_scharnagl_backrank(scharnagl / 960);

        // Set white pieces
        for (pc, file) in white_backrank.iter().zip(File::iter()) {
            let sq = Square::from(file, Rank::One);
            board.toggle_sq(sq, *pc, Side::White)
        }

        // Set white pawns
        for file in File::iter() {
            let sq = Square::from(file, Rank::Two);
            board.toggle_sq(sq, Piece::Pawn, Side::White);
        }

        // Set black pieces
        for (pc, file) in black_backrank.iter().zip(File::iter()) {
            let sq = Square::from(file, Rank::Eight);
            board.toggle_sq(sq, *pc, Side::Black)
        }

        // Set black pawns
        for file in File::iter() {
            let sq = Square::from(file, Rank::Seven);
            board.toggle_sq(sq, Piece::Pawn, Side::Black);
        }

        let mut w_rook_indices = white_backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == Piece::Rook {
                Some(i)
            } else {
                None
            }
        });
        let w_queenside_file = w_rook_indices.next().unwrap();
        let w_kingside_file = w_rook_indices.next().unwrap();

        let mut b_rook_indices = black_backrank.iter().enumerate().filter_map(|(i, &piece)| {
            if piece == Piece::Rook {
                Some(i)
            } else {
                None
            }
        });
        let b_queenside_file = b_rook_indices.next().unwrap();
        let b_kingside_file = b_rook_indices.next().unwrap();

        board.rights = Rights::new(
            Some(File::parse(w_kingside_file)),
            Some(File::parse(w_queenside_file)),
            Some(File::parse(b_kingside_file)),
            Some(File::parse(b_queenside_file)),
        );

        board.keys.hash = Zobrist::get_hash(&board);
        board.keys.pawn_hash = Zobrist::get_pawn_hash(&board);
        board.keys.non_pawn_hashes = Zobrist::get_non_pawn_hashes(&board);
        board.keys.major_hash = Zobrist::get_major_hash(&board);
        board.keys.minor_hash = Zobrist::get_minor_hash(&board);
        board.threats = board.calc_threats(board.stm);
        board.checkers = board.calc_checkers(board.stm);
        board.pinned = board.calc_both_pinned();

        board
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_frc_idx_unique_hashes() {
        let mut hashes = HashSet::new();

        for scharnagl in 0..960 {
            let board = Board::from_frc_idx(scharnagl);
            let hash = board.keys.hash;

            assert!(
                hashes.insert(hash),
                "Duplicate hash found for FRC index {}: hash = {}",
                scharnagl,
                hash
            );
        }

        assert_eq!(
            hashes.len(),
            960,
            "Expected 960 unique hashes for FRC positions"
        );
    }

    #[test]
    fn test_frc_idx_bounds() {
        let _board = Board::from_frc_idx(0);
        let _board = Board::from_frc_idx(959);
    }

    #[test]
    #[should_panic(expected = "Scharnagl index must be in the range [0, 959)")]
    fn test_frc_idx_invalid_bound() {
        Board::from_frc_idx(960);
    }

    #[test]
    fn test_dfrc_idx_bounds() {
        let _board = Board::from_dfrc_idx(0);
        let _board = Board::from_dfrc_idx(921599);
    }

    #[test]
    #[should_panic(expected = "Double scharnagl index must be in the range [0, 921599)")]
    fn test_dfrc_idx_invalid_bound() {
        Board::from_dfrc_idx(921600);
    }
}
