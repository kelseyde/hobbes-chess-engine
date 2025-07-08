use crate::attacks;
use crate::board::Board;
use crate::consts::{Piece, Side};
use crate::moves::Move;
use crate::types::bitboard::Bitboard;
use crate::types::square::Square;

const PIECE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];

pub fn value(pc: Piece) -> i32 {
    PIECE_VALUES[pc]
}

pub fn see(board: &Board, mv: &Move, threshold: i32) -> bool {

    let from = mv.from();
    let to = mv.to();

    let next_victim = mv.promo_piece().map_or_else(
        || board.piece_at(from).unwrap(),
        |promo| promo,
    );

    let mut balance = move_value(board, mv) - threshold;

    if balance < 0 {
        return false;
    }

    balance -= PIECE_VALUES[next_victim];

    if balance >= 0 {
        return true;
    }

    let mut occ = board.occ() ^ Bitboard::of_sq(from) ^ Bitboard::of_sq(to);

    if let Some(ep_sq) = board.ep_sq {
        occ ^= Bitboard::of_sq(ep_sq);
    }

    let mut attackers = attackers_to(board, to, occ) & occ;
    let diagonal = board.pcs(Piece::Bishop) | board.pcs(Piece::Queen);
    let orthogonal = board.pcs(Piece::Rook) | board.pcs(Piece::Queen);

    let mut stm = board.stm.flip();

    loop {
        let our_attackers = attackers & board.side(stm);
        if our_attackers.is_empty() {
            break;
        }

        let attacker = least_valuable_attacker(board, our_attackers);

        if attacker == Piece::King && !(attackers & board.side(stm.flip())).is_empty() {
            break;
        }

        // Make the capture
        let pcs = board.pcs(attacker) & our_attackers;
        let sq = (our_attackers & pcs).lsb();
        occ = occ.pop_bit(sq);
        stm = stm.flip();

        balance = -balance - 1 - PIECE_VALUES[attacker];
        if balance >= 0 {
            break;
        }

        // Capturing may reveal a new slider
        if [Piece::Pawn, Piece::Bishop, Piece::Queen].contains(&attacker) {
            attackers |= attacks::bishop(to, occ) & diagonal;
        }
        if [Piece::Rook, Piece::Queen].contains(&attacker) {
            attackers |= attacks::rook(to, occ) & orthogonal;
        }
        attackers &= occ;
    }

    stm != board.stm
}

fn move_value(board: &Board, mv: &Move) -> i32 {
    let mut value = board.piece_at(mv.to())
        .map_or(0, |captured| PIECE_VALUES[captured]);

    if let Some(promo) = mv.promo_piece() {
        value += PIECE_VALUES[promo];
    } else if mv.is_ep() {
        value = PIECE_VALUES[Piece::Pawn];
    }
    value
}

fn least_valuable_attacker(board: &Board, our_attackers: Bitboard) -> Piece {
    if !(our_attackers & board.pcs(Piece::Pawn)).is_empty() {
        return Piece::Pawn;
    }
    if !(our_attackers & board.pcs(Piece::Knight)).is_empty() {
        return Piece::Knight;
    }
    if !(our_attackers & board.pcs(Piece::Bishop)).is_empty() {
        return Piece::Bishop;
    }
    if !(our_attackers & board.pcs(Piece::Rook)).is_empty() {
        return Piece::Rook;
    }
    if !(our_attackers & board.pcs(Piece::Queen)).is_empty() {
        return Piece::Queen;
    }
    if !(our_attackers & board.pcs(Piece::King)).is_empty() {
        return Piece::King;
    }
    panic!("No attackers found");
}

fn attackers_to(board: &Board, square: Square, occupancies: Bitboard) -> Bitboard {
    let diagonals = board.pcs(Piece::Bishop) | board.pcs(Piece::Queen);
    let orthogonals = board.pcs(Piece::Rook) | board.pcs(Piece::Queen);
    let white_pawn_attacks = attacks::pawn(square, Side::Black) & board.pawns(Side::White);
    let black_pawn_attacks = attacks::pawn(square, Side::White) & board.pawns(Side::Black);
    let knight_attacks = attacks::knight(square) & board.pcs(Piece::Knight);
    let diagonal_attacks = attacks::bishop(square, occupancies) & diagonals;
    let orthogonal_attacks = attacks::rook(square, occupancies) & orthogonals;
    let king_attacks = attacks::king(square) & board.pcs(Piece::King);
    white_pawn_attacks
        | black_pawn_attacks
        | knight_attacks
        | diagonal_attacks
        | orthogonal_attacks
        | king_attacks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen;
    use crate::movegen::MoveFilter;
    use std::fs;

    #[test]
    fn test_see_suite() {

        let see_suite = fs::read_to_string("resources/see_suite.epd").unwrap();

        let mut tried = 0;
        let mut passed = 0;

        for see_test in see_suite.lines() {
            let parts: Vec<&str> = see_test.split("|").collect();
            let fen = parts[0].trim();
            let mv_uci = parts[1].trim();
            let threshold_str = parts[2].trim();
            let threshold: i32 = threshold_str.parse().unwrap();

            let board = Board::from_fen(fen);
            let moves = movegen::gen_moves(&board, MoveFilter::All);
            let mv = moves.iter()
                .map(|entry| entry.mv)
                .find(|m| m.to_uci() == mv_uci)
                .expect("Move not found in generated moves");

            tried += 1;
            if see(&board, &mv, threshold) {
                passed += 1;
            } else {
                println!("Failed SEE test for FEN: {} and move: {}", fen, mv_uci);
            }

        }

        assert_eq!(passed, tried, "Passed {} out of {} SEE tests", passed, tried);


    }

    #[test]
    fn debug_single() {

        let line = "r2qkbn1/ppp1pp1p/3p1rp1/3Pn3/4P1b1/2N2N2/PPP2PPP/R1BQKB1R b KQq - | g4f3 | 100 | N - B + P";
        let parts: Vec<&str> = line.split("|").collect();
        let fen = parts[0].trim();
        let mv_uci = parts[1].trim();
        let threshold_str = parts[2].trim();
        let threshold: i32 = threshold_str.parse().unwrap();

        let board = Board::from_fen(fen);
        let moves = movegen::gen_moves(&board, MoveFilter::All);
        let mv = moves.iter()
            .map(|entry| entry.mv)
            .find(|m| m.to_uci() == mv_uci)
            .expect("Move not found in generated moves");


        assert!(see(&board, &mv, threshold), "Failed SEE test for FEN: {} and move: {}", fen, mv_uci);

    }
}