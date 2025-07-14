use crate::attacks;
use crate::board::{Board, CastleSafety, CastleTravel};
use crate::movegen::MoveFilter::Quiets;
use crate::moves::{MoveFlag, MoveList};
use crate::types::bitboard::Bitboard;
use crate::types::piece::Piece;
use crate::types::side::Side;
use crate::types::side::Side::White;
use crate::types::square::Square;
use crate::types::{File, Rank};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MoveFilter {
    All,
    Quiets,
    Noisies,
    Captures
}

pub fn gen_moves(board: &Board, filter: MoveFilter) -> MoveList {
    let side = board.stm;
    let mut moves = MoveList::new();

    let us = board.us();
    let them = board.them();
    let occ = us | them;

    // handle special moves first (en passant, promo, castling etc.)
    gen_pawn_moves(board, side, occ, them, filter, &mut moves);
    if filter != MoveFilter::Captures && filter != MoveFilter::Noisies {
        gen_castle_moves(board, side, &mut moves);
    }

    let filter_mask = match filter {
        MoveFilter::All => Bitboard::ALL,
        MoveFilter::Quiets => !them,
        MoveFilter::Noisies => them,
        MoveFilter::Captures => them
    };

    // handle standard moves
    for &pc in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King].iter() {
        for from in board.pcs(pc) & us {
            let attacks = attacks::attacks(from, pc, side, occ) & !us & filter_mask;
            for to in attacks {
                moves.add_move(from, to, MoveFlag::Standard);
            }
        }
    }

    moves
}

#[inline(always)]
fn gen_pawn_moves(board: &Board, side: Side, occ: Bitboard, them: Bitboard, filter: MoveFilter, moves: &mut MoveList) {

    let pawns = board.pcs(Piece::Pawn) & board.side(side);

    if filter != MoveFilter::Captures && filter != MoveFilter::Noisies {

        for to in single_push(pawns, side, occ) {
            let from = if side == White { to.minus(8) } else { to.plus(8) };
            moves.add_move(from, to, MoveFlag::Standard);
        }

        for to in double_push(pawns, side, occ) {
            let from = if side == White { to.minus(16) } else { to.plus(16) };
            moves.add_move(from, to, MoveFlag::DoublePush);
        }

        for to in push_promos(pawns, side, occ) {
            let from = if side == White { to.minus(8) } else { to.plus(8) };
            add_promos(moves, from, to);
        }

    }

    if filter != Quiets {
        for to in left_capture(pawns, side, them) {
            let from = if side == White { to.minus(7) } else { to.plus(9) };
            moves.add_move(from, to, MoveFlag::Standard);
        }

        for to in right_capture(pawns, side, them) {
            let from = if side == White { to.minus(9) } else { to.plus(7) };
            moves.add_move(from, to, MoveFlag::Standard);
        }

        if let Some(ep_sq) = board.ep_sq {
            let ep_bb = Bitboard::of_sq(ep_sq);

            for to in left_capture(pawns, side, ep_bb) {
                let from = if side == White { to.minus(7) } else { to.plus(9) };
                moves.add_move(from, to, MoveFlag::EnPassant);
            }

            for to in right_capture(pawns, side, ep_bb) {
                let from = if side == White { to.minus(9) } else { to.plus(7) };
                moves.add_move(from, to, MoveFlag::EnPassant);
            }

        }

        for to in left_capture_promos(pawns, side, them) {
            let from = if side == White { to.minus(7) } else { to.plus(9) };
            add_promos(moves, from, to);
        }

        for to in right_capture_promos(pawns, side, them) {
            let from = if side == White { to.minus(9) } else { to.plus(7) };
            add_promos(moves, from, to);
        }

    }

}

#[inline(always)]
fn gen_castle_moves(board: &Board, side: Side, moves: &mut MoveList) {
    let king_sq = board.king_sq(side);
    let occ = board.occ();
    if board.has_kingside_rights(side) {
        let travel_mask = if side == White { CastleTravel::WKS } else { CastleTravel::BKS };
        let safety_mask = if side == White { CastleSafety::WKS } else { CastleSafety::BKS };
        if (occ & travel_mask).is_empty() && !is_attacked(safety_mask, side, occ, board) {
            moves.add_move(king_sq, Square(king_sq.0 + 2), MoveFlag::CastleK);
        }
    }
    if board.has_queenside_rights(side) {
        let travel_mask = if side == White { CastleTravel::WQS } else { CastleTravel::BQS };
        let safety_mask = if side == White { CastleSafety::WQS } else { CastleSafety::BQS };
        if (occ & travel_mask).is_empty() && !is_attacked(safety_mask, side, occ, board) {
            moves.add_move(king_sq, Square(king_sq.0 - 2), MoveFlag::CastleQ);
        }
    }
}

#[inline(always)]
fn single_push(pawns: Bitboard, side: Side, occ: Bitboard) -> Bitboard {
    match side {
        White => pawns.north() & !occ & !Rank::Eight.to_bb(),
        _ => pawns.south() & !occ & !Rank::One.to_bb()
    }
}

#[inline(always)]
fn double_push(pawns: Bitboard, side: Side, occ: Bitboard) -> Bitboard {
    let single_push = single_push(pawns, side, occ);
    match side {
        White => single_push.north() & !occ & Rank::Four.to_bb(),
        _ => single_push.south() & !occ & Rank::Five.to_bb()
    }
}

#[inline(always)]
fn left_capture(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_west() & them & !File::H.to_bb() & !Rank::Eight.to_bb(),
        _ => pawns.south_west() & them & !File::H.to_bb() & !Rank::One.to_bb()
    }
}

#[inline(always)]
fn right_capture(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_east() & them & !File::A.to_bb() & !Rank::Eight.to_bb(),
        _ => pawns.south_east() & them & !File::A.to_bb() & !Rank::One.to_bb()
    }
}

#[inline(always)]
fn push_promos(pawns: Bitboard, side: Side, occ: Bitboard) -> Bitboard {
    match side {
        White => pawns.north() & !occ & Rank::Eight.to_bb(),
        _ => pawns.south() & !occ & Rank::One.to_bb()
    }
}

#[inline(always)]
fn left_capture_promos(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_west() & them & !File::H.to_bb() & Rank::Eight.to_bb(),
        _ => pawns.south_west() & them & !File::H.to_bb() & Rank::One.to_bb()
    }
}

#[inline(always)]
fn right_capture_promos(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_east() & them & !File::A.to_bb() & Rank::Eight.to_bb(),
        _ => pawns.south_east() & them & !File::A.to_bb() & Rank::One.to_bb()
    }
}

#[inline(always)]
fn add_promos(moves: &mut MoveList, from: Square, to: Square) {
    moves.add_move(from, to, MoveFlag::PromoQ);
    moves.add_move(from, to, MoveFlag::PromoR);
    moves.add_move(from, to, MoveFlag::PromoB);
    moves.add_move(from, to, MoveFlag::PromoN);
}

#[inline(always)]
pub fn is_attacked(bb: Bitboard, side: Side, occ: Bitboard, board: &Board) -> bool {
    for sq in bb {
        if is_sq_attacked(sq, side, occ, board) { return true; }
    }
    false
}

#[inline(always)]
pub fn is_sq_attacked(sq: Square, side: Side, occ: Bitboard, board: &Board) -> bool {

    let knight_attacks = attacks::knight(sq) & board.knights(side.flip());
    if !knight_attacks.is_empty() { return true; }

    let king_attacks = attacks::king(sq) & board.king(side.flip());
    if !king_attacks.is_empty() { return true; }

    let pawn_attacks = attacks::pawn(sq, side) & board.pawns(side.flip());
    if !pawn_attacks.is_empty() { return true; }

    let diag_attacks = attacks::rook(sq, occ) & (board.rooks(side.flip()) | board.queens(side.flip()));
    if !diag_attacks.is_empty() { return true; }

    let ortho_attacks = attacks::bishop(sq, occ) & (board.bishops(side.flip()) | board.queens(side.flip()));
    if !ortho_attacks.is_empty() { return true; }

    false
}

#[inline(always)]
pub fn calc_threats(board: &Board, side: Side) -> Bitboard {
    let occ = board.occ();
    let mut threats = Bitboard::empty();

    for sq in board.knights(side.flip()) {
        threats |= attacks::knight(sq);
    }

    for sq in board.bishops(side.flip()) {
        threats |= attacks::bishop(sq, occ);
    }

    for sq in board.rooks(side.flip()) {
        threats |= attacks::rook(sq, occ);
    }

    for sq in board.queens(side.flip()) {
        threats |= attacks::queen(sq, occ);
    }

    threats |= attacks::pawn_attacks(board.pawns(side.flip()), side.flip());
    threats |= attacks::king(board.king_sq(side.flip()));

    threats
}

#[inline(always)]
pub fn is_check(board: &Board, side: Side) -> bool {
    let occ = board.occ();
    let king_sq = board.king_sq(side);
    is_sq_attacked(king_sq, side, occ, board)
}