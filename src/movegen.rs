use bits::{CastleSafetyMask, CastleTravelMask};

use crate::bits::{FILE_A, FILE_H, RANK_1, RANK_4, RANK_5, RANK_8};
use crate::board::Board;
use crate::consts::Side::White;
use crate::consts::{Piece, Side};
use crate::moves::{MoveFlag, MoveList};
use crate::{attacks, bits};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MoveFilter {
    All,
    Captures
}

pub fn gen_moves(board: &Board, filter: MoveFilter) -> MoveList {
    let side = board.stm;
    let mut moves = MoveList::new();

    let us = board.side(side);
    let them = board.side(side.flip());
    let occ = us | them;

    // handle special moves first (en passant, promo, castling etc.)
    gen_pawn_moves(board, side, occ, them, filter, &mut moves);
    if filter != MoveFilter::Captures {
        gen_castle_moves(board, side, &mut moves);
    }

    let filter_mask = match filter {
        MoveFilter::All => bits::ALL_SQUARES,
        MoveFilter::Captures => them
    };

    // handle standard moves
    for &pc in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King].iter() {
        let mut pcs = board.pcs(pc) & us;
        while pcs != 0 {
            let from = bits::lsb(pcs);
            let mut attacks = attacks::attacks(from, pc, side, occ) & !us & filter_mask;
            while attacks != 0 {
                let to = bits::lsb(attacks);
                moves.add_move(from, to, MoveFlag::Standard);
                attacks = bits::pop(attacks);
            }
            pcs = bits::pop(pcs);
        }
    }

    moves
}

#[inline(always)]
fn gen_pawn_moves(board: &Board, side: Side, occ: u64, them: u64, filter: MoveFilter, moves: &mut MoveList) {

    let pawns = board.pcs(Piece::Pawn) & board.side(side);

    if filter != MoveFilter::Captures {
        let mut single_push_moves = single_push(pawns, side, occ);
        while single_push_moves != 0 {
            let to = bits::lsb(single_push_moves);
            let from = if side == White { to - 8 } else { to + 8 };
            moves.add_move(from, to, MoveFlag::Standard);
            single_push_moves = bits::pop(single_push_moves);
        }

        let mut double_push_moves = double_push(pawns, side, occ);
        while double_push_moves != 0 {
            let to = bits::lsb(double_push_moves);
            let from = if side == White { to - 16 } else { to + 16 };
            moves.add_move(from, to, MoveFlag::DoublePush);
            double_push_moves = bits::pop(double_push_moves);
        }
    }

    let mut left_capture_moves = left_capture(pawns, side, them);
    while left_capture_moves != 0 {
        let to = bits::lsb(left_capture_moves);
        let from = if side == White { to - 7 } else { to + 9 };
        moves.add_move(from, to, MoveFlag::Standard);
        left_capture_moves = bits::pop(left_capture_moves);
    }

    let mut right_capture_moves = right_capture(pawns, side, them);
    while right_capture_moves != 0 {
        let to = bits::lsb(right_capture_moves);
        let from = if side == White { to - 9 } else { to + 7 };
        moves.add_move(from, to, MoveFlag::Standard);
        right_capture_moves = bits::pop(right_capture_moves);
    }

    if board.ep_sq.is_some() {
        let ep_sq = board.ep_sq.unwrap();
        let ep_bb = bits::bb(ep_sq);
        let left_ep_captures = left_capture(pawns, side, ep_bb);
        if left_ep_captures != 0 {
            let to = bits::lsb(left_ep_captures);
            let from = if side == White { to - 7 } else { to + 9 };
            moves.add_move(from, to, MoveFlag::EnPassant);
        }
        let right_ep_captures = right_capture(pawns, side, ep_bb);
        if right_ep_captures != 0 {
            let to = bits::lsb(right_ep_captures);
            let from = if side == White { to - 9 } else { to + 7 };
            moves.add_move(from, to, MoveFlag::EnPassant);
        }
    }

    if filter != MoveFilter::Captures {
        let mut push_promo_moves = push_promos(pawns, side, occ);
        while push_promo_moves != 0 {
            let to = bits::lsb(push_promo_moves);
            let from = if side == White { to - 8 } else { to + 8 };
            add_promos(moves, from, to);
            push_promo_moves = bits::pop(push_promo_moves);
        }
    }

    let mut left_capture_promo_moves = left_capture_promos(pawns, side, them);
    while left_capture_promo_moves != 0 {
        let to = bits::lsb(left_capture_promo_moves);
        let from = if side == White { to - 7 } else { to + 9 };
        add_promos(moves, from, to);
        left_capture_promo_moves = bits::pop(left_capture_promo_moves);
    }

    let mut right_capture_promo_moves = right_capture_promos(pawns, side, them);
    while right_capture_promo_moves != 0 {
        let to = bits::lsb(right_capture_promo_moves);
        let from = if side == White { to - 9 } else { to + 7 };
        add_promos(moves, from, to);
        right_capture_promo_moves = bits::pop(right_capture_promo_moves);
    }

}

#[inline(always)]
fn gen_castle_moves(board: &Board, side: Side, moves: &mut MoveList) {
    let king_sq = bits::lsb(board.king(side));
    let occ = board.occ();
    if board.has_kingside_rights(side) {
        let travel_mask = if side == White { CastleTravelMask::WKS } else { CastleTravelMask::BKS } as u64;
        let safety_mask = if side == White { CastleSafetyMask::WKS } else { CastleSafetyMask::BKS } as u64;
        if occ & travel_mask == 0 && !is_attacked(safety_mask, side, occ, board) {
            moves.add_move(king_sq, king_sq + 2, MoveFlag::CastleK);
        }
    }
    if board.has_queenside_rights(side) {
        let travel_mask = if side == White { CastleTravelMask::WQS } else { CastleTravelMask::BQS } as u64;
        let safety_mask = if side == White { CastleSafetyMask::WQS } else { CastleSafetyMask::BQS } as u64;
        if occ & travel_mask == 0 && !is_attacked(safety_mask, side, occ, board) {
            moves.add_move(king_sq, king_sq - 2, MoveFlag::CastleQ);
        }
    }
}

#[inline(always)]
const fn single_push(pawns: u64, side: Side, occ: u64) -> u64 {
    match side {
        White => bits::north(pawns) & !occ & !RANK_8,
        _ => bits::south(pawns) & !occ & !RANK_1
    }
}

#[inline(always)]
const fn double_push(pawns: u64, side: Side, occ: u64) -> u64 {
    let single_push = single_push(pawns, side, occ);
    match side {
        White => bits::north(single_push) & !occ & RANK_4,
        _ => bits::south(single_push) & !occ & RANK_5
    }
}

#[inline(always)]
const fn left_capture(pawns: u64, side: Side, them: u64) -> u64 {
    match side {
        White => bits::north_west(pawns) & them & !FILE_H & !RANK_8,
        _ => bits::south_west(pawns) & them & !FILE_H & !RANK_1
    }
}

#[inline(always)]
const fn right_capture(pawns: u64, side: Side, them: u64) -> u64 {
    match side {
        White => bits::north_east(pawns) & them & !FILE_A & !RANK_8,
        _ => bits::south_east(pawns) & them & !FILE_A & !RANK_1
    }
}

#[inline(always)]
const fn push_promos(pawns: u64, side: Side, occ: u64) -> u64 {
    match side {
        White => bits::north(pawns) & !occ & RANK_8,
        _ => bits::south(pawns) & !occ & RANK_1
    }
}

#[inline(always)]
const fn left_capture_promos(pawns: u64, side: Side, them: u64) -> u64 {
    match side {
        White => bits::north_west(pawns) & them & !FILE_H & RANK_8,
        _ => bits::south_west(pawns) & them & !FILE_H & RANK_1
    }
}

#[inline(always)]
const fn right_capture_promos(pawns: u64, side: Side, them: u64) -> u64 {
    match side {
        White => bits::north_east(pawns) & them & !FILE_A & RANK_8,
        _ => bits::south_east(pawns) & them & !FILE_A & RANK_1
    }
}

#[inline(always)]
fn add_promos(moves: &mut MoveList, from: u8, to: u8) {
    moves.add_move(from, to, MoveFlag::PromoQ);
    moves.add_move(from, to, MoveFlag::PromoR);
    moves.add_move(from, to, MoveFlag::PromoB);
    moves.add_move(from, to, MoveFlag::PromoN);
}

#[inline(always)]
pub fn is_attacked(mut bb: u64, side: Side, occ: u64, board: &Board) -> bool {
    while bb != 0 {
        let sq = bits::lsb(bb);
        if is_sq_attacked(sq, side, occ, board) { return  true; }
        bb = bits::pop(bb);
    }
    false
}

#[inline(always)]
pub fn is_sq_attacked(sq: u8, side: Side, occ: u64, board: &Board) -> bool {
    if attacks::knight(sq)      & board.knights(side.flip()) != 0 { return true; }
    if attacks::king(sq)        & board.king(side.flip()) != 0 { return true; }
    if attacks::pawn(sq, side)  & board.pawns(side.flip()) != 0 { return true; }
    if attacks::rook(sq, occ)   & (board.rooks(side.flip()) | board.queens(side.flip())) != 0 { return true; }
    if attacks::bishop(sq, occ) & (board.bishops(side.flip()) | board.queens(side.flip())) != 0 { return true; }
    false
}

#[inline(always)]
pub fn is_check(board: &Board, side: Side) -> bool {
    let occ = board.occ();
    let king_sq = bits::lsb(board.king(side));
    if king_sq > 63 {
        println!("fen: {}, king_sq: {}", board.to_fen(), king_sq);
        return false;
    }
    is_sq_attacked(king_sq, side, occ, board)
}

#[cfg(test)]
mod test {

}




