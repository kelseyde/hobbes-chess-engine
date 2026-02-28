use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;
use crate::board::{attacks, ray};
use crate::search::parameters::{see_value_bishop_ordering, see_value_bishop_pruning, see_value_knight_ordering, see_value_knight_pruning, see_value_pawn_ordering, see_value_pawn_pruning, see_value_queen_ordering, see_value_queen_pruning, see_value_rook_ordering, see_value_rook_pruning};
use SeeType::{Ordering, Pruning};

#[derive(Clone, Copy)]
pub enum SeeType {
    Pruning,
    Ordering
}

pub fn value(pc: Piece, see_type: SeeType) -> i32 {
    match pc {
        Piece::Pawn => pawn_value(see_type),
        Piece::Knight => knight_value(see_type),
        Piece::Bishop => bishop_value(see_type),
        Piece::Rook => rook_value(see_type),
        Piece::Queen => queen_value(see_type),
        Piece::King => 0,
    }
}

pub fn see(board: &Board, mv: &Move, threshold: i32, see_type: SeeType) -> bool {
    let from = mv.from();
    let to = mv.to();

    let next_victim = mv
        .promo_piece()
        .map_or_else(|| board.piece_at(from).unwrap(), |promo| promo);

    let mut balance = move_value(board, mv, see_type) - threshold;

    if balance < 0 {
        return false;
    }

    balance -= value(next_victim, see_type);

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

    let white_pinned = board.pinned[Side::White];
    let black_pinned = board.pinned[Side::Black];
    let pinned = white_pinned | black_pinned;
    attackers &= !pinned
        | (white_pinned & ray::extending(board.king_sq(Side::White), to))
        | (black_pinned & ray::extending(board.king_sq(Side::Black), to));

    let mut stm = !board.stm;

    loop {
        let our_attackers = attackers & board.side(stm);
        if our_attackers.is_empty() {
            break;
        }

        let attacker = least_valuable_attacker(board, our_attackers);

        if attacker == Piece::King && !(attackers & board.side(!stm)).is_empty() {
            break;
        }

        // Make the capture
        let pcs = board.pcs(attacker) & our_attackers;
        let sq = (our_attackers & pcs).lsb();
        occ = occ.pop_bit(sq);
        stm = !stm;

        balance = -balance - 1 - value(attacker, see_type);
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

#[allow(clippy::redundant_closure)]
fn move_value(board: &Board, mv: &Move, see_type: SeeType) -> i32 {
    let mut see_value = board
        .piece_at(mv.to())
        .map_or(0, |captured| value(captured, see_type));

    if let Some(promo) = mv.promo_piece() {
        see_value += value(promo, see_type);
    } else if mv.is_ep() {
        see_value = value(Piece::Pawn, see_type);
    }
    see_value
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

#[inline]
fn pawn_value(see_type: SeeType) -> i32 {
    match see_type {
        Pruning => see_value_pawn_pruning(),
        Ordering => see_value_pawn_ordering()
    }
}

#[inline]
fn knight_value(see_type: SeeType) -> i32 {
    match see_type {
        Pruning => see_value_knight_pruning(),
        Ordering => see_value_knight_ordering()
    }
}

#[inline]
fn bishop_value(see_type: SeeType) -> i32 {
    match see_type {
        Pruning => see_value_bishop_pruning(),
        Ordering => see_value_bishop_ordering()
    }
}

#[inline]
fn rook_value(see_type: SeeType) -> i32 {
    match see_type {
        Pruning => see_value_rook_pruning(),
        Ordering => see_value_rook_ordering()
    }
}

#[inline]
fn queen_value(see_type: SeeType) -> i32 {
    match see_type {
        Pruning => see_value_queen_pruning(),
        Ordering => see_value_queen_ordering()
    }
}
