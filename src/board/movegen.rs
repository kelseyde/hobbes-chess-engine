use crate::board::bitboard::Bitboard;
use crate::board::castling::{CastleSafety, CastleTravel};
use crate::board::file::File;
use crate::board::moves::{MoveFlag, MoveList, MoveListEntry};
use crate::board::piece::Piece;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::side::Side::White;
use crate::board::square::Square;
use crate::board::Board;
use crate::board::{attacks, castling, ray};

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MoveFilter {
    All,
    Quiets,
    Noisies,
    Captures,
}

impl Board {
    pub fn gen_legal_moves(&self) -> MoveList {
        let mut moves = self.gen_moves(MoveFilter::All);
        let mut legal_moves = MoveList::new();
        for entry in moves.iter() {
            if self.is_legal(&entry.mv) {
                legal_moves.add(MoveListEntry {
                    mv: entry.mv,
                    score: 0,
                })
            }
        }
        legal_moves
    }

    pub fn gen_moves(&self, filter: MoveFilter) -> MoveList {
        const STANDARD_PIECES: [Piece; 4] = [
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
        ];
        let side = self.stm;
        let mut moves = MoveList::new();

        let us = self.us();
        let them = self.them();
        let occ = us | them;

        let filter_mask = match filter {
            MoveFilter::All => Bitboard::ALL,
            MoveFilter::Quiets => !them,
            MoveFilter::Noisies => them,
            MoveFilter::Captures => them,
        };

        // Generate king moves first
        self.gen_standard_moves(Piece::King, side, occ, us, filter_mask, &mut moves);

        // If we are in double-check, the only legal moves are king moves
        if self.checkers.count() == 2 {
            return moves;
        }

        // handle special moves first (en passant, promo, castling etc.)
        gen_pawn_moves(self, side, occ, them, filter, &mut moves);
        if filter != MoveFilter::Captures && filter != MoveFilter::Noisies {
            gen_castle_moves(self, side, &mut moves);
        }

        // handle standard moves
        for &pc in STANDARD_PIECES.iter() {
            self.gen_standard_moves(pc, side, occ, us, filter_mask, &mut moves);
        }

        moves
    }

    #[inline(always)]
    pub fn calc_threats(&self, side: Side) -> Bitboard {
        // The king is excluded from the occupancy bitboard to include slider threats 'through' the
        // king, allowing us to detect illegal moves where the king steps along the checking ray.
        let king = self.king(side);
        let occ = self.occ() ^ king;
        let mut threats = Bitboard::empty();

        for sq in self.knights(!side) {
            threats |= attacks::knight(sq);
        }

        for sq in self.bishops(!side) {
            threats |= attacks::bishop(sq, occ);
        }

        for sq in self.rooks(!side) {
            threats |= attacks::rook(sq, occ);
        }

        for sq in self.queens(!side) {
            threats |= attacks::queen(sq, occ);
        }

        threats |= attacks::pawn_attacks(self.pawns(!side), !side);
        threats |= attacks::king(self.king_sq(!side));

        threats
    }

    #[inline(always)]
    pub fn calc_checkers(&self, side: Side) -> Bitboard {
        let occ = self.occ();
        let king_sq = self.king_sq(side);
        let mut checkers = Bitboard::empty();

        let pawn_attacks = attacks::pawn(king_sq, side) & self.pawns(!side);
        checkers |= pawn_attacks;

        let knight_attacks = attacks::knight(king_sq) & self.knights(!side);
        checkers |= knight_attacks;

        let ortho_attacks = attacks::rook(king_sq, occ) & (self.rooks(!side) | self.queens(!side));
        checkers |= ortho_attacks;

        let diag_attacks =
            attacks::bishop(king_sq, occ) & (self.bishops(!side) | self.queens(!side));
        checkers |= diag_attacks;

        checkers
    }

    fn gen_standard_moves(
        &self,
        pc: Piece,
        side: Side,
        occ: Bitboard,
        us: Bitboard,
        filter_mask: Bitboard,
        moves: &mut MoveList) {

        for from in self.pcs(pc) & us {
            let attacks = attacks::attacks(from, pc, side, occ) & !us & filter_mask;
            for to in attacks {
                moves.add_move(from, to, MoveFlag::Standard);
            }
        }

    }
}

#[inline(always)]
fn gen_pawn_moves(
    board: &Board,
    side: Side,
    occ: Bitboard,
    them: Bitboard,
    filter: MoveFilter,
    moves: &mut MoveList,
) {
    let pawns = board.pcs(Piece::Pawn) & board.side(side);

    if filter != MoveFilter::Captures && filter != MoveFilter::Noisies {
        for to in single_push(pawns, side, occ) {
            let from = if side == White {
                to.minus(8)
            } else {
                to.plus(8)
            };
            moves.add_move(from, to, MoveFlag::Standard);
        }

        for to in double_push(pawns, side, occ) {
            let from = if side == White {
                to.minus(16)
            } else {
                to.plus(16)
            };
            moves.add_move(from, to, MoveFlag::DoublePush);
        }
    }

    if filter != MoveFilter::Quiets {
        for to in left_capture(pawns, side, them) {
            let from = if side == White {
                to.minus(7)
            } else {
                to.plus(9)
            };
            moves.add_move(from, to, MoveFlag::Standard);
        }

        for to in right_capture(pawns, side, them) {
            let from = if side == White {
                to.minus(9)
            } else {
                to.plus(7)
            };
            moves.add_move(from, to, MoveFlag::Standard);
        }

        if let Some(ep_sq) = board.ep_sq {
            let ep_bb = Bitboard::of_sq(ep_sq);

            for to in left_capture(pawns, side, ep_bb) {
                let from = if side == White {
                    to.minus(7)
                } else {
                    to.plus(9)
                };
                moves.add_move(from, to, MoveFlag::EnPassant);
            }

            for to in right_capture(pawns, side, ep_bb) {
                let from = if side == White {
                    to.minus(9)
                } else {
                    to.plus(7)
                };
                moves.add_move(from, to, MoveFlag::EnPassant);
            }
        }

        for to in push_promos(pawns, side, occ) {
            let from = if side == White {
                to.minus(8)
            } else {
                to.plus(8)
            };
            add_promos(moves, from, to);
        }

        for to in left_capture_promos(pawns, side, them) {
            let from = if side == White {
                to.minus(7)
            } else {
                to.plus(9)
            };
            add_promos(moves, from, to);
        }

        for to in right_capture_promos(pawns, side, them) {
            let from = if side == White {
                to.minus(9)
            } else {
                to.plus(7)
            };
            add_promos(moves, from, to);
        }
    }
}

#[inline(always)]
fn gen_castle_moves(board: &Board, side: Side, moves: &mut MoveList) {
    if board.is_frc() {
        gen_frc_castle_moves(board, side, moves);
    } else {
        gen_standard_castle_moves(board, side, moves);
    }
}

#[inline(always)]
pub fn gen_standard_castle_moves(board: &Board, side: Side, moves: &mut MoveList) {
    let king_sq = board.king_sq(side);
    let occ = board.occ();
    if board.has_kingside_rights(side) {
        let travel_mask = if side == White {
            CastleTravel::WKS
        } else {
            CastleTravel::BKS
        };
        let safety_mask = if side == White {
            CastleSafety::WKS
        } else {
            CastleSafety::BKS
        };
        if (occ & travel_mask).is_empty() && !is_attacked(safety_mask, side, occ, board) {
            moves.add_move(king_sq, Square(king_sq.0 + 2), MoveFlag::CastleK);
        }
    }
    if board.has_queenside_rights(side) {
        let travel_mask = if side == White {
            CastleTravel::WQS
        } else {
            CastleTravel::BQS
        };
        let safety_mask = if side == White {
            CastleSafety::WQS
        } else {
            CastleSafety::BQS
        };
        if (occ & travel_mask).is_empty() && !is_attacked(safety_mask, side, occ, board) {
            moves.add_move(king_sq, Square(king_sq.0 - 2), MoveFlag::CastleQ);
        }
    }
}

#[inline(always)]
pub fn gen_frc_castle_moves(board: &Board, side: Side, moves: &mut MoveList) {
    gen_frc_castle_moves_side(board, side, true, moves);
    gen_frc_castle_moves_side(board, side, false, moves);
}

#[inline(always)]
pub fn gen_frc_castle_moves_side(board: &Board, side: Side, kingside: bool, moves: &mut MoveList) {
    let occ = board.occ();
    let rook_file = if kingside {
        board.rights.kingside(side)
    } else {
        board.rights.queenside(side)
    };

    if let Some(rook_file) = rook_file {
        let king_from = board.king_sq(side);
        let king_to = castling::king_to(side, kingside);

        let rank = if side == White {
            Rank::One
        } else {
            Rank::Eight
        };
        let rook_from = Square::from(rook_file, rank);
        let rook_to = castling::rook_to(side, kingside);

        let king_travel_sqs = ray::between(king_from, king_to) | Bitboard::of_sq(king_to);
        let rook_travel_sqs = ray::between(rook_from, rook_to) | Bitboard::of_sq(rook_to);

        let travel_sqs = (king_travel_sqs | rook_travel_sqs)
            & !Bitboard::of_sq(rook_from)
            & !Bitboard::of_sq(king_from);

        let blocked_sqs = travel_sqs & occ;
        let safe_sqs = Bitboard::of_sq(king_from)
            | ray::between(king_from, king_to)
            | Bitboard::of_sq(king_to);

        if blocked_sqs.is_empty() && !is_attacked(safe_sqs, side, occ, board) {
            let flag = if kingside {
                MoveFlag::CastleK
            } else {
                MoveFlag::CastleQ
            };
            moves.add_move(king_from, rook_from, flag);
        }
    }
}

#[inline(always)]
fn single_push(pawns: Bitboard, side: Side, occ: Bitboard) -> Bitboard {
    match side {
        White => pawns.north() & !occ & !Rank::Eight.to_bb(),
        _ => pawns.south() & !occ & !Rank::One.to_bb(),
    }
}

#[inline(always)]
fn double_push(pawns: Bitboard, side: Side, occ: Bitboard) -> Bitboard {
    let single_push = single_push(pawns, side, occ);
    match side {
        White => single_push.north() & !occ & Rank::Four.to_bb(),
        _ => single_push.south() & !occ & Rank::Five.to_bb(),
    }
}

#[inline(always)]
fn left_capture(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_west() & them & !File::H.to_bb() & !Rank::Eight.to_bb(),
        _ => pawns.south_west() & them & !File::H.to_bb() & !Rank::One.to_bb(),
    }
}

#[inline(always)]
fn right_capture(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_east() & them & !File::A.to_bb() & !Rank::Eight.to_bb(),
        _ => pawns.south_east() & them & !File::A.to_bb() & !Rank::One.to_bb(),
    }
}

#[inline(always)]
fn push_promos(pawns: Bitboard, side: Side, occ: Bitboard) -> Bitboard {
    match side {
        White => pawns.north() & !occ & Rank::Eight.to_bb(),
        _ => pawns.south() & !occ & Rank::One.to_bb(),
    }
}

#[inline(always)]
fn left_capture_promos(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_west() & them & !File::H.to_bb() & Rank::Eight.to_bb(),
        _ => pawns.south_west() & them & !File::H.to_bb() & Rank::One.to_bb(),
    }
}

#[inline(always)]
fn right_capture_promos(pawns: Bitboard, side: Side, them: Bitboard) -> Bitboard {
    match side {
        White => pawns.north_east() & them & !File::A.to_bb() & Rank::Eight.to_bb(),
        _ => pawns.south_east() & them & !File::A.to_bb() & Rank::One.to_bb(),
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
        if is_sq_attacked(sq, side, occ, board) {
            return true;
        }
    }
    false
}

#[inline(always)]
pub fn is_sq_attacked(sq: Square, side: Side, occ: Bitboard, board: &Board) -> bool {
    let knight_attacks = attacks::knight(sq) & board.knights(!side);
    if !knight_attacks.is_empty() {
        return true;
    }

    let king_attacks = attacks::king(sq) & board.king(!side);
    if !king_attacks.is_empty() {
        return true;
    }

    let pawn_attacks = attacks::pawn(sq, side) & board.pawns(!side);
    if !pawn_attacks.is_empty() {
        return true;
    }

    let ortho_attacks = attacks::rook(sq, occ) & (board.rooks(!side) | board.queens(!side));
    if !ortho_attacks.is_empty() {
        return true;
    }

    let diag_attacks = attacks::bishop(sq, occ) & (board.bishops(!side) | board.queens(!side));
    if !diag_attacks.is_empty() {
        return true;
    }

    false
}

#[inline(always)]
pub fn is_check(board: &Board, side: Side) -> bool {
    let occ = board.occ();
    let king_sq = board.king_sq(side);
    is_sq_attacked(king_sq, side, occ, board)
}
