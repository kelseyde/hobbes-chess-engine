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
    /// Generate all legal moves for the current position.
    /// This is *not* optimized for speed, and is intended only as a utility method. Actual move
    /// generation used during search is pseudo-legal, with legality checks performed on the fly.
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

    /// Generate all pseudo-legal moves for the current position.
    pub fn gen_moves(&self, filter: MoveFilter) -> MoveList {
        // 'Standard' meaning non-pawn, since pawn moves are calculated setwise rather than piece-wise.
        // The king is technically also a standard piece, but its moves are generated first for efficiency.
        const STANDARD_PIECES: [Piece; 4] =
            [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
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

    /// Compute the squares attacked by the opponent's pieces.
    #[inline(always)]
    pub fn calc_threats(&self, side: Side) -> Bitboard {
        // The king is excluded from the occupancy bitboard to include slider threats 'through' the
        // king, allowing us to detect illegal moves where the king steps along the checking ray.
        let king = self.king(side);
        let occ = self.occ() ^ king;
        let them = !side;
        let mut threats = Bitboard::empty();

        threats |= attacks::pawn_attacks(self.pawns(them), them);

        for sq in self.knights(them) {
            threats |= attacks::knight(sq);
        }

        for sq in self.diags(them) {
            threats |= attacks::bishop(sq, occ);
        }

        for sq in self.orthos(them) {
            threats |= attacks::rook(sq, occ);
        }

        threats |= attacks::king(self.king_sq(them));

        threats
    }

    /// Compute the pieces checking the king of the given side.
    #[inline(always)]
    pub fn calc_checkers(&self, side: Side) -> Bitboard {
        let occ = self.occ();
        let king_sq = self.king_sq(side);
        let them = !side;
        let mut checkers = Bitboard::empty();

        checkers |= attacks::pawn(king_sq, side) & self.pawns(them);
        checkers |= attacks::knight(king_sq) & self.knights(them);
        checkers |= attacks::rook(king_sq, occ) & self.orthos(them);
        checkers |= attacks::bishop(king_sq, occ) & self.diags(them);

        checkers
    }

    #[inline(always)]
    fn gen_standard_moves(
        &self,
        pc: Piece,
        side: Side,
        occ: Bitboard,
        us: Bitboard,
        filter_mask: Bitboard,
        moves: &mut MoveList,
    ) {
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

    // Quiet pawn moves (single and double pushes).
    if filter != MoveFilter::Captures && filter != MoveFilter::Noisies {
        add_pawn_moves(single_push(pawns, side, occ), side, 8, 8, MoveFlag::Standard, moves);
        add_pawn_moves(double_push(pawns, side, occ), side, 16, 16, MoveFlag::DoublePush, moves);
    }

    // Noisy pawn moves (captures, promos, en passant).
    if filter != MoveFilter::Quiets {
        add_pawn_moves(left_capture(pawns, side, them), side, 7, 9, MoveFlag::Standard, moves);
        add_pawn_moves(right_capture(pawns, side, them), side, 9, 7, MoveFlag::Standard, moves);

        if let Some(ep_sq) = board.ep_sq {
            let ep_bb = Bitboard::of_sq(ep_sq);
            add_pawn_moves(left_capture(pawns, side, ep_bb), side, 7, 9, MoveFlag::EnPassant, moves);
            add_pawn_moves(right_capture(pawns, side, ep_bb), side, 9, 7, MoveFlag::EnPassant, moves);
        }

        add_pawn_promos(push_promos(pawns, side, occ), side, 8, 8, moves);
        add_pawn_promos(left_capture_promos(pawns, side, them), side, 7, 9, moves);
        add_pawn_promos(right_capture_promos(pawns, side, them), side, 9, 7, moves);
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
#[rustfmt::skip]
pub fn gen_standard_castle_moves(board: &Board, side: Side, moves: &mut MoveList) {
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
fn add_pawn_moves(targets: Bitboard, side: Side, w_off: u8, b_off: u8, flag: MoveFlag, moves: &mut MoveList) {
    for to in targets {
        let from = sq_offset(to, side, w_off, b_off);
        moves.add_move(from, to, flag);
    }
}

#[inline(always)]
fn add_pawn_promos(targets: Bitboard, side: Side, w_off: u8, b_off: u8, moves: &mut MoveList) {
    for to in targets {
        let from = sq_offset(to, side, w_off, b_off);
        add_promos(moves, from, to);
    }
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
    let them = !side;

    if !(attacks::pawn(sq, side) & board.pawns(them)).is_empty() {
        return true;
    }

    if !(attacks::knight(sq) & board.knights(them)).is_empty() {
        return true;
    }

    if !(attacks::rook(sq, occ) & board.orthos(them)).is_empty() {
        return true;
    }

    if !(attacks::bishop(sq, occ) & board.diags(them)).is_empty() {
        return true;
    }

    if !(attacks::king(sq) & board.king(them)).is_empty() {
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

#[inline(always)]
const fn sq_offset(sq: Square, side: Side, w: u8, b: u8) -> Square {
    match side {
        White => sq.minus(w),
        _ => sq.plus(b),
    }
}
