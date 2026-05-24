use crate::board::bitboard::Bitboard;
use crate::board::castling::{CastleSafety, CastleTravel};
use crate::board::file::File;
use crate::board::moves::{Move, MoveFlag, MoveList};
use crate::board::piece::Piece;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::side::Side::White;
use crate::board::square::Square;
use crate::board::{attacks, castling, ray};
use crate::board::{setwise, Board};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum MoveFilter {
    All,
    Quiets,
    Noisies,
    Captures,
}

impl MoveFilter {
    pub fn gen_quiets(self) -> bool {
        matches!(self, MoveFilter::All | MoveFilter::Quiets)
    }

    pub fn gen_noisies(self) -> bool {
        matches!(self, MoveFilter::All | MoveFilter::Noisies)
    }

    pub fn gen_captures(self) -> bool {
        matches!(self, MoveFilter::All | MoveFilter::Noisies | MoveFilter::Captures)
    }
}

impl Board {

    /// Generate all legal moves for the current position.
    #[rustfmt::skip]
    pub fn gen_moves(&self, filter: MoveFilter, moves: &mut MoveList) {
        let side = self.stm;
        let us = self.us();
        let them = self.them();
        let occ = us | them;
        let threats = self.threats;
        let king_sq = self.king_sq(side);
        let pinned = self.pinned[side];
        let in_check = !self.checkers.is_empty();

        let mut filter_mask = match filter {
            MoveFilter::All => Bitboard::ALL,
            MoveFilter::Quiets => !them,
            MoveFilter::Noisies | MoveFilter::Captures => them,
        };

        // Generate king moves first, because...
        self.gen_king_moves(side, us, threats, filter_mask, moves);

        // ...if we are in double-check, the only legal moves are king moves. So we can exit early.
        if self.checkers.is_multiple() {
            return;
        }

        let evasion_mask = if in_check {
            // If we are in single-check, we can only generate moves that block or capture the checker.
            self.checkers | ray::between(king_sq, self.checkers.lsb())
        } else {
            Bitboard::ALL
        };
        filter_mask &= evasion_mask;
        filter_mask &= if filter == MoveFilter::Quiets { !occ } else { !us };

        self.gen_pawn_moves(filter, evasion_mask, moves);
        self.gen_knight_moves(side, pinned, filter_mask, moves);
        self.gen_sliding_moves(self.diags(side), pinned, filter_mask, moves, |sq| attacks::bishop(sq, occ));
        self.gen_sliding_moves(self.orthos(side), pinned, filter_mask, moves, |sq| attacks::rook(sq, occ));
    }

    #[inline(always)]
    fn gen_king_moves(
        &self,
        side: Side,
        us: Bitboard,
        threats: Bitboard,
        filter_mask: Bitboard,
        moves: &mut MoveList,
    ) {
        let king_sq = self.king_sq(side);
        let attacks = attacks::king(king_sq) & !us & !threats & filter_mask;
        moves.add_moves(king_sq, attacks, MoveFlag::Standard);
    }

    fn gen_knight_moves(
        &self,
        side: Side,
        pinned: Bitboard,
        filter_mask: Bitboard,
        moves: &mut MoveList,
    ) {
        for knight in self.knights(side) & !pinned {
            let attacks = attacks::knight(knight) & filter_mask;
            moves.add_moves(knight, attacks, MoveFlag::Standard);
        }
    }

    #[inline(always)]
    fn gen_sliding_moves<F: Fn(Square) -> Bitboard>(
        &self,
        pieces: Bitboard,
        pinned: Bitboard,
        filter_mask: Bitboard,
        moves: &mut MoveList,
        attacks: F,
    ) {
        for from in pieces & !pinned {
            let attacks = attacks(from) & filter_mask;
            moves.add_moves(from, attacks, MoveFlag::Standard);
        }

        let king_sq = self.king_sq(self.stm);
        for from in pieces & pinned {
            let pin_ray = ray::extending(king_sq, from);
            let attacks = attacks(from) & filter_mask & pin_ray;
            moves.add_moves(from, attacks, MoveFlag::Standard);
        }
    }

    #[inline(always)]
    fn gen_pawn_moves(
        &self,
        filter: MoveFilter,
        filter_mask: Bitboard,
        moves: &mut MoveList,
    ) {
        let side = self.stm;
        let pinned = self.pinned[side];
        let pawns = self.pcs(Piece::Pawn) & self.side(side);
        let king_sq = self.king_sq(side);
        let king_file = king_sq.file().to_bb();
        let third_rank = Rank::BB[if side == White { 2 } else { 5 }];
        let seventh_rank = Rank::BB[if side == White { 6 } else { 1 }];
        let up = Square::UP[side];
        let empty = !self.occ();
        let pushable_pawns = pawns & (!pinned | king_file);

        // Quiet pawn moves (single and double pushes).
        if filter.gen_quiets() {
            let non_promotions = pushable_pawns & !seventh_rank;
            let single_pushes = non_promotions.shift(up) & empty;
            let double_pushes = (single_pushes & third_rank).shift(up) & empty;
            moves.add_pawn_moves(single_pushes & filter_mask, up, MoveFlag::Standard);
            moves.add_pawn_moves(double_pushes & filter_mask, up * 2, MoveFlag::DoublePush);
        }

        if filter.gen_noisies() {
            // Push promotions (noisy, but not captures).
            let push_promos = (pushable_pawns & seventh_rank).shift(up) & empty;
            moves.add_pawn_promos(push_promos & filter_mask, up);
        }

        if filter.gen_captures() {
            let filter_mask = filter_mask & self.them();
            let up_right = up + Square::RIGHT;
            let up_left = up + Square::LEFT;
            let right_pin_mask = ray::relative_diagonal(side, king_sq);
            let left_pin_mask = ray::relative_diagonal(!side, king_sq);

            // Pawns which are capable of capturing left/right (not pinned unless capturing along the
            // pin ray, and not on the edge of the board).
            let left_pawns = pawns & (!pinned | left_pin_mask) & !File::A.to_bb();
            let right_pawns = pawns & (!pinned | right_pin_mask) & !File::H.to_bb();

            // Non-promotion captures
            let left_caps = (left_pawns & !seventh_rank).shift(up_left);
            let right_caps = (right_pawns & !seventh_rank).shift(up_right);

            moves.add_pawn_moves(right_caps & filter_mask, up_right, MoveFlag::Standard);
            moves.add_pawn_moves(left_caps & filter_mask, up_left, MoveFlag::Standard);

            // Promotion captures
            let right_cap_promos = (right_pawns & seventh_rank).shift(up_right);
            let left_cap_promos = (left_pawns & seventh_rank).shift(up_left);
            moves.add_pawn_promos(left_cap_promos & filter_mask, up_left);
            moves.add_pawn_promos(right_cap_promos & filter_mask, up_right);

            // En passant
            if let Some(ep_sq) = self.ep_sq {
                let ep_bb = Bitboard::of_sq(ep_sq);
                let right_attacker = right_pawns & ep_bb.shift(-up_right);
                let left_attacker = left_pawns & ep_bb.shift(-up_left);
                for pawn in right_attacker | left_attacker {
                    let mv = Move::new(pawn, ep_sq, MoveFlag::EnPassant);
                    if self.is_legal(&mv) {
                        moves.add_single(mv);
                    }
                }
            }
        }

        if filter.gen_quiets() && self.checkers.is_empty() {
            gen_castle_moves(self, side, moves);
        }
    }

    /// Compute the squares attacked by the opponent's pieces.
    #[inline(always)]
    pub fn calc_threats(&self, side: Side) -> Bitboard {
        // The king is excluded from the occupancy bitboard to include slider threats 'through' the
        // king, allowing us to detect illegal moves where the king steps along the checking ray.
        let king = self.king(side);
        let occ = self.occ() ^ king;
        let them = !side;

        let pawns = self.pawns(them);
        let knights = self.knights(them);
        let diags = self.diags(them);
        let orthos = self.orthos(them);

        attacks::pawn_attacks(pawns, them)
            | setwise::knights_and_sliders_setwise(knights, orthos, diags, occ)
            | attacks::king(self.king_sq(them))
    }

    /// Compute the pieces checking the king of the given side.
    #[inline(always)]
    pub fn calc_checkers(&self, side: Side) -> Bitboard {
        let occ = self.occ();
        let king_sq = self.king_sq(side);
        let them = !side;

        attacks::pawn(king_sq, side) & self.pawns(them)
            | attacks::knight(king_sq) & self.knights(them)
            | attacks::rook(king_sq, occ) & self.orthos(them)
            | attacks::bishop(king_sq, occ) & self.diags(them)
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
        if (occ & travel_mask).is_empty() && (board.threats & safety_mask).is_empty() {
            moves.add_move(king_sq, Square(king_sq.0 + 2), MoveFlag::CastleK);
        }
    }
    if board.has_queenside_rights(side) {
        let travel_mask = if side == White { CastleTravel::WQS } else { CastleTravel::BQS };
        let safety_mask = if side == White { CastleSafety::WQS } else { CastleSafety::BQS };
        if (occ & travel_mask).is_empty() && (board.threats & safety_mask).is_empty() {
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

#[cfg(test)]
mod tests {
    use crate::board::{ray, Board};
    use crate::board::movegen::MoveFilter;
    use crate::board::moves::MoveList;

    #[test]
    fn test_filters() {
        ray::init();
        let board = Board::from_fen("8/2p5/1p2kPp1/p5Pp/P1P1KRnR/6P1/4P3/r7 b - - 0 44").unwrap();

        let mut legal_moves = MoveList::new();
        let mut quiet_moves = MoveList::new();
        let mut noisy_moves = MoveList::new();
        let mut capture_moves = MoveList::new();
        board.gen_moves(MoveFilter::All, &mut legal_moves);
        board.gen_moves(MoveFilter::Quiets, &mut quiet_moves);
        board.gen_moves(MoveFilter::Noisies, &mut noisy_moves);
        board.gen_moves(MoveFilter::Captures, &mut capture_moves);

        let legal_mv_strings = legal_moves.iter().map(|mv| mv.mv.to_uci()).collect::<Vec<_>>();
        let quiet_mv_strings = quiet_moves.iter().map(|mv| mv.mv.to_uci()).collect::<Vec<_>>();
        let noisy_mv_strings = noisy_moves.iter().map(|mv| mv.mv.to_uci()).collect::<Vec<_>>();

        println!("legals: {:?}", legal_mv_strings);
        println!("quiets: {:?}", quiet_mv_strings);
        println!("noisies: {:?}", noisy_mv_strings);
        println!("captures: {:?}", capture_moves);

        let extra_quiets = quiet_mv_strings.iter().filter(|mv| !legal_mv_strings.contains(mv)).collect::<Vec<_>>();
        let extra_noisies = noisy_mv_strings.iter().filter(|mv| !legal_mv_strings.contains(mv)).collect::<Vec<_>>();
        let extra_captures = capture_moves.iter().map(|mv| mv.mv.to_uci()).filter(|mv| !legal_mv_strings.contains(mv)).collect::<Vec<_>>();

        println!("extra_quiets: {:?}", extra_quiets);
        println!("extra_noisies: {:?}", extra_noisies);
        println!("extra_captures: {:?}", extra_captures);

        assert_eq!(legal_mv_strings.len(), quiet_mv_strings.len() + noisy_mv_strings.len());
        assert_eq!(extra_quiets.len(), 0);
        assert_eq!(extra_noisies.len(), 0);
        assert_eq!(extra_captures.len(), 0);

        let noisy_quiets = quiet_moves.iter().filter(|mv| board.captured(&mv.mv).is_some()).count();
        let quiet_noisies = noisy_moves.iter().filter(|mv| board.is_noisy(&mv.mv)).count();
        let quiet_captures = quiet_moves.iter().filter(|mv| board.captured(&mv.mv).is_some()).count();

        assert_eq!(noisy_quiets, 0);
        assert_eq!(quiet_noisies, 0);
        assert_eq!(quiet_captures, 0);
    }

    #[test]
    fn test_perft_standard_epd() {
        run_perft_epd(include_str!("../../resources/standard.epd"), false);
    }

    #[test]
    fn test_perft_frc_epd() {
        run_perft_epd(include_str!("../../resources/frc.epd"), true);
    }

    fn run_perft_epd(epd: &str, frc: bool) {
        use crate::board::Board;
        use crate::tools::perft::perft;
        ray::init();

        let mut failures = Vec::new();

        for line in epd.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }

            let fen = match line.split_once(';') {
                Some((f, _)) => f.trim(),
                None => continue,
            };

            let expected_nodes: Option<u64> = line.split(';')
                .find_map(|tok| {
                    tok.trim().strip_prefix("D6 ").and_then(|n| n.trim().parse().ok())
                });

            let expected_nodes = match expected_nodes {
                Some(n) => n,
                None => continue,
            };

            let mut board = Board::from_fen(fen).expect("valid fen");
            board.set_frc(frc);
            let actual_nodes = perft::<true>(&board, 6);

            if actual_nodes != expected_nodes {
                failures.push(format!(
                    "FAIL fen='{}' expected={} actual={}",
                    fen, expected_nodes, actual_nodes
                ));
            }
        }

        if !failures.is_empty() {
            for f in &failures {
                println!("{}", f);
            }
            panic!("{} perft position(s) failed", failures.len());
        }
    }

}