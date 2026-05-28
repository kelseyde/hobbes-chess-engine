use crate::board::bitboard::Bitboard;

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
        matches!(
            self,
            MoveFilter::All | MoveFilter::Noisies | MoveFilter::Captures
        )
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
        let king_sq = self.king_sq(side);
        let pinned = self.pinned[side];
        let in_check = !self.checkers.is_empty();

        // During search, we generate moves in stages - noisies first, then quiets. We therefore
        // apply a filter mask to exclude quiet moves in the noisy stage and vice versa.
        let mut filter_mask = match filter {
            MoveFilter::All => Bitboard::ALL,
            MoveFilter::Quiets => !them,
            MoveFilter::Noisies | MoveFilter::Captures => them,
        };

        // Generate king moves first, because if we are in double-check, the only legal moves are
        // king moves, so we can exit early.
        self.gen_king_moves(side, us, filter_mask, moves);

        if self.checkers.is_multiple() {
            return;
        }

        // If we are in single-check, we can only generate moves that block or capture the checker.
        let evasion_mask = if in_check {
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

    /// Generate the legal king moves in the position. We re-use the pre-computed opponent threat
    /// bitboard to prevent the king from stepping into check.
    #[inline(always)]
    fn gen_king_moves(
        &self,
        side: Side,
        us: Bitboard,
        filter_mask: Bitboard,
        moves: &mut MoveList,
    ) {
        let king_sq = self.king_sq(side);
        let threats = self.threats;
        let attacks = attacks::king(king_sq) & !us & !threats & filter_mask;
        moves.add_moves(king_sq, attacks, MoveFlag::Standard);
    }

    /// Generate the legal knight moves in the position. Pinned knights are skipped, since a pinned
    /// knight cannot move along the pin ray, and so by definition has zero legal moves.
    #[inline(always)]
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

    /// Generate the legal moves for the given sliding pieces. Diagonal sliders (bishops and queens)
    /// and orthogonal sliders (rooks and queens) are handled together. Pinned sliders can move only
    /// along the pin ray, so we mask their attacks with the ray between the king and the slider.
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

    /// Generate the legal pawn moves in the position. Pawn moves are generated setwise rather than
    /// piecewise for efficiency.
    #[inline(always)]
    fn gen_pawn_moves(&self, filter: MoveFilter, filter_mask: Bitboard, moves: &mut MoveList) {
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

        // Push promotions (noisy, but not captures).
        if filter.gen_noisies() {
            let push_promos = (pushable_pawns & seventh_rank).shift(up) & empty;
            moves.add_pawn_promos(push_promos & filter_mask, up);
        }

        // Captures (standard captures, promo captures, and en passant).
        if filter.gen_captures() {
            let filter_mask = filter_mask & self.them();
            let dirs = [up + Square::RIGHT, up + Square::LEFT];
            let pin_masks = [
                ray::relative_diagonal(side, king_sq),
                ray::relative_diagonal(!side, king_sq),
            ];
            let shift_masks = [!File::H.to_bb(), !File::A.to_bb()];

            for i in 0..2 {
                // Pawns which are capable of capturing in the given direction (not pinned unless
                // capturing along the pin ray, and not on the edge of the board).
                let sided_pawns = pawns & (!pinned | pin_masks[i]) & shift_masks[i];

                let non_promo_captures = (sided_pawns & !seventh_rank).shift(dirs[i]);
                moves.add_pawn_moves(
                    non_promo_captures & filter_mask,
                    dirs[i],
                    MoveFlag::Standard,
                );

                let promo_captures = (sided_pawns & seventh_rank).shift(dirs[i]);
                moves.add_pawn_promos(promo_captures & filter_mask, dirs[i]);

                // En passant.
                if let Some(ep_sq) = self.ep_sq {
                    let ep_bb = Bitboard::of_sq(ep_sq);
                    let attacker = sided_pawns & ep_bb.shift(-dirs[i]);
                    let mv = (!attacker.is_empty())
                        .then(|| Move::new(attacker.lsb(), ep_sq, MoveFlag::EnPassant))
                        .filter(|mv| self.is_legal(mv));
                    if let Some(mv) = mv {
                        moves.add_single(mv);
                    }
                }
            }
        }

        if filter.gen_quiets() && self.checkers.is_empty() {
            self.gen_castle_moves(side, moves);
        }
    }

    #[inline(always)]
    fn gen_castle_moves(&self, side: Side, moves: &mut MoveList) {
        if self.is_frc() {
            self.gen_frc_castle_moves(side, moves);
        } else {
            self.gen_standard_castle_moves(side, moves);
        }
    }

    #[inline(always)]
    fn gen_standard_castle_moves(&self, side: Side, moves: &mut MoveList) {
        const FLAGS: [MoveFlag; 2] = [MoveFlag::CastleK, MoveFlag::CastleQ];
        const OFFSETS: [i8; 2] = [castling::KS_CASTLE_OFFSET, castling::QS_CASTLE_OFFSET];

        let occ = self.occ();
        let king_sq = self.king_sq(side);
        let has_rights = [
            self.has_kingside_rights(side),
            self.has_queenside_rights(side),
        ];

        for i in 0..2 {
            if has_rights[i] {
                let travel_mask = castling::TRAVEL_MASKS[side][i];
                let safety_mask = castling::SAFETY_MASKS[side][i];
                if (occ & travel_mask).is_empty() && (self.threats & safety_mask).is_empty() {
                    moves.add_move(king_sq, king_sq.shift(OFFSETS[i]), FLAGS[i]);
                }
            }
        }
    }

    #[inline(always)]
    fn gen_frc_castle_moves(&self, side: Side, moves: &mut MoveList) {
        self.gen_frc_castle_moves_side(side, true, moves);
        self.gen_frc_castle_moves_side(side, false, moves);
    }

    #[inline(always)]
    fn gen_frc_castle_moves_side(&self, side: Side, kingside: bool, moves: &mut MoveList) {
        let occ = self.occ();
        let rook_file = if kingside {
            self.rights.kingside(side)
        } else {
            self.rights.queenside(side)
        };

        if let Some(rook_file) = rook_file {
            let king_from = self.king_sq(side);
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

            if blocked_sqs.is_empty() && !is_attacked(safe_sqs, side, occ, &self) {
                let flag = if kingside {
                    MoveFlag::CastleK
                } else {
                    MoveFlag::CastleQ
                };
                moves.add_move(king_from, rook_from, flag);
            }
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

#[cfg(test)]
mod tests {
    use crate::board::movegen::MoveFilter;
    use crate::board::moves::MoveList;
    use crate::board::{ray, Board};

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

        let legal_mv_strings = legal_moves
            .iter()
            .map(|mv| mv.mv.to_uci())
            .collect::<Vec<_>>();
        let quiet_mv_strings = quiet_moves
            .iter()
            .map(|mv| mv.mv.to_uci())
            .collect::<Vec<_>>();
        let noisy_mv_strings = noisy_moves
            .iter()
            .map(|mv| mv.mv.to_uci())
            .collect::<Vec<_>>();

        println!("legals: {:?}", legal_mv_strings);
        println!("quiets: {:?}", quiet_mv_strings);
        println!("noisies: {:?}", noisy_mv_strings);
        println!("captures: {:?}", capture_moves);

        let extra_quiets = quiet_mv_strings
            .iter()
            .filter(|mv| !legal_mv_strings.contains(mv))
            .collect::<Vec<_>>();
        let extra_noisies = noisy_mv_strings
            .iter()
            .filter(|mv| !legal_mv_strings.contains(mv))
            .collect::<Vec<_>>();
        let extra_captures = capture_moves
            .iter()
            .map(|mv| mv.mv.to_uci())
            .filter(|mv| !legal_mv_strings.contains(mv))
            .collect::<Vec<_>>();

        println!("extra_quiets: {:?}", extra_quiets);
        println!("extra_noisies: {:?}", extra_noisies);
        println!("extra_captures: {:?}", extra_captures);

        assert_eq!(
            legal_mv_strings.len(),
            quiet_mv_strings.len() + noisy_mv_strings.len()
        );
        assert_eq!(extra_quiets.len(), 0);
        assert_eq!(extra_noisies.len(), 0);
        assert_eq!(extra_captures.len(), 0);

        let noisy_quiets = quiet_moves
            .iter()
            .filter(|mv| board.captured(&mv.mv).is_some())
            .count();
        let quiet_noisies = noisy_moves
            .iter()
            .filter(|mv| board.is_noisy(&mv.mv))
            .count();
        let quiet_captures = quiet_moves
            .iter()
            .filter(|mv| board.captured(&mv.mv).is_some())
            .count();

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
            if line.is_empty() {
                continue;
            }

            let fen = match line.split_once(';') {
                Some((f, _)) => f.trim(),
                None => continue,
            };

            let expected_nodes: Option<u64> = line.split(';').find_map(|tok| {
                tok.trim()
                    .strip_prefix("D6 ")
                    .and_then(|n| n.trim().parse().ok())
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