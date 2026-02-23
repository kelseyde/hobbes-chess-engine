pub mod attacks;
pub mod bitboard;
pub mod castling;
pub mod file;
pub mod legal;
pub mod magics;
pub mod movegen;
pub mod moves;
pub mod piece;
pub mod rank;
pub mod ray;
pub mod side;
pub mod square;
pub mod zobrist;

use crate::board::castling::Rights;
use crate::board::zobrist::{Keys, Zobrist};
use crate::tools::fen;
use bitboard::Bitboard;
use moves::{Move, MoveFlag};
use piece::Piece;
use side::Side;
use side::Side::{Black, White};
use square::Square;

/// Represents the current state of the chess board, including the positions of the pieces, the side
/// to move, en passant rights, fifty-move counter, and the move counter. Includes functions to 'make'
/// and 'unmake' moves on the board. Uses bitboards to represent the pieces and 'toggling' functions
/// to set and unset pieces.
#[derive(Clone, Copy)]
pub struct Board {
    pub bb: [Bitboard; 8], // bitboards for each piece type (0-5) and for both colours (6-7)
    pub pcs: [Option<Piece>; 64], // piece type on each square
    pub stm: Side,         // side to move (White or Black)
    pub hm: u8,            // number of half moves since last capture or pawn move
    pub fm: u8,            // number of full moves
    pub ep_sq: Option<Square>, // en passant square (0-63)
    pub recapture_sq: Option<Square>, // square where a recapture can occur
    pub rights: Rights,    // encoded castle rights
    pub keys: Keys,        // zobrist hashes
    pub frc: bool,         // whether the game is Fischer Random Chess
    pub threats: Bitboard, // squares attacked by the opponent
    pub checkers: Bitboard, // opponent pieces checking the king
    pub pinned: [Bitboard; 2], // pinned pieces for both sides
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Board {
        Board::from_fen(fen::STARTPOS).unwrap()
    }

    pub fn empty() -> Board {
        Board {
            bb: [Bitboard::empty(); 8],
            pcs: [None; 64],
            stm: White,
            hm: 0,
            fm: 0,
            ep_sq: None,
            recapture_sq: None,
            rights: Rights::default(),
            keys: Keys::default(),
            frc: false,
            threats: Bitboard::empty(),
            checkers: Bitboard::empty(),
            pinned: [Bitboard::empty(); 2],
        }
    }

    pub fn make(&mut self, m: &Move) {
        let side = self.stm;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let pc = self.piece_at(from).unwrap();
        let captured = self.captured(m);
        let new_pc = self.new_pc(m, pc);
        let new_to = self.new_to(m, from, to);

        self.toggle_sq(from, pc, side);
        if let Some(captured) = captured {
            let capture_sq = if flag == MoveFlag::EnPassant {
                self.ep_capture_sq(to)
            } else {
                to
            };
            self.toggle_sq(capture_sq, captured, !side);
        }
        if self.is_frc() && m.is_castle() {
            // In the case of FRC castling, we first unset the rook to cover the
            // scenario where the king moves to the occupied rook square.
            self.toggle_sq(to, Piece::Rook, side);
        }
        self.toggle_sq(new_to, new_pc, side);

        if m.is_castle() {
            let kingside = castling::is_kingside(from, to);
            let (rook_from, rook_to) = self.rook_sqs(to, kingside);
            if self.is_frc() {
                self.toggle_sq(rook_to, Piece::Rook, side);
            } else {
                self.toggle_sqs(rook_from, rook_to, Piece::Rook, side);
            }
        }

        self.ep_sq = self.calc_ep(flag, to);
        self.recapture_sq = if captured.is_some() {
            Some(m.to())
        } else {
            None
        };
        self.rights = self.calc_castle_rights(from, to, pc);
        self.fm += if side == Black { 1 } else { 0 };
        self.hm = if captured.is_some() || pc == Piece::Pawn {
            0
        } else {
            self.hm + 1
        };
        self.keys.hash ^= Zobrist::stm();
        self.stm = !self.stm;
        self.threats = self.calc_threats(self.stm);
        self.checkers = self.calc_checkers(self.stm);
        self.pinned = self.calc_both_pinned();
    }

    #[inline(always)]
    pub fn toggle_sq(&mut self, sq: Square, pc: Piece, side: Side) {
        let bb: Bitboard = Bitboard::of_sq(sq);
        self.bb[pc] ^= bb;
        self.bb[side.idx()] ^= bb;
        let cur = self.pcs[sq];
        self.pcs[sq] = if cur == Some(pc) { None } else { Some(pc) };

        let hash = Zobrist::sq(pc, side, sq);
        self.keys.hash ^= hash;
        if pc == Piece::Pawn {
            self.keys.pawn_hash ^= hash;
        } else {
            self.keys.non_pawn_hashes[side] ^= hash;
            if pc.is_major() {
                self.keys.major_hash ^= hash;
            }
            if pc.is_minor() {
                self.keys.minor_hash ^= hash;
            }
        }
    }

    #[inline]
    pub fn toggle_sqs(&mut self, from: Square, to: Square, piece: Piece, side: Side) {
        self.toggle_sq(from, piece, side);
        self.toggle_sq(to, piece, side);
    }

    #[inline]
    fn rook_sqs(&self, king_to_sq: Square, kingside: bool) -> (Square, Square) {
        let rook_from = if self.is_frc() {
            // In Chess960, castling moves are encoded as king captures rook
            king_to_sq
        } else {
            castling::rook_from(self.stm, kingside)
        };
        let rook_to = castling::rook_to(self.stm, kingside);
        (rook_from, rook_to)
    }

    #[inline]
    fn ep_capture_sq(&self, to: Square) -> Square {
        if self.stm == White {
            Square(to.0 - 8)
        } else {
            Square(to.0 + 8)
        }
    }

    #[inline]
    fn new_pc(&self, m: &Move, pc: Piece) -> Piece {
        if let Some(promo) = m.promo_piece() {
            promo
        } else {
            pc
        }
    }

    #[inline]
    fn new_to(&self, m: &Move, from: Square, to: Square) -> Square {
        if m.is_castle() && self.is_frc() {
            let kingside = castling::is_kingside(from, to);
            castling::king_to(self.stm, kingside)
        } else {
            to
        }
    }

    #[inline]
    fn calc_castle_rights(&mut self, from: Square, to: Square, piece_type: Piece) -> Rights {
        let original_rights = self.rights;
        let mut new_rights = self.rights;
        // Both sides already lost castling rights, so nothing to calculate.
        if new_rights.is_empty() {
            return new_rights;
        }
        // Any move by the king removes castling rights.
        if piece_type == Piece::King {
            new_rights.clear(self.stm);
        }
        // Any move starting from/ending at a rook square removes castling rights for that corner.
        if self.rights.wk_sq() == Some(from) || self.rights.wk_sq() == Some(to) {
            new_rights.clear_side(White, true);
        }
        if self.rights.bk_sq() == Some(from) || self.rights.bk_sq() == Some(to) {
            new_rights.clear_side(Black, true);
        }
        if self.rights.wq_sq() == Some(from) || self.rights.wq_sq() == Some(to) {
            new_rights.clear_side(White, false);
        }
        if self.rights.bq_sq() == Some(from) || self.rights.bq_sq() == Some(to) {
            new_rights.clear_side(Black, false);
        }

        self.keys.hash ^=
            Zobrist::castle(original_rights.hash()) ^ Zobrist::castle(new_rights.hash());
        new_rights
    }

    #[inline]
    fn calc_ep(&mut self, flag: MoveFlag, sq: Square) -> Option<Square> {
        if let Some(old_ep) = self.ep_sq {
            self.keys.hash ^= Zobrist::ep(old_ep);
        }
        let ep_sq = if flag == MoveFlag::DoublePush {
            Some(self.ep_capture_sq(sq))
        } else {
            None
        };
        if let Some(new_ep) = ep_sq {
            self.keys.hash ^= Zobrist::ep(new_ep);
        }
        ep_sq
    }

    #[inline]
    pub fn calc_both_pinned(&self) -> [Bitboard; 2] {
        [self.calc_pinned(White), self.calc_pinned(Black)]
    }

    #[inline]
    pub fn calc_pinned(&self, side: Side) -> Bitboard {
        let king = self.king_sq(side);
        let us = self.side(side);
        let them = self.side(!side);

        let their_diags = self.diags(!side);
        let their_orthos = self.orthos(!side);

        if their_diags.is_empty() && their_orthos.is_empty() {
            return Bitboard::empty();
        }

        let potential_attackers =
            attacks::bishop(king, them) & their_diags | attacks::rook(king, them) & their_orthos;

        let mut pinned = Bitboard::empty();
        for attacker in potential_attackers {
            let between = ray::between(king, attacker);
            let maybe_pinned = us & between;
            // Exactly one piece between king and attacker means it's pinned
            if !maybe_pinned.is_empty() && maybe_pinned.pop().is_empty() {
                pinned |= maybe_pinned;
            }
        }

        pinned
    }

    pub fn has_kingside_rights(&self, side: Side) -> bool {
        self.rights.kingside(side).is_some()
    }

    pub fn has_queenside_rights(&self, side: Side) -> bool {
        self.rights.queenside(side).is_some()
    }

    pub fn make_null_move(&mut self) {
        self.hm = 0;
        self.stm = !self.stm;
        self.keys.hash ^= Zobrist::stm();
        if let Some(ep_sq) = self.ep_sq {
            self.keys.hash ^= Zobrist::ep(ep_sq);
            self.ep_sq = None;
        }
        self.threats = self.calc_threats(self.stm);
        self.checkers = self.calc_checkers(self.stm);
    }

    pub const fn hash(&self) -> u64 {
        self.keys.hash
    }

    #[inline(always)]
    pub fn pawns(&self, side: Side) -> Bitboard {
        self.bb[Piece::Pawn] & self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn knights(&self, side: Side) -> Bitboard {
        self.bb[Piece::Knight] & self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn bishops(&self, side: Side) -> Bitboard {
        self.bb[Piece::Bishop] & self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn rooks(&self, side: Side) -> Bitboard {
        self.bb[Piece::Rook] & self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn queens(&self, side: Side) -> Bitboard {
        self.bb[Piece::Queen] & self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn diags(&self, side: Side) -> Bitboard {
        self.bishops(side) | self.queens(side)
    }

    #[inline(always)]
    pub fn orthos(&self, side: Side) -> Bitboard {
        self.rooks(side) | self.queens(side)
    }

    #[inline(always)]
    pub fn king(&self, side: Side) -> Bitboard {
        self.bb[Piece::King] & self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn king_sq(&self, side: Side) -> Square {
        self.king(side).lsb()
    }

    #[inline(always)]
    pub fn occ(&self) -> Bitboard {
        self.bb[White.idx()] | self.bb[Black.idx()]
    }

    #[inline(always)]
    pub fn pcs(&self, piece: Piece) -> Bitboard {
        self.bb[piece]
    }

    #[inline(always)]
    pub fn side(&self, side: Side) -> Bitboard {
        self.bb[side.idx()]
    }

    #[inline(always)]
    pub fn white(&self) -> Bitboard {
        self.bb[White.idx()]
    }

    #[inline(always)]
    pub fn black(&self) -> Bitboard {
        self.bb[Black.idx()]
    }

    #[inline(always)]
    pub fn us(&self) -> Bitboard {
        self.bb[self.stm.idx()]
    }

    #[inline(always)]
    pub fn them(&self) -> Bitboard {
        self.bb[(!self.stm).idx()]
    }

    #[inline(always)]
    pub fn our(&self, piece: Piece) -> Bitboard {
        self.bb[piece] & self.bb[self.stm.idx()]
    }

    #[inline(always)]
    pub fn their(&self, piece: Piece) -> Bitboard {
        self.bb[piece] & self.bb[(!self.stm).idx()]
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.pcs[sq]
    }

    pub fn pieces(&self, pc: Piece) -> Bitboard {
        self.bb[pc]
    }

    #[inline(always)]
    pub fn captured(&self, mv: &Move) -> Option<Piece> {
        let flag = mv.flag();
        if matches!(flag, MoveFlag::CastleK | MoveFlag::CastleQ) {
            return None;
        }
        if flag == MoveFlag::EnPassant {
            return Some(Piece::Pawn);
        }
        self.piece_at(mv.to())
    }

    #[inline(always)]
    pub fn is_noisy(&self, mv: &Move) -> bool {
        let flag = mv.flag();
        matches!(flag, MoveFlag::PromoQ | MoveFlag::PromoR | MoveFlag::PromoB | MoveFlag::PromoN)
            || self.captured(mv).is_some()
    }

    pub fn side_at(&self, sq: Square) -> Option<Side> {
        if !(self.bb[White.idx()] & Bitboard::of_sq(sq)).is_empty() {
            Some(White)
        } else if !(self.bb[Black.idx()] & Bitboard::of_sq(sq)).is_empty() {
            Some(Black)
        } else {
            None
        }
    }

    pub fn has_non_pawns(&self) -> bool {
        self.our(Piece::King) | self.our(Piece::Pawn) != self.us()
    }

    pub const fn is_fifty_move_rule(&self) -> bool {
        self.hm >= 100
    }

    pub fn is_insufficient_material(&self) -> bool {
        let pawns = self.bb[Piece::Pawn];
        let rooks = self.bb[Piece::Rook];
        let queens = self.bb[Piece::Queen];

        if !(pawns | rooks | queens).is_empty() {
            return false;
        }

        let knights = self.bb[Piece::Knight];
        let bishops = self.bb[Piece::Bishop];

        let minor_pieces = knights | bishops;
        let minor_count = minor_pieces.count();
        if minor_count <= 1 {
            return true;
        }

        if knights.is_empty() && !bishops.is_empty() && (bishops & self.white()).count() == 2
            || (bishops & self.black()).count() == 2
        {
            return false;
        }
        minor_count <= 3
    }

    pub const fn is_frc(&self) -> bool {
        self.frc
    }

    pub const fn set_frc(&mut self, frc: bool) {
        self.frc = frc;
    }
}

#[cfg(test)]
mod tests {
    use crate::board::bitboard::Bitboard;
    use crate::board::moves::{Move, MoveFlag};
    use crate::board::side::Side;
    use crate::board::{ray, Board};

    #[test]
    fn computing_correct_pins() {
        ray::init();
        assert_eq!(
            Board::from_fen("2k5/6r1/6N1/8/8/8/6K1/8 b - - 0 1")
                .unwrap()
                .calc_pinned(Side::White),
            Bitboard(70368744177664),
        );
        assert_eq!(
            Board::from_fen("2k5/7b/6N1/8/8/8/2K5/8 b - - 0 1")
                .unwrap()
                .calc_pinned(Side::White),
            Bitboard(70368744177664),
        );
    }

    #[test]
    fn standard_move() {
        assert_make_move(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
            Move::parse_uci("g1f3"),
        );
    }

    #[test]
    fn capture_move() {
        assert_make_move(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
            Move::parse_uci("e4d5"),
        );
    }

    #[test]
    fn double_push() {
        assert_make_move(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
            Move::parse_uci_with_flag("c7c5", MoveFlag::DoublePush),
        );
    }

    #[test]
    fn en_passant() {
        assert_make_move(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
            Move::parse_uci_with_flag("e5f6", MoveFlag::EnPassant),
        );
    }

    #[test]
    fn castle_kingside_white() {
        assert_make_move(
            "r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
            Move::parse_uci_with_flag("e1g1", MoveFlag::CastleK),
        );
    }

    #[test]
    fn castle_kingside_black() {
        assert_make_move(
            "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 4",
            "rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQ - 1 5",
            Move::parse_uci_with_flag("e8g8", MoveFlag::CastleQ),
        );
    }

    #[test]
    fn castle_queenside_white() {
        assert_make_move(
            "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5",
            "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/2KR1BNR b kq - 7 5",
            Move::parse_uci_with_flag("e1c1", MoveFlag::CastleQ),
        );
    }

    #[test]
    fn castle_queenside_black() {
        assert_make_move(
            "r3kbnr/pppqpppp/2n5/3p1b2/8/2N2NP1/PPPPPPBP/R1BQ1K1R b kq - 6 5",
            "2kr1bnr/pppqpppp/2n5/3p1b2/8/2N2NP1/PPPPPPBP/R1BQ1K1R w - - 7 6",
            Move::parse_uci_with_flag("e8c8", MoveFlag::CastleQ),
        );
    }

    #[test]
    fn queen_promotion() {
        assert_make_move(
            "rn1q1bnr/pppbkPpp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQ - 1 5",
            "rn1q1bQr/pppbk1pp/8/8/8/8/PPPP1PPP/RNBQKBNR b KQ - 0 5",
            Move::parse_uci("f7g8q"),
        );
    }

    #[test]
    fn pseudo_legal_pawn_capture() {
        let board =
            Board::from_fen("1R6/2p2ppk/4q2p/r1p1Pb2/5P2/2PrNQ2/P5PP/4R1K1 w - - 2 26").unwrap();
        assert!(!board.is_pseudo_legal(&Move::parse_uci("e5f5")));
    }

    #[test]
    fn insufficient_material() {
        assert!(Board::from_fen("8/1k6/2n5/8/8/5N2/6K1/8 w - - 0 1")
            .unwrap()
            .is_insufficient_material());
        assert!(!Board::from_fen("8/1k6/2np4/8/8/5N2/6K1/8 w - - 0 1")
            .unwrap()
            .is_insufficient_material());
        assert!(Board::from_fen("8/1k6/2b5/8/8/5B2/6K1/8 w - - 0 1")
            .unwrap()
            .is_insufficient_material());
        assert!(Board::from_fen("8/1k6/2b5/8/8/5N2/6K1/8 w - - 0 1")
            .unwrap()
            .is_insufficient_material());
        assert!(Board::from_fen("8/1k6/2bN4/8/8/5N2/6K1/8 w - - 0 1")
            .unwrap()
            .is_insufficient_material());
        assert!(!Board::from_fen("8/1k6/2bb4/8/8/8/6K1/8 w - - 0 1")
            .unwrap()
            .is_insufficient_material());
    }

    fn assert_make_move(start_fen: &str, end_fen: &str, m: Move) {
        let mut board = Board::from_fen(start_fen).unwrap();
        board.make(&m);
        assert_eq!(board.to_fen(), end_fen);
    }
}
