pub mod attacks;
pub mod bitboard;
pub mod castling;
pub mod cuckoo;
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
pub mod observer;

pub mod setwise {
    #[cfg(target_feature = "avx512f")]
    mod avx512;
    #[cfg(target_feature = "avx512f")]
    pub use crate::board::setwise::avx512::*;

    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
    mod avx2;
    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
    pub use crate::board::setwise::avx2::*;

    #[cfg(all(
        target_feature = "neon",
        not(any(target_feature = "avx2", target_feature = "avx512f"))
    ))]
    mod neon;
    #[cfg(all(
        target_feature = "neon",
        not(any(target_feature = "avx2", target_feature = "avx512f"))
    ))]
    pub use neon::*;

    #[cfg(not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "neon"
    )))]
    mod scalar;
    #[cfg(not(any(
        target_feature = "avx512f",
        target_feature = "avx2",
        target_feature = "neon"
    )))]
    pub use crate::board::setwise::scalar::*;
}

use crate::board::castling::Rights;
use crate::board::zobrist::{Hashes, Keys};
use crate::tools::fen;
use bitboard::Bitboard;
use moves::{Move, MoveFlag};
use piece::Piece;
use side::Side;
use side::Side::{Black, White};
use square::Square;
use crate::board::observer::BoardObserver;

/// Represents the current state of the chess board, including the positions of the pieces, the side
/// to move, en passant rights, fifty-move counter, and the move counter. Includes functions to 'make'
/// and 'unmake' moves on the board. Uses bitboards to represent the pieces and 'toggling' functions
/// to set and unset pieces.
#[rustfmt::skip]
#[derive(Clone, Copy)]
pub struct Board {
    pub pieces: [Bitboard; 6],        // bitboards for each piece type
    pub colours: [Bitboard; 2],       // bitboards for each colour
    pub mailbox: [Option<Piece>; 64], // piece type on each square
    pub stm: Side,                    // side to move (White or Black)
    pub hm: u8,                       // number of half moves since last capture or pawn move
    pub fm: u8,                       // number of full moves
    pub ep_sq: Option<Square>,        // en passant square (0-63)
    pub recapture_sq: Option<Square>, // square where a recapture can occur
    pub rights: Rights,               // encoded castle rights
    pub hashes: Hashes,               // zobrist hashes
    pub frc: bool,                    // whether the game is Fischer Random Chess
    pub threats: Bitboard,            // squares attacked by the opponent
    pub checkers: Bitboard,           // opponent pieces checking the king
    pub pinned: [Bitboard; 2],        // pinned pieces for both sides
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// Creates a new board in the standard chess starting position.
    pub fn new() -> Board {
        Board::from_fen(fen::STARTPOS).unwrap()
    }

    /// Creates a completely empty board with no pieces, no castling rights, and no en passant.
    pub fn empty() -> Board {
        Board {
            pieces: [Bitboard::empty(); 6],
            colours: [Bitboard::empty(); 2],
            mailbox: [None; 64],
            stm: White,
            hm: 0,
            fm: 0,
            ep_sq: None,
            recapture_sq: None,
            rights: Rights::default(),
            hashes: Hashes::default(),
            frc: false,
            threats: Bitboard::empty(),
            checkers: Bitboard::empty(),
            pinned: [Bitboard::empty(); 2],
        }
    }

    /// Applies a move to the board, updating all state: piece positions, side to move, castling
    /// rights, en passant square, half-move clock, Zobrist hashes, threats, checkers, and pinned
    /// pieces. Handles promotions, en passant, and both standard and Fischer Random castling.
    #[rustfmt::skip]
    pub fn make<T: BoardObserver>(&mut self, m: &Move, observer: &mut T) {
        let side = self.stm;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let pc = self.piece_at(from).expect("No piece on starting square");
        let captured = self.captured(m);
        let (new_pc, new_to) = (self.new_pc(m, pc), self.new_to(m, from, to));
        let is_ep = flag == MoveFlag::EnPassant;
        let is_frc_castle = self.is_frc() && m.is_castle();

        // Move can be described with a single 'teleport' observer callback, with no additional
        // pieces created/destroyed/transformed (e.g., no castling, ep, or promos).
        let mover_teleports = captured.is_none() && !m.is_promo() && !is_frc_castle;

        // Unset the piece from the starting square.
        self.toggle_sq(from, pc, side);
        if !mover_teleports {
            // If the move can't be described with a simple teleport, we first make the observer
            // destroy the piece on the starting square
            observer.on_piece_destroy(self, pc, side, from);
        }

        if let Some(captured) = captured {
            // Unset the captured piece.
            let capture_sq = self.capture_sq(flag, to);
            self.toggle_sq(capture_sq, captured, !side);
            if is_ep {
                // Tell the observer to destroy the en passant pawn
                observer.on_piece_destroy(self, captured, !side, capture_sq);
            }
        }
        if is_frc_castle {
            // In the case of FRC castling, we first unset the rook to cover the
            // scenario where the king moves to the occupied rook square.
            self.toggle_sq(to, Piece::Rook, side);
            // Tell the observer to destroy the rook
            observer.on_piece_destroy(self, Piece::Rook, side, to);
        }
        // Set the piece on the destination square.
        self.toggle_sq(new_to, new_pc, side);
        if let Some(captured) = captured.filter(|_| !is_ep) {
            // Normal capture, we simply transform the captured piece in-place to the new piece.
            observer.on_piece_transform(self, captured, !side, new_pc, side, new_to);
        } else if mover_teleports {
            // Standard move, simply teleport the piece
            observer.on_piece_teleport(self, pc, side, from, new_to);
        } else {
            // Quiet promo, ep, castling: we already destroyed the old piece, now create the new one.
            observer.on_piece_create(self, new_pc, side, new_to);
        }

        // Handle rook movement for castling.
        if m.is_castle() {
            let kingside = castling::is_kingside(from, to);
            let (rook_from, rook_to) = self.rook_sqs(to, kingside);
            if self.is_frc() {
                self.toggle_sq(rook_to, Piece::Rook, side);
                observer.on_piece_create(self, Piece::Rook, side, rook_to);
            } else {
                self.toggle_sqs(rook_from, rook_to, Piece::Rook, side);
                observer.on_piece_teleport(self, Piece::Rook, side, rook_from, rook_to);
            }
        }

        // Update remaining board state.
        self.ep_sq = self.calc_ep(flag, to);
        self.recapture_sq = captured.map(|_| m.to());
        self.rights = self.calc_castle_rights(from, to, pc);
        self.fm += (side == Black) as u8;
        self.hm = if captured.is_some() || pc == Piece::Pawn { 0 } else { self.hm + 1 };
        self.hashes.flip_stm();
        self.stm = !self.stm;
        self.threats = self.calc_threats(self.stm);
        self.checkers = self.calc_checkers(self.stm);
        self.pinned = self.calc_both_pinned();
    }

    /// Toggles a single piece on or off a given square for the given side.
    ///
    /// Flips the relevant piece-type and colour bitboards, updates the per-square piece array,
    /// and XORs all affected Zobrist keys.
    #[inline]
    pub fn toggle_sq(&mut self, sq: Square, pc: Piece, side: Side) {
        self.pieces[pc] ^= sq;
        self.colours[side] ^= sq;
        let cur = self.mailbox[sq];
        self.mailbox[sq] = if cur == Some(pc) { None } else { Some(pc) };

        let hash = Keys::sq(pc, side, sq);
        self.hashes.update_piece(pc, side, hash);
    }

    /// Toggles a piece off `from` and on to `to` in a single call.
    #[inline]
    pub fn toggle_sqs(&mut self, from: Square, to: Square, piece: Piece, side: Side) {
        self.toggle_sq(from, piece, side);
        self.toggle_sq(to, piece, side);
    }

    /// Returns the `(rook_from, rook_to)` squares needed to relocate the rook during castling.
    ///
    /// In standard chess the rook starts on its usual corner square. In Fischer Random Chess,
    /// castling moves are encoded as "king captures rook".
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
    fn capture_sq(&self, flag: MoveFlag, to: Square) -> Square {
        if flag == MoveFlag::EnPassant {
            self.ep_capture_sq(to)
        } else {
            to
        }
    }

    /// Returns the square of the pawn that would be captured by an en passant capture.
    #[inline]
    fn ep_capture_sq(&self, to: Square) -> Square {
        let offset: i8 = if self.stm == White { -8 } else { 8 };
        Square((to.0 as i8 + offset) as u8)
    }

    /// Returns the piece that will occupy the destination square after the move.
    /// For promotions this is the promotion piece; for all other moves it is `pc` unchanged.
    #[inline]
    fn new_pc(&self, m: &Move, pc: Piece) -> Piece {
        m.promo_piece().unwrap_or(pc)
    }

    /// Returns the square the king will land on after the move.
    ///
    /// In Fischer Random Chess, a castling move is encoded as "king captures rook", so the
    /// encoded destination square is the rook's square, not the king's final square. This
    /// method resolves the true king destination in that case.
    #[inline]
    fn new_to(&self, m: &Move, from: Square, to: Square) -> Square {
        if m.is_castle() && self.is_frc() {
            let kingside = castling::is_kingside(from, to);
            castling::king_to(self.stm, kingside)
        } else {
            to
        }
    }

    /// Recomputes castling rights after a move and updates the Zobrist hash accordingly.
    ///
    /// Rights are removed when:
    /// - The king moves (both rights for that side are cleared).
    /// - Any move originates from or lands on a rook's starting square (the corresponding
    ///   corner right is cleared).
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

        let castle_hash = Keys::castle(original_rights.hash()) ^ Keys::castle(new_rights.hash());
        self.hashes.update_hash(castle_hash);
        new_rights
    }

    /// Recomputes the en passant square after a move and keeps the Zobrist hash in sync.
    #[inline]
    fn calc_ep(&mut self, flag: MoveFlag, sq: Square) -> Option<Square> {
        if let Some(old_ep) = self.ep_sq {
            self.hashes.update_hash(Keys::ep(old_ep));
        }
        let ep_sq = if flag == MoveFlag::DoublePush {
            Some(self.ep_capture_sq(sq))
        } else {
            None
        };
        if let Some(new_ep) = ep_sq {
            self.hashes.update_hash(Keys::ep(new_ep));
        }
        ep_sq
    }

    /// Returns pinned-piece bitboards for both sides as `[white_pinned, black_pinned]`.
    #[inline]
    pub fn calc_both_pinned(&self) -> [Bitboard; 2] {
        [self.calc_pinned(White), self.calc_pinned(Black)]
    }

    /// Computes the set of `side`'s pieces that are absolutely pinned to their king.
    /// A piece is pinned iff it is the only piece standing between the king and a sliding attacker.
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
            if maybe_pinned.count() == 1 {
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

    /// Makes a null move: passes the turn to the opponent without moving any piece.
    pub fn make_null_move(&mut self) {
        self.hm = 0;
        self.stm = !self.stm;
        self.hashes.flip_stm();
        if let Some(ep_sq) = self.ep_sq {
            self.hashes.update_hash(Keys::ep(ep_sq));
            self.ep_sq = None;
        }
        self.threats = self.calc_threats(self.stm);
        self.checkers = self.calc_checkers(self.stm);
    }

    pub fn hash(&self) -> u64 {
        self.hashes.hash()
    }

    pub fn hash_with_50mr_bucket(&self) -> u64 {
        self.hashes.hash() ^ Keys::hm(self.hm_bucket())
    }

    pub fn hm_bucket(&self) -> usize {
        if self.hm < 50 {
            0
        } else {
            (self.hm.saturating_sub(8) as usize / 8).min(15)
        }
    }

    #[inline]
    pub fn pawns(&self, side: Side) -> Bitboard {
        self.pieces[Piece::Pawn] & self.colours[side]
    }

    #[inline]
    pub fn knights(&self, side: Side) -> Bitboard {
        self.pieces[Piece::Knight] & self.colours[side]
    }

    #[inline]
    pub fn bishops(&self, side: Side) -> Bitboard {
        self.pieces[Piece::Bishop] & self.colours[side]
    }

    #[inline]
    pub fn rooks(&self, side: Side) -> Bitboard {
        self.pieces[Piece::Rook] & self.colours[side]
    }

    #[inline]
    pub fn queens(&self, side: Side) -> Bitboard {
        self.pieces[Piece::Queen] & self.colours[side]
    }

    #[inline]
    pub fn diags(&self, side: Side) -> Bitboard {
        self.bishops(side) | self.queens(side)
    }

    #[inline]
    pub fn orthos(&self, side: Side) -> Bitboard {
        self.rooks(side) | self.queens(side)
    }

    #[inline]
    pub fn king(&self, side: Side) -> Bitboard {
        self.pieces[Piece::King] & self.colours[side]
    }

    #[inline]
    pub fn king_sq(&self, side: Side) -> Square {
        self.king(side).lsb()
    }

    #[inline]
    pub fn occ(&self) -> Bitboard {
        self.colours[White] | self.colours[Black]
    }

    #[inline]
    pub fn pieces(&self, pc: Piece) -> Bitboard {
        self.pieces[pc]
    }

    #[inline]
    pub fn side(&self, side: Side) -> Bitboard {
        self.colours[side]
    }

    #[inline]
    pub fn white(&self) -> Bitboard {
        self.colours[White]
    }

    #[inline]
    pub fn black(&self) -> Bitboard {
        self.colours[Black]
    }

    #[inline]
    pub fn us(&self) -> Bitboard {
        self.colours[self.stm]
    }

    #[inline]
    pub fn them(&self) -> Bitboard {
        self.colours[!self.stm]
    }

    #[inline]
    pub fn our(&self, piece: Piece) -> Bitboard {
        self.pieces[piece] & self.colours[self.stm]
    }

    #[inline]
    pub fn their(&self, piece: Piece) -> Bitboard {
        self.pieces[piece] & self.colours[!self.stm]
    }

    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.mailbox[sq]
    }

    /// Returns the piece captured by `mv`, or `None` if the move is not a capture.
    #[inline]
    pub fn captured(&self, mv: &Move) -> Option<Piece> {
        if mv.is_castle() {
            return None;
        }
        if mv.is_ep() {
            return Some(Piece::Pawn);
        }
        self.piece_at(mv.to())
    }

    #[inline]
    pub fn is_noisy(&self, mv: &Move) -> bool {
        mv.is_promo() || self.captured(mv).is_some()
    }

    /// Returns the square occupied by the side to move's king.
    #[inline]
    pub fn our_king_sq(&self) -> Square {
        self.king_sq(self.stm)
    }

    /// Returns the set of the side to move's pieces that are absolutely pinned to their king.
    #[inline]
    pub fn our_pinned(&self) -> Bitboard {
        self.pinned[self.stm]
    }

    /// Returns `true` if `mv` lands on the square of the last capture, making it a recapture.
    #[inline]
    pub fn is_recapture(&self, mv: &Move) -> bool {
        self.recapture_sq.is_some_and(|sq| sq == mv.to())
    }

    /// Returns the side that occupies `sq`, or `None` if the square is empty.
    pub fn side_at(&self, sq: Square) -> Option<Side> {
        if !(self.colours[White] & sq).is_empty() {
            Some(White)
        } else if !(self.colours[Black] & sq).is_empty() {
            Some(Black)
        } else {
            None
        }
    }

    /// Returns `true` if the side to move has at least one piece other than the king and pawns.
    ///
    /// Used in Null Move Pruning to avoid making null moves in king-and-pawn endings, where
    /// the assumption that passing the turn is always worse breaks down due to zugzwang.
    pub fn has_non_pawns(&self) -> bool {
        (self.our(Piece::King) | self.our(Piece::Pawn)) != self.us()
    }

    pub const fn is_fifty_move_rule(&self) -> bool {
        self.hm >= 100
    }

    pub fn is_insufficient_material(&self) -> bool {
        let pawns = self.pieces[Piece::Pawn];
        let rooks = self.pieces[Piece::Rook];
        let queens = self.pieces[Piece::Queen];

        if !(pawns | rooks | queens).is_empty() {
            return false;
        }

        let knights = self.pieces[Piece::Knight];
        let bishops = self.pieces[Piece::Bishop];

        let minor_pieces = knights | bishops;
        let minor_count = minor_pieces.count();
        if minor_count <= 1 {
            return true;
        }

        let white_bishops = bishops & self.white();
        let black_bishops = bishops & self.black();
        if white_bishops.count() >= 2 || black_bishops.count() >= 2 {
            return false;
        }

        let white_knights = knights & self.white();
        let black_knights = knights & self.black();
        if (!white_knights.is_empty() && !white_bishops.is_empty())
            || (!black_knights.is_empty() && !black_bishops.is_empty())
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
    use crate::board::observer::NullBoardObserver;

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
        board.make(&m, &mut NullBoardObserver);
        assert_eq!(board.to_fen(), end_fen);
    }
}
