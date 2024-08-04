use crate::{bits, consts::Piece, consts::Side, moves::Move, moves::MoveFlag};
use crate::bits::Rights;
use crate::consts::Side::{Black, White};
use crate::fen;
use crate::zobrist::Zobrist;

#[derive(Clone, Copy)]
pub struct Board {
    pub bb: [u64; 8],              // bitboards for each piece type (0-5) and for both colours (6-7)
    pub pcs: [Option<Piece>; 64],  // piece type on each square
    pub stm: Side,                 // side to move (White or Black)
    pub hm: u8,                    // number of half moves since last capture or pawn move
    pub fm: u8,                    // number of full moves
    pub ep_sq: Option<u8>,         // en passant square (0-63)
    pub castle: u8,                // encoded castle rights
    pub hash: u64,                 // Zobrist hash
}

impl Board {

    pub fn new() -> Board {
        Board::from_fen(fen::STARTPOS)
    }

    pub fn empty() -> Board {
        Board {
            bb: [0; 8], pcs: [None; 64], stm: White, hm: 0, fm: 0, ep_sq: None, castle: 0, hash: 0
        }
    }

    pub fn make(&mut self, m: &Move) {

        let side = self.stm;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let piece = self.piece_at(from).unwrap();
        let new_piece = if m.is_promo() { m.promo_piece() } else { piece };
        let captured_piece = if flag == MoveFlag::EnPassant { Some(Piece::Pawn) } else { self.pcs[to as usize] };

        self.toggle_sq(from, piece, side);
        if captured_piece.is_some() {
            let capture_sq = if flag == MoveFlag::EnPassant { self.ep_capture_sq(to) } else { to };
            self.toggle_sq(capture_sq, captured_piece.unwrap(), side.flip());
        }
        self.toggle_sq(to, new_piece, side);

        if m.is_castle() {
            let (rook_from, rook_to) = self.rook_sqs(to);
            self.toggle_sqs(rook_from, rook_to, Piece::Rook, side);
        }

        self.ep_sq = self.calc_ep(flag, to);
        self.castle = self.calc_castle_rights(from, to, piece);
        self.fm += if side == Black { 1 } else { 0 };
        self.hm = if captured_piece.is_some() || piece == Piece::Pawn { 0 } else { self.hm + 1 };
        self.hash = Zobrist::toggle_stm(self.hash);
        self.stm = self.stm.flip();

    }

    #[inline]
    pub fn toggle_sq(&mut self, sq: u8, piece: Piece, side: Side) {
        let bb = bits::bb(sq);
        self.bb[piece as usize] ^= bb;
        self.bb[side.idx()] ^= bb;
        self.pcs[sq as usize] = if self.pcs[sq as usize] == Some(piece) { None } else { Some(piece) };
        self.hash = Zobrist::toggle_sq(self.hash, piece, side, sq);
    }

    #[inline]
    pub fn toggle_sqs(&mut self, from: u8, to: u8, piece: Piece, side: Side) {
        self.toggle_sq(from, piece, side);
        self.toggle_sq(to, piece, side);
    }

    #[inline]
    fn rook_sqs(self, king_to_sq: u8) -> (u8, u8) {
        match king_to_sq {
            2 => (0, 3),
            6 => (7, 5),
            58 => (56, 59),
            62 => (63, 61),
            _ => unreachable!()
        }
    }

    #[inline]
    fn ep_capture_sq(&self, to: u8) -> u8 {
        if self.stm == White { to - 8 } else { to + 8 }
    }

    #[inline]
    fn calc_castle_rights(&mut self, from: u8, to: u8, piece_type: Piece) -> u8 {
        self.hash = Zobrist::toggle_castle(self.hash, self.castle);
        let mut new_rights = self.castle;
        if new_rights == Rights::None as u8 {
            // Both sides already lost castling rights, so nothing to calculate.
            return new_rights;
        }
        // Any move by the king removes castling rights.
        if piece_type == Piece::King {
            new_rights &= if self.stm == White { Rights::Black as u8 } else { Rights::White as u8 };
        }
        // Any move starting from/ending at a rook square removes castling rights for that corner.
        if from == 7 || to == 7    { new_rights &= !(Rights::WKS as u8); }
        if from == 63 || to == 63  { new_rights &= !(Rights::BKS as u8); }
        if from == 0 || to == 0    { new_rights &= !(Rights::WQS as u8); }
        if from == 56 || to == 56  { new_rights &= !(Rights::BQS as u8); }
        self.hash = Zobrist::toggle_castle(self.hash, new_rights);
        new_rights
    }

    #[inline]
    fn calc_ep(&mut self, flag: MoveFlag, sq: u8) -> Option<u8>{
        if self.ep_sq.is_some() {
            self.hash = Zobrist::toggle_ep(self.hash, self.ep_sq.unwrap());
        }
        let ep_sq = if flag == MoveFlag::DoublePush { Some(self.ep_capture_sq(sq)) } else { None };
        if ep_sq.is_some() {
            self.hash = Zobrist::toggle_ep(self.hash, ep_sq.unwrap());
        }
        ep_sq
    }

    pub fn has_kingside_rights(&self, side: Side) -> bool {
        if side == White {
            self.castle & Rights::WKS as u8 != 0
        } else {
            self.castle & Rights::BKS as u8 != 0
        }
    }

    pub fn has_queenside_rights(&self, side: Side) -> bool {
        if side == White {
            self.castle & Rights::WQS as u8 != 0
        } else {
            self.castle & Rights::BQS as u8 != 0
        }
    }

    pub fn pawns(self, side: Side) -> u64 {
        self.bb[Piece::Pawn as usize] & self.bb[side.idx()]
    }

    pub fn knights(self, side: Side) -> u64 {
        self.bb[Piece::Knight as usize] & self.bb[side.idx()]
    }

    pub fn bishops(self, side: Side) -> u64 {
        self.bb[Piece::Bishop as usize] & self.bb[side.idx()]
    }

    pub fn rooks(self, side: Side) -> u64 {
        self.bb[Piece::Rook as usize] & self.bb[side.idx()]
    }

    pub fn queens(self, side: Side) -> u64 {
        self.bb[Piece::Queen as usize] & self.bb[side.idx()]
    }

    pub fn king(self, side: Side) -> u64 {
        self.bb[Piece::King as usize] & self.bb[side.idx()]
    }

    pub fn occ(self) -> u64 {
        self.bb[White.idx()] | self.bb[Black.idx()]
    }

    pub fn pcs(self, piece: Piece) -> u64 {
        self.bb[piece as usize]
    }

    pub fn side(self, side: Side) -> u64 {
        self.bb[side.idx()]
    }

    pub fn piece_at(self, sq: u8) -> Option<Piece> {
        self.pcs[sq as usize]
    }

    pub fn side_at(self, sq: u8) -> Option<Side> {
        if self.bb[White.idx()] & bits::bb(sq) != 0 { Some(White) }
        else if self.bb[Black.idx()] & bits::bb(sq) != 0 { Some(Black) }
        else { None }
    }

    pub fn rank(sq: u8) -> u8 {
        sq >> 3
    }

    pub fn file(sq: u8) -> u8 {
        sq & 7
    }

    pub fn sq_idx(rank: u8, file: u8) -> u8 {
        rank << 3 | file
    }

    pub fn is_valid_sq(sq: u8) -> bool {
        sq < 64
    }

}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::moves::{Move, MoveFlag};

    #[test]
    fn standard_move() {
        assert_make_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                         "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
                         Move::parse_uci("g1f3"));
    }

    #[test]
    fn capture_move() {
        assert_make_move("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                         "rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
                         Move::parse_uci("e4d5"));
    }

    #[test]
    fn double_push() {
        assert_make_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                         "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
                         Move::parse_uci_with_flag("c7c5", MoveFlag::DoublePush));
    }

    #[test]
    fn en_passant() {
        assert_make_move("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                         "rnbqkbnr/ppp1p1pp/5P2/3p4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
                         Move::parse_uci_with_flag("e5f6", MoveFlag::EnPassant));
    }

    #[test]
    fn castle_kingside_white() {
        assert_make_move("r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                         "r1bqk1nr/pppp1ppp/2n5/1Bb1p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4",
                         Move::parse_uci_with_flag("e1g1", MoveFlag::CastleK));
    }

    #[test]
    fn castle_kingside_black() {
        assert_make_move("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R b KQkq - 0 4",
                         "rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQ - 1 5",
                         Move::parse_uci_with_flag("e8g8", MoveFlag::CastleQ));
    }

    #[test]
    fn castle_queenside_white() {
        assert_make_move("r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/R3KBNR w KQkq - 6 5",
                         "r3kbnr/pppqpppp/2n5/3p1b2/3P1B2/2N5/PPPQPPPP/2KR1BNR b kq - 7 5",
                         Move::parse_uci_with_flag("e1c1", MoveFlag::CastleQ));
    }

    #[test]
    fn castle_queenside_black() {
        assert_make_move("r3kbnr/pppqpppp/2n5/3p1b2/8/2N2NP1/PPPPPPBP/R1BQ1K1R b kq - 6 5",
                         "2kr1bnr/pppqpppp/2n5/3p1b2/8/2N2NP1/PPPPPPBP/R1BQ1K1R w - - 7 6",
                         Move::parse_uci_with_flag("e8c8", MoveFlag::CastleQ));
    }

    #[test]
    fn queen_promotion() {
        assert_make_move("rn1q1bnr/pppbkPpp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQ - 1 5",
                         "rn1q1bQr/pppbk1pp/8/8/8/8/PPPP1PPP/RNBQKBNR b KQ - 0 5",
                         Move::parse_uci("f7g8q"));
    }

    fn assert_make_move(start_fen: &str, end_fen: &str, m: Move) {
        let mut board = Board::from_fen(start_fen);
        board.make(&m);
        assert_eq!(board.to_fen(), end_fen);
    }

}