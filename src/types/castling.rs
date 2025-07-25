use crate::types::bitboard::Bitboard;
use crate::types::side::Side;
use crate::types::square::Square;
use crate::types::{File, Rank};

// Check if the castling move is kingside or queenside
pub fn is_kingside(from: Square, to: Square) -> bool {
    from.0 < to.0
}

// Starting square (in classical chess) for the rook
pub fn rook_from(side: Side, kingside: bool) -> Square {
    match (side, kingside) {
        (Side::White, true)  => Square(7),
        (Side::White, false) => Square(0),
        (Side::Black, true)  => Square(63),
        (Side::Black, false) => Square(56),
    }
}

// Target square for the rook
pub fn rook_to(side: Side, kingside: bool) -> Square {
    match (side, kingside) {
        (Side::White, true)  => Square(5),
        (Side::White, false) => Square(3),
        (Side::Black, true)  => Square(61),
        (Side::Black, false) => Square(59),
    }
}

pub fn king_to(side: Side, kingside: bool) -> Square {
    match (side, kingside) {
        (Side::White, true)  => Square(6),
        (Side::White, false) => Square(2),
        (Side::Black, true)  => Square(62),
        (Side::Black, false) => Square(58),
    }
}

// Packed representation of castling rights, inspired by Viridithas.
// The starting file of each rook is required for DFRC-compatibility.
// 8 possible starting files means 3 bits per rook. The bottom 4 bits
// represent the presence/absence of castling rights for each side.
// [ 3 (wk) ][ 3 (wq) ][ 3 (bk) ][ 3 (bq) ][ 4 (flags) ]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Rights {
    data: u16
}

impl Rights {

    // Bits representing the presence/absence of castling rights for colour/side
    pub const WKCA: u16 = 0b0001;
    pub const WQCA: u16 = 0b0010;
    pub const BKCA: u16 = 0b0100;
    pub const BQCA: u16 = 0b1000;

    // Bits representing the starting file of each of the four rooks
    pub const WK_MASK: u16 = 0b111_000_000_000_0000;
    pub const WQ_MASK: u16 = 0b000_111_000_000_0000;
    pub const BK_MASK: u16 = 0b000_000_111_000_0000;
    pub const BQ_MASK: u16 = 0b000_000_000_111_0000;

    // Bits used to shift rook starting files into/out of their appropriate positions
    pub const WK_SHIFT: u8 = 4 + 3 + 3 + 3;
    pub const WQ_SHIFT: u8 = 4 + 3 + 3;
    pub const BK_SHIFT: u8 = 4 + 3;
    pub const BQ_SHIFT: u8 = 4;

    // Mask for the rights part of the encoding, used in zobrist updates
    pub const KEY_MASK: u16 = 0b1111;

    pub const fn new(wk: Option<File>,
                     wq: Option<File>,
                     bk: Option<File>,
                     bq: Option<File>) -> Self {
        let mut data = 0;

        if let Some(file) = wk {
            data |= ((file as u16) << Self::WK_SHIFT) | Self::WKCA;
        }
        if let Some(file) = wq {
            data |= ((file as u16) << Self::WQ_SHIFT) | Self::WQCA;
        }
        if let Some(file) = bk {
            data |= ((file as u16) << Self::BK_SHIFT) | Self::BKCA;
        }
        if let Some(file) = bq {
            data |= ((file as u16) << Self::BQ_SHIFT) | Self::BQCA;
        }

        Rights { data }
    }

    // Default castling rights for classical chess
    pub const fn classical() -> Self {
        Rights::new(Some(File::H), Some(File::A), Some(File::H), Some(File::A))
    }

    pub const fn hash(self) -> u8 {
        (self.data & Self::KEY_MASK) as u8
    }

    pub const fn is_empty(self) -> bool {
        (self.data & Self::KEY_MASK) == 0
    }

    pub fn wk_sq(self) -> Option<Square> {
        let file_value = (self.data & Self::WK_MASK) >> Self::WK_SHIFT;
        let file = File::parse(file_value as usize);
        Some(Square::from(file, Rank::One))
    }

    pub fn wq_sq(self) -> Option<Square> {
        let file_value = (self.data & Self::WQ_MASK) >> Self::WQ_SHIFT;
        let file = File::parse(file_value as usize);
        Some(Square::from(file, Rank::One))
    }

    pub fn bk_sq(self) -> Option<Square> {
        let file_value = (self.data & Self::BK_MASK) >> Self::BK_SHIFT;
        let file = File::parse(file_value as usize);
        Some(Square::from(file, Rank::Eight))
    }

    pub fn bq_sq(self) -> Option<Square> {
        let file_value = (self.data & Self::BQ_MASK) >> Self::BQ_SHIFT;
        let file = File::parse(file_value as usize);
        Some(Square::from(file, Rank::Eight))
    }

    pub fn clear(&mut self, side: Side) {
        self.data &= match side {
            Side::White => !(Self::WK_MASK | Self::WQ_MASK | Self::WKCA | Self::WQCA),
            Side::Black => !(Self::BK_MASK | Self::BQ_MASK | Self::BKCA | Self::BQCA)
        };
    }

    pub fn clear_side(&mut self, side: Side, kingside: bool) {
        self.data &= !match (side, kingside) {
            (Side::White, true)  => Self::WK_MASK | Self::WKCA,
            (Side::White, false) => Self::WQ_MASK | Self::WQCA,
            (Side::Black, true)  => Self::BK_MASK | Self::BKCA,
            (Side::Black, false) => Self::BQ_MASK | Self::BQCA,
        };
    }

    pub fn remove(&mut self, side: Side, file: File) {
        if self.kingside(side) == Some(file) {
            self.clear_side(side, true);
        } else if self.queenside(side) == Some(file) {
            self.clear_side(side, false);
        }
    }

    pub fn kingside(self, side: Side) -> Option<File> {
        let presence = [Self::WKCA, Self::BKCA][side];
        if self.data & presence == 0 {
          return None;
        }
        let shift = [Self::WK_SHIFT, Self::BK_SHIFT][side];
        let mask = [Self::WK_MASK, Self::BK_MASK][side];
        let value = (self.data & mask) >> shift;
        Some(File::parse(value as usize))
    }

    pub fn queenside(self, side: Side) -> Option<File> {
        let presence = [Self::WQCA, Self::BQCA][side];
        if self.data & presence == 0 {
          return None;
        }
        let shift = [Self::WQ_SHIFT, Self::BQ_SHIFT][side];
        let mask = [Self::WQ_MASK, Self::BQ_MASK][side];
        let value = (self.data & mask) >> shift;
        Some(File::parse(value as usize))
    }

    pub fn set_kingside(&mut self, side: Side, file: File) {
        let presence = [Self::WKCA, Self::BKCA][side];
        let shift = [Self::WK_SHIFT, Self::BK_SHIFT][side];
        let mask = [!Self::WK_MASK, !Self::BK_MASK][side];
        let value = file as u16;
        self.data &= mask;
        self.data |= (value << shift) | presence;
    }

    pub fn set_queenside(&mut self, side: Side, file: File) {
        let presence = [Self::WQCA, Self::BQCA][side];
        let shift = [Self::WQ_SHIFT, Self::BQ_SHIFT][side];
        let mask = [!Self::WQ_MASK, !Self::BQ_MASK][side];
        let value = file as u16;
        self.data &= mask;
        self.data |= (value << shift) | presence;
    }

}

// Squares that must not be attacked when the king castles
pub struct CastleSafety;

impl CastleSafety {
    pub const WQS: Bitboard = Bitboard(0x000000000000001C);
    pub const WKS: Bitboard = Bitboard(0x0000000000000070);
    pub const BQS: Bitboard = Bitboard(0x1C00000000000000);
    pub const BKS: Bitboard = Bitboard(0x7000000000000000);
}

// Squares that must be unoccupied when the king castles
pub struct CastleTravel;

impl CastleTravel {
    pub const WKS: Bitboard = Bitboard(0x0000000000000060);
    pub const WQS: Bitboard = Bitboard(0x000000000000000E);
    pub const BKS: Bitboard = Bitboard(0x6000000000000000);
    pub const BQS: Bitboard = Bitboard(0x0E00000000000000);
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::moves::Move;
    use crate::types::castling::Rights;
    use crate::types::side::Side;
    use crate::types::File;

    #[test]
    fn test_kingside_basics() {
        let mut rights = Rights::default();

        assert_eq!(rights.kingside(Side::White), None);
        rights.set_kingside(Side::White, File::H);
        assert_eq!(rights.kingside(Side::White), Some(File::H));

        assert_eq!(rights.kingside(Side::Black), None);
        rights.set_kingside(Side::Black, File::H);
        assert_eq!(rights.kingside(Side::Black), Some(File::H));

        rights.set_kingside(Side::White, File::G);
        assert_eq!(rights.kingside(Side::White), Some(File::G));
    }

    #[test]
    fn test_queenside_basics() {
        let mut rights = Rights::default();

        assert_eq!(rights.queenside(Side::White), None);
        rights.set_queenside(Side::White, File::A);
        assert_eq!(rights.queenside(Side::White), Some(File::A));

        assert_eq!(rights.queenside(Side::Black), None);
        rights.set_queenside(Side::Black, File::A);
        assert_eq!(rights.queenside(Side::Black), Some(File::A));

        rights.set_queenside(Side::White, File::B);
        assert_eq!(rights.queenside(Side::White), Some(File::B));
    }

    #[test]
    fn test_clear_rights() {
        let mut rights = Rights::new(Some(File::H), Some(File::A), Some(File::H), Some(File::A));

        assert_eq!(rights.kingside(Side::White), Some(File::H));
        assert_eq!(rights.queenside(Side::White), Some(File::A));
        assert_eq!(rights.kingside(Side::Black), Some(File::H));
        assert_eq!(rights.queenside(Side::Black), Some(File::A));

        rights.clear(Side::White);
        assert_eq!(rights.kingside(Side::White), None);
        assert_eq!(rights.queenside(Side::White), None);
        assert_eq!(rights.kingside(Side::Black), Some(File::H));
        assert_eq!(rights.queenside(Side::Black), Some(File::A));

        rights.clear(Side::Black);
        assert_eq!(rights.kingside(Side::Black), None);
        assert_eq!(rights.queenside(Side::Black), None);
    }

    #[test]
    fn test_remove_rights() {
        let mut rights =
            Rights::new(Some(File::H), Some(File::A), Some(File::H), Some(File::A));

        rights.remove(Side::White, File::H);
        assert_eq!(rights.kingside(Side::White), None);
        assert_eq!(rights.queenside(Side::White), Some(File::A));

        rights.remove(Side::Black, File::A);
        assert_eq!(rights.kingside(Side::Black), Some(File::H));
        assert_eq!(rights.queenside(Side::Black), None);

        rights.remove(Side::White, File::G);
        assert_eq!(rights.queenside(Side::White), Some(File::A));
    }

    #[test]
    fn test_debug() {

        let mut board =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        assert!(board.has_kingside_rights(Side::White));
        assert!(board.has_queenside_rights(Side::White));
        assert!(board.has_kingside_rights(Side::Black));
        assert!(board.has_queenside_rights(Side::Black));

        board.make(&Move::parse_uci("a1b1"));
        assert!(board.has_kingside_rights(Side::White));
        assert!(!board.has_queenside_rights(Side::White));
        assert!(board.has_kingside_rights(Side::Black));
        assert!(board.has_queenside_rights(Side::Black));

        board.make(&Move::parse_uci("h8f8"));
        assert!(board.has_kingside_rights(Side::White));
        assert!(!board.has_queenside_rights(Side::White));
        assert!(!board.has_kingside_rights(Side::Black));
        assert!(board.has_queenside_rights(Side::Black));

    }

}