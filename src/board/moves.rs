use arrayvec::ArrayVec;
use std::fmt;

use crate::board::piece::Piece;
use crate::board::square::Square;

/// Represents a chess move, encoded as a 16-bit unsigned integer.
/// The encoding is as follows:
/// - Bits 0-5: From square (0-63)
/// - Bits 6-11: To square (0-63)
/// - Bits 12-15: Special flag (0-8)
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Move(pub u16);

/// The maximum number of legal moves in any chess position.
pub const MAX_MOVES: usize = 218;

/// A list of moves, each with an associated score for move ordering purposes.
#[derive(Debug, Clone)]
pub struct MoveList {
    pub list: ArrayVec<ScoredMove, MAX_MOVES>,
}

#[derive(Debug, Copy, Clone)]
pub struct ScoredMove {
    pub mv: Move,
    pub score: i32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MoveFlag {
    Standard,
    DoublePush,
    EnPassant,
    CastleK,
    CastleQ,
    PromoQ,
    PromoR,
    PromoB,
    PromoN,
}

const FROM_MASK: u16 = 0x3F;
const TO_MASK: u16 = 0xFC0;

const TO_SHIFT: u16 = 6;
const FLAG_SHIFT: u16 = 12;

impl Move {
    pub const NONE: Move = Move(0);

    pub fn new(from: Square, to: Square, flag: MoveFlag) -> Move {
        Move((from.0 as u16) | ((to.0 as u16) << TO_SHIFT) | ((flag as u16) << FLAG_SHIFT))
    }

    pub const fn from(self) -> Square {
        Square((self.0 & FROM_MASK) as u8)
    }

    pub const fn to(self) -> Square {
        Square(((self.0 & TO_MASK) >> TO_SHIFT) as u8)
    }

    pub const fn flag(self) -> MoveFlag {
        unsafe { std::mem::transmute((self.0 >> FLAG_SHIFT) as u8) }
    }

    pub fn is_double_push(self) -> bool {
        self.flag() == MoveFlag::DoublePush
    }

    pub fn is_ep(self) -> bool {
        self.flag() == MoveFlag::EnPassant
    }

    pub fn is_castle(self) -> bool {
        matches!(self.flag(), MoveFlag::CastleK | MoveFlag::CastleQ)
    }

    pub fn is_promo(self) -> bool {
        matches!(
            self.flag(),
            MoveFlag::PromoQ | MoveFlag::PromoR | MoveFlag::PromoB | MoveFlag::PromoN
        )
    }

    pub const fn promo_piece(self) -> Option<Piece> {
        match self.flag() {
            MoveFlag::PromoQ => Some(Piece::Queen),
            MoveFlag::PromoR => Some(Piece::Rook),
            MoveFlag::PromoB => Some(Piece::Bishop),
            MoveFlag::PromoN => Some(Piece::Knight),
            _ => None,
        }
    }

    pub fn parse_uci(notation: &str) -> Move {
        let from = Self::parse_uci_sq(&notation[0..2]);
        let to = Self::parse_uci_sq(&notation[2..4]);

        let flag = if notation.len() == 5 {
            let piece_code = &notation[4..5];
            Self::get_promotion_flag(piece_code.chars().nth(0).unwrap())
        } else {
            MoveFlag::Standard
        };

        Move::new(from, to, flag)
    }

    pub fn parse_uci_with_flag(notation: &str, flag: MoveFlag) -> Move {
        let from = Self::parse_uci_sq(&notation[0..2]);
        let to = Self::parse_uci_sq(&notation[2..4]);
        Move::new(from, to, flag)
    }

    fn parse_uci_sq(notation: &str) -> Square {
        let file = notation.chars().nth(0).unwrap() as u8 - b'a';
        let rank = notation.chars().nth(1).unwrap() as u8 - b'1';
        Square(rank * 8 + file)
    }

    fn get_promotion_flag(c: char) -> MoveFlag {
        match c {
            'q' => MoveFlag::PromoQ,
            'r' => MoveFlag::PromoR,
            'b' => MoveFlag::PromoB,
            'n' => MoveFlag::PromoN,
            _ => panic!("Invalid promotion flag"),
        }
    }

    pub fn to_uci(self) -> String {
        let from = Self::uci_sq(self.from());
        let to = Self::uci_sq(self.to());
        let promo = if let Some(promo) = self.promo_piece() {
            match promo {
                Piece::Queen => "q",
                Piece::Rook => "r",
                Piece::Bishop => "b",
                Piece::Knight => "n",
                _ => panic!("Invalid promo piece"),
            }
        } else {
            ""
        };
        format!("{}{}{}", from, to, promo)
    }

    pub fn uci_sq(sq: Square) -> String {
        let file = (sq.0 % 8) + b'a';
        let rank = (sq.0 / 8) + b'1';
        format!("{}{}", file as char, rank as char)
    }

    pub fn matches(self, m: &Move) -> bool {
        let square_match = self.from() == m.from() && self.to() == m.to();
        let promo_match = if self.is_promo() && m.is_promo() {
            self.promo_piece() == m.promo_piece()
        } else {
            true
        };
        square_match && promo_match
    }

    pub fn exists(self) -> bool {
        self != Move::NONE
    }

    pub fn is_null(self) -> bool {
        self == Move::NONE
    }

    pub const fn encoded(self) -> usize {
        (self.0 & 0b0000_1111_1111_1111) as usize
    }

    pub fn rook_to(kingside: bool, white: bool) -> Square {
        // Castling target for rooks
        if kingside {
            if white {
                Square(5)
            } else {
                Square(61)
            }
        } else if white {
            Square(3)
        } else {
            Square(59)
        }
    }

    pub fn rook_from(kingside: bool, white: bool) -> Square {
        // Castling starting squares for rooks
        if kingside {
            if white {
                Square(7)
            } else {
                Square(63)
            }
        } else if white {
            Square(0)
        } else {
            Square(56)
        }
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            list: ArrayVec::new(),
        }
    }

    #[inline(always)]
    pub fn add_move(&mut self, from: Square, to: Square, flag: MoveFlag) {
        unsafe {
            self.list.push_unchecked(ScoredMove {
                mv: Move::new(from, to, flag),
                score: 0,
            });
        }
    }

    #[inline(always)]
    pub fn add(&mut self, entry: ScoredMove) {
        unsafe {
            self.list.push_unchecked(entry);
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub const fn len(&self) -> usize {
        self.list.len()
    }

    pub fn get(&self, idx: usize) -> Option<&ScoredMove> {
        if idx < self.list.len() {
            Some(&self.list[idx])
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &ScoredMove> {
        self.list.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ScoredMove> {
        self.list.iter_mut()
    }
}
