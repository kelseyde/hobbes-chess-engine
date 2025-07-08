use arrayvec::ArrayVec;

use crate::types::piece::Piece;
use crate::types::square::Square;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Move(pub u16);

pub const MAX_MOVES: usize = 256;

#[derive(Clone)]
pub struct MoveList {
    pub list: ArrayVec<MoveListEntry, MAX_MOVES>,
    pub len: usize,
}

#[derive(Clone)]
pub struct MoveListEntry {
    pub mv: Move,
    pub score: i32,
}

#[derive(Eq, PartialEq)]
pub enum MoveFlag {
    Standard = 0,
    DoublePush = 1,
    EnPassant = 2,
    CastleK = 3,
    CastleQ = 4,
    PromoQ = 5,
    PromoR = 6,
    PromoB = 7,
    PromoN = 8
}

const FROM_MASK: u16 = 0x3F;
const TO_MASK: u16 = 0xFC0;
const FLAG_MASK: u16 = 0xF000;
const PROMO_FLAGS: [MoveFlag; 4] = [MoveFlag::PromoQ, MoveFlag::PromoR, MoveFlag::PromoB, MoveFlag::PromoN];

impl Move {

    pub const NONE: Move = Move(0);

    pub fn new(from: Square, to: Square, flag: MoveFlag) -> Move {
        Move((from.0 as u16) | ((to.0 as u16) << 6) | ((flag as u16) << 12))
    }

    pub fn from(self) -> Square {
        Square((self.0 & FROM_MASK) as u8)
    }

    pub fn to(self) -> Square {
        Square(((self.0 & TO_MASK) >> 6) as u8)
    }

    pub fn flag(self) -> MoveFlag {
        match (self.0 & FLAG_MASK) >> 12 {
            0 => MoveFlag::Standard,
            1 => MoveFlag::DoublePush,
            2 => MoveFlag::EnPassant,
            3 => MoveFlag::CastleK,
            4 => MoveFlag::CastleQ,
            5 => MoveFlag::PromoQ,
            6 => MoveFlag::PromoR,
            7 => MoveFlag::PromoB,
            8 => MoveFlag::PromoN,
            _ => panic!("Invalid move flag")
        }
    }

    pub fn is_double_push(self) -> bool {
        self.flag() == MoveFlag::DoublePush
    }

    pub fn is_ep(self) -> bool {
        self.flag() == MoveFlag::EnPassant
    }

    pub fn is_castle(self) -> bool {
        self.flag() == MoveFlag::CastleK || self.flag() == MoveFlag::CastleQ
    }

    pub fn is_promo(self) -> bool {
        PROMO_FLAGS.contains(&self.flag())
    }

    pub fn promo_piece(self) -> Option<Piece> {
        match self.flag() {
            MoveFlag::PromoQ => Some(Piece::Queen),
            MoveFlag::PromoR => Some(Piece::Rook),
            MoveFlag::PromoB => Some(Piece::Bishop),
            MoveFlag::PromoN => Some(Piece::Knight),
            _ => None
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
            _ => panic!("Invalid promotion flag")
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
                _ => panic!("Invalid promo piece")
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

    pub fn rook_to(kingside: bool, white: bool) -> Square {
        // Castling target for rooks
        if kingside {
            if white { Square(5) } else { Square(61) }
        } else if white { Square(3) } else { Square(59) }
    }

    pub fn rook_from(kingside: bool, white: bool) -> Square {
        // Castling starting squares for rooks
        if kingside {
            if white { Square(7) } else { Square(63) }
        } else if white { Square(0) } else { Square(56) }
    }

}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {

    pub fn new() -> Self {
        MoveList { list: ArrayVec::new(), len: 0 }
    }

    pub fn add_move(&mut self, from: Square, to: Square, flag: MoveFlag) {
        self.list.push(MoveListEntry { mv: Move::new(from, to, flag), score: 0 });
        self.len += 1;
    }

    pub fn iter(&mut self) -> impl Iterator<Item = &mut MoveListEntry> {
        self.list.iter_mut().take(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, m: &Move) -> bool {
        self.list.iter().take(self.len).any(|entry| entry.mv.matches(m))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&mut self, idx: usize) -> Option<&mut MoveListEntry> {
        if idx < self.len {
            Some(&mut self.list[idx])
        } else {
            None
        }
    }

    pub fn pick(&mut self, scores: &mut [i32; MAX_MOVES]) -> Option<Move> {
        if self.len == 0 {
            return None;
        }

        let mut idx = 0;
        let mut best = i32::MIN;
        for (i, &score) in scores.iter().enumerate().take(self.len) {
            if score > best {
                best = score;
                idx = i;
            }
        }
        self.len -= 1;
        scores.swap(idx, self.len);
        self.list.swap(idx, self.len);
        Some(self.list[self.len].mv)
    }

    pub fn sort(&mut self, scores: &[i32; MAX_MOVES]) {
        let mut indices: Vec<usize> = (0..self.len).collect();

        indices.sort_unstable_by_key(|&i| -scores[i]); // sort descending by score

        let sorted: ArrayVec<MoveListEntry, MAX_MOVES> = indices
            .into_iter()
            .map(|i| self.list[i].clone())
            .collect();

        self.list = sorted;
    }

}