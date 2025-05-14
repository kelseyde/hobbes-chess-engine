use arrayvec::ArrayVec;

use crate::consts::Piece;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Move(pub u16);

pub const MAX_MOVES: usize = 256;

#[derive(Clone)]
pub struct MoveList {
    pub list: ArrayVec<Move, MAX_MOVES>,
    pub len: usize,
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

    pub fn new(from: u8, to: u8, flag: MoveFlag) -> Move {
        Move((from as u16) | ((to as u16) << 6) | ((flag as u16) << 12))
    }

    pub fn from(self) -> u8 {
        (self.0 & FROM_MASK) as u8
    }

    pub fn to(self) -> u8 {
        ((self.0 & TO_MASK) >> 6) as u8
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

    pub fn promo_piece(self) -> Piece {
        match self.flag() {
            MoveFlag::PromoQ => Piece::Queen,
            MoveFlag::PromoR => Piece::Rook,
            MoveFlag::PromoB => Piece::Bishop,
            MoveFlag::PromoN => Piece::Knight,
            _ => panic!("Invalid promo piece")
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

    fn parse_uci_sq(notation: &str) -> u8 {
        let file = notation.chars().nth(0).unwrap() as u8 - b'a';
        let rank = notation.chars().nth(1).unwrap() as u8 - b'1';
        rank * 8 + file
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
        let promo = if self.is_promo() {
            match self.promo_piece() {
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

    pub fn uci_sq(sq: u8) -> String {
        let file = (sq % 8) + b'a';
        let rank = (sq / 8) + b'1';
        format!("{}{}", file as char, rank as char)
    }

    pub fn matches(self, m: &Move) -> bool {
        self.from() ==  m.from() && self.to() == m.to()
    }

    pub fn exists(self) -> bool {
        self != Move::NONE
    }

}

impl MoveList {

    pub fn new() -> Self {
        MoveList { list: ArrayVec::new(), len: 0 }
    }

    pub fn add_move(&mut self, from: u8, to: u8, flag: MoveFlag) {
        self.list.push(Move::new(from, to, flag));
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.list.iter().take(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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
        Some(self.list[self.len])
    }

}