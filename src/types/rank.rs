use crate::types::bitboard::Bitboard;
use crate::types::square::Square;

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Rank {
    One, Two, Three, Four, Five, Six, Seven, Eight
}

impl Rank {

    pub const COUNT: usize = 8;

    pub const CHARS: [char; Rank::COUNT] = ['1', '2', '3', '4', '5', '6', '7', '8'];

    pub const BB: [Bitboard; Rank::COUNT] = [
        Bitboard(0x00000000000000FF),
        Bitboard(0x000000000000FF00),
        Bitboard(0x0000000000FF0000),
        Bitboard(0x00000000FF000000),
        Bitboard(0x000000FF00000000),
        Bitboard(0x0000FF0000000000),
        Bitboard(0x00FF000000000000),
        Bitboard(0xFF00000000000000),
    ];

    pub fn parse(rank: usize) -> Rank {
        match rank {
            0 => Rank::One,
            1 => Rank::Two,
            2 => Rank::Three,
            3 => Rank::Four,
            4 => Rank::Five,
            5 => Rank::Six,
            6 => Rank::Seven,
            7 => Rank::Eight,
            _ => panic!("Invalid rank index: {}", rank),
        }
    }

    pub fn of(sq: Square) -> Rank {
        match sq.0 >> 3 {
            0 => Rank::One,
            1 => Rank::Two,
            2 => Rank::Three,
            3 => Rank::Four,
            4 => Rank::Five,
            5 => Rank::Six,
            6 => Rank::Seven,
            7 => Rank::Eight,
            _ => unreachable!(),
        }
    }

    pub const fn to_bb(self) -> Bitboard {
        Rank::BB[self as usize]
    }

    pub const fn to_char(self) -> char {
        Rank::CHARS[self as usize]
    }

    pub const fn from_char(c: char) -> Option<Rank> {
        match c {
            '1' => Some(Rank::One),
            '2' => Some(Rank::Two),
            '3' => Some(Rank::Three),
            '4' => Some(Rank::Four),
            '5' => Some(Rank::Five),
            '6' => Some(Rank::Six),
            '7' => Some(Rank::Seven),
            '8' => Some(Rank::Eight),
            _ => None,
        }
    }

}