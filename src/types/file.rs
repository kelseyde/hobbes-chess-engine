use crate::types::bitboard::Bitboard;
use crate::types::square::Square;

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum File {
    A, B, C, D, E, F, G, H
}

impl File {

    pub const COUNT: usize = 8;

    pub const CHARS: [char; File::COUNT] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

    pub const BB: [Bitboard; File::COUNT] = [
        Bitboard(0x0101010101010101),
        Bitboard(0x0202020202020202),
        Bitboard(0x0404040404040404),
        Bitboard(0x0808080808080808),
        Bitboard(0x1010101010101010),
        Bitboard(0x2020202020202020),
        Bitboard(0x4040404040404040),
        Bitboard(0x8080808080808080),
    ];

    pub fn parse(file: usize) -> File {
        match file {
            0 => File::A,
            1 => File::B,
            2 => File::C,
            3 => File::D,
            4 => File::E,
            5 => File::F,
            6 => File::G,
            7 => File::H,
            _ => panic!("Invalid file index: {}", file),
        }
    }

    pub fn of(sq: Square) -> File {
        match sq.0 & 7 {
            0 => File::A,
            1 => File::B,
            2 => File::C,
            3 => File::D,
            4 => File::E,
            5 => File::F,
            6 => File::G,
            7 => File::H,
            _ => unreachable!(),
        }
    }

    pub const fn to_bb(self) -> Bitboard {
        File::BB[self as usize]
    }

    pub fn to_char(self) -> char {
        File::CHARS[self as usize]
    }

    pub fn from_char(c: char) -> Option<File> {
        match c {
            'a' => Some(File::A),
            'b' => Some(File::B),
            'c' => Some(File::C),
            'd' => Some(File::D),
            'e' => Some(File::E),
            'f' => Some(File::F),
            'g' => Some(File::G),
            'h' => Some(File::H),
            _ => None,
        }
    }

}