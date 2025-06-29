use crate::types::square::Square;
use crate::types::File;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

#[derive(Copy, Clone, Eq, PartialEq, Default, Debug)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {

    pub const ALL: Self = Self(0xFFFFFFFFFFFFFFFF);

    pub const NONE: Self = Self(0);

    pub fn new(bb: u64) -> Self {
        Self(bb)
    }

    pub fn empty() -> Self {
        Self::NONE
    }

    pub fn of_sq(sq: Square) -> Self {
        Self(1 << sq.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub const fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    pub const fn shift(self, offset: i8) -> Self {
        if offset > 0 {
            Self(self.0 << offset)
        } else {
            Self(self.0 >> -offset)
        }
    }

    pub const fn pop(self) -> Self {
        Self(self.0 & (self.0 - 1))
    }

    pub const fn pop_bit(self, sq: Square) -> Self {
        Self(self.0 ^ (1 << sq.0))
    }

    pub const fn north(self) -> Self {
        Bitboard(self.0 << 8)
    }

    pub const fn south(self) -> Self {
        Bitboard(self.0 >> 8)
    }

    pub const fn north_east(self) -> Self {
        Bitboard(self.0 << 9 & !File::A.to_bb().0)
    }

    pub const fn north_west(self) -> Self {
        Bitboard(self.0 << 7 & !File::H.to_bb().0)
    }

    pub fn south_east(self) -> Self {
        Bitboard(self.0 >> 7 & !File::A.to_bb().0)
    }

    pub fn south_west(self) -> Self {
        Bitboard(self.0 >> 9 & !File::H.to_bb().0)
    }

    pub fn print(self) {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                let bit = (self.0 >> sq) & 1;
                if bit == 1 {
                    print!("X ");
                } else {
                    print!("0 ");
                }
            }
            println!();
        }
    }

}

impl Iterator for Bitboard {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_empty() {
            None
        } else {
            let lsb = self.lsb();
            self.0 &= self.0 - 1;
            Some(lsb)
        }
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Shl<u8> for Bitboard {
    type Output = Self;

    fn shl(self, rhs: u8) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl Shr<u8> for Bitboard {
    type Output = Self;

    fn shr(self, rhs: u8) -> Self::Output {
        Self(self.0 >> rhs)
    }
}