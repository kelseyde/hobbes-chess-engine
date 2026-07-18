use crate::board::file::File;
use crate::board::square::Square;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

/// A Bitboard is a 64-bit integer where each bit represents a square on the chessboard: the least
/// significant bit corresponds to A1, and the most significant bit corresponds to H8. A set bit
/// can represent e.g. the presence of a piece on that square, or a square being attacked, etc.
/// Bitboards are the main data structure used to represent the state of the chessboard. They allow
/// for efficient querying and manipulation of the board state using bitwise operations.
#[derive(Copy, Clone, Eq, PartialEq, Default, Debug)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const ALL: Self = Self(0xFFFFFFFFFFFFFFFF);

    pub const NONE: Self = Self(0);

    pub fn empty() -> Self {
        Self::NONE
    }

    /// Is the square `sq` set in this bitboard?
    #[inline(always)]
    pub fn contains(self, sq: Square) -> bool {
        (self.0 >> sq.0) & 1 != 0
    }

    /// Create a bitboard with only the square `sq` set.
    #[inline(always)]
    pub fn of_sq(sq: Square) -> Self {
        Self(1 << sq.0)
    }

    /// Is the bitboard empty (i.e., no squares are set)?
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Count the number of set squares in the bitboard.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Are multiple squares set in the bitboard?
    #[inline(always)]
    pub const fn is_multiple(self) -> bool {
        self.0 & (self.0.wrapping_sub(1)) != 0
    }

    /// Get the least significant set square in the bitboard.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        Square(self.0.trailing_zeros() as u8)
    }

    /// Shift all set bits in the bitboard by `offset` squares.
    #[inline(always)]
    pub const fn shift(self, offset: i8) -> Self {
        if offset > 0 {
            Self(self.0 << offset)
        } else {
            Self(self.0 >> -offset)
        }
    }

    /// Unset the least significant set bit from the bitboard.
    #[inline(always)]
    pub const fn pop(self) -> Self {
        Self(self.0 & (self.0 - 1))
    }

    /// Unset the square `sq` from the bitboard (i.e., set it to 0).
    #[inline(always)]
    pub const fn pop_bit(self, sq: Square) -> Self {
        Self(self.0 ^ (1 << sq.0))
    }

    /// Shift all set bits north by one rank (from white's perspective).
    #[inline(always)]
    pub const fn north(self) -> Self {
        Bitboard(self.0 << 8)
    }

    /// Shift all set bits south by one rank (from white's perspective).
    #[inline(always)]
    pub const fn south(self) -> Self {
        Bitboard(self.0 >> 8)
    }

    /// Shift all set bits north-east by one square, taking care to not wrap around the board.
    #[inline(always)]
    pub const fn north_east(self) -> Self {
        Bitboard(self.0 << 9 & !File::A.to_bb().0)
    }

    /// Shift all set bits north-west by one square, taking care to not wrap around the board.
    #[inline(always)]
    pub const fn north_west(self) -> Self {
        Bitboard(self.0 << 7 & !File::H.to_bb().0)
    }

    /// Shift all set bits south-east by one square, taking care to not wrap around the board.
    #[inline(always)]
    pub const fn south_east(self) -> Self {
        Bitboard(self.0 >> 7 & !File::A.to_bb().0)
    }

    /// Shift all set bits south-west by one square, taking care to not wrap around the board.
    #[inline(always)]
    pub const fn south_west(self) -> Self {
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

impl BitOr<Square> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Square) -> Self::Output {
        Self(self.0 | (1u64 << rhs.0))
    }
}

impl BitOrAssign<Square> for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Square) {
        self.0 |= 1u64 << rhs.0;
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitAnd<Square> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Square) -> Self::Output {
        Self(self.0 & (1u64 << rhs.0))
    }
}

impl BitAndAssign<Square> for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Square) {
        self.0 &= 1u64 << rhs.0;
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0 ^= rhs.0;
    }
}

impl BitXor<Square> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Square) -> Self::Output {
        Self(self.0 ^ (1u64 << rhs.0))
    }
}

impl BitXorAssign<Square> for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Square) {
        self.0 ^= 1u64 << rhs.0;
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
