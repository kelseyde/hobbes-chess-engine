use std::ops::{Index, IndexMut};
use crate::board::file::File;
use crate::board::rank::Rank;

#[derive(Copy, Clone, Eq, PartialEq, Default)]
#[repr(transparent)]
pub struct Square(pub u8);

impl Square {

    pub const COUNT: u8 = 64;

    #[inline(always)]
    pub const fn from(file: File, rank: Rank) -> Square {
        Square((rank as u8) << 3 | (file as u8))
    }

    #[inline(always)]
    pub fn file(self) -> File {
        File::of(self)
    }

    #[inline(always)]
    pub fn rank(self) -> Rank {
        Rank::of(self)
    }

    #[inline(always)]
    pub const fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    #[inline(always)]
    pub const fn flip_file(self) -> Square {
        Square(self.0 ^ 7)
    }

    #[inline(always)]
    pub const fn plus(self, offset: u8) -> Square {
        Square(self.0 + offset)
    }

    #[inline(always)]
    pub const fn minus(self, offset: u8) -> Square {
        Square(self.0 - offset)
    }

    pub fn iter() -> impl Iterator<Item = Square> {
        (0..Self::COUNT).map(Square)
    }

}

impl<T, const N: usize> Index<Square> for [T; N] {
    type Output = T;

    fn index(&self, sq: Square) -> &Self::Output {
        &self[sq.0 as usize]
    }
}

impl<T, const N: usize> IndexMut<Square> for [T; N] {
    fn index_mut(&mut self, sq: Square) -> &mut Self::Output {
        &mut self[sq.0 as usize]
    }
}