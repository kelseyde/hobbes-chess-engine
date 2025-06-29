use crate::types::{File, Rank};

#[derive(Copy, Clone, Eq, PartialEq, Default)]
#[repr(transparent)]
pub struct Square(pub u8);

impl Square {

    pub const COUNT: u8 = 64;

    pub fn from(file: File, rank: Rank) -> Square {
        Square((rank as u8) << 3 | (file as u8))
    }

    pub fn file(self) -> File {
        File::of(self)
    }

    pub fn rank(self) -> Rank {
        Rank::of(self)
    }

    pub fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    pub fn flip_file(self) -> Square {
        Square(self.0 ^ 7)
    }

    pub fn plus(self, offset: u8) -> Square {
        Square(self.0 + offset)
    }

    pub fn minus(self, offset: u8) -> Square {
        Square(self.0 - offset)
    }

    pub fn iter() -> impl Iterator<Item = Square> {
        (0..Self::COUNT).map(Square)
    }

}