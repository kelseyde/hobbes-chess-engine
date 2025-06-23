pub struct Square(pub u8);

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum File {
    A, B, C, D, E, F, G, H
}

#[derive(PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Rank {
    One, Two, Three, Four, Five, Six, Seven, Eight
}

impl Square {

    pub fn from(rank: u8, file: u8) -> Square {
        Square((rank << 3) + file)
    }

}