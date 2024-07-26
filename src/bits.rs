
// File bitboards
pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_B: u64 = 0x0202020202020202;
pub const FILE_C: u64 = 0x0404040404040404;
pub const FILE_D: u64 = 0x0808080808080808;
pub const FILE_E: u64 = 0x1010101010101010;
pub const FILE_F: u64 = 0x2020202020202020;
pub const FILE_G: u64 = 0x4040404040404040;
pub const FILE_H: u64 = 0x8080808080808080;

// Rank bitboards
pub const RANK_1: u64 = 0x00000000000000FF;
pub const RANK_2: u64 = 0x000000000000FF00;
pub const RANK_3: u64 = 0x0000000000FF0000;
pub const RANK_4: u64 = 0x00000000FF000000;
pub const RANK_5: u64 = 0x000000FF00000000;
pub const RANK_6: u64 = 0x0000FF0000000000;
pub const RANK_7: u64 = 0x00FF000000000000;
pub const RANK_8: u64 = 0xFF00000000000000;

pub enum Rights {
    NONE = 0b0000,
    WKS = 0b0001,
    WQS = 0b0010,
    BKS = 0b0100,
    BQS = 0b1000,
    WHITE = 0b0011,
    BLACK = 0b1100,
}

// Squares that must not be attacked when the king castles
pub enum CastleSafetyMask {
    WQS = 0x000000000000001C,
    WKS = 0x0000000000000060,
    BQS = 0x1C00000000000000,
    BKS = 0x6000000000000000,
}

// Squares that must be unoccupied when the king castles
pub enum CastleTravelMask {
    WKS = 0x0000000000000006,
    WQS = 0x000000000000000E,
    BKS = 0x0600000000000000,
    BQS = 0x0E00000000000000,
}

pub fn pop(b: u64) -> u64 {
    b & (b - 1)
}

pub fn lsb(b: u64) -> u8 {
    b.trailing_zeros() as u8
}

pub fn bb(sq: u8) -> u64 {
    1 << sq
}

pub fn north(bb: u64) -> u64 {
    bb << 8
}

pub fn south(bb: u64) -> u64 {
    bb >> 8
}

pub fn east(bb: u64) -> u64 {
    (bb & !FILE_H) << 1
}

pub fn west(bb: u64) -> u64 {
    (bb & !FILE_A) >> 1
}

pub fn north_east(bb: u64) -> u64 {
    (bb & !FILE_H) << 9
}

pub fn north_west(bb: u64) -> u64 {
    (bb & !FILE_A) << 7
}

pub fn south_east(bb: u64) -> u64 {
    (bb & !FILE_H) >> 7
}

pub fn south_west(bb: u64) -> u64 {
    (bb & !FILE_A) >> 9
}


