
// File bitboards
pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = 0x8080808080808080;

// Rank bitboards
pub const RANK_1: u64 = 0x00000000000000FF;
pub const RANK_4: u64 = 0x00000000FF000000;
pub const RANK_5: u64 = 0x000000FF00000000;
pub const RANK_8: u64 = 0xFF00000000000000;

pub enum Rights {
    None = 0b0000,
    WKS = 0b0001,
    WQS = 0b0010,
    BKS = 0b0100,
    BQS = 0b1000,
    White = 0b0011,
    Black = 0b1100,
}

// Squares that must not be attacked when the king castles
pub enum CastleSafetyMask {
    WQS = 0x000000000000001C,
    WKS = 0x0000000000000070,
    BQS = 0x1C00000000000000,
    BKS = 0x7000000000000000,
}

// Squares that must be unoccupied when the king castles
pub enum CastleTravelMask {
    WKS = 0x0000000000000060,
    WQS = 0x000000000000000E,
    BKS = 0x6000000000000000,
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

pub fn north_east(bb: u64) -> u64 {
    (bb << 9) & !FILE_A
}

pub fn north_west(bb: u64) -> u64 {
    (bb << 7) & !FILE_H
}

pub fn south_east(bb: u64) -> u64 {
    (bb >> 7) & !FILE_A
}

pub fn south_west(bb: u64) -> u64 {
    (bb >> 9) & !FILE_H
}

pub fn print(bb: u64) {
    for rank in (0..8).rev() {  // Print ranks from 8 to 1
        for file in 0..8 {      // Print files from a to h
            let sq = rank * 8 + file;
            let bit = (bb >> sq) & 1;
            if bit == 1 {
                print!("X ");
            } else {
                print!("0 ");
            }
        }
        println!(); // Newline at the end of each rank
    }
    println!("-----------------");
}


