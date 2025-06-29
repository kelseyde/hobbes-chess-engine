use crate::types::bitboard::Bitboard;

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
pub struct CastleSafety;

impl CastleSafety {
    pub const WQS: Bitboard = Bitboard(0x000000000000001C);
    pub const WKS: Bitboard = Bitboard(0x0000000000000070);
    pub const BQS: Bitboard = Bitboard(0x1C00000000000000);
    pub const BKS: Bitboard = Bitboard(0x7000000000000000);
}

// Squares that must be unoccupied when the king castles
pub struct CastleTravel;

impl CastleTravel {
    pub const WKS: Bitboard = Bitboard(0x0000000000000060);
    pub const WQS: Bitboard = Bitboard(0x000000000000000E);
    pub const BKS: Bitboard = Bitboard(0x6000000000000000);
    pub const BQS: Bitboard = Bitboard(0x0E00000000000000);
}


