use crate::types::bitboard::Bitboard;
use crate::types::side::Side;
use crate::types::side::Side::White;
use crate::types::square::Square;

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

pub fn is_kingside(from: Square, to: Square) -> bool {
    from.0 < to.0
}

pub fn rook_sqs(king_to_sq: Square) -> (Square, Square) {
    match king_to_sq.0 {
        2 => (Square(0), Square(3)),
        6 => (Square(7), Square(5)),
        58 => (Square(56), Square(59)),
        62 => (Square(63), Square(61)),
        _ => unreachable!()
    }
}

pub fn king_to(kingside: bool, side: Side) -> Square {
    // Castling target for kings
    if kingside {
        if side == White { Square(6) } else { Square(62) }
    } else {
        if side == White { Square(2) } else { Square(58) }
    }
}

pub fn rook_to(kingside: bool, side: Side) -> Square {
    // Castling target for rooks
    if kingside {
        if side == White { Square(5) } else { Square(61) }
    } else {
        if side == White { Square(3) } else { Square(59) }
    }
}

pub fn rook_from(kingside: bool, side: Side) -> Square {
    // Castling starting squares for rooks
    if kingside {
        if side == White { Square(7) } else { Square(63) }
    } else {
        if side == White { Square(0) } else { Square(56) }
    }
}