use crate::board::bitboard::Bitboard;
use crate::board::magics::{BISHOP_ATTACKS, BISHOP_MAGICS, ROOK_ATTACKS, ROOK_MAGICS};
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::side::Side::White;
use crate::board::square::Square;

#[inline(always)]
pub fn attacks(sq: Square, piece: Piece, side: Side, occ: Bitboard) -> Bitboard {
    match piece {
        Piece::Pawn => pawn(sq, side),
        Piece::Knight => knight(sq),
        Piece::Bishop => bishop(sq, occ),
        Piece::Rook => rook(sq, occ),
        Piece::Queen => queen(sq, occ),
        Piece::King => king(sq),
    }
}

#[inline(always)]
pub fn pawn_attacks(pawns: Bitboard, side: Side) -> Bitboard {
    if side == White {
        pawns.north_east() | pawns.north_west()
    } else {
        pawns.south_east() | pawns.south_west()
    }
}

#[inline(always)]
pub fn pawn(sq: Square, side: Side) -> Bitboard {
    let bb = Bitboard::of_sq(sq);
    pawn_attacks(bb, side)
}

#[inline(always)]
pub fn knight(sq: Square) -> Bitboard {
    KNIGHT[sq]
}

#[inline(always)]
pub fn bishop(sq: Square, blockers: Bitboard) -> Bitboard {
    let magic = BISHOP_MAGICS[sq];
    let idx = magic.index(blockers.0);
    BISHOP_ATTACKS[idx]
}

#[inline(always)]
pub fn rook(sq: Square, blockers: Bitboard) -> Bitboard {
    let magic = ROOK_MAGICS[sq];
    let idx = magic.index(blockers.0);
    ROOK_ATTACKS[idx]
}

#[inline(always)]
pub fn queen(sq: Square, occ: Bitboard) -> Bitboard {
    rook(sq, occ) | bishop(sq, occ)
}

#[inline(always)]
pub fn king(sq: Square) -> Bitboard {
    KING[sq]
}

pub const KING: [Bitboard; 64] = [
    Bitboard(0x0000000000000302),
    Bitboard(0x0000000000000705),
    Bitboard(0x0000000000000e0a),
    Bitboard(0x0000000000001c14),
    Bitboard(0x0000000000003828),
    Bitboard(0x0000000000007050),
    Bitboard(0x000000000000e0a0),
    Bitboard(0x000000000000c040),
    Bitboard(0x0000000000030203),
    Bitboard(0x0000000000070507),
    Bitboard(0x00000000000e0a0e),
    Bitboard(0x00000000001c141c),
    Bitboard(0x0000000000382838),
    Bitboard(0x0000000000705070),
    Bitboard(0x0000000000e0a0e0),
    Bitboard(0x0000000000c040c0),
    Bitboard(0x0000000003020300),
    Bitboard(0x0000000007050700),
    Bitboard(0x000000000e0a0e00),
    Bitboard(0x000000001c141c00),
    Bitboard(0x0000000038283800),
    Bitboard(0x0000000070507000),
    Bitboard(0x00000000e0a0e000),
    Bitboard(0x00000000c040c000),
    Bitboard(0x0000000302030000),
    Bitboard(0x0000000705070000),
    Bitboard(0x0000000e0a0e0000),
    Bitboard(0x0000001c141c0000),
    Bitboard(0x0000003828380000),
    Bitboard(0x0000007050700000),
    Bitboard(0x000000e0a0e00000),
    Bitboard(0x000000c040c00000),
    Bitboard(0x0000030203000000),
    Bitboard(0x0000070507000000),
    Bitboard(0x00000e0a0e000000),
    Bitboard(0x00001c141c000000),
    Bitboard(0x0000382838000000),
    Bitboard(0x0000705070000000),
    Bitboard(0x0000e0a0e0000000),
    Bitboard(0x0000c040c0000000),
    Bitboard(0x0003020300000000),
    Bitboard(0x0007050700000000),
    Bitboard(0x000e0a0e00000000),
    Bitboard(0x001c141c00000000),
    Bitboard(0x0038283800000000),
    Bitboard(0x0070507000000000),
    Bitboard(0x00e0a0e000000000),
    Bitboard(0x00c040c000000000),
    Bitboard(0x0302030000000000),
    Bitboard(0x0705070000000000),
    Bitboard(0x0e0a0e0000000000),
    Bitboard(0x1c141c0000000000),
    Bitboard(0x3828380000000000),
    Bitboard(0x7050700000000000),
    Bitboard(0xe0a0e00000000000),
    Bitboard(0xc040c00000000000),
    Bitboard(0x0203000000000000),
    Bitboard(0x0507000000000000),
    Bitboard(0x0a0e000000000000),
    Bitboard(0x141c000000000000),
    Bitboard(0x2838000000000000),
    Bitboard(0x5070000000000000),
    Bitboard(0xa0e0000000000000),
    Bitboard(0x40c0000000000000),
];

pub const KNIGHT: [Bitboard; 64] = [
    Bitboard(0x0000000000020400),
    Bitboard(0x0000000000050800),
    Bitboard(0x00000000000a1100),
    Bitboard(0x0000000000142200),
    Bitboard(0x0000000000284400),
    Bitboard(0x0000000000508800),
    Bitboard(0x0000000000a01000),
    Bitboard(0x0000000000402000),
    Bitboard(0x0000000002040004),
    Bitboard(0x0000000005080008),
    Bitboard(0x000000000a110011),
    Bitboard(0x0000000014220022),
    Bitboard(0x0000000028440044),
    Bitboard(0x0000000050880088),
    Bitboard(0x00000000a0100010),
    Bitboard(0x0000000040200020),
    Bitboard(0x0000000204000402),
    Bitboard(0x0000000508000805),
    Bitboard(0x0000000a1100110a),
    Bitboard(0x0000001422002214),
    Bitboard(0x0000002844004428),
    Bitboard(0x0000005088008850),
    Bitboard(0x000000a0100010a0),
    Bitboard(0x0000004020002040),
    Bitboard(0x0000020400040200),
    Bitboard(0x0000050800080500),
    Bitboard(0x00000a1100110a00),
    Bitboard(0x0000142200221400),
    Bitboard(0x0000284400442800),
    Bitboard(0x0000508800885000),
    Bitboard(0x0000a0100010a000),
    Bitboard(0x0000402000204000),
    Bitboard(0x0002040004020000),
    Bitboard(0x0005080008050000),
    Bitboard(0x000a1100110a0000),
    Bitboard(0x0014220022140000),
    Bitboard(0x0028440044280000),
    Bitboard(0x0050880088500000),
    Bitboard(0x00a0100010a00000),
    Bitboard(0x0040200020400000),
    Bitboard(0x0204000402000000),
    Bitboard(0x0508000805000000),
    Bitboard(0x0a1100110a000000),
    Bitboard(0x1422002214000000),
    Bitboard(0x2844004428000000),
    Bitboard(0x5088008850000000),
    Bitboard(0xa0100010a0000000),
    Bitboard(0x4020002040000000),
    Bitboard(0x0400040200000000),
    Bitboard(0x0800080500000000),
    Bitboard(0x1100110a00000000),
    Bitboard(0x2200221400000000),
    Bitboard(0x4400442800000000),
    Bitboard(0x8800885000000000),
    Bitboard(0x100010a000000000),
    Bitboard(0x2000204000000000),
    Bitboard(0x0004020000000000),
    Bitboard(0x0008050000000000),
    Bitboard(0x00110a0000000000),
    Bitboard(0x0022140000000000),
    Bitboard(0x0044280000000000),
    Bitboard(0x0088500000000000),
    Bitboard(0x0010a00000000000),
    Bitboard(0x0020400000000000),
];
