use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;

/// A compact encoding of a single threat change, packed into a `u32`.
///
/// Bit layout:
/// bits  0– 7 : threatener     (piece_type (0-5) * 2 + side (0-1))
/// bits  8–15 : from square    (0–63)
/// bits 16–23 : threatened     (piece_type (0-5) * 2 + side (0-1))
/// bits 24–30 : to square      (0–63)
/// bit     31 : add            (1 = add threat, 0 = remove threat)
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct ThreatDelta(u32);

impl ThreatDelta {
    const FROM_SHIFT: u32 = 8;
    const ATTACKED_SHIFT: u32 = 16;
    const TO_SHIFT: u32 = 24;
    const ADD_SHIFT: u32 = 31;

    pub fn new(
        side: Side,
        from: Square,
        to: Square,
        piece: Piece,
        attacked: Piece,
        attacked_side: Side,
        add: bool,
    ) -> Self {
        let threatener = Self::encode_piece(piece, side);
        let threatened = Self::encode_piece(attacked, attacked_side);
        Self(
            threatener
                | (from.0 as u32) << Self::FROM_SHIFT
                | threatened << Self::ATTACKED_SHIFT
                | (to.0 as u32) << Self::TO_SHIFT
                | (add as u32) << Self::ADD_SHIFT,
        )
    }

    #[inline(always)]
    const fn encode_piece(piece: Piece, side: Side) -> u32 {
        piece as u32 * 2 + side as u32
    }

    #[inline(always)]
    const fn decode_piece(val: u8) -> Piece {
        unsafe { std::mem::transmute(val / 2) }
    }

    #[inline(always)]
    const fn decode_side(val: u8) -> Side {
        unsafe { std::mem::transmute(val & 1) }
    }

    #[inline(always)]
    pub const fn piece(self) -> Piece {
        Self::decode_piece(self.0 as u8)
    }

    #[inline(always)]
    pub const fn side(self) -> Side {
        Self::decode_side(self.0 as u8)
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        unsafe { std::mem::transmute((self.0 >> Self::FROM_SHIFT) as u8) }
    }

    #[inline(always)]
    pub const fn attacked(self) -> Piece {
        Self::decode_piece((self.0 >> Self::ATTACKED_SHIFT) as u8)
    }

    #[inline(always)]
    pub const fn attacked_side(self) -> Side {
        Self::decode_side((self.0 >> Self::ATTACKED_SHIFT) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        unsafe { std::mem::transmute(((self.0 >> Self::TO_SHIFT) & 0x7F) as u8) }
    }

    #[inline(always)]
    pub const fn add(self) -> bool {
        self.0 >> Self::ADD_SHIFT != 0
    }
}