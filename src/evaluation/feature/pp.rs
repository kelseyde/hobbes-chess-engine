use crate::board::bitboard::Bitboard;
use crate::board::file::File;
use crate::board::side::Side;
use crate::board::side::Side::*;
use crate::board::square::Square;

/// Pawn pair inputs describe unordered pairs of pawns, of either colour, standing on the same or
/// adjacent files (3-wide band). They are a strict superset of 'pawn-attacks-pawn' inputs, since
/// pawn attacks are diagonal and therefore on adjacent files. Therefore those inputs are removed
/// from the standard threat input accumulator and described here.
///
/// Each pawn is given a distinct identity based in its colour (perspective relative), and the
/// square it occupies. Since pawns can only occupy ranks 2-7, there are 48 sqaures x 2 colours, so
/// 96 identities in total. 96 * 95 gives 9120 ordered pairs, but we divide by 2 to get unordered
/// pairs, resulting in 4560 features.
///
/// Credit to chef, Pawnocchio author, for inventing this technique, upon whose implementation this
/// is heavily based.

/// The squares a pawn can stand on: ranks 2-7
const PAWN_SQUARES: u32 = 48;

/// The lowest square a pawn can stand on (a2)
const FIRST_PAWN_SQ: Square = Square(8);

/// Distinct pawn identities (relative colour, square)
const PAWN_IDS: u32 = 2 * PAWN_SQUARES;

/// The total number of pawn-pair features (every unordered pair of pawn identities)
/// PAWN_IDS * PAWN_IDS - 1 gives ordered features, / 2 gives unordered features.
pub const PAWN_PAIR_FEATURES: usize = (PAWN_IDS * (PAWN_IDS - 1) / 2) as usize;

/// Compute the 3-wide band of same-or-adjacent-files for each square on the board.
pub static PP_BANDS: [Bitboard; 64] = init_pp_band();

const fn init_pp_band() -> [Bitboard; 64] {
    let mut table = [Bitboard::NONE; 64];
    let mut sq = FIRST_PAWN_SQ.0 as usize;
    while sq < FIRST_PAWN_SQ.0 as usize + PAWN_SQUARES as usize {
        let file = sq & 7;
        let mut mask = File::BB[file].0;
        // Include the left adjacent file, if it exists
        if file > 0 {
            mask |= File::BB[file - 1].0;
        }
        // Include the right adjacent file, if it exists
        if file < 7 {
            mask |= File::BB[file + 1].0;
        }
        table[sq] = Bitboard(mask);
        sq += 1;
    }
    table
}

/// Compute the identity of a pawn for a given side on a given square, from a given perspective.
#[inline(always)]
fn pawn_id(sq: Square, pawn_side: Side, perspective: Side, mirror: bool) -> u32 {
    let mut sq = sq;
    // Threat indices are reversed for black.
    if perspective == Black {
        sq = sq.flip_rank();
    }
    // Threat indices are horizontally mirrored if the king is on the right side of the board.
    if mirror {
        sq = sq.flip_file();
    }
    // Friendly pawns occupy 0..48, enemy pawns occupy 48..96
    let enemy_offset = if pawn_side == perspective { 0 } else { PAWN_SQUARES };
    enemy_offset + (sq.0 - FIRST_PAWN_SQ.0) as u32
}

/// Compute the index of the unordered pawn pair (a, b) among all pawn features. We compute the
/// higher of the two ids, then skip over every pair formed by a lower high id (of which there are
/// `hi * (hi - 1) / 2`), then offset by the lower id. Doing so results in a unique index for each
/// unordered pawn pair.
#[inline(always)]
fn pp_index(id_a: u32, id_b: u32) -> u32 {
    debug_assert!(id_a != id_b, "A pawn cannot pair with itself!");
    let hi = id_a.max(id_b);
    let lo = id_a.min(id_b);
    hi * (hi - 1) / 2 + lo
}

/// An encoding of one pawn-pair input change: the pair of pawns on `sq_a` and `sq_b`, either
/// created (add = true) or destroyed (add = false). Relative to feature
///
/// Bit layout:
/// 0-7:  square a (0..64)
/// 8-15: square b (0..64)
/// 16:   side a (0 = white, 1 = black)
/// 17:   side b
/// 18:   add (1 = pair created, 0 = pair destroyed)
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct PawnPairFeature(u32);

impl PawnPairFeature {
    const SQ_B_SHIFT: u32 = 8;
    const SIDE_A_SHIFT: u32 = 16;
    const SIDE_B_SHIFT: u32 = 17;
    const ADD_SHIFT: u32 = 18;

    #[inline(always)]
    pub fn new(sq_a: Square, side_a: Side, sq_b: Square, side_b: Side, add: bool) -> Self {
        Self(
            sq_a.0 as u32
                | (sq_b.0 as u32) << Self::SQ_B_SHIFT
                | (side_a as u32) << Self::SIDE_A_SHIFT
                | (side_b as u32) << Self::SIDE_B_SHIFT
                | (add as u32) << Self::ADD_SHIFT,
        )
    }

    /// Compute the index of this pawn pair from the given perspective.
    #[inline(always)]
    pub fn index(&self, perspective: Side, king_sq: Square) -> u32 {
        let mirror = king_sq.file() >= File::E;
        let id_a = pawn_id(self.sq_a(), self.side_a(), perspective, mirror);
        let id_b = pawn_id(self.sq_b(), self.side_b(), perspective, mirror);
        pp_index(id_a, id_b)
    }

    #[inline(always)]
    pub const fn sq_a(self) -> Square {
        Square(self.0 as u8)
    }

    #[inline(always)]
    pub const fn sq_b(self) -> Square {
        Square((self.0 >> Self::SQ_B_SHIFT) as u8)
    }

    #[inline(always)]
    pub const fn side_a(self) -> Side {
        Self::decode_side(self.0 >> Self::SIDE_A_SHIFT)
    }

    #[inline(always)]
    pub const fn side_b(self) -> Side {
        Self::decode_side(self.0 >> Self::SIDE_B_SHIFT)
    }

    #[inline(always)]
    pub const fn add(self) -> bool {
        (self.0 >> Self::ADD_SHIFT) & 1 != 0
    }

    #[inline(always)]
    const fn decode_side(bits: u32) -> Side {
        if bits & 1 == 0 { White } else { Black }
    }
}