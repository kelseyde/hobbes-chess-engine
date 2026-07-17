use std::ptr::null;
use crate::board::file::File;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;

/// The total number of threat features encoded in the network.
///
/// Why 60144? Naively, the total possible space of threat features is side * piece * square * side
/// * piece * square = 2 * 6 * 64 * 2 * 6 * 64 = 589824 inputs. However, encoding the entire space
/// would be prohibitively slow. Fortunately for us, many of these encodings are redundant, for
/// reasons explained below. After deduplicating the redundant inputs, we arrive at a total of 60144
/// threat features.
pub const THREAT_FEATURES: usize = 60144;

/// Lookup table indexed by [attacker][victim].
/// This table tells us whether a given attacker/victim combination is included in the threat inputs.
/// Some combinations are redundant: e.g., pawn-attacking-bishop is implied by pawn-attacking-pawn.
/// We can therefore decrease the number of threat features by deduplicating these redundant inputs.
///
/// All king-threats are fully excluded following this logic, in addition to the following list:
///     - PAWN-attacking-BISHOP
///     - PAWN-attacking-ROOK
///     - PAWN-attacking-QUEEN
///     - BISHOP-attacking-QUEEN
///     - ROOK-attacking-QUEEN
///
/// Same-type threats, e.g. knight-attacks-knight, are 'semi-excluded' later in the process, but
/// included here.
///
/// -1 tells us the threat is fully excluded. Otherwise, we return the index of the victim in the
/// attacker's list of valid targets.
#[rustfmt::skip]
const PIECE_TARGET_MAP: [[i32; 6]; 6] = [
    [ 0,  1, -1,  2, -1, -1], // pawn    -> P N R
    [ 0,  1,  2,  3,  4, -1], // knight  -> P N B R Q
    [ 0,  1,  2,  3, -1, -1], // bishop  -> P N B R
    [ 0,  1,  2,  3, -1, -1], // rook    -> P N B R
    [ 0,  1,  2,  3,  4, -1], // queen   -> P N B R Q
    [-1, -1, -1, -1, -1, -1], // king    -> nothing
];

/// For each attacker piece type, tell me how many valid victim types it has, counting each colour
/// separately. This is essentially a pre-computed summary of the `PIECE_TARGET_MAP` table, with each
/// entry multiplied by 2 to account for the two sides.
const PIECE_TARGET_COUNT: [i32; 6] = [6, 10, 8, 8, 10, 8];

static ATTACK_INDEX: [[[u32; 2]; 12]; 12] = [[[0; 2]; 12]; 12];
static OFFSETS: [[u32; 64]; 12] = [[0; 64]; 12];
static PIECE_INDEX: [[[u8; 64]; 64]; 12] = [[[0; 64]; 64]; 12];




fn is_threat_included(attacker: Piece, victim: Piece) -> bool {
    PIECE_TARGET_MAP[attacker as usize][victim as usize] >= 0
}

pub fn threat_index(
    side: Side,
    king_sq: Square,
    mut attacker: Piece,
    mut attacker_side: Side,
    mut victim: Piece,
    mut victim_side: Side,
    mut from: Square,
    mut to: Square
) -> (bool, i32) {
    // Threat indices are reversed for black.
    if side == Side::Black {
        attacker_side = !attacker_side;
        victim_side = !victim_side;
        from = from.flip_rank();
        to = to.flip_rank();
    }
    // Threat indices are horizontally mirrored if the king is on the right side of the board.
    if king_sq.file() >= File::E {
        from = from.flip_file();
        to = to.flip_file();
    }

    let is_forward_threat = from.0 < to.0;

    // relative-colour coloured-piece indices
    let att = piece_index(attacker, attacker_side);
    let vic = piece_index(victim, victim_side);

    let base   = ATTACK_INDEX[att][vic];           // precomputed global base
    let offset = OFFSETS[att][from];               // from-square offset within attacker block
    let victim_idx = PIECE_INDEX[att][from][to];   // compressed victim square

    (
        base != u32::MAX,
        base.wrapping_add(offset).wrapping_add(victim_idx as u32) as i32
    )
}

const fn piece_index(pc: Piece, side: Side) -> usize {
    pc as usize + 6 * side as usize
}

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