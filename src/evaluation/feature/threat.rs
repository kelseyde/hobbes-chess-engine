use crate::board::bitboard::Bitboard;
use crate::board::file::File;
use crate::board::piece::Piece;
use crate::board::rank::Rank;
use crate::board::side::Side;
use crate::board::side::Side::*;
use crate::board::square::Square;
use crate::board::attacks;

/// Below is a jumbled mess of code used to compute the index of a given threat in the threat inputs
/// accumulator.
/// N.B. Everything here is heavily inspired by other engines, specifically Viridithas, Reckless, &
/// Stormphrax. My only contribution is a demented level of comments to aid my own understanding.

/// The total number of threat features encoded in the network is 60144.
///
/// Why 60144? Naively, the total possible space of threat features is side * piece * square * side
/// * piece * square = 2 * 6 * 64 * 2 * 6 * 64 = 589824 inputs. However, encoding the entire space
/// would be prohibitively slow. Fortunately for us, many of these encodings are redundant, for
/// reasons explained below. After deduplicating the redundant inputs, we arrive at a total of 60144.

/// This table tells us whether a given attacker/victim combination is included in the threat inputs.
/// Some combinations are redundant: e.g., pawn-attacking-bishop is implied by pawn-attacking-pawn.
/// We can therefore decrease the number of threat features by deduplicating these redundant inputs.
///
/// All king-threats are fully excluded, in addition to the following list:
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
const PIECE_TARGET_COUNT: [i32; 6] = [6, 10, 8, 8, 10, 0];

/// Lookup table containing a tuple for each piece/side combination, containing (total pseudo-attacks,
/// global pseudo-attack offset). The first value is simply how many squares that piece pseudo-attacks;
/// the second value is the running total of all pseudo-attacks for all pieces of lower ordinal value.
static mut PIECE_TOTALS: [(i32, i32); 12] = [(0, 0); 12];

/// The following three tables are used to build the index into the threat inputs accumulator for a
/// given threat. They are ordered from big to small: the first table gives us the chunk for a given
/// (attacker, victim, direction); the second gives us the sub-chunk for a given (attacker, from-square);
/// the third table gives us the number of squares the attacker pseudo-attacks from `from` that are
/// below `to`.

/// Provides the base index into the threat inputs array for this combination of attacker, victim,
/// and direction ('forwards' or 'backwards'). This is where exclusions are handled: a fully excluded
/// threat returns `u32::MAX` for both directions; a semi-excluded threat returns `u32::MAX` for the
/// forward direction, but a valid index for the backward direction.
static mut PAIR_BASE: [[[u32; 2]; 12]; 12] = [[[u32::MAX; 2]; 12]; 12];

/// Within a single (attacker, victim) pair block selected by `PAIR_BASE`, this supplies the offset
/// to the sub-block for one particular from-square. It is the running total of the number of squares
/// the attacker pseudo-attacks, summed over all from-squares strictly before `from`.
static mut FROM_OFFSET: [[u32; 64]; 12] = [[0; 64]; 12];

/// For each (attacker, from-square, to-square) combination, store the number of pseudo-attacked
/// squares below `to`. Not every square can be attacked by a given piece on a given square, and so
/// we can drastically reduce the size of our inputs by only indexing squares which are actually
/// pseudo-attackable, ordered low-to-high. We use this number as the compressed victim square index
/// within the from-square's attack set.
static mut VICTIM_ORDINAL: [[[u8; 64]; 64]; 12] = [[[0; 64]; 64]; 12];

/// Initialise the threat-feature lookup tables.
pub fn init() {
    unsafe {
        init_victim_ordinal();
        init_piece_totals();
        init_from_offset();
        init_pair_base();
    }
}

unsafe fn init_victim_ordinal() {
    for (side, pc) in Piece::coloured_pieces() {
        let pc_idx = pc.coloured_index(side);
        for from in Square::iter() {
            let attacks = attacks::attacks(from, pc, side, Bitboard::NONE);
            for to in Square::iter() {
                // The compressed ordinal of `to` is the number of squares this piece attacks that
                // lie strictly below `to` in index order.
                let attacks_below = attacks & Bitboard::below(to);
                VICTIM_ORDINAL[pc_idx][from][to] = attacks_below.count() as u8;
            }
        }
    }
}

unsafe fn init_piece_totals() {
    let mut global_offset: i32 = 0;
    for (side, pc) in Piece::coloured_pieces() {
        let mut piece_offset: u32 = 0;
        for sq in Square::iter() {
            if is_valid_piece_placement(pc, sq) {
                piece_offset += attacks::attacks(sq, pc, side, Bitboard::NONE).count();
            }
        }
        PIECE_TOTALS[pc.coloured_index(side)] = (piece_offset as i32, global_offset);
        global_offset += PIECE_TARGET_COUNT[pc] * piece_offset as i32;
    }
}

unsafe fn init_from_offset() {
    for (side, pc) in Piece::coloured_pieces() {
        let pc_idx = pc.coloured_index(side);
        let mut piece_offset: u32 = 0;
        for sq in Square::iter() {
            FROM_OFFSET[pc_idx][sq] = piece_offset;
            if is_valid_piece_placement(pc, sq) {
                piece_offset += attacks::attacks(sq, pc, side, Bitboard::NONE).count();
            }
        }
    }
}

unsafe fn init_pair_base() {
    for (attacker_side, attacker_pc) in Piece::coloured_pieces() {
        let attacker_idx = attacker_pc.coloured_index(attacker_side);
        let (attack_total, block_base) = PIECE_TOTALS[attacker_idx];

        for (victim_side, victim_pc) in Piece::coloured_pieces() {
            let victim_idx = victim_pc.coloured_index(victim_side);

            let map = PIECE_TARGET_MAP[attacker_pc][victim_pc];
            let fully_excluded = map == -1;

            // Handle semi-exclusions: threats where the attacker and victim are the same type are
            // excluded from the forward direction, due to being redundant. The exception are pawns
            // of the same colour, since pawn attacks are directional, and so white pawn A attacking
            // white pawn B does not imply the reverse (in fact the reverse is impossible).
            let opposed = attacker_side != victim_side;
            let semi_excluded = attacker_pc == victim_pc && (opposed || attacker_pc != Piece::Pawn);

            let colour_base = victim_side as i32 * (PIECE_TARGET_COUNT[attacker_pc] / 2);
            let feature = block_base + (colour_base + map) * attack_total;

            let backward_idx = if fully_excluded {
                u32::MAX
            } else {
                feature as u32
            };
            PAIR_BASE[attacker_idx][victim_idx][0] = backward_idx;

            let forward_idx = if fully_excluded || semi_excluded {
                u32::MAX
            } else {
                feature as u32
            };
            PAIR_BASE[attacker_idx][victim_idx][1] = forward_idx;
        }
    }
}

/// Pawns on the first or eighth are excluded because they cannot exist
fn is_valid_piece_placement(pc: Piece, sq: Square) -> bool {
    let rank = sq.rank();
    !(pc == Piece::Pawn && (rank == Rank::One || rank == Rank::Eight))
}

/// An encoding of one threat input change: attacker on `from` threatens victim on `to`, either
/// created (add = true) or destroyed (add = false).
///
/// Deltas are perspective-neutral, since the actual index into the threat inputs accumulator will
/// depend on the perspective.
///
/// Bit layout:
/// 0–7:    attacker (0..12)
/// 8–15:   from square (0..64)
/// 16–23:  victim (0..12)
/// 24–30:  to square (0..64)
/// 31:     add (1 = threat created, 0 = threat destroyed)
///
/// Implementation inspired by Reckless.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct ThreatFeature(u32);

impl ThreatFeature {
    const FROM_SHIFT: u32 = 8;
    const VICTIM_SHIFT: u32 = 16;
    const TO_SHIFT: u32 = 24;
    const ADD_SHIFT: u32 = 31;
    const TO_MASK: u32 = 0x7F;

    #[inline(always)]
    pub fn new(
        attacker: Piece,
        attacker_side: Side,
        from: Square,
        victim: Piece,
        victim_side: Side,
        to: Square,
        add: bool,
    ) -> Self {
        let attacker_idx = attacker.coloured_index(attacker_side) as u32;
        let victim_idx = victim.coloured_index(victim_side) as u32;
        Self(
            attacker_idx
                | (from.0 as u32) << Self::FROM_SHIFT
                | victim_idx << Self::VICTIM_SHIFT
                | (to.0 as u32) << Self::TO_SHIFT
                | (add as u32) << Self::ADD_SHIFT,
        )
    }

    /// Compute the index of the given threat. We return a tuple containing the index itself, and a bool
    /// indicating whether the threat is included in the threat inputs. We could return an `Option<i32>`,
    /// but this way allows for branchless execution.
    pub fn index(&self, perspective: Side, king_sq: Square) -> (bool, i32) {
        let (mut from, mut to) = (self.from(), self.to());
        let (atk_pc, mut atk_side) = self.attacker();
        let (vic_pc, mut vic_side) = self.victim();
        
        // Threat indices are reversed for black.
        if perspective == Black {
            atk_side = !atk_side;
            vic_side = !vic_side;
            from = from.flip_rank();
            to = to.flip_rank();
        }
        // Threat indices are horizontally mirrored if the king is on the right side of the board.
        if king_sq.file() >= File::E {
            from = from.flip_file();
            to = to.flip_file();
        }

        // Whether this is a forward or backward threat, relevant for semi-exclusions.
        let direction = (from.0 < to.0) as usize;

        // Get the indices of the attacking and defending piece (white 0-5, black 6-11).
        let attacker_idx = atk_pc.coloured_index(atk_side);
        let victim_idx = vic_pc.coloured_index(vic_side);

        // Get the block for the (attacker, victim, direction) combination
        let base = unsafe { PAIR_BASE[attacker_idx][victim_idx][direction] };

        // Get the sub-block for the (attacker, from) combination.
        let offset = unsafe { FROM_OFFSET[attacker_idx][from] as i32 };

        // Get the number of squares the attacker threatens from `from` that are below `to`.
        let slot = unsafe { VICTIM_ORDINAL[attacker_idx][from][to] as i32 };

        let index = (base as i32).wrapping_add(offset).wrapping_add(slot);

        (base != u32::MAX, index)
    }

    #[inline(always)]
    pub const fn attacker(self) -> (Piece, Side) {
        Self::decode_coloured(self.0 as u8)
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        Square((self.0 >> Self::FROM_SHIFT) as u8)
    }

    #[inline(always)]
    pub const fn victim(self) -> (Piece, Side) {
        Self::decode_coloured((self.0 >> Self::VICTIM_SHIFT) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        Square(((self.0 >> Self::TO_SHIFT) as u8) & Self::TO_MASK as u8)
    }

    #[inline(always)]
    pub const fn add(self) -> bool {
        (self.0 >> Self::ADD_SHIFT) != 0
    }

    #[inline(always)]
    const fn decode_coloured(idx: u8) -> (Piece, Side) {
        let pc = idx % 6;
        let side = idx / 6;
        unsafe { (std::mem::transmute(pc), std::mem::transmute(side)) }
    }
}