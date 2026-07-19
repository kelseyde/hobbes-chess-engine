use hobbes_nnue_arch::L1_SIZE;
use arrayvec::ArrayVec;
use crate::board::bitboard::Bitboard;
use crate::board::{attacks, ray, Board};
use crate::board::piece::Piece;
use crate::board::piece::Piece::{Bishop, Knight, Queen, Rook};
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::evaluation::feature::threat::threat_index;
use crate::evaluation::NETWORK;

#[repr(C, align(64))]
pub struct ThreatAccumulator {
    features: [[i16; L1_SIZE]; 2],
    pub deltas: ArrayVec<ThreatDelta, 80>,
}

impl Default for ThreatAccumulator {
    fn default() -> Self {
        Self {
            features: [[0; L1_SIZE]; 2],
            deltas: ArrayVec::new(),
        }
    }
}

impl ThreatAccumulator {

    pub fn features(&self, perspective: Side) -> &[i16; L1_SIZE] {
        &self.features[perspective]
    }

    pub fn features_mut(&mut self, perspective: Side) -> &mut [i16; L1_SIZE] {
        &mut self.features[perspective]
    }

    pub fn refresh_threats(&mut self, board: &Board, perspective: Side) {
        let mut indices = ArrayVec::new();
        Self::collect_threat_indices(board, perspective, &mut indices);
        let out = &mut self.features[perspective];
        out.fill(0);
        for &idx in &indices {
            let base = idx as usize * L1_SIZE;
            let row = &NETWORK.l0_threat_weights[base..base + L1_SIZE];
            for i in 0..L1_SIZE {
                out[i] += row[i] as i16;
            }
        }
    }

    /// Update accumulator threat deltas when a piece is added on a square
    pub fn push_piece_create(&mut self, board: &Board, pc: Piece, side: Side, sq: Square) {
        self.push_piece_single(board, board.occ(), pc, side, sq, true);
    }

    /// Update accumulator threat deltas when a piece is removed from a square.
    pub fn push_piece_destroy(&mut self, board: &Board, pc: Piece, side: Side, sq: Square) {
        self.push_piece_single(board, board.occ(), pc, side, sq, false);
    }

    /// Update accumulator threat deltas when a piece is moved from one square to another.
    pub fn push_piece_teleport(&mut self, board: &Board, pc: Piece, side: Side, from: Square, to: Square) {
        let occ = board.occ() ^ Bitboard::of_sq(to);
        self.push_piece_single(board, occ, pc, side, from, false);
        self.push_piece_single(board, occ, pc, side, to, true);
    }

    /// Update accumulator threat deltas when a piece type is changed on a single square.
    pub fn push_piece_transform(
        &mut self,
        board: &Board,
        old_pc: Piece,
        old_side: Side,
        new_pc: Piece,
        new_side: Side,
        sq: Square
    ) {

        let deltas = &mut self.deltas;
        let occ = board.occ();
        let attacked = attacks::attacks(sq, old_pc, old_side, occ) & occ;
        for to in attacked {
            let vic_pc = board.piece_at(to).unwrap();
            let vic_side = board.side_at(to).unwrap();
            deltas.push(ThreatDelta::new(old_pc, old_side, sq, vic_pc, vic_side, to, false));
        }
        let attacked = attacks::attacks(sq, new_pc, new_side, occ) & occ;
        for to in attacked {
            let vic_pc = board.piece_at(to).unwrap();
            let vic_side = board.side_at(to).unwrap();
            deltas.push(ThreatDelta::new(new_pc, new_side, sq, vic_pc, vic_side, to, true));
        }

        let rook_attacks = attacks::rook(sq, occ);
        let bishop_attacks = attacks::bishop(sq, occ);

        let diags = (board.pieces(Bishop) | board.pieces(Queen)) & bishop_attacks;
        let orthos = (board.pieces(Rook) | board.pieces(Queen)) & rook_attacks;

        let white_pawns = board.pawns(White) & attacks::pawn(sq, Black);
        let black_pawns = board.pawns(Black) & attacks::pawn(sq, White);
        let knights = board.pieces(Knight) & attacks::knight(sq);
        let kings = board.pieces(Piece::King) & attacks::king(sq);

        for from in (black_pawns | white_pawns | knights | diags | orthos | kings) & occ {
            let atk_pc = board.piece_at(from).unwrap();
            let atk_side = board.side_at(from).unwrap();
            deltas.push(ThreatDelta::new(atk_pc, atk_side, from, old_pc, old_side, sq, false));
            deltas.push(ThreatDelta::new(atk_pc, atk_side, from, new_pc, new_side, sq, true));
        }

    }

    fn push_piece_single(&mut self, board: &Board, occ: Bitboard, pc: Piece, side: Side, sq: Square, add: bool) {

        let deltas = &mut self.deltas;
        let attacked = attacks::attacks(sq, pc, side, occ) & occ;
        for to in attacked {
            let vic_pc = board.piece_at(to).unwrap();
            let vic_side = board.side_at(to).unwrap();
            deltas.push(ThreatDelta::new(pc, side, sq, vic_pc, vic_side, to, add));
        }

        let bishop_attacks = attacks::bishop(sq, occ);
        let rook_attacks = attacks::rook(sq, occ);
        let queen_attacks = bishop_attacks | rook_attacks;

        let diags = (board.pieces(Bishop) | board.pieces(Queen)) & bishop_attacks;
        let orthos = (board.pieces(Rook) | board.pieces(Queen)) & rook_attacks;
        let sliders = diags | orthos;

        for from in sliders & occ {
            let slider_pc = board.piece_at(from).unwrap();
            let slider_side = board.side_at(from).unwrap();
            let threatened = ray::between(from, sq) & occ & queen_attacks; // TODO raypass
            if let Some(to) = threatened.into_iter().next() {
                let vic_pc = board.piece_at(to).unwrap();
                let vic_side = board.side_at(to).unwrap();
                deltas.push(ThreatDelta::new(slider_pc, slider_side, from, vic_pc, vic_side, to, !add));
            }
        }

        let white_pawns = board.pawns(White) & attacks::pawn(sq, Black);
        let black_pawns = board.pawns(Black) & attacks::pawn(sq, White);
        let knights = board.pieces(Knight) & attacks::knight(sq);
        let kings = board.pieces(Piece::King) & attacks::king(sq);

        for from in (black_pawns | white_pawns | knights | kings) & occ {
            let atk_pc = board.piece_at(from).unwrap();
            let atk_side = board.side_at(from).unwrap();
            deltas.push(ThreatDelta::new(atk_pc, atk_side, from, pc, side, sq, add));
        }

    }

    fn collect_threat_indices(board: &Board, pov: Side, out: &mut ArrayVec<u32, 4096>) {
        let occ = board.occ();
        let king_sq = board.king_sq(pov);
        for from in occ {
            let (atk, atk_side) = (board.piece_at(from).unwrap(), board.side_at(from).unwrap());
            let attacks = attacks::attacks(from, atk, atk_side, occ) & occ;
            for to in attacks {
                let (vic, vic_side) = (board.piece_at(to).unwrap(), board.side_at(to).unwrap());
                let (valid, idx) = threat_index(pov, king_sq, atk, atk_side, vic, vic_side, from, to);
                if valid {
                    out.push(idx as u32);
                }
            }
        }
    }

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
pub struct ThreatDelta(u32);

impl ThreatDelta {
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