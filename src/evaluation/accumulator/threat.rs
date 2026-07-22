use crate::board::bitboard::Bitboard;
use crate::board::piece::Piece;
use crate::board::piece::Piece::{Bishop, Knight, Queen, Rook};
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::board::{attacks, ray, Board};
use crate::evaluation::feature::threat::ThreatFeature;
use crate::evaluation::{simd, NETWORK, NNUE};
use arrayvec::ArrayVec;
use hobbes_nnue_arch::L1_SIZE;

const MAX_DELTA_INDICES: usize = 80;
const MAX_ACTIVE_INDICES: usize = 4096;

#[cfg(target_feature = "avx512f")]
const REGISTERS: usize = L1_SIZE / simd::I16_LANES;
#[cfg(not(target_feature = "avx512f"))]
const REGISTERS: usize = 8;

const STEP: usize = REGISTERS * simd::I16_LANES;
const _: () = assert!(L1_SIZE.is_multiple_of(STEP), "step must divide by the accumulator evenly");

#[repr(C, align(64))]
pub struct ThreatAccumulator {
    features: [[i16; L1_SIZE]; 2],
    pub deltas: ArrayVec<ThreatFeature, MAX_DELTA_INDICES>,
    pub needs_refresh: [bool; 2],
    pub computed: [bool; 2],
}

impl Default for ThreatAccumulator {
    fn default() -> Self {
        Self {
            features: [[0; L1_SIZE]; 2],
            deltas: ArrayVec::new(),
            needs_refresh: [false; 2],
            computed: [false; 2],
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

    pub fn refresh(&mut self, board: &Board, pov: Side) {
        let mut adds = ArrayVec::<u32, MAX_ACTIVE_INDICES>::new();
        Self::collect_threat_indices(board, pov, &mut adds);
        unsafe { accumulate(&mut self.features[pov], None, &adds, &[]) };
        self.computed[pov] = true;
    }

    pub fn apply(&mut self, parent: &ThreatAccumulator, king_sq: Square, pov: Side) {
        let mut adds = ArrayVec::<u32, MAX_DELTA_INDICES>::new();
        let mut subs = ArrayVec::<u32, MAX_DELTA_INDICES>::new();

        for delta in &self.deltas {
            let (valid, idx) = delta.index(pov, king_sq);
            if !valid {
                continue;
            }
            if delta.add() {
                adds.push(idx as u32);
            } else {
                subs.push(idx as u32);
            }
        }

        unsafe { accumulate(&mut self.features[pov], Some(&parent.features[pov]), &adds, &subs) };
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
            deltas.push(ThreatFeature::new(old_pc, old_side, sq, vic_pc, vic_side, to, false));
        }
        let attacked = attacks::attacks(sq, new_pc, new_side, occ) & occ;
        for to in attacked {
            let vic_pc = board.piece_at(to).unwrap();
            let vic_side = board.side_at(to).unwrap();
            deltas.push(ThreatFeature::new(new_pc, new_side, sq, vic_pc, vic_side, to, true));
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
            deltas.push(ThreatFeature::new(atk_pc, atk_side, from, old_pc, old_side, sq, false));
            deltas.push(ThreatFeature::new(atk_pc, atk_side, from, new_pc, new_side, sq, true));
        }

    }

    fn push_piece_single(&mut self, board: &Board, occ: Bitboard, pc: Piece, side: Side, sq: Square, add: bool) {

        let deltas = &mut self.deltas;
        let attacked = attacks::attacks(sq, pc, side, occ) & occ;
        for (vic_side, targets) in [
            (White, attacked & board.side(White)),
            (Black, attacked & board.side(Black)),
        ] {
            for to in targets {
                let vic_pc = board.piece_at(to).unwrap();
                deltas.push(ThreatFeature::new(pc, side, sq, vic_pc, vic_side, to, add));
            }
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
            let threatened = ray::beyond(from, sq) & occ & queen_attacks;
            if let Some(to) = threatened.into_iter().next() {
                let vic_pc = board.piece_at(to).unwrap();
                let vic_side = board.side_at(to).unwrap();
                deltas.push(ThreatFeature::new(slider_pc, slider_side, from, vic_pc, vic_side, to, !add));
            }
            deltas.push(ThreatFeature::new(slider_pc, slider_side, from, pc, side, sq, add));
        }

        let white_pawns = board.pawns(White) & attacks::pawn(sq, Black);
        let black_pawns = board.pawns(Black) & attacks::pawn(sq, White);
        let knights = board.pieces(Knight) & attacks::knight(sq);
        let kings = board.pieces(Piece::King) & attacks::king(sq);
        let leapers = black_pawns | white_pawns | knights | kings;

        for from in leapers & occ {
            let atk_pc = board.piece_at(from).unwrap();
            let atk_side = board.side_at(from).unwrap();
            deltas.push(ThreatFeature::new(atk_pc, atk_side, from, pc, side, sq, add));
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
                let delta = ThreatFeature::new(atk, atk_side, from, vic, vic_side, to, true);
                let (valid, idx) = delta.index(pov, king_sq);
                if valid {
                    out.push(idx as u32);
                }
            }
        }
    }

}

pub fn apply_lazy_updates(nnue: &mut NNUE, board: &Board) {
    for pov in [White, Black] {
        if nnue.stack[nnue.current].threat.computed[pov] {
            continue;
        }

        if nnue.stack[nnue.current].threat.needs_refresh[pov] {
            let threat = &mut nnue.stack[nnue.current].threat;
            threat.refresh(board, pov);
            threat.needs_refresh[pov] = false;
            threat.computed[pov] = true;
            continue;
        }

        let mut curr = nnue.current;
        while !nnue.stack[curr].threat.computed[pov] {
            curr -= 1;
        }

        let king_sq = board.king_sq(pov);
        while curr < nnue.current {
            let (parents, currents) = nnue.stack.split_at_mut(curr + 1);
            let parent = &parents[curr].threat;
            let child = &mut currents[0].threat;
            child.apply(parent, king_sq, pov);
            child.computed[pov] = true;
            curr += 1;
        }
    }
}

#[inline(always)]
unsafe fn accumulate(
    out: &mut [i16; L1_SIZE],
    parent: Option<&[i16; L1_SIZE]>,
    adds: &[u32],
    subs: &[u32],
) {
    let weights = NETWORK.l0_threat_weights.as_ptr();
    let out_ptr = out.as_mut_ptr();

    for offset in (0..L1_SIZE).step_by(STEP) {

        let mut regs = [simd::zero_i16(); REGISTERS];
        if let Some(p) = parent {
            let in_ptr = p.as_ptr();
            for (i, reg) in regs.iter_mut().enumerate() {
                *reg = simd::load_i16(in_ptr.add(offset + i * simd::I16_LANES));
            }
        }

        let (mut added, mut subtracted) = (0, 0);

        while added < adds.len() && subtracted < subs.len() {
            let add_row = weights.add(adds[added] as usize * L1_SIZE + offset);
            let sub_row = weights.add(subs[subtracted] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                let lane = i * simd::I16_LANES;
                let add_w = simd::load_i8_as_i16(add_row.add(lane));
                let sub_w = simd::load_i8_as_i16(sub_row.add(lane));
                *reg = simd::add_i16(*reg, simd::sub_i16(add_w, sub_w));
            }
            added += 1;
            subtracted += 1;
        }

        while added + 1 < adds.len() {
            let row1 = weights.add(adds[added] as usize * L1_SIZE + offset);
            let row2 = weights.add(adds[added + 1] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                let lane = i * simd::I16_LANES;
                let w1 = simd::load_i8_as_i16(row1.add(lane));
                let w2 = simd::load_i8_as_i16(row2.add(lane));
                *reg = simd::add_i16(*reg, simd::add_i16(w1, w2));
            }
            added += 2;
        }

        while added < adds.len() {
            let row = weights.add(adds[added] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                *reg = simd::add_i16(*reg, simd::load_i8_as_i16(row.add(i * simd::I16_LANES)));
            }
            added += 1;
        }

        while subtracted < subs.len() {
            let row = weights.add(subs[subtracted] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                *reg = simd::sub_i16(*reg, simd::load_i8_as_i16(row.add(i * simd::I16_LANES)));
            }
            subtracted += 1;
        }

        for (i, reg) in regs.iter().enumerate() {
            simd::store_i16(out_ptr.add(offset + i * simd::I16_LANES), *reg);
        }
    }
}

