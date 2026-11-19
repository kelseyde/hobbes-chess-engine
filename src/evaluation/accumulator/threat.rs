use crate::board::bitboard::Bitboard;
use crate::board::piece::Piece;
use crate::board::piece::Piece::{Bishop, Knight, Pawn, Queen, Rook};
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::square::Square;
use crate::board::{attacks, ray, Board};
use crate::evaluation::accumulator::should_mirror;
use crate::evaluation::feature::pp;
use crate::evaluation::feature::pp::{PawnPairFeature, PP_BANDS};
use crate::evaluation::feature::threat::ThreatFeature;
use crate::evaluation::{simd, NETWORK, NNUE};
use arrayvec::ArrayVec;
use hobbes_nnue_arch::L1_SIZE;

const MAX_THREAT_FEATURES: usize = 80;
const MAX_PAWN_PAIR_FEATURES: usize = 64;
const MAX_FEATURES: usize = MAX_THREAT_FEATURES + MAX_PAWN_PAIR_FEATURES;
const MAX_ACTIVE_INDICES: usize = 4096;

#[cfg(target_feature = "avx512f")]
const REGISTERS: usize = L1_SIZE / simd::i16::LANES;
#[cfg(not(target_feature = "avx512f"))]
const REGISTERS: usize = 8;

#[repr(C, align(64))]
pub struct ThreatAccumulator {
    features: [[i16; L1_SIZE]; 2],
    pub threat_fts: ArrayVec<ThreatFeature, MAX_THREAT_FEATURES>,
    pub pawn_pair_fts: ArrayVec<PawnPairFeature, MAX_PAWN_PAIR_FEATURES>,
    pub needs_refresh: [bool; 2],
    pub computed: [bool; 2],
}

impl Default for ThreatAccumulator {
    fn default() -> Self {
        Self {
            features: [[0; L1_SIZE]; 2],
            threat_fts: ArrayVec::new(),
            pawn_pair_fts: ArrayVec::new(),
            needs_refresh: [false; 2],
            computed: [false; 2],
        }
    }
}

impl ThreatAccumulator {
    #[inline(always)]
    pub fn features(&self, perspective: Side) -> &[i16; L1_SIZE] {
        &self.features[perspective]
    }

    #[inline(always)]
    pub fn features_relative(&self, perspective: Side) -> (&[i16; L1_SIZE], &[i16; L1_SIZE]) {
        (self.features(perspective), self.features(!perspective))
    }

    #[inline(always)]
    pub fn features_mut(&mut self, perspective: Side) -> &mut [i16; L1_SIZE] {
        &mut self.features[perspective]
    }

    pub fn refresh(&mut self, board: &Board, pov: Side) {
        let mut adds = ArrayVec::<u32, MAX_ACTIVE_INDICES>::new();
        Self::collect_threat_indices(board, pov, &mut adds);
        Self::collect_pp_indices(board, pov, &mut adds);
        unsafe { accumulate(None, &mut self.features[pov], &adds, &[]) };
        self.computed[pov] = true;
    }

    pub fn apply(&mut self, parent: &ThreatAccumulator, king_sq: Square, pov: Side) {
        let mut adds = ArrayVec::<u32, MAX_FEATURES>::new();
        let mut subs = ArrayVec::<u32, MAX_FEATURES>::new();

        for delta in &self.threat_fts {
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

        for pair in &self.pawn_pair_fts {
            let idx = pair.index(pov, king_sq);
            if pair.add() {
                adds.push(idx)
            } else {
                subs.push(idx)
            }
        }

        unsafe {
            accumulate(
                Some(&parent.features[pov]),
                &mut self.features[pov],
                &adds,
                &subs,
            )
        };
    }

    /// Update accumulator threat deltas when a piece is added on a square
    pub fn push_piece_create(&mut self, board: &Board, pc: Piece, side: Side, sq: Square) {
        self.push_piece_single(board, board.occ(), pc, side, sq, true);
        if pc == Pawn {
            self.push_pawn_pairs(board, side, sq, true);
        }
    }

    /// Update accumulator threat deltas when a piece is removed from a square.
    pub fn push_piece_destroy(&mut self, board: &Board, pc: Piece, side: Side, sq: Square) {
        self.push_piece_single(board, board.occ(), pc, side, sq, false);
        if pc == Pawn {
            self.push_pawn_pairs(board, side, sq, false);
        }
    }

    /// Update accumulator threat deltas when a piece is moved from one square to another.
    pub fn push_piece_teleport(
        &mut self,
        board: &Board,
        pc: Piece,
        side: Side,
        from: Square,
        to: Square,
    ) {
        let occ = board.occ() ^ Bitboard::of_sq(to);
        self.push_piece_single(board, occ, pc, side, from, false);
        self.push_piece_single(board, occ, pc, side, to, true);
        if pc == Pawn {
            // board is already updated with the teleporter on the 'to' square,
            // so we need to mask it out here
            let others = board.all_pawns() ^ Bitboard::of_sq(to);
            self.push_pawn_pairs_with(board, side, from, others, false);
            self.push_pawn_pairs_with(board, side, to, others, true);
        }
    }

    /// Update accumulator threat deltas when a piece type is changed on a single square.
    pub fn push_piece_transform(
        &mut self,
        board: &Board,
        old_pc: Piece,
        old_side: Side,
        new_pc: Piece,
        new_side: Side,
        sq: Square,
    ) {
        let deltas = &mut self.threat_fts;
        let occ = board.occ();
        let attacked = attacks::attacks(sq, old_pc, old_side, occ) & occ;
        for to in attacked {
            let victim_pc = board.piece_at(to).unwrap();
            let victim_side = board.side_at(to).unwrap();
            deltas.push(ThreatFeature::new(
                old_pc,
                old_side,
                sq,
                victim_pc,
                victim_side,
                to,
                false,
            ));
        }
        let attacked = attacks::attacks(sq, new_pc, new_side, occ) & occ;
        for to in attacked {
            let victim_pc = board.piece_at(to).unwrap();
            let victim_side = board.side_at(to).unwrap();
            deltas.push(ThreatFeature::new(
                new_pc,
                new_side,
                sq,
                victim_pc,
                victim_side,
                to,
                true,
            ));
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
            let attacker_pc = board.piece_at(from).unwrap();
            let attacker_side = board.side_at(from).unwrap();
            deltas.push(ThreatFeature::new(
                attacker_pc,
                attacker_side,
                from,
                old_pc,
                old_side,
                sq,
                false,
            ));
            deltas.push(ThreatFeature::new(
                attacker_pc,
                attacker_side,
                from,
                new_pc,
                new_side,
                sq,
                true,
            ));
        }

        if old_pc == Pawn {
            self.push_pawn_pairs(board, old_side, sq, false);
        }
        if new_pc == Pawn {
            self.push_pawn_pairs(board, new_side, sq, true);
        }
    }

    fn push_piece_single(
        &mut self,
        board: &Board,
        occ: Bitboard,
        pc: Piece,
        side: Side,
        sq: Square,
        add: bool,
    ) {
        let deltas = &mut self.threat_fts;
        let attacked = attacks::attacks(sq, pc, side, occ) & occ;
        for (victim_side, targets) in [
            (White, attacked & board.side(White)),
            (Black, attacked & board.side(Black)),
        ] {
            for to in targets {
                let victim_pc = board.piece_at(to).unwrap();
                deltas.push(ThreatFeature::new(
                    pc,
                    side,
                    sq,
                    victim_pc,
                    victim_side,
                    to,
                    add,
                ));
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
                let victim_pc = board.piece_at(to).unwrap();
                let victim_side = board.side_at(to).unwrap();
                deltas.push(ThreatFeature::new(
                    slider_pc,
                    slider_side,
                    from,
                    victim_pc,
                    victim_side,
                    to,
                    !add,
                ));
            }
            deltas.push(ThreatFeature::new(
                slider_pc,
                slider_side,
                from,
                pc,
                side,
                sq,
                add,
            ));
        }

        let white_pawns = board.pawns(White) & attacks::pawn(sq, Black);
        let black_pawns = board.pawns(Black) & attacks::pawn(sq, White);
        let knights = board.pieces(Knight) & attacks::knight(sq);
        let kings = board.pieces(Piece::King) & attacks::king(sq);
        let leapers = black_pawns | white_pawns | knights | kings;

        for from in leapers & occ {
            let attacker_pc = board.piece_at(from).unwrap();
            let attacker_side = board.side_at(from).unwrap();
            deltas.push(ThreatFeature::new(
                attacker_pc,
                attacker_side,
                from,
                pc,
                side,
                sq,
                add,
            ));
        }
    }

    fn push_pawn_pairs(&mut self, board: &Board, side: Side, sq: Square, add: bool) {
        let others = board.all_pawns() & !Bitboard::of_sq(sq);
        self.push_pawn_pairs_with(board, side, sq, others, add);
    }

    fn push_pawn_pairs_with(
        &mut self,
        board: &Board,
        side: Side,
        sq: Square,
        others: Bitboard,
        add: bool,
    ) {
        let partners = others & PP_BANDS[sq];
        for (side_b, pawns) in [(White, board.pawns(White)), (Black, board.pawns(Black))] {
            for b in partners & pawns {
                self.pawn_pair_fts
                    .push(PawnPairFeature::new(sq, side, b, side_b, add));
            }
        }
    }

    /// Collect the indices of the active threat features on the board.
    fn collect_threat_indices(board: &Board, pov: Side, output: &mut ArrayVec<u32, 4096>) {
        let occ = board.occ();
        let king_sq = board.king_sq(pov);
        for from in occ {
            let (attacker_pc, attacker_side) =
                (board.piece_at(from).unwrap(), board.side_at(from).unwrap());
            let attacks = attacks::attacks(from, attacker_pc, attacker_side, occ) & occ;
            for to in attacks {
                let (victim_pc, victim_side) =
                    (board.piece_at(to).unwrap(), board.side_at(to).unwrap());
                let delta = ThreatFeature::new(
                    attacker_pc,
                    attacker_side,
                    from,
                    victim_pc,
                    victim_side,
                    to,
                    true,
                );
                let (valid, idx) = delta.index(pov, king_sq);
                if valid {
                    output.push(idx as u32);
                }
            }
        }
    }

    /// Collect the indices of the active pawn pair features on the board.
    fn collect_pp_indices(
        board: &Board,
        pov: Side,
        output: &mut ArrayVec<u32, MAX_ACTIVE_INDICES>,
    ) {
        let mirror = should_mirror(board.king_sq(pov));
        let (w_pawns, b_pawns) = (board.pawns(White), board.pawns(Black));

        for (side_a, side_b, outer, inner) in [
            (White, White, w_pawns, w_pawns),
            (Black, Black, b_pawns, b_pawns),
            (White, Black, w_pawns, b_pawns),
        ] {
            let same = side_a == side_b;
            for a in outer {
                let mut partners = inner & PP_BANDS[a];
                if same {
                    partners &= Bitboard::below(a);
                }
                if partners.is_empty() {
                    continue;
                }
                let id_a = pp::pawn_id(a, side_a, pov, mirror);
                for b in partners {
                    let id_b = pp::pawn_id(b, side_b, pov, mirror);
                    let idx = pp::pp_index(id_a, id_b);
                    output.push(idx);
                }
            }
        }
    }
}

/// Apply any pending updates lazily to the threat accumulator before evaluation. We scan forward
/// from the last computed accumulator, applying the updates one by one. If at any point we detect
/// a change which requires a refresh, we cancel all updates and immediately refresh.
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
    input: Option<&[i16; L1_SIZE]>,
    output: &mut [i16; L1_SIZE],
    adds: &[u32],
    subs: &[u32],
) {
    const STEP: usize = REGISTERS * simd::i16::LANES;

    let weights = NETWORK.l0_threat_pp_weights.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for offset in (0..L1_SIZE).step_by(STEP) {
        let mut regs = [simd::i16::zero(); REGISTERS];
        if let Some(p) = input {
            let in_ptr = p.as_ptr();
            for (i, reg) in regs.iter_mut().enumerate() {
                *reg = simd::i16::load(in_ptr.add(offset + i * simd::i16::LANES));
            }
        }

        let (mut added, mut subtracted) = (0, 0);

        while added < adds.len() && subtracted < subs.len() {
            let add_row = weights.add(adds[added] as usize * L1_SIZE + offset);
            let sub_row = weights.add(subs[subtracted] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                let lane = i * simd::i16::LANES;
                let add_weight = simd::i16::load_i8(add_row.add(lane));
                let sub_weight = simd::i16::load_i8(sub_row.add(lane));
                *reg = simd::i16::add(*reg, simd::i16::sub(add_weight, sub_weight));
            }
            added += 1;
            subtracted += 1;
        }

        while added + 1 < adds.len() {
            let row1 = weights.add(adds[added] as usize * L1_SIZE + offset);
            let row2 = weights.add(adds[added + 1] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                let lane = i * simd::i16::LANES;
                let add_weight1 = simd::i16::load_i8(row1.add(lane));
                let add_weight2 = simd::i16::load_i8(row2.add(lane));
                *reg = simd::i16::add(*reg, simd::i16::add(add_weight1, add_weight2));
            }
            added += 2;
        }

        while added < adds.len() {
            let row = weights.add(adds[added] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                let add_weight = simd::i16::load_i8(row.add(i * simd::i16::LANES));
                *reg = simd::i16::add(*reg, add_weight);
            }
            added += 1;
        }

        while subtracted < subs.len() {
            let row = weights.add(subs[subtracted] as usize * L1_SIZE + offset);
            for (i, reg) in regs.iter_mut().enumerate() {
                let sub_weight = simd::i16::load_i8(row.add(i * simd::i16::LANES));
                *reg = simd::i16::sub(*reg, sub_weight);
            }
            subtracted += 1;
        }

        for (i, reg) in regs.iter().enumerate() {
            simd::i16::store(out_ptr.add(offset + i * simd::i16::LANES), *reg);
        }
    }
}
