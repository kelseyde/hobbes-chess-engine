use crate::board::piece::Piece::{Bishop, King, Knight, Pawn, Queen, Rook};
use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::board::Board;
use crate::evaluation::accumulator::psq;
use crate::evaluation::cache::InputBucketCache;
use crate::evaluation::feature::psq::PieceSquareFeature;
use crate::evaluation::{king_bucket, should_mirror, simd, NETWORK, NNUE};
use arrayvec::ArrayVec;
use hobbes_nnue_arch::{PieceSquareWeights, L1_SIZE};

/// The `PieceSquareAccumulator` holds the pre-activations of the first layer of the neural network.
/// The input layer just encodes the positions of pieces on the board, from the perspective of both
/// sides. The accumulator is updated incrementally when a move is made or unmade during search, so
/// that we can efficiently compute the evaluation without needing to recompute the board each time.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct PieceSquareAccumulator {
    features: [[i16; L1_SIZE]; 2],
    pub update: PieceSquareAccumulatorUpdate,
    pub computed: [bool; 2],
    pub needs_refresh: [bool; 2],
    pub mirrored: [bool; 2],
}

/// A single update to the `PieceSquareAccumulator` caused by a move. A standard move will add one
/// feature (the piece moving to its new square) and remove one feature (the piece leaving its old
/// square). Captures require one extra feature (removing the captured piece), and castling requires
/// two extra features (moving the rook).
#[derive(Clone, Copy, Default)]
pub enum PieceSquareAccumulatorUpdate {
    #[default]
    None,
    AddSub(PieceSquareFeature, PieceSquareFeature),
    AddSubSub(PieceSquareFeature, PieceSquareFeature, PieceSquareFeature),
    AddAddSubSub(PieceSquareFeature, PieceSquareFeature, PieceSquareFeature, PieceSquareFeature),
}

impl Default for PieceSquareAccumulator {
    fn default() -> Self {
        PieceSquareAccumulator {
            features: [NETWORK.l0_biases, NETWORK.l0_biases],
            update: PieceSquareAccumulatorUpdate::default(),
            computed: [false, false],
            needs_refresh: [false, false],
            mirrored: [false, false],
        }
    }
}

impl PieceSquareAccumulator {
    /// Get a reference to the features for the given perspective.
    #[inline(always)]
    pub fn features(&self, perspective: Side) -> &[i16; L1_SIZE] {
        &self.features[perspective]
    }

    /// Get a mutable reference to the features for the given perspective.
    #[inline(always)]
    pub fn features_mut(&mut self, perspective: Side) -> &mut [i16; L1_SIZE] {
        &mut self.features[perspective]
    }

    /// Reset the features for the given perspective to the initial biases.
    #[inline]
    pub fn reset(&mut self, perspective: Side) {
        let feats = &mut self.features[perspective];
        *feats = NETWORK.l0_biases;
    }

    /// Copy the features from another accumulator into this one, for the given perspective.
    #[inline]
    pub fn copy_from(&mut self, side: Side, features: &[i16; L1_SIZE]) {
        self.features[side] = *features;
    }

    /// Get a mutable and immutable reference to the features for the given perspective.
    #[inline]
    pub fn features_inplace(&mut self, side: Side) -> (&[i16; L1_SIZE], &mut [i16; L1_SIZE]) {
        let p = self.features_mut(side).as_mut_ptr();
        unsafe {
            (
                &*(p as *const [i16; L1_SIZE]),
                &mut *(p as *mut [i16; L1_SIZE]),
            )
        }
    }

    /// Refresh the accumulator for the given perspective, mirror state, and bucket. Retrieves
    /// the cached state for this accumulator, bucket, and perspective, and refreshes only the
    /// features of the board that have changed since the last refresh.
    pub fn refresh(
        &mut self,
        board: &Board,
        side: Side,
        cache: &mut InputBucketCache,
    ) {
        let king_sq = board.king_sq(side);
        let mirror = should_mirror(king_sq);
        let bucket = king_bucket(king_sq, side);

        self.mirrored[side] = mirror;
        let cache_entry = cache.get(side, mirror, bucket);
        self.copy_from(side, &cache_entry.features);

        let mut adds = ArrayVec::<_, 32>::new();
        let mut subs = ArrayVec::<_, 32>::new();

        for side in [White, Black] {
            for pc in [Pawn, Knight, Bishop, Rook, Queen, King] {
                let pieces = board.pieces(pc) & board.side(side);
                let cached_pieces = cache_entry.pieces[pc] & cache_entry.colours[side];

                let added = pieces & !cached_pieces;
                for add in added {
                    adds.push(PieceSquareFeature::new(pc, add, side));
                }

                let removed = cached_pieces & !pieces;
                for sub in removed {
                    subs.push(PieceSquareFeature::new(pc, sub, side));
                }
            }
        }

        let weights = &NETWORK.l0_psq_weights[bucket];
        let mirror = self.mirrored[side as usize];

        // Fuse together updates to the accumulator for efficiency using iterators.
        for chunk in adds.as_slice().chunks_exact(4) {
            let (input, output) = self.features_inplace(side);
            add4(
                input,
                output,
                chunk.try_into().unwrap(),
                weights,
                side,
                mirror,
            );
        }
        for &add in adds.as_slice().chunks_exact(4).remainder() {
            let (input, output) = self.features_inplace(side);
            add1(input, output, add, weights, side, mirror);
        }

        for chunk in subs.as_slice().chunks_exact(4) {
            let (input, output) = self.features_inplace(side);
            sub4(
                input,
                output,
                chunk.try_into().unwrap(),
                weights,
                side,
                mirror,
            );
        }
        for &sub in subs.as_slice().chunks_exact(4).remainder() {
            let (input, output) = self.features_inplace(side);
            sub1(input, output, sub, weights, side, mirror);
        }

        self.computed[side] = true;
        self.needs_refresh[side] = false;

        cache_entry.pieces = board.pieces;
        cache_entry.colours = board.colours;
        cache_entry.features = *self.features(side);
    }
}

/// Apply any pending lazy updates to the current accumulator. For each perspective, scan
/// backwards to find the nearest computed accumulator, and move forward applying all updates
/// one by one. If at any point we encounter an accumulator that requires a refresh - due to
/// bucket or mirror change - we bail out and perform a full refresh instead.
pub fn apply_lazy_updates(nnue: &mut NNUE, board: &Board) {
    for side in [White, Black] {
        // If already up-to-date for this perspective, then there is nothing to do.
        if nnue.stack[nnue.current].psq.computed[side] {
            continue;
        }

        let king_sq = board.king_sq(side);
        let mirror = should_mirror(king_sq);
        let bucket = king_bucket(king_sq, side);

        // If the current accumulator requires a full refresh, skip lazy updates and do a refresh.
        if nnue.stack[nnue.current].psq.needs_refresh[side] {
            let acc = &mut nnue.stack[nnue.current].psq;
            acc.refresh(board, side, &mut nnue.cache);
            continue;
        }

        // Scan backwards to find the nearest parent accumulator that is computed for this
        // perspective, or requires a refresh.
        let mut curr = nnue.current - 1;
        while !nnue.stack[curr].psq.computed[side] && !nnue.stack[curr].psq.needs_refresh[side] {
            if curr == 0 {
                break;
            }
            curr -= 1;
        }

        if nnue.stack[curr].psq.needs_refresh[side] {
            // If we found an accumulator that requires a full refresh, do that instead.
            let acc = &mut nnue.stack[nnue.current].psq;
            acc.refresh(board, side, &mut nnue.cache);
        } else {
            // Otherwise, move forward through the stack applying all updates one by one.
            let weights = &NETWORK.l0_psq_weights[bucket];
            while curr < nnue.current {
                let (front, back) = nnue.stack.split_at_mut(curr + 1);
                let prev_acc = front.last().unwrap();
                let next_acc = back.first_mut().unwrap();
                let update = next_acc.psq.update;
                let prev_fts = prev_acc.psq.features(side);
                let next_fts = next_acc.psq.features_mut(side);
                psq::apply_update(prev_fts, next_fts, weights, &update, side, mirror);
                next_acc.psq.computed[side] = true;
                curr += 1;
            }
        }
    }
}

/// Apply the given update to the accumulator, modifying the output features in place.
#[rustfmt::skip]
pub fn apply_update(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    weights: &PieceSquareWeights,
    update: &PieceSquareAccumulatorUpdate,
    perspective: Side,
    mirror: bool,
) {
    match update {
        PieceSquareAccumulatorUpdate::None => {}
        PieceSquareAccumulatorUpdate::AddSub(add, sub) => {
            add1_sub1(input_features, output_features, *add, *sub, weights, perspective, mirror);
        }
        PieceSquareAccumulatorUpdate::AddSubSub(add, sub1, sub2) => {
            add1_sub2(input_features, output_features, *add, *sub1, *sub2, weights, perspective, mirror);
        }
        PieceSquareAccumulatorUpdate::AddAddSubSub(add1, add2, sub1, sub2) => {
            add2_sub2(input_features, output_features, *add1, *add2, *sub1, *sub2, weights, perspective, mirror);
        }
    }
}

#[inline]
pub fn add1(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: PieceSquareFeature,
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa = weight_ptr(weights, add, perspective, mirror);
    unsafe { update_features::<1, 0>(in_ptr, out_ptr, [wa], []) };
}

#[inline]
pub fn sub1(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    sub: PieceSquareFeature,
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let ws = weight_ptr(weights, sub, perspective, mirror);
    unsafe { update_features::<0, 1>(in_ptr, out_ptr, [], [ws]) };
}

#[inline]
pub fn add1_sub1(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: PieceSquareFeature,
    sub: PieceSquareFeature,
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa = weight_ptr(weights, add, perspective, mirror);
    let ws = weight_ptr(weights, sub, perspective, mirror);
    unsafe { update_features::<1, 1>(in_ptr, out_ptr, [wa], [ws]) };
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn add1_sub2(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: PieceSquareFeature,
    sub1: PieceSquareFeature,
    sub2: PieceSquareFeature,
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa = weight_ptr(weights, add, perspective, mirror);
    let ws1 = weight_ptr(weights, sub1, perspective, mirror);
    let ws2 = weight_ptr(weights, sub2, perspective, mirror);
    unsafe { update_features::<1, 2>(in_ptr, out_ptr, [wa], [ws1, ws2]) };
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn add2_sub2(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add1: PieceSquareFeature,
    add2: PieceSquareFeature,
    sub1: PieceSquareFeature,
    sub2: PieceSquareFeature,
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa1 = weight_ptr(weights, add1, perspective, mirror);
    let wa2 = weight_ptr(weights, add2, perspective, mirror);
    let ws1 = weight_ptr(weights, sub1, perspective, mirror);
    let ws2 = weight_ptr(weights, sub2, perspective, mirror);
    unsafe { update_features::<2, 2>(in_ptr, out_ptr, [wa1, wa2], [ws1, ws2]) };
}

#[inline]
pub fn add4(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    adds: &[PieceSquareFeature; 4],
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa1 = weight_ptr(weights, adds[0], perspective, mirror);
    let wa2 = weight_ptr(weights, adds[1], perspective, mirror);
    let wa3 = weight_ptr(weights, adds[2], perspective, mirror);
    let wa4 = weight_ptr(weights, adds[3], perspective, mirror);
    unsafe { update_features::<4, 0>(in_ptr, out_ptr, [wa1, wa2, wa3, wa4], []) };
}

#[inline]
pub fn sub4(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    subs: &[PieceSquareFeature; 4],
    weights: &PieceSquareWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let ws1 = weight_ptr(weights, subs[0], perspective, mirror);
    let ws2 = weight_ptr(weights, subs[1], perspective, mirror);
    let ws3 = weight_ptr(weights, subs[2], perspective, mirror);
    let ws4 = weight_ptr(weights, subs[3], perspective, mirror);
    unsafe { update_features::<0, 4>(in_ptr, out_ptr, [], [ws1, ws2, ws3, ws4]) };
}

#[inline(always)]
pub unsafe fn update_features<const ADDS: usize, const SUBS: usize>(
    input: *const i16,
    output: *mut i16,
    adds: [*const i16; ADDS],
    subs: [*const i16; SUBS],
) {
    let mut i = 0;
    while i + 4 * simd::I16_LANES <= L1_SIZE {
        for k in 0..4 {
            let off = i + k * simd::I16_LANES;
            let mut val = simd::load_i16(input.add(off));
            for &a in adds.iter() {
                val = simd::add_i16(val, simd::load_i16(a.add(off)));
            }
            for &s in subs.iter() {
                val = simd::sub_i16(val, simd::load_i16(s.add(off)));
            }
            simd::store_i16(output.add(off), val);
        }
        i += 4 * simd::I16_LANES;
    }
}

#[inline(always)]
pub fn weight_ptr(
    weights: &PieceSquareWeights,
    feature: PieceSquareFeature,
    perspective: Side,
    mirror: bool,
) -> *const i16 {
    unsafe {
        weights
            .as_ptr()
            .add(feature.index(perspective, mirror) * L1_SIZE)
    }
}