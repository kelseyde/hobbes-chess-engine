use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::evaluation::feature::Feature;
use crate::evaluation::{simd, NETWORK};
use hobbes_nnue_arch::{FeatureWeights, L1_SIZE};

/// The `Accumulator` holds the pre-activations of the first layer of the neural network. The input
/// layer just encodes the positions of pieces on the board, from the perspective of both sides.
/// The accumulator is updated incrementally when a move is made or unmade during search, so that we
/// can efficiently compute the evaluation without needing to recompute the board state each time.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub sides: [AccumulatorSide; 2],
    pub update: AccumulatorUpdate,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct AccumulatorSide {
    pub features: [i16; L1_SIZE],
    pub computed: bool,
    pub needs_refresh: bool,
    pub mirrored: bool,
}

impl Default for AccumulatorSide {
    fn default() -> Self {
        AccumulatorSide {
            features: NETWORK.l0_biases,
            computed: false,
            needs_refresh: false,
            mirrored: false,
        }
    }
}

/// A single update to the `Accumulator` caused by a move. A standard move will add one feature (the
/// piece moving to its new square) and remove one feature (the piece leaving its old square).
/// Captures require one extra feature (removing the captured piece), and castling requires two extra
/// features (moving the rook).
#[derive(Clone, Copy, Default)]
pub struct AccumulatorUpdate {
    pub add_count: usize,
    pub sub_count: usize,
    pub adds: [Option<Feature>; 2],
    pub subs: [Option<Feature>; 2],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorUpdateType {
    None,
    Add,
    Sub,
    AddSub,
    AddSubSub,
    AddAddSubSub,
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            sides: [AccumulatorSide::default(), AccumulatorSide::default()],
            update: AccumulatorUpdate::default(),
        }
    }
}

impl AccumulatorUpdate {
    pub fn push_add(&mut self, feature: Feature) {
        if self.add_count < 2 {
            self.adds[self.add_count] = Some(feature);
            self.add_count += 1;
        }
    }

    pub fn push_sub(&mut self, feature: Feature) {
        if self.sub_count < 2 {
            self.subs[self.sub_count] = Some(feature);
            self.sub_count += 1;
        }
    }

    pub fn update_type(&self) -> AccumulatorUpdateType {
        match (self.add_count, self.sub_count) {
            (0, 0) => AccumulatorUpdateType::None,
            (1, 0) => AccumulatorUpdateType::Add,
            (0, 1) => AccumulatorUpdateType::Sub,
            (1, 1) => AccumulatorUpdateType::AddSub,
            (1, 2) => AccumulatorUpdateType::AddSubSub,
            (2, 2) => AccumulatorUpdateType::AddAddSubSub,
            _ => AccumulatorUpdateType::None,
        }
    }
}

impl Accumulator {
    /// Get a reference to the `AccumulatorSide` for the given perspective.
    #[inline(always)]
    pub fn side(&self, perspective: Side) -> &AccumulatorSide {
        &self.sides[perspective]
    }

    /// Get a mutable reference to the `AccumulatorSide` for the given perspective.
    #[inline(always)]
    pub fn side_mut(&mut self, perspective: Side) -> &mut AccumulatorSide {
        &mut self.sides[perspective]
    }

    /// Get a reference to the features for the given perspective.
    #[inline(always)]
    pub fn features(&self, perspective: Side) -> &[i16; L1_SIZE] {
        &self.side(perspective).features
    }

    /// Get a mutable reference to the features for the given perspective.
    #[inline(always)]
    pub fn features_mut(&mut self, perspective: Side) -> &mut [i16; L1_SIZE] {
        &mut self.side_mut(perspective).features
    }

    /// Reset the features for the given perspective to the initial biases.
    #[inline]
    pub fn reset(&mut self, perspective: Side) {
        self.side_mut(perspective).features = NETWORK.l0_biases;
    }

    /// Copy the features from another accumulator into this one, for the given perspective.
    #[inline]
    pub fn copy_from(&mut self, side: Side, features: &[i16; L1_SIZE]) {
        self.side_mut(side).features = *features;
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
}

/// Apply the given update to the accumulator, modifying the output features in place.
#[rustfmt::skip]
pub fn apply_update(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    weights: &FeatureWeights,
    update: &AccumulatorUpdate,
    perspective: Side,
    mirror: bool,
) {
    match update.update_type() {
        AccumulatorUpdateType::None => {}
        AccumulatorUpdateType::Add => {
            let add = unsafe { update.adds[0].unwrap_unchecked() };
            add1(input_features, output_features, add, weights, perspective, mirror);
        }
        AccumulatorUpdateType::Sub => {
            let sub = unsafe { update.subs[0].unwrap_unchecked() };
            sub1(input_features, output_features, sub, weights, perspective, mirror);
        }
        AccumulatorUpdateType::AddSub => {
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            add1_sub1(input_features, output_features, add1, sub1, weights, perspective, mirror);
        }
        AccumulatorUpdateType::AddSubSub => {
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            let sub2 = unsafe { update.subs[1].unwrap_unchecked() };
            add1_sub2(input_features, output_features, add1, sub1, sub2, weights, perspective, mirror);
        }
        AccumulatorUpdateType::AddAddSubSub => {
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            let add2 = unsafe { update.adds[1].unwrap_unchecked() };
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            let sub2 = unsafe { update.subs[1].unwrap_unchecked() };
            add2_sub2(input_features, output_features, add1, add2, sub1, sub2, weights, perspective, mirror);
        }
    }
}

#[inline]
pub fn add1(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: Feature,
    weights: &FeatureWeights,
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
    sub: Feature,
    weights: &FeatureWeights,
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
    add: Feature,
    sub: Feature,
    weights: &FeatureWeights,
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
    add: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
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
    add1: Feature,
    add2: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
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
    adds: [Feature; 4],
    weights: &FeatureWeights,
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
    subs: [Feature; 4],
    weights: &FeatureWeights,
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
            for a in 0..ADDS {
                val = simd::add_i16(val, simd::load_i16(adds[a].add(off)));
            }
            for s in 0..SUBS {
                val = simd::sub_i16(val, simd::load_i16(subs[s].add(off)));
            }
            simd::store_i16(output.add(off), val);
        }
        i += 4 * simd::I16_LANES;
    }
}

#[inline(always)]
pub fn weight_ptr(
    weights: &FeatureWeights,
    feature: Feature,
    perspective: Side,
    mirror: bool,
) -> *const i16 {
    unsafe {
        weights
            .as_ptr()
            .add(feature.index(perspective, mirror) * L1_SIZE)
    }
}
