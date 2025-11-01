use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::evaluation::feature::Feature;
use crate::evaluation::network::{Align64, Block, FeatureWeights, HIDDEN, NETWORK};
use crate::evaluation::update::{AccumulatorUpdate, AccumulatorUpdateType};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: Align64<Block>,
    pub black_features: Align64<Block>,
    pub mirrored: [bool; 2],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            white_features: NETWORK.feature_bias,
            black_features: NETWORK.feature_bias,
            mirrored: [false, false],
        }
    }
}

impl Accumulator {
    #[inline(always)]
    pub fn features(&self, perspective: Side) -> &Align64<Block> {
        match perspective {
            White => &self.white_features,
            Black => &self.black_features,
        }
    }

    #[inline]
    pub fn reset(&mut self, perspective: Side) {
        let feats = match perspective {
            White => &mut self.white_features,
            Black => &mut self.black_features,
        };
        *feats = NETWORK.feature_bias;
    }

    #[inline]
    pub fn copy_from(&mut self, side: Side, features: &Align64<Block>) {
        match side {
            White => self.white_features = *features,
            Black => self.black_features = *features,
        }
    }

    #[inline(always)]
    pub fn features_mut(&mut self, perspective: Side) -> &mut Align64<Block> {
        match perspective {
            White => &mut self.white_features,
            Black => &mut self.black_features,
        }
    }
}

pub fn apply_update(
    features: &mut Align64<Block>,
    update: &AccumulatorUpdate,
    weights: &FeatureWeights
) {
    match update.update_type() {
        AccumulatorUpdateType::None => {}
        AccumulatorUpdateType::Add => {
            add(features, weights, update.adds[0]);
        }
        AccumulatorUpdateType::Sub => {
            sub(features, weights, update.subs[0]);
        }
        AccumulatorUpdateType::AddSub => {
            add_sub(features, weights, update.adds[0], update.subs[0]);
        }
        AccumulatorUpdateType::AddSubSub => {
            add_sub_sub(features, weights, update.adds[0], update.subs[0], update.subs[1]);
        }
        AccumulatorUpdateType::AddAddSubSub => {
            add_add_sub_sub(features, weights, update.adds[0], update.adds[1], update.subs[0], update.subs[1]);
        }
    }
}

#[inline]
pub fn add(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    add1: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            let weight = *weights.get_unchecked(i + add1 * HIDDEN);
            *feat_ptr = feat_ptr.wrapping_add(weight);
        }
        i += 1;
    }
}

#[inline]
pub fn sub(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    sub1: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            let weight = *weights.get_unchecked(i + sub1 * HIDDEN);
            *feat_ptr = feat_ptr.wrapping_sub(weight);
        }
        i += 1;
    }
}

#[inline]
pub fn add_sub(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    add1: usize,
    sub1: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub1 * HIDDEN));
        }
        i += 1;
    }
}

#[inline]
pub fn add_sub_sub(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    add1: usize,
    sub1: usize,
    sub2: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub1 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub2 * HIDDEN));
        }
        i += 1;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn add_add_sub_sub(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    add1: usize,
    add2: usize,
    sub1: usize,
    sub2: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1 * HIDDEN))
                .wrapping_add(*weights.get_unchecked(i + add2 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub1 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub2 * HIDDEN));
        }
        i += 1;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn add_add_add_add(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    add1: usize,
    add2: usize,
    add3: usize,
    add4: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1 * HIDDEN))
                .wrapping_add(*weights.get_unchecked(i + add2 * HIDDEN))
                .wrapping_add(*weights.get_unchecked(i + add3 * HIDDEN))
                .wrapping_add(*weights.get_unchecked(i + add4 * HIDDEN));
        }
        i += 1;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn sub_sub_sub_sub(
    features: &mut Align64<Block>,
    weights: &FeatureWeights,
    sub1: usize,
    sub2: usize,
    sub3: usize,
    sub4: usize
) {
    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_sub(*weights.get_unchecked(i + sub1 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub2 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub3 * HIDDEN))
                .wrapping_sub(*weights.get_unchecked(i + sub4 * HIDDEN));
        }
        i += 1;
    }
}
