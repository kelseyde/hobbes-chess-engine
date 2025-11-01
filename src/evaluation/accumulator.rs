use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::evaluation::feature::Feature;
use crate::evaluation::network::{Align64, Block, FeatureWeights, HIDDEN, NETWORK};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: Align64<Block>,
    pub black_features: Align64<Block>,
    pub mirrored: [bool; 2],
}

#[derive(Clone, Copy)]
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
    AddAddSubSub
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

impl Default for AccumulatorUpdate {
    fn default() -> Self {
        AccumulatorUpdate {
            add_count: 0,
            sub_count: 0,
            adds: [None, None],
            subs: [None, None],
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
    fn features_mut(&mut self, perspective: Side) -> &mut Align64<Block> {
        match perspective {
            White => &mut self.white_features,
            Black => &mut self.black_features,
        }
    }

    pub fn apply_update(
        &mut self,
        update: &AccumulatorUpdate,
        weights: &FeatureWeights,
        perspective: Side) {
        match update.update_type() {
            AccumulatorUpdateType::None => {},
            AccumulatorUpdateType::Add => {
                if let Some(add1) = update.adds[0] {
                    add(self, add1, weights, perspective);
                }
            },
            AccumulatorUpdateType::Sub => {
                if let Some(sub1) = update.subs[0] {
                    sub(self, sub1, weights, perspective);
                }
            },
            AccumulatorUpdateType::AddSub => {
                if let (Some(add1), Some(sub1)) = (update.adds[0], update.subs[0]) {
                    add_sub(self, add1, sub1, weights, perspective);
                }
            },
            AccumulatorUpdateType::AddSubSub => {
                if let (Some(add1), Some(sub1), Some(sub2)) =
                    (update.adds[0], update.subs[0], update.subs[1]) {
                    add_sub_sub(self, add1, sub1, sub2, weights, perspective);
                }
            },
            AccumulatorUpdateType::AddAddSubSub => {
                if let (Some(add1), Some(add2), Some(sub1), Some(sub2)) =
                    (update.adds[0], update.adds[1], update.subs[0], update.subs[1]) {
                    add_add_sub_sub(self, add1, add2, sub1, sub2, weights, perspective);
                }
            },
        }
    }

}

#[inline]
pub fn add(
    acc: &mut Accumulator,
    add: Feature,
    weights: &FeatureWeights,
    perspective: Side
) {
    let mirror = acc.mirrored[perspective];
    let idx = add.index(perspective, mirror);
    let feats = acc.features_mut(perspective);
    let weight_offset = idx * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = feats.get_unchecked_mut(i);
            let weight = *weights.get_unchecked(i + weight_offset);
            *feat_ptr = feat_ptr.wrapping_add(weight);
        }
        i += 1;
    }
}

#[inline]
pub fn sub(
    acc: &mut Accumulator,
    sub: Feature,
    weights: &FeatureWeights,
    perspective: Side
) {
    let mirror = acc.mirrored[perspective];
    let idx = sub.index(perspective, mirror);
    let feats = acc.features_mut(perspective);
    let weight_offset = idx * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = feats.get_unchecked_mut(i);
            let weight = *weights.get_unchecked(i + weight_offset);
            *feat_ptr = feat_ptr.wrapping_sub(weight);
        }
        i += 1;
    }
}

#[inline]
pub fn add_sub(
    acc: &mut Accumulator,
    add: Feature,
    sub: Feature,
    weights: &FeatureWeights,
    perspective: Side,
) {
    let mirror = acc.mirrored[perspective];

    let add_offset = add.index(perspective, mirror) * HIDDEN;
    let sub_offset = sub.index(perspective, mirror) * HIDDEN;

    let features = acc.features_mut(perspective);

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub_offset));

        }
        i += 1;
    }
}

#[inline]
pub fn add_sub_sub(
    acc: &mut Accumulator,
    add: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
    perspective: Side,
) {
    let mirror = acc.mirrored[perspective];

    let add_offset = add.index(perspective, mirror) * HIDDEN;
    let sub1_offset = sub1.index(perspective, mirror) * HIDDEN;
    let sub2_offset = sub2.index(perspective, mirror) * HIDDEN;

    let features = acc.features_mut(perspective);

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub1_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub2_offset));
        }
        i += 1;
    }
}

#[inline]
pub fn add_add_sub_sub(
    acc: &mut Accumulator,
    add1: Feature,
    add2: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
    perspective: Side,
) {
    let mirror = acc.mirrored[perspective];

    let add1_offset = add1.index(perspective, mirror) * HIDDEN;
    let add2_offset = add2.index(perspective, mirror) * HIDDEN;
    let sub1_offset = sub1.index(perspective, mirror) * HIDDEN;
    let sub2_offset = sub2.index(perspective, mirror) * HIDDEN;

    let features = acc.features_mut(perspective);

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1_offset))
                .wrapping_add(*weights.get_unchecked(i + add2_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub1_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub2_offset));
        }
        i += 1;
    }
}

#[inline]
pub fn add_add_add_add(
    acc: &mut Accumulator,
    add1: Feature,
    add2: Feature,
    add3: Feature,
    add4: Feature,
    weights: &FeatureWeights,
    perspective: Side,
) {
    let mirror = acc.mirrored[perspective];

    let add1_offset = add1.index(perspective, mirror) * HIDDEN;
    let add2_offset = add2.index(perspective, mirror) * HIDDEN;
    let add3_offset = add3.index(perspective, mirror) * HIDDEN;
    let add4_offset = add4.index(perspective, mirror) * HIDDEN;

    let features = acc.features_mut(perspective);

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1_offset))
                .wrapping_add(*weights.get_unchecked(i + add2_offset))
                .wrapping_add(*weights.get_unchecked(i + add3_offset))
                .wrapping_add(*weights.get_unchecked(i + add4_offset));
        }
        i += 1;
    }
}

pub fn sub_sub_sub_sub(
    acc: &mut Accumulator,
    sub1: Feature,
    sub2: Feature,
    sub3: Feature,
    sub4: Feature,
    weights: &FeatureWeights,
    perspective: Side,
) {
    let mirror = acc.mirrored[perspective];

    let sub1_offset = sub1.index(perspective, mirror) * HIDDEN;
    let sub2_offset = sub2.index(perspective, mirror) * HIDDEN;
    let sub3_offset = sub3.index(perspective, mirror) * HIDDEN;
    let sub4_offset = sub4.index(perspective, mirror) * HIDDEN;

    let features = acc.features_mut(perspective);

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let feat_ptr = features.get_unchecked_mut(i);
            *feat_ptr = feat_ptr
                .wrapping_sub(*weights.get_unchecked(i + sub1_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub2_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub3_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub4_offset));
        }
        i += 1;
    }
}
