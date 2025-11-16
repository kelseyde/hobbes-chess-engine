use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::evaluation::feature::Feature;
use crate::evaluation::network::{FeatureWeights, HIDDEN, NETWORK};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: [i16; HIDDEN],
    pub black_features: [i16; HIDDEN],
    pub update: AccumulatorUpdate,
    pub computed: [bool; 2],
    pub needs_refresh: [bool; 2],
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
    AddAddSubSub,
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            white_features: NETWORK.feature_bias,
            black_features: NETWORK.feature_bias,
            update: AccumulatorUpdate::default(),
            computed: [false, false],
            needs_refresh: [false, false],
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
    pub fn features(&self, perspective: Side) -> &[i16; HIDDEN] {
        match perspective {
            White => &self.white_features,
            Black => &self.black_features,
        }
    }


    #[inline(always)]
    pub fn features_mut(&mut self, perspective: Side) -> &mut [i16; HIDDEN] {
        match perspective {
            White => &mut self.white_features,
            Black => &mut self.black_features,
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
    pub fn copy_from(&mut self, side: Side, features: &[i16; HIDDEN]) {
        match side {
            White => self.white_features = *features,
            Black => self.black_features = *features,
        }
    }

}

pub fn apply_update(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    weights: &FeatureWeights,
    update: &AccumulatorUpdate,
    perspective: Side,
    mirror: bool
) {
    match update.update_type() {
        AccumulatorUpdateType::None => {},
        AccumulatorUpdateType::Add => {
            if let Some(add1) = update.adds[0] {
                add(input_features, output_features, add1, weights, perspective, mirror);
            }
        },
        AccumulatorUpdateType::Sub => {
            if let Some(sub1) = update.subs[0] {
                sub(input_features, output_features, sub1, weights, perspective, mirror);
            }
        },
        AccumulatorUpdateType::AddSub => {
            if let (Some(add1), Some(sub1)) = (update.adds[0], update.subs[0]) {
                add_sub(input_features, output_features, add1, sub1, weights, perspective, mirror);
            }
        },
        AccumulatorUpdateType::AddSubSub => {
            if let (Some(add), Some(sub1), Some(sub2)) =
                (update.adds[0], update.subs[0], update.subs[1]) {
                add_sub_sub(input_features, output_features, add, sub1, sub2, weights, perspective, mirror);
            }
        },
        AccumulatorUpdateType::AddAddSubSub => {
            if let (Some(add1), Some(add2), Some(sub1), Some(sub2)) =
                (update.adds[0], update.adds[1], update.subs[0], update.subs[1]) {
                add_add_sub_sub(input_features, output_features, add1, add2, sub1, sub2, weights, perspective, mirror);
            }
        },
    }
}

#[inline]
pub fn add(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    add: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let idx = add.index(perspective, mirror);
    let weight_offset = idx * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            let weight = *weights.get_unchecked(i + weight_offset);
            *out_feat_ptr = in_feat_ptr.wrapping_add(weight);
        }
        i += 1;
    }
}

#[inline]
pub fn sub(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    sub: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let idx = sub.index(perspective, mirror);
    let weight_offset = idx * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            let weight = *weights.get_unchecked(i + weight_offset);
            *out_feat_ptr = in_feat_ptr.wrapping_sub(weight);
        }
        i += 1;
    }
}

#[inline]
pub fn add_sub(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    add: Feature,
    sub: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let add_offset = add.index(perspective, mirror) * HIDDEN;
    let sub_offset = sub.index(perspective, mirror) * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            *out_feat_ptr = in_feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub_offset));
        }
        i += 1;
    }
}

#[inline]
pub fn add_sub_sub(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    add: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let add_offset = add.index(perspective, mirror) * HIDDEN;
    let sub1_offset = sub1.index(perspective, mirror) * HIDDEN;
    let sub2_offset = sub2.index(perspective, mirror) * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            *out_feat_ptr = in_feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub1_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub2_offset));
        }
        i += 1;
    }
}

#[inline]
pub fn add_add_sub_sub(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    add1: Feature,
    add2: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let add1_offset = add1.index(perspective, mirror) * HIDDEN;
    let add2_offset = add2.index(perspective, mirror) * HIDDEN;
    let sub1_offset = sub1.index(perspective, mirror) * HIDDEN;
    let sub2_offset = sub2.index(perspective, mirror) * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            *out_feat_ptr = in_feat_ptr
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
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    add1: Feature,
    add2: Feature,
    add3: Feature,
    add4: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let add1_offset = add1.index(perspective, mirror) * HIDDEN;
    let add2_offset = add2.index(perspective, mirror) * HIDDEN;
    let add3_offset = add3.index(perspective, mirror) * HIDDEN;
    let add4_offset = add4.index(perspective, mirror) * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            *out_feat_ptr = in_feat_ptr
                .wrapping_add(*weights.get_unchecked(i + add1_offset))
                .wrapping_add(*weights.get_unchecked(i + add2_offset))
                .wrapping_add(*weights.get_unchecked(i + add3_offset))
                .wrapping_add(*weights.get_unchecked(i + add4_offset));
        }
        i += 1;
    }
}

pub fn sub_sub_sub_sub(
    input_features: &[i16; HIDDEN],
    output_features: &mut [i16; HIDDEN],
    sub1: Feature,
    sub2: Feature,
    sub3: Feature,
    sub4: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool
) {
    let sub1_offset = sub1.index(perspective, mirror) * HIDDEN;
    let sub2_offset = sub2.index(perspective, mirror) * HIDDEN;
    let sub3_offset = sub3.index(perspective, mirror) * HIDDEN;
    let sub4_offset = sub4.index(perspective, mirror) * HIDDEN;

    let mut i = 0;
    while i < HIDDEN {
        unsafe {
            let in_feat_ptr = input_features.get_unchecked(i);
            let out_feat_ptr = output_features.get_unchecked_mut(i);
            *out_feat_ptr = in_feat_ptr
                .wrapping_sub(*weights.get_unchecked(i + sub1_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub2_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub3_offset))
                .wrapping_sub(*weights.get_unchecked(i + sub4_offset));
        }
        i += 1;
    }
}

