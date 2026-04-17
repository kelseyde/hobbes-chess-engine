use crate::board::side::Side;
use crate::board::side::Side::{Black, White};
use crate::evaluation::feature::Feature;
use crate::evaluation::{simd, NETWORK};
use hobbes_nnue_arch::{FeatureWeights, L1_SIZE};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: [i16; L1_SIZE],
    pub black_features: [i16; L1_SIZE],
    pub update: AccumulatorUpdate,
    pub computed: [bool; 2],
    pub needs_refresh: [bool; 2],
    pub mirrored: [bool; 2],
}

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
            white_features: NETWORK.l0_biases,
            black_features: NETWORK.l0_biases,
            update: AccumulatorUpdate::default(),
            computed: [false, false],
            needs_refresh: [false, false],
            mirrored: [false, false],
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
    pub fn features(&self, perspective: Side) -> &[i16; L1_SIZE] {
        match perspective {
            White => &self.white_features,
            Black => &self.black_features,
        }
    }

    #[inline(always)]
    pub fn features_mut(&mut self, perspective: Side) -> &mut [i16; L1_SIZE] {
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
        *feats = NETWORK.l0_biases;
    }

    #[inline]
    pub fn copy_from(&mut self, side: Side, features: &[i16; L1_SIZE]) {
        match side {
            White => self.white_features = *features,
            Black => self.black_features = *features,
        }
    }

    #[inline]
    pub fn add(&mut self, add: Feature, weights: &FeatureWeights, perspective: Side) {
        let mirror = self.mirrored[perspective as usize];
        let feats  = self.features_mut(perspective).as_mut_ptr();
        let wa     = unsafe { weights.as_ptr().add(add.index(perspective, mirror) * L1_SIZE) };
        unsafe { update_features::<1, 0>(feats, feats, [wa], []) };
    }

    #[inline]
    pub fn sub(&mut self, sub: Feature, weights: &FeatureWeights, perspective: Side) {
        let mirror = self.mirrored[perspective as usize];
        let feats  = self.features_mut(perspective).as_mut_ptr();
        let ws     = unsafe { weights.as_ptr().add(sub.index(perspective, mirror) * L1_SIZE) };
        unsafe { update_features::<0, 1>(feats, feats, [], [ws]) };
    }

    #[inline]
    pub fn add_add_add_add(
        &mut self,
        add1: Feature,
        add2: Feature,
        add3: Feature,
        add4: Feature,
        weights: &FeatureWeights,
        perspective: Side,
    ) {
        let mirror = self.mirrored[perspective as usize];
        let feats  = self.features_mut(perspective).as_mut_ptr();
        let wa1    = unsafe { weights.as_ptr().add(add1.index(perspective, mirror) * L1_SIZE) };
        let wa2    = unsafe { weights.as_ptr().add(add2.index(perspective, mirror) * L1_SIZE) };
        let wa3    = unsafe { weights.as_ptr().add(add3.index(perspective, mirror) * L1_SIZE) };
        let wa4    = unsafe { weights.as_ptr().add(add4.index(perspective, mirror) * L1_SIZE) };
        unsafe { update_features::<4, 0>(feats, feats, [wa1, wa2, wa3, wa4], []) };
    }

    #[inline]
    pub fn sub_sub_sub_sub(
        &mut self,
        sub1: Feature,
        sub2: Feature,
        sub3: Feature,
        sub4: Feature,
        weights: &FeatureWeights,
        perspective: Side,
    ) {
        let mirror = self.mirrored[perspective as usize];
        let feats  = self.features_mut(perspective).as_mut_ptr();
        let ws1    = unsafe { weights.as_ptr().add(sub1.index(perspective, mirror) * L1_SIZE) };
        let ws2    = unsafe { weights.as_ptr().add(sub2.index(perspective, mirror) * L1_SIZE) };
        let ws3    = unsafe { weights.as_ptr().add(sub3.index(perspective, mirror) * L1_SIZE) };
        let ws4    = unsafe { weights.as_ptr().add(sub4.index(perspective, mirror) * L1_SIZE) };
        unsafe { update_features::<0, 4>(feats, feats, [], [ws1, ws2, ws3, ws4]) };
    }
}

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
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            add(input_features, output_features, add1, weights, perspective, mirror);
        }
        AccumulatorUpdateType::Sub => {
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            sub(input_features, output_features, sub1, weights, perspective, mirror);
        }
        AccumulatorUpdateType::AddSub => {
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            add_sub(input_features, output_features, add1, sub1, weights, perspective, mirror);
        }
        AccumulatorUpdateType::AddSubSub => {
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            let sub2 = unsafe { update.subs[1].unwrap_unchecked() };
            add_sub_sub(input_features, output_features, add1, sub1, sub2, weights, perspective, mirror);
        }
        AccumulatorUpdateType::AddAddSubSub => {
            let add1 = unsafe { update.adds[0].unwrap_unchecked() };
            let add2 = unsafe { update.adds[1].unwrap_unchecked() };
            let sub1 = unsafe { update.subs[0].unwrap_unchecked() };
            let sub2 = unsafe { update.subs[1].unwrap_unchecked() };
            add_add_sub_sub(input_features, output_features, add1, add2, sub1, sub2, weights, perspective, mirror);
        }
    }
}

#[inline]
pub fn add(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr  = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa      = weight_ptr(weights, add, perspective, mirror);
    unsafe { update_features::<1, 0>(in_ptr, out_ptr, [wa], []) };
}

#[inline]
pub fn sub(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    sub: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr  = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let ws      = weight_ptr(weights, sub, perspective, mirror);
    unsafe { update_features::<0, 1>(in_ptr, out_ptr, [], [ws]) };
}

#[inline]
pub fn add_sub(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: Feature,
    sub: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr  = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa      = weight_ptr(weights, add, perspective, mirror);
    let ws      = weight_ptr(weights, sub, perspective, mirror);
    unsafe { update_features::<1, 1>(in_ptr, out_ptr, [wa], [ws]) };
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn add_sub_sub(
    input_features: &[i16; L1_SIZE],
    output_features: &mut [i16; L1_SIZE],
    add: Feature,
    sub1: Feature,
    sub2: Feature,
    weights: &FeatureWeights,
    perspective: Side,
    mirror: bool,
) {
    let in_ptr  = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa      = weight_ptr(weights, add,  perspective, mirror);
    let ws1     = weight_ptr(weights, sub1, perspective, mirror);
    let ws2     = weight_ptr(weights, sub2, perspective, mirror);
    unsafe { update_features::<1, 2>(in_ptr, out_ptr, [wa], [ws1, ws2]) };
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn add_add_sub_sub(
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
    let in_ptr  = input_features.as_ptr();
    let out_ptr = output_features.as_mut_ptr();
    let wa1     = weight_ptr(weights, add1, perspective, mirror);
    let wa2     = weight_ptr(weights, add2, perspective, mirror);
    let ws1     = weight_ptr(weights, sub1, perspective, mirror);
    let ws2     = weight_ptr(weights, sub2, perspective, mirror);
    unsafe { update_features::<2, 2>(in_ptr, out_ptr, [wa1, wa2], [ws1, ws2]) };
}

#[inline(always)]
unsafe fn update_features<const ADDS: usize, const SUBS: usize>(
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
fn weight_ptr(weights: &FeatureWeights, feature: Feature, perspective: Side, mirror: bool) -> *const i16 {
    unsafe { weights.as_ptr().add(feature.index(perspective, mirror) * L1_SIZE) }
}
