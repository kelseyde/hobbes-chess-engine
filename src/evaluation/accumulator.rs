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
        let mirror = unsafe { *self.mirrored.get_unchecked(perspective as usize) };
        let idx = add.index(perspective, mirror);
        let feats = self.features_mut(perspective);
        let weight_offset = idx * L1_SIZE;

        let mut i = 0;
        while i < L1_SIZE {
            unsafe {
                let feat_ptr = feats.get_unchecked_mut(i);
                let weight = *weights.get_unchecked(i + weight_offset);
                *feat_ptr = feat_ptr.wrapping_add(weight);
            }
            i += 1;
        }
    }

    #[inline]
    pub fn sub(&mut self, sub: Feature, weights: &FeatureWeights, perspective: Side) {
        let mirror = unsafe { *self.mirrored.get_unchecked(perspective as usize) };
        let idx = sub.index(perspective, mirror);
        let feats = self.features_mut(perspective);
        let weight_offset = idx * L1_SIZE;

        let mut i = 0;
        while i < L1_SIZE {
            unsafe {
                let feat_ptr = feats.get_unchecked_mut(i);
                let weight = *weights.get_unchecked(i + weight_offset);
                *feat_ptr = feat_ptr.wrapping_sub(weight);
            }
            i += 1;
        }
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

        let add1_offset = add1.index(perspective, mirror) * L1_SIZE;
        let add2_offset = add2.index(perspective, mirror) * L1_SIZE;
        let add3_offset = add3.index(perspective, mirror) * L1_SIZE;
        let add4_offset = add4.index(perspective, mirror) * L1_SIZE;

        let feats = self.features_mut(perspective);

        let mut i = 0;
        while i < L1_SIZE {
            unsafe {
                let feat_ptr = feats.get_unchecked_mut(i);
                *feat_ptr = feat_ptr
                    .wrapping_add(*weights.get_unchecked(i + add1_offset))
                    .wrapping_add(*weights.get_unchecked(i + add2_offset))
                    .wrapping_add(*weights.get_unchecked(i + add3_offset))
                    .wrapping_add(*weights.get_unchecked(i + add4_offset));
            }
            i += 1;
        }
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

        let sub1_offset = sub1.index(perspective, mirror) * L1_SIZE;
        let sub2_offset = sub2.index(perspective, mirror) * L1_SIZE;
        let sub3_offset = sub3.index(perspective, mirror) * L1_SIZE;
        let sub4_offset = sub4.index(perspective, mirror) * L1_SIZE;

        let feats = self.features_mut(perspective);

        let mut i = 0;
        while i < L1_SIZE {
            unsafe {
                let feat_ptr = feats.get_unchecked_mut(i);
                *feat_ptr = feat_ptr
                    .wrapping_sub(*weights.get_unchecked(i + sub1_offset))
                    .wrapping_sub(*weights.get_unchecked(i + sub2_offset))
                    .wrapping_sub(*weights.get_unchecked(i + sub3_offset))
                    .wrapping_sub(*weights.get_unchecked(i + sub4_offset));
            }
            i += 1;
        }
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
    let w = unsafe {
        weights
            .as_ptr()
            .add(add.index(perspective, mirror) * L1_SIZE)
    };
    unsafe {
        let mut i = 0;
        while i + 4 * simd::I16_LANES <= L1_SIZE {
            simd::store_i16(
                output_features.as_mut_ptr().add(i),
                simd::add_i16(
                    simd::load_i16(input_features.as_ptr().add(i)),
                    simd::load_i16(w.add(i)),
                ),
            );
            simd::store_i16(
                output_features.as_mut_ptr().add(i + simd::I16_LANES),
                simd::add_i16(
                    simd::load_i16(input_features.as_ptr().add(i + simd::I16_LANES)),
                    simd::load_i16(w.add(i + simd::I16_LANES)),
                ),
            );
            simd::store_i16(
                output_features.as_mut_ptr().add(i + 2 * simd::I16_LANES),
                simd::add_i16(
                    simd::load_i16(input_features.as_ptr().add(i + 2 * simd::I16_LANES)),
                    simd::load_i16(w.add(i + 2 * simd::I16_LANES)),
                ),
            );
            simd::store_i16(
                output_features.as_mut_ptr().add(i + 3 * simd::I16_LANES),
                simd::add_i16(
                    simd::load_i16(input_features.as_ptr().add(i + 3 * simd::I16_LANES)),
                    simd::load_i16(w.add(i + 3 * simd::I16_LANES)),
                ),
            );
            i += 4 * simd::I16_LANES;
        }
    }
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
    let w = unsafe {
        weights
            .as_ptr()
            .add(sub.index(perspective, mirror) * L1_SIZE)
    };
    unsafe {
        let mut i = 0;
        while i + 4 * simd::I16_LANES <= L1_SIZE {
            simd::store_i16(
                output_features.as_mut_ptr().add(i),
                simd::sub_i16(
                    simd::load_i16(input_features.as_ptr().add(i)),
                    simd::load_i16(w.add(i)),
                ),
            );
            simd::store_i16(
                output_features.as_mut_ptr().add(i + simd::I16_LANES),
                simd::sub_i16(
                    simd::load_i16(input_features.as_ptr().add(i + simd::I16_LANES)),
                    simd::load_i16(w.add(i + simd::I16_LANES)),
                ),
            );
            simd::store_i16(
                output_features.as_mut_ptr().add(i + 2 * simd::I16_LANES),
                simd::sub_i16(
                    simd::load_i16(input_features.as_ptr().add(i + 2 * simd::I16_LANES)),
                    simd::load_i16(w.add(i + 2 * simd::I16_LANES)),
                ),
            );
            simd::store_i16(
                output_features.as_mut_ptr().add(i + 3 * simd::I16_LANES),
                simd::sub_i16(
                    simd::load_i16(input_features.as_ptr().add(i + 3 * simd::I16_LANES)),
                    simd::load_i16(w.add(i + 3 * simd::I16_LANES)),
                ),
            );
            i += 4 * simd::I16_LANES;
        }
    }
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
    let wa = unsafe {
        weights
            .as_ptr()
            .add(add.index(perspective, mirror) * L1_SIZE)
    };
    let ws = unsafe {
        weights
            .as_ptr()
            .add(sub.index(perspective, mirror) * L1_SIZE)
    };
    unsafe {
        let mut i = 0;
        while i + 4 * simd::I16_LANES <= L1_SIZE {
            for k in 0..4usize {
                let off = i + k * simd::I16_LANES;
                let f = simd::load_i16(input_features.as_ptr().add(off));
                simd::store_i16(
                    output_features.as_mut_ptr().add(off),
                    simd::sub_i16(
                        simd::add_i16(f, simd::load_i16(wa.add(off))),
                        simd::load_i16(ws.add(off)),
                    ),
                );
            }
            i += 4 * simd::I16_LANES;
        }
    }
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
    let wa = unsafe {
        weights
            .as_ptr()
            .add(add.index(perspective, mirror) * L1_SIZE)
    };
    let ws1 = unsafe {
        weights
            .as_ptr()
            .add(sub1.index(perspective, mirror) * L1_SIZE)
    };
    let ws2 = unsafe {
        weights
            .as_ptr()
            .add(sub2.index(perspective, mirror) * L1_SIZE)
    };
    unsafe {
        let mut i = 0;
        while i + 4 * simd::I16_LANES <= L1_SIZE {
            for k in 0..4usize {
                let off = i + k * simd::I16_LANES;
                let f = simd::load_i16(input_features.as_ptr().add(off));
                simd::store_i16(
                    output_features.as_mut_ptr().add(off),
                    simd::sub_i16(
                        simd::sub_i16(
                            simd::add_i16(f, simd::load_i16(wa.add(off))),
                            simd::load_i16(ws1.add(off)),
                        ),
                        simd::load_i16(ws2.add(off)),
                    ),
                );
            }
            i += 4 * simd::I16_LANES;
        }
    }
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
    let wa1 = unsafe {
        weights
            .as_ptr()
            .add(add1.index(perspective, mirror) * L1_SIZE)
    };
    let wa2 = unsafe {
        weights
            .as_ptr()
            .add(add2.index(perspective, mirror) * L1_SIZE)
    };
    let ws1 = unsafe {
        weights
            .as_ptr()
            .add(sub1.index(perspective, mirror) * L1_SIZE)
    };
    let ws2 = unsafe {
        weights
            .as_ptr()
            .add(sub2.index(perspective, mirror) * L1_SIZE)
    };
    unsafe {
        let mut i = 0;
        while i + 4 * simd::I16_LANES <= L1_SIZE {
            for k in 0..4usize {
                let off = i + k * simd::I16_LANES;
                let f = simd::load_i16(input_features.as_ptr().add(off));
                simd::store_i16(
                    output_features.as_mut_ptr().add(off),
                    simd::sub_i16(
                        simd::sub_i16(
                            simd::add_i16(
                                simd::add_i16(f, simd::load_i16(wa1.add(off))),
                                simd::load_i16(wa2.add(off)),
                            ),
                            simd::load_i16(ws1.add(off)),
                        ),
                        simd::load_i16(ws2.add(off)),
                    ),
                );
            }
            i += 4 * simd::I16_LANES;
        }
    }
}