use crate::evaluation::feature::Feature;
use crate::evaluation::network::{FeatureWeights, HIDDEN, NETWORK};
use crate::types::side::Side;
use crate::types::side::Side::{Black, White};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: [i16; HIDDEN],
    pub black_features: [i16; HIDDEN],
    pub mirrored: [bool; 2]
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
    pub fn features(&self, perspective: Side) -> &[i16; HIDDEN] {
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
    pub fn copy_from(&mut self, side: Side, features: &[i16; HIDDEN]) {
        match side {
            White => self.white_features = *features,
            Black => self.black_features = *features,
        }
    }

    #[inline(always)]
    fn features_mut(&mut self, perspective: Side) -> &mut [i16; HIDDEN] {
        match perspective {
            White => &mut self.white_features,
            Black => &mut self.black_features,
        }
    }

    #[inline]
    pub fn add(&mut self, add: Feature, weights: &FeatureWeights, perspective: Side) {
        let mirror = unsafe { *self.mirrored.get_unchecked(perspective as usize) };
        let idx = add.index(perspective, mirror);
        let feats = self.features_mut(perspective);
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
    pub fn sub(&mut self, sub: Feature, weights: &FeatureWeights, perspective: Side) {
        let mirror = unsafe { *self.mirrored.get_unchecked(perspective as usize) };
        let idx = sub.index(perspective, mirror);
        let feats = self.features_mut(perspective);
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
    pub fn add_sub(&mut self,
                   add: Feature,
                   sub: Feature,
                   w_weights: &FeatureWeights,
                   b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[0];
        let b_mirror = self.mirrored[1];

        let w_add_offset = add.index(White, w_mirror) * HIDDEN;
        let b_add_offset = add.index(Black, b_mirror) * HIDDEN;
        let w_sub_offset = sub.index(White, w_mirror) * HIDDEN;
        let b_sub_offset = sub.index(Black, b_mirror) * HIDDEN;

        let mut i = 0;
        while i < HIDDEN {
            unsafe {
                let w_feat_ptr = self.white_features.get_unchecked_mut(i);
                *w_feat_ptr = w_feat_ptr
                    .wrapping_add(*w_weights.get_unchecked(i + w_add_offset))
                    .wrapping_sub(*w_weights.get_unchecked(i + w_sub_offset));

                let b_feat_ptr = self.black_features.get_unchecked_mut(i);
                *b_feat_ptr = b_feat_ptr
                    .wrapping_add(*b_weights.get_unchecked(i + b_add_offset))
                    .wrapping_sub(*b_weights.get_unchecked(i + b_sub_offset));
            }
            i += 1;
        }
    }

    #[inline]
    pub fn add_sub_sub(&mut self,
                       add: Feature,
                       sub1: Feature,
                       sub2: Feature,
                       w_weights: &FeatureWeights,
                       b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[0];
        let b_mirror = self.mirrored[1];

        let w_add_offset = add.index(White, w_mirror) * HIDDEN;
        let b_add_offset = add.index(Black, b_mirror) * HIDDEN;
        let w_sub1_offset = sub1.index(White, w_mirror) * HIDDEN;
        let b_sub1_offset = sub1.index(Black, b_mirror) * HIDDEN;
        let w_sub2_offset = sub2.index(White, w_mirror) * HIDDEN;
        let b_sub2_offset = sub2.index(Black, b_mirror) * HIDDEN;

        let mut i = 0;
        while i < HIDDEN {
            unsafe {
                let w_feat_ptr = self.white_features.get_unchecked_mut(i);
                *w_feat_ptr = w_feat_ptr
                    .wrapping_add(*w_weights.get_unchecked(i + w_add_offset))
                    .wrapping_sub(*w_weights.get_unchecked(i + w_sub1_offset))
                    .wrapping_sub(*w_weights.get_unchecked(i + w_sub2_offset));

                let b_feat_ptr = self.black_features.get_unchecked_mut(i);
                *b_feat_ptr = b_feat_ptr
                    .wrapping_add(*b_weights.get_unchecked(i + b_add_offset))
                    .wrapping_sub(*b_weights.get_unchecked(i + b_sub1_offset))
                    .wrapping_sub(*b_weights.get_unchecked(i + b_sub2_offset));
            }
            i += 1;
        }
    }

    #[inline]
    pub fn add_add_sub_sub(&mut self,
                           add1: Feature,
                           add2: Feature,
                           sub1: Feature,
                           sub2: Feature,
                           w_weights: &FeatureWeights,
                           b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[0];
        let b_mirror = self.mirrored[1];

        let w_add1_offset = add1.index(White, w_mirror) * HIDDEN;
        let b_add1_offset = add1.index(Black, b_mirror) * HIDDEN;
        let w_add2_offset = add2.index(White, w_mirror) * HIDDEN;
        let b_add2_offset = add2.index(Black, b_mirror) * HIDDEN;
        let w_sub1_offset = sub1.index(White, w_mirror) * HIDDEN;
        let b_sub1_offset = sub1.index(Black, b_mirror) * HIDDEN;
        let w_sub2_offset = sub2.index(White, w_mirror) * HIDDEN;
        let b_sub2_offset = sub2.index(Black, b_mirror) * HIDDEN;

        let mut i = 0;
        while i < HIDDEN {
            unsafe {
                let w_feat_ptr = self.white_features.get_unchecked_mut(i);
                *w_feat_ptr = w_feat_ptr
                    .wrapping_add(*w_weights.get_unchecked(i + w_add1_offset))
                    .wrapping_add(*w_weights.get_unchecked(i + w_add2_offset))
                    .wrapping_sub(*w_weights.get_unchecked(i + w_sub1_offset))
                    .wrapping_sub(*w_weights.get_unchecked(i + w_sub2_offset));

                let b_feat_ptr = self.black_features.get_unchecked_mut(i);
                *b_feat_ptr = b_feat_ptr
                    .wrapping_add(*b_weights.get_unchecked(i + b_add1_offset))
                    .wrapping_add(*b_weights.get_unchecked(i + b_add2_offset))
                    .wrapping_sub(*b_weights.get_unchecked(i + b_sub1_offset))
                    .wrapping_sub(*b_weights.get_unchecked(i + b_sub2_offset));
            }
            i += 1;
        }
    }

}