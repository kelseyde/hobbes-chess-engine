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

    pub fn features(&self, perspective: Side) -> &[i16; HIDDEN] {
        if perspective == White {
            &self.white_features
        } else {
            &self.black_features
        }
    }

    pub fn reset(&mut self, perspective: Side) {
        let feats = if perspective == White { &mut self.white_features } else { &mut self.black_features };
        *feats = NETWORK.feature_bias;
    }

    pub fn copy_from(&mut self, side: Side, features: &[i16; HIDDEN]) {
        if side == White {
            self.white_features = *features;
        } else {
            self.black_features = *features;
        }
    }

    pub fn add(&mut self,
               add: Feature,
               weights: &FeatureWeights,
               perspective: Side) {

        let mirror = self.mirrored[perspective as usize];
        let idx = add.index(perspective, mirror);
        let feats = if perspective == White { &mut self.white_features } else { &mut self.black_features };

        for i in 0..feats.len() {
            feats[i] = feats[i].wrapping_add(weights[i + idx * HIDDEN]);
        }
    }

    pub fn sub(&mut self,
               add: Feature,
               weights: &FeatureWeights,
               perspective: Side) {

        let mirror = self.mirrored[perspective as usize];
        let idx = add.index(perspective, mirror);
        let feats = if perspective == White { &mut self.white_features } else { &mut self.black_features };

        for i in 0..feats.len() {
            feats[i] = feats[i].wrapping_sub(weights[i + idx * HIDDEN]);
        }
    }

    pub fn add_sub(&mut self,
                   add: Feature,
                   sub: Feature,
                   w_weights: &FeatureWeights,
                   b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[White];
        let b_mirror = self.mirrored[Black];

        let w_idx_1 = add.index(White, w_mirror);
        let b_idx_1 = add.index(Black, b_mirror);

        let w_idx_2 = sub.index(White, w_mirror);
        let b_idx_2 = sub.index(Black, b_mirror);

        for i in 0..self.white_features.len() {
            self.white_features[i] = self.white_features[i]
                .wrapping_add(w_weights[i + w_idx_1 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_2 * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] = self.black_features[i]
                .wrapping_add(b_weights[i + b_idx_1 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_2 * HIDDEN]);
        }
    }

    pub fn add_sub_sub(&mut self,
                       add: Feature,
                       sub1: Feature,
                       sub2: Feature,
                       w_weights: &FeatureWeights,
                       b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[White];
        let b_mirror = self.mirrored[Black];

        let w_idx_1 = add.index(White, w_mirror);
        let b_idx_1 = add.index(Black, b_mirror);

        let w_idx_2 = sub1.index(White, w_mirror);
        let b_idx_2 = sub1.index(Black, b_mirror);

        let w_idx_3 = sub2.index(White, w_mirror);
        let b_idx_3 = sub2.index(Black, b_mirror);

        for i in 0..self.white_features.len() {
            self.white_features[i] = self.white_features[i]
                .wrapping_add(w_weights[i + w_idx_1 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_2 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_3 * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] = self.black_features[i]
                .wrapping_add(b_weights[i + b_idx_1 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_2 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_3 * HIDDEN]);
        }
    }

    pub fn add_add_sub_sub(&mut self,
                           add1: Feature,
                           add2: Feature,
                           sub1: Feature,
                           sub2: Feature,
                           w_weights: &FeatureWeights,
                           b_weights: &FeatureWeights) {

        let w_mirror = self.mirrored[White];
        let b_mirror = self.mirrored[Black];

        let w_idx_1 = add1.index(White, w_mirror);
        let b_idx_1 = add1.index(Black, b_mirror);

        let w_idx_2 = add2.index(White, w_mirror);
        let b_idx_2 = add2.index(Black, b_mirror);

        let w_idx_3 = sub1.index(White, w_mirror);
        let b_idx_3 = sub1.index(Black, b_mirror);

        let w_idx_4 = sub2.index(White, w_mirror);
        let b_idx_4 = sub2.index(Black, b_mirror);

        for i in 0..self.white_features.len() {
            self.white_features[i] = self.white_features[i]
                .wrapping_add(w_weights[i + w_idx_1 * HIDDEN])
                .wrapping_add(w_weights[i + w_idx_2 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_3 * HIDDEN])
                .wrapping_sub(w_weights[i + w_idx_4 * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] = self.black_features[i]
                .wrapping_add(b_weights[i + b_idx_1 * HIDDEN])
                .wrapping_add(b_weights[i + b_idx_2 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_3 * HIDDEN])
                .wrapping_sub(b_weights[i + b_idx_4 * HIDDEN]);
        }
    }
}