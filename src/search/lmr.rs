use crate::search::parameters::{lmr_factor_1, lmr_factor_2, lmr_factor_3, lmr_noisy_base, lmr_noisy_div, lmr_quiet_base, lmr_quiet_div};
use crate::tools::utils::{boxed_and_zeroed, convolve};

// Number of (unordered) factorised LMR features per order (singular, pairs, triplets)
pub const LMR_FEATURES: usize = 9;
pub const LMR_ORDER_1: usize = LMR_FEATURES;
pub const LMR_ORDER_2: usize = LMR_FEATURES * (LMR_FEATURES - 1) / 2;
pub const LMR_ORDER_3: usize = LMR_FEATURES * (LMR_FEATURES - 1) * (LMR_FEATURES - 2) / 6;
pub const LMR_COMBINATIONS: usize = 1 << LMR_FEATURES;

pub struct LmrTable {
    base: Box<[[[i32; 2]; 64]; 256]>,
    factorised: [i32; LMR_COMBINATIONS]
}

impl LmrTable {

    pub fn init(&mut self) {
        let quiet_base = lmr_quiet_base() as f32 / 100.0;
        let quiet_divisor = lmr_quiet_div() as f32 / 100.0;
        let noisy_base = lmr_noisy_base() as f32 / 100.0;
        let noisy_divisor = lmr_noisy_div() as f32 / 100.0;

        for depth in 1..256 {
            for move_count in 1..64 {
                for is_quiet in [true, false] {
                    let base = if is_quiet { quiet_base } else { noisy_base };
                    let divisor = if is_quiet {
                        quiet_divisor
                    } else {
                        noisy_divisor
                    };
                    let ln_depth = (depth as f32).ln();
                    let ln_move_count = (move_count as f32).ln();
                    let reduction = (base + (ln_depth * ln_move_count / divisor)) as i32;
                    self.base[depth as usize][move_count as usize][is_quiet as usize] = reduction;
                }
            }
        }
        for index in 0..LMR_COMBINATIONS {
            let mut features = [false; LMR_FEATURES];
            for (i, feature) in features.iter_mut().enumerate() {
                *feature = index & (1 << i) != 0;
            }
            self.factorised[index] = convolve::<LMR_FEATURES>(features, lmr_factor_1, lmr_factor_2, lmr_factor_3);
        }
    }

    #[inline(always)]
    pub fn base(&self, depth: i32, move_count: i32, is_quiet: bool) -> i32 {
        self.base[depth.min(255) as usize][move_count.min(63) as usize][is_quiet as usize]
    }

    /// Look up the total factorised LMR adjustment for the given set of active features.
    #[inline(always)]
    pub fn factorised(&self, features: [bool; LMR_FEATURES]) -> i32 {
        let index = self.factorised_index(features);
        self.factorised[index]
    }

    fn factorised_index(&self, features: [bool; LMR_FEATURES]) -> usize {
        features
            .iter()
            .enumerate()
            .fold(0, |index, (i, &feature)| index | ((feature as usize) << i))
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        unsafe {
            Self {
                base: boxed_and_zeroed(),
                factorised: [0; LMR_COMBINATIONS]
            }
        }
    }
}