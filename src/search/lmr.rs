use crate::search::parameters::{
    lmr_factor_1, lmr_factor_2, lmr_factor_3, lmr_noisy_base, lmr_noisy_div, lmr_quiet_base,
    lmr_quiet_div,
};
use crate::search::MAX_PLY;
use crate::tools::utils::{boxed_and_zeroed, convolve};

// The number of LMR features included in the factorised LMR table.
pub const LMR_QUANTISATION: i32 = 1024;
pub const LMR_FEATURES: usize = 9;
pub const LMR_COMBINATIONS: usize = 1 << LMR_FEATURES;

pub struct LmrTable {
    base: Box<[[[i32; 2]; 64]; 256]>,
    factorised: [i32; LMR_COMBINATIONS],
}

impl LmrTable {
    pub fn init(&mut self) {
        self.init_base();
        self.init_factorised();
    }

    /// Initialise the base LMR table, which provides a lookup for the base reduction for a given
    /// depth, move count, and whether the move is quiet/noisy.
    #[inline(always)]
    fn init_base(&mut self) {
        for depth in 1..MAX_PLY {
            for move_count in 1..64 {
                for is_quiet in [true, false] {
                    let base = if is_quiet {
                        lmr_quiet_base() as f32 / 100.0
                    } else {
                        lmr_noisy_base() as f32 / 100.0
                    };
                    let divisor = if is_quiet {
                        lmr_quiet_div() as f32 / 100.0
                    } else {
                        lmr_noisy_div() as f32 / 100.0
                    };
                    let ln_depth = (depth as f32).ln();
                    let ln_move_count = (move_count as f32).ln();
                    let reduction = ((base + ln_depth * ln_move_count / divisor)
                        * LMR_QUANTISATION as f32) as i32;
                    self.base[depth as usize][move_count as usize][is_quiet as usize] = reduction;
                }
            }
        }
    }

    /// Initialise the factorised LMR table, which provides a lookup for the total reduction for
    /// all features included in the factorised feature set.
    #[inline(always)]
    fn init_factorised(&mut self) {
        for index in 0..LMR_COMBINATIONS {
            let mut features = [false; LMR_FEATURES];
            for (i, feature) in features.iter_mut().enumerate() {
                *feature = index & (1 << i) != 0;
            }
            self.factorised[index] =
                convolve::<LMR_FEATURES>(features, lmr_factor_1, lmr_factor_2, lmr_factor_3);
        }
    }

    /// Get the base LMR reduction for a given depth, movecount, and quiet/noisy bool.
    #[inline(always)]
    pub fn base(&self, depth: i32, move_count: i32, is_quiet: bool) -> i32 {
        self.base[depth.min(255) as usize][move_count.min(63) as usize][is_quiet as usize]
    }

    /// Get the total factorised LMR reduction for the given set of features.
    #[inline(always)]
    pub fn factorised(&self, features: [bool; LMR_FEATURES]) -> i32 {
        let index = self.factorised_index(features);
        self.factorised[index]
    }

    #[inline(always)]
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
                factorised: [0; LMR_COMBINATIONS],
            }
        }
    }
}
