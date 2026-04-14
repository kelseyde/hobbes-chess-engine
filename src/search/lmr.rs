use crate::search::parameters::{lmr_noisy_base, lmr_noisy_div, lmr_quiet_base, lmr_quiet_div};
use crate::tools::utils::boxed_and_zeroed;

pub struct LmrTable {
    table: Box<[[[i32; 2]; 64]; 256]>,
}

impl LmrTable {
    pub fn reduction(&self, depth: i32, move_count: i32, is_quiet: bool) -> i32 {
        self.table[depth.min(255) as usize][move_count.min(63) as usize][is_quiet as usize]
    }

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
                    self.table[depth as usize][move_count as usize][is_quiet as usize] = reduction;
                }
            }
        }
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        unsafe {
            Self {
                table: boxed_and_zeroed(),
            }
        }
    }
}

#[rustfmt::skip]
mod reductions {
use crate::search::parameters::*;

macro_rules! lmr_reduction {
    ($fn_name:ident ($base:ident, $depth_mult:ident, $max:ident)) => {
        pub fn $fn_name(condition: bool, depth: i32) -> i32 {
            if !condition {
                return 0;
            }
            let base = $base();
            let scaled = ($depth_mult() * depth * depth) >> 6;
            (base + scaled).min($max())
        }
    };
}

    lmr_reduction!(lmr_ttpv         (lmr_ttpv_base,       lmr_ttpvth_mult,         lmr_ttpv_max));
    lmr_reduction!(lmr_ttpv_score   (lmr_ttpv_score_base, lmr_ttpv_score_mult,     lmr_ttpv_score_max));
    lmr_reduction!(lmr_ttpv_depth   (lmr_ttpv_depth_base, lmr_ttpv_depth_mult,     lmr_ttpv_depth_max));
    lmr_reduction!(lmr_cut_node     (lmr_cut_node_base,   lmr_cut_node_mult,       lmr_cut_node_max));
    lmr_reduction!(lmr_capture      (lmr_capture_base,    lmr_capture_mult,        lmr_capture_max));
    lmr_reduction!(lmr_improving    (lmr_improving_base,  lmr_improving_mult,      lmr_improving_max));
    lmr_reduction!(lmr_killer       (lmr_killer_base,     lmr_killer_mult,         lmr_killer_max));
    lmr_reduction!(lmr_quiet_see    (lmr_quiet_see_base,  lmr_quiet_see_mult,      lmr_quiet_see_max));
}
pub use reductions::*;
