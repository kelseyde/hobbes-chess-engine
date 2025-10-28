use std::time::Duration;
use crate::search::parameters::{node_tm_base, node_tm_scale, time_base_mult, time_hard_mult, time_inc_mult, time_soft_mult};

pub struct SearchLimits {
    pub hard_time: Option<Duration>,
    pub soft_time: Option<Duration>,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub depth: Option<u64>,
}

pub enum LimitType {
    Soft,
    Hard,
}

pub type FischerTime = (u64, u64);

impl SearchLimits {
    pub fn new(
        fischer: Option<FischerTime>,
        movetime: Option<u64>,
        soft_nodes: Option<u64>,
        hard_nodes: Option<u64>,
        depth: Option<u64>,
    ) -> SearchLimits {
        let (soft_time, hard_time) = match (fischer, movetime) {
            (Some(f), _) => {
                let (soft, hard) = Self::calc_time_limits(f);
                (Some(soft), Some(hard))
            }
            (None, Some(mt)) => {
                let duration = Duration::from_millis(mt);
                (Some(duration), Some(duration))
            }
            (None, None) => (None, None),
        };

        SearchLimits {
            hard_time,
            soft_time,
            soft_nodes,
            hard_nodes,
            depth,
        }
    }

    pub fn scaled_soft_limit(
        &self,
        depth: i32,
        nodes: u64,
        best_move_nodes: u64,
    ) -> Option<Duration> {
        self.soft_time.map(|soft_time| {
            let scaled =
                soft_time.as_secs_f32() * self.node_tm_scale(depth, nodes, best_move_nodes);
            Duration::from_secs_f32(scaled)
        })
    }

    fn node_tm_scale(&self, depth: i32, nodes: u64, best_move_nodes: u64) -> f32 {
        if depth < 4 || best_move_nodes == 0 {
            return 1.0;
        }
        let fraction = best_move_nodes as f32 / nodes as f32;
        let base = node_tm_base() as f32 / 100.0;
        let scale = node_tm_scale() as f32 / 100.0;
        (base - fraction) * scale
    }

    fn calc_time_limits(fischer: FischerTime) -> (Duration, Duration) {
        let (time, inc) = (fischer.0 as f64, fischer.1 as f64);

        let base_mult = time_base_mult() as f64 / 100.0;
        let inc_mult = time_inc_mult() as f64 / 100.0;
        let base = time * base_mult + inc * inc_mult;

        let soft_mult = time_soft_mult() as f64 / 100.0;
        let hard_mult = time_hard_mult() as f64 / 100.0;
        let soft_time = base * soft_mult;
        let hard_time = base * hard_mult;

        let soft = soft_time.min(time - 50.0);
        let hard = hard_time.min(time - 50.0);
        (
            Duration::from_millis(soft as u64),
            Duration::from_millis(hard as u64),
        )
    }
}
