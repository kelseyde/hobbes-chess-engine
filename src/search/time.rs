use std::time::Duration;

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
        fm_clock: usize
    ) -> SearchLimits {
        let (soft_time, hard_time) = match (fischer, movetime) {
            (Some(f), _) => {
                let (soft, hard) = Self::calc_time_limits(f, fm_clock);
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

    const fn node_tm_scale(&self, depth: i32, nodes: u64, best_move_nodes: u64) -> f32 {
        if depth < 4 || best_move_nodes == 0 {
            return 1.0;
        }
        let fraction = best_move_nodes as f32 / nodes as f32;
        (1.5 - fraction) * 1.35
    }

    fn calc_time_limits(fischer: FischerTime, fm_clock: usize) -> (Duration, Duration) {
        let (time, inc) = (fischer.0, fischer.1);
        // Credit to Reckless for this formula for calculating the hard/soft bounds
        let soft_scale = 0.024 + 0.042 * (1.0 - (-0.045 * fm_clock as f64).exp());
        let hard_scale = 0.742;
        let soft_bound = (soft_scale * time.saturating_sub(50) as f64 + 0.75 * inc as f64) as u64;
        let hard_bound = (hard_scale * time.saturating_sub(50) as f64 + 0.75 * inc as f64) as u64;
        (
            Duration::from_millis(soft_bound),
            Duration::from_millis(hard_bound),
        )
    }
}
