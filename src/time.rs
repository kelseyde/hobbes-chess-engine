use std::time::Duration;

pub struct SearchLimits {
    pub hard_time:  Option<Duration>,
    pub soft_time:  Option<Duration>,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub depth:      Option<u64>,
}

pub enum LimitType { Soft, Hard }

pub type FischerTime = (u64, u64);

impl SearchLimits {

    pub fn new(fischer:    Option<FischerTime>,
               movetime:   Option<u64>,
               soft_nodes: Option<u64>,
               hard_nodes: Option<u64>,
               depth:      Option<u64>) -> SearchLimits {

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

    pub fn scaled_soft_limit(&self, depth: i32, nodes: u64, best_move_nodes: u64) -> Option<Duration> {
        self.soft_time.map(|soft_time| {
            if depth < 4 {
                soft_time
            } else {
                let nodes_fraction = best_move_nodes as f32 / nodes as f32;
                let nodes_factor = 2.15 - 1.5 * nodes_fraction;
                Duration::from_secs_f32(soft_time.as_secs_f32() * nodes_factor)
            }
        })
    }

    fn calc_time_limits(fischer: FischerTime) -> (Duration, Duration) {
        let (time, inc) = (fischer.0 as f64, fischer.1 as f64);
        let base = time * 0.05 + inc * 0.08;
        let soft_time = base * 0.66;
        let hard_time = base * 2.0;
        let soft = soft_time.min(time - 50.0);
        let hard = hard_time.min(time - 50.0);
        (Duration::from_millis(soft as u64), Duration::from_millis(hard as u64))
    }

}