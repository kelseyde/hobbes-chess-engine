use std::time::Duration;

pub const SOFT_TM_BASE: f64 = 0.024;
pub const SOFT_TM_SCALE: f64 = 0.042;
pub const SOFT_TM_INC_SCALE: f64 = 0.75;
pub const SOFT_TM_FM_SCALE: f64 = 0.045;
pub const HARD_TM_SCALE: f64 = 0.742;
pub const HARD_TM_INC_SCALE: f64 = 0.75;
pub const NODE_TM_BASE: f32 = 1.5;
pub const NODE_TM_SCALE: f32 = 1.35;
pub const BEST_MOVE_TM_BASE: f32 = 1.8;
pub const BEST_MOVE_TM_SCALE: f32 = 0.1;
pub const BEST_MOVE_TM_MIN: f32 = 0.9;
pub const SCORE_TM_BASE: f32 = 1.2;
pub const SCORE_TM_SCALE: f32 = 0.04;
pub const SCORE_TM_MIN: f32 = 0.88;
pub const UCI_OVERHEAD_MS: u64 = 50;

/// The amount of time the engine chooses to search is split into to two limits: hard and soft. The
/// hard limit is checked regularly during search, and the search is aborted as soon as it is reached.
/// The soft limit is checked at the start of each iterative deepening loop, and the engine does not
/// bother starting a new search if it is reached.
///
/// We can then scale the soft limit up or down during search, based on how stable the search is.
/// 'How stable the search is' can be captured a few ways: how many iterations the best move has
/// remained the same, how stable the search score has been across iterations, or what portion of
/// nodes have been spent searching the current best move.
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
        fm_clock: usize,
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

    /// Scale the soft limit based on how stable we think the search is, and therefore how confident
    /// we are that the current best move is likely to be correct.
    pub fn scaled_soft_limit(
        &self,
        depth: i32,
        nodes: u64,
        best_move_nodes: u64,
        best_move_stability: u64,
        score_stability: u64,
    ) -> Option<Duration> {
        self.soft_time.map(|soft_time| {
            let scaled = soft_time.as_secs_f32()
                * Self::node_tm_scale(depth, nodes, best_move_nodes)
                * Self::best_move_stability_scale(best_move_stability)
                * Self::score_stability_scale(score_stability);
            Duration::from_secs_f32(scaled)
        })
    }

    /// 'Node TM': scale the soft limit based on the fraction of nodes that have been spent searching
    /// the current best move. If we've spent a large fraction of our nodes on the current best move,
    /// then we should be more confident in it and spend less time, and vice versa.
    const fn node_tm_scale(depth: i32, nodes: u64, best_move_nodes: u64) -> f32 {
        if depth < 4 || best_move_nodes == 0 {
            return 1.0;
        }
        let fraction = best_move_nodes as f32 / nodes as f32;
        (NODE_TM_BASE - fraction) * NODE_TM_SCALE
    }

    /// 'Best move stability': scale the soft limit based on how many iterations the best move has
    /// remained the same. If the best move has been stable for many iterations, then we should be
    /// more confident in it and spend less time, and vice versa.
    const fn best_move_stability_scale(stability: u64) -> f32 {
        (BEST_MOVE_TM_BASE - BEST_MOVE_TM_SCALE * (stability as f32)).max(BEST_MOVE_TM_MIN)
    }

    /// 'Score stability': scale the soft limit based on how stable the search score has been across
    /// iterations. If the score has been stable for many iterations, then we should be more confident
    /// in the search score and spend less time, and vice versa.
    const fn score_stability_scale(stability: u64) -> f32 {
        (SCORE_TM_BASE - SCORE_TM_SCALE * stability as f32).max(SCORE_TM_MIN)
    }

    fn calc_time_limits(fischer: FischerTime, fm_clock: usize) -> (Duration, Duration) {
        let (time, inc) = (fischer.0, fischer.1 as f64);
        let max_time = time.saturating_sub(UCI_OVERHEAD_MS);
        // Credit to Reckless for this formula for calculating the hard/soft bounds
        let soft_scale =
            SOFT_TM_BASE + SOFT_TM_SCALE * (1.0 - (-SOFT_TM_FM_SCALE * fm_clock as f64).exp());
        let hard_scale = HARD_TM_SCALE;
        let soft_bound = (soft_scale * max_time as f64 + SOFT_TM_INC_SCALE * inc) as u64;
        let hard_bound =
            ((hard_scale * max_time as f64 + HARD_TM_INC_SCALE * inc) as u64).min(max_time);
        (
            Duration::from_millis(soft_bound),
            Duration::from_millis(hard_bound),
        )
    }
}
